import torch
from dataclasses import dataclass, field
import dataclasses
from typing import Optional, Generic, TypeVar, cast, Any, TypedDict, Union, Mapping, Literal
import numpy as np
from abc import abstractmethod

from ..visualizer import MPLVisualizer
from ..tasks._base import Task, ObjectT, RewriteSpecT, ExtraMetrics
from ..object_collection import ObjectCollection
from ..objects.arclines import Shape, ShapeCollection, PrimitiveHelper, ShapeRewriteSpec
from ..objects.ur import UR, URCollection, URRewrite
from ..third_party.pid import PID
from ..third_party.topopt import SensitivityAnalysisArgs, SensitivityAnalysis
from ..losses._base import LossArgs
from ..util import set_grad


@dataclass
class TopoptArgs(LossArgs):
    sens: SensitivityAnalysisArgs
    occ_blur: float = 0  # float(1 / np.sqrt(2)) / 4 # / 8
    grad_blur: float = 1  # pixel within grad_blur of the boundary will take gradient
    vol_frac: float = 0.5
    enable_pid: bool = True
    pid_kp: float = 1.0
    pid_ki: float = 0.2
    pid_kd: float = 0.0
    warmup_steps: int = 500
    init_setup: Literal["a", "cantileaver", "mbb", "mbb_half"] = "a"


@dataclass
class TopoptState:
    pids: list[PID]
    it: int


class TopoptMixin(Task[ObjectT, RewriteSpecT, TopoptState]):
    init_vol: float
    target_vol: float

    def __init__(self, args: TopoptArgs, lim: tuple[float, float], device: Union[str, torch.device]):
        self.loss_args = args
        self.lim = lim
        size = args.sens.nelx
        assert (
            args.sens.nelx == args.sens.nely
        ), f"Topopt only supports square grids, got {args.sens.nelx} != {args.sens.nely}"
        self.size = size

        # setup sentivity analysis

        if self.loss_args.init_setup == "a":
            # fixed left 1/2 middle edge; 1/16 from left
            cond_y = np.arange(size // 2 - round(size / 4), size // 2 + round(size / 4) + 1)
            cond_x = np.full(len(cond_y), size // 16)
            boundary_cond = np.concat(
                [
                    np.stack([cond_x, cond_y, np.zeros(len(cond_y))], axis=-1),
                    np.stack([cond_x, cond_y, np.ones(len(cond_y))], axis=-1),
                ],
                axis=0,
            )
            # vertical load right 1/8 middle edge; 1/16 from right
            load_y = np.arange(size // 2 - round(size / 16), size // 2 + round(size / 16) + 1)
            load_x = size - np.full(len(load_y), size // 16)
            load = np.stack([load_x, load_y, np.ones(len(load_y)), np.ones(len(load_y))], axis=-1)
            self.init_vol = 7 / 16  # 1 / 2 * 7 / 8
        elif self.loss_args.init_setup == "cantileaver":
            # left 7 / 16 middle edge; 1/16 from left
            cond_y = np.arange(size // 2 - round(size * 7 / 32), size // 2 + round(size * 7 / 32) + 1)
            cond_x = np.full(len(cond_y), size // 16)
            boundary_cond = np.concat(
                [
                    np.stack([cond_x, cond_y, np.zeros(len(cond_y))], axis=-1),
                    np.stack([cond_x, cond_y, np.ones(len(cond_y))], axis=-1),
                ],
                axis=0,
            )
            # vertical load right middle; 1/16 from right
            load_y = np.arange(size // 2, size // 2 + 1)
            load_x = np.full(len(load_y), size - size // 16)
            load = np.stack([load_x, load_y, np.ones(len(load_y)), -np.ones(len(load_y))], axis=-1)
            self.init_vol = 49 / 128  # 7 / 16 * 7 / 8
        elif self.loss_args.init_setup == "mbb":
            # beam 1:6; 1/16 from left and right
            # fix bottom left and bottom right in y-direction
            boundary_cond = np.array(
                [[size // 16, round(size * 41 / 96), 1], [size - size // 16, round(size * 41 / 96), 1]]
            )
            # vertical load top middle; downward
            load = np.array([[size // 2, size - round(size * 41 / 96), 1, -1]])
            self.init_vol = 49 / 384
        elif self.loss_args.init_setup == "mbb_half":
            # beam 1:3; 1/16 from left and right
            # 7 / 8 ; 7 / 24 // 17 / 48
            # fix left edge in x-direction; and bottom right in y-direction
            boundary_cond = np.array(
                [
                    # *[[size // 16, y, 0] for y in range(round(size * 17 / 48), size - round(size * 17 / 48) + 1)],
                    *[[size // 16, y, 0] for y in range(0, size + 1)],
                    [size - size // 16, round(size * 17 / 48), 1],
                ]
            )
            # vertical load top middle; downward
            load = np.array([[size // 16, size - round(size * 17 / 48), 1, -1]])
            self.init_vol = 49 / 192
        else:
            raise ValueError(f"Invalid init_setup: {self.loss_args.init_setup}")

        self.target_vol = self.loss_args.vol_frac * self.init_vol

        self.sens = SensitivityAnalysis(args.sens, boundary_cond, load)
        lim0, lim1 = lim
        self.occ_v = args.occ_blur * (lim1 - lim0) / size
        self.grad_v = args.grad_blur * (lim1 - lim0) / size
        self.total_area = (lim1 - lim0) ** 2

    @abstractmethod
    def initialize_object(self) -> ObjectT: ...

    def initialize_state(self) -> TopoptState:
        return TopoptState(
            pids=[PID(self.loss_args.pid_kp, self.loss_args.pid_ki, self.loss_args.pid_kd, self.target_vol)],
            it=0,
        )

    def update_state_for_proposals(self, state: TopoptState, proposals: ObjectCollection[ObjectT]) -> TopoptState:
        pids = state.pids
        assert len(pids) == 1, f"len(pids) != 1, got {len(pids)} != 1"
        return TopoptState(pids=[pids[0].clone() for _ in range(len(proposals))], it=state.it)

    def step_state(self, state: TopoptState) -> TopoptState:
        return dataclasses.replace(state, it=state.it + 1)

    def _compute_losses(
        self, collection: ObjectCollection[ObjectT], state: TopoptState
    ) -> tuple[torch.Tensor, ExtraMetrics]:
        """
        - imgs: (n_shapes, size, size)
        """
        imgs = collection.render(
            self.render_args.size,
            self.render_args.lim,
            center_pixel=self.render_args.center_pixel,
        )
        pids = state.pids
        assert len(pids) == len(imgs), f"len(pids) != len(imgs), got {len(pids)} != {len(imgs)}"

        vlim = self.occ_v
        if vlim > 0:
            occ = (-(imgs).clamp(-vlim, vlim) + vlim) / (2 * vlim)  # (n_shapes, size, size)
        else:
            occ = (-(imgs) > 0).float()
        occ_wo_offset = occ  # (-(imgs).clamp(-vlim, vlim) + vlim) / (2 * vlim)  # (n_shapes, size, size)
        losses: list[torch.Tensor] = []
        cur_vols: list[float] = []
        set_points: list[float] = []
        for i, img in enumerate(imgs):
            sed, sed_grad = self.sens.get_sensitivity(occ[i])
            cur_vol = occ_wo_offset[i].mean()

            set_point = self.init_vol + (self.target_vol - self.init_vol) * min(
                1, state.it / self.loss_args.warmup_steps
            )

            cur_vols.append(cur_vol.item())
            set_points.append(set_point)

            if self.loss_args.enable_pid:
                lambda_ = pids[i].update(cur_vol.item(), setpoint=set_point)
            else:
                lambda_ = 0.0
            # lambda_ = np.percentile(sed, 100 - (self.target_vol) * 100)
            # pids[i].last_value = float(lambda_)

            # loss_ = (
            #     (self.sens.args.Emin + occ[i] * (self.sens.args.E0 - self.sens.args.Emin)) * sed_mat
            # ).sum() + lambda_ * (cur_vol - set_point)

            grad = sed_grad - lambda_  # (size, size)
            loss_ = set_grad(
                img.clamp(-self.grad_v, self.grad_v),  # (size, size)
                grad,  # (size, size)
                sed.sum() + lambda_ * (cur_vol - set_point),  # ()
            )  # ()

            losses.append(loss_)

        return torch.stack(losses, dim=0), {"vols": tuple(cur_vols), "setpoints": tuple(set_points)}  # (n_shapes,)


class TopoptArclinesMixin(TopoptMixin[Shape, ShapeRewriteSpec]):
    def initialize_object(self) -> Shape:
        lim0, lim1 = self.lim
        xrange = lim1 - lim0
        match self.loss_args.init_setup:
            case "a":
                return Shape.create_rectangle(
                    (lim0 + xrange / 16, lim0 + xrange * 0.75),
                    (lim1 - xrange / 16, lim1 - xrange * 0.75),
                    self.device(),
                )
            case "cantileaver":
                return Shape.create_rectangle(
                    (lim0 + xrange / 16, lim0 + xrange * 9 / 32),
                    (lim1 - xrange / 16, lim1 - xrange * 9 / 32),
                    self.device(),
                )
            case "mbb":
                return Shape.create_rectangle(
                    (lim0 + xrange / 16, lim0 + xrange * 41 / 96),
                    (lim1 - xrange / 16, lim1 - xrange * 41 / 96),
                    self.device(),
                )
            case "mbb_half":
                return Shape.create_rectangle(
                    (lim0 + xrange / 16, lim0 + xrange * 17 / 48),
                    (lim1 - xrange / 16, lim1 - xrange * 17 / 48),
                    self.device(),
                )
            case _:
                raise ValueError(f"Invalid init_setup: {self.loss_args.init_setup}")

    def compute_constraints(self, collection: ObjectCollection[Shape], state: TopoptState) -> torch.Tensor:
        assert isinstance(collection, ShapeCollection)
        return (collection.compute_area() / self.total_area - self.target_vol).abs()

    def visualize(self, collection: ObjectCollection[Shape], step: int, loss: float, state: TopoptState) -> np.ndarray:
        assert isinstance(collection, ShapeCollection)
        assert len(collection) == 1
        shape = collection[0]
        fig = MPLVisualizer(1, 1, 10.8, 10.8, xlim=self.render_args.lim, ylim=self.render_args.lim, notebook=False)
        lambda_ = state.pids[0].last_value or 0.0
        img = collection.render(
            self.render_args.size,
            self.render_args.lim,
            center_pixel=self.render_args.center_pixel,
        )[0]

        area = sum(x[1] for x in shape.find_loops())

        vlim = self.occ_v
        if vlim > 0:
            occ = (-img.clamp(-vlim, vlim) + vlim) / (2 * vlim)  # (n_shapes, size, size)
        else:
            occ = (-img > 0).float()
        sed, sed_grad = self.sens.get_sensitivity(occ)  # (size, size)
        ax = fig[0]
        ax.ax.set_title(
            f"{self.get_elapsed_time():.0f}s: {shape.id}: {loss:.2e}: L{len(collection.shapes[0].line_idx)}A{len(collection.shapes[0].arcs_idx)}"
            f"\n$\\lambda$: {lambda_:.2f} occ: {occ.mean():.3f}/{state.pids[0].setpoint:.3f}"
            f"\narea: {area:.2f}/{self.total_area * self.target_vol:.2f}"
        )
        ax.visualize_shape(collection[0])
        # ax.ax.imshow(
        #     img.clamp(-self.grad_v, self.grad_v).detach().cpu().numpy(),
        #     extent=(self.lim[0], self.lim[1], self.lim[1], self.lim[0]),
        #     cmap="RdBu",
        #     vmin=-1,
        #     vmax=1,
        #     # alpha=0.2,
        # )
        # ax.ax.imshow(
        #     occ.detach().cpu().numpy(),
        #     extent=(self.lim[0], self.lim[1], self.lim[1], self.lim[0]),
        #     cmap="YlOrRd",
        #     # alpha=0.2,
        # )
        lim = min(sed_grad.abs().max().item(), 3)
        ax.ax.imshow(
            sed_grad.detach().cpu().numpy(),  # - lambda_,
            extent=(self.lim[0], self.lim[1], self.lim[1], self.lim[0]),
            cmap="RdBu",
            vmin=-lim,  # np.abs(grad).max(),
            vmax=lim,  # np.abs(grad).max(),
            # alpha=0.2,
        )
        return fig.get_image()


class TopoptURMixin(TopoptMixin[UR, URRewrite]):
    def initialize_object(self) -> UR:
        lim0, lim1 = self.lim
        xrange = lim1 - lim0
        device = self.device()
        match self.loss_args.init_setup:
            case "a":
                return UR(
                    xs=torch.tensor([[0.0, 0.0]], device=device),
                    sizes=torch.tensor([[xrange * 7 / 16, xrange / 4]], device=device),
                    rots=torch.tensor([0.0], device=device),
                    is_subs=torch.tensor([False], device=device),
                )
            case "cantileaver":
                return UR(
                    xs=torch.tensor([[0.0, 0.0]], device=device),
                    sizes=torch.tensor([[xrange * 7 / 16, xrange * 7 / 32]], device=device),
                    rots=torch.tensor([0.0], device=device),
                    is_subs=torch.tensor([False], device=device),
                )
            case "mbb":
                return UR(
                    xs=torch.tensor([[0.0, 0.0]], device=device),
                    sizes=torch.tensor([[xrange * 7 / 16, xrange * 7 / 96]], device=device),
                    rots=torch.tensor([0.0], device=device),
                    is_subs=torch.tensor([False], device=device),
                )
            case "mbb_half":
                return UR(
                    xs=torch.tensor([[0.0, 0.0]], device=device),
                    sizes=torch.tensor([[xrange * 7 / 16, xrange * 7 / 48]], device=device),
                    rots=torch.tensor([0.0], device=device),
                    is_subs=torch.tensor([False], device=device),
                )
            case _:
                raise ValueError(f"Invalid init_setup: {self.loss_args.init_setup}")

    def compute_constraints(self, collection: ObjectCollection[UR], state: TopoptState) -> torch.Tensor:
        raise NotImplementedError()
