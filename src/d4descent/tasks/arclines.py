import torch
from dataclasses import dataclass, field
from typing import Literal, Optional, Literal, Union, cast
from itertools import combinations
import numpy as np
import random

from ..object_collection import ObjectCollection
from ..objects.arclines import (
    Shape,
    ShapeCollection,
    ShapeRewriteArgs,
    ShapeRewriteType,
    ShapeRewriteSpec,
    ShapeCollectionArgs,
)
from ._base import Task, ObjectT, TaskArgs, RenderArgs, StateT
from ..visualizer import MPLVisualizer
from ..losses._base import LossArgs
from ..losses.raster import RasterLossMixin, RasterLossArgs
from ..losses.sds import SDSLossMixin, SDSLossArgs
from ..losses.topopt import TopoptArclinesMixin, TopoptArgs, TopoptState


@dataclass
class ArclinesArgs(TaskArgs):
    # init args
    n_source_pts: int = 4
    # metrics
    arc_weight: float = 1.5e-7
    line_weight: float = 1e-7
    # rewrite args
    rewrite_args: ShapeRewriteArgs = field(default_factory=ShapeRewriteArgs)  # only used if rewrite_algo == "rewrite"
    # arclines
    arclines_args: ShapeCollectionArgs = field(default_factory=ShapeCollectionArgs)
    # cleanup
    cleanup_area: float = 3e-4
    # better
    better_rel_eps: float = 1e-2
    better_abs_eps: float = 1e-8

    def create(
        self,
        render_args: RenderArgs,
        loss_args: LossArgs,
        device: Union[torch.device, str],
        target_img: Optional[torch.Tensor] = None,
    ) -> "Task":
        if isinstance(loss_args, RasterLossArgs):
            assert target_img is not None, "target_img must be provided for RasterLossArgs"
            return ArclinesRasterTask(self, render_args, loss_args, target_img)
        elif isinstance(loss_args, SDSLossArgs):
            return ArclinesSDSTask(self, render_args, loss_args, device)
        elif isinstance(loss_args, TopoptArgs):
            return ArclinesTopoptTask(self, render_args, loss_args, device)
        else:
            raise NotImplementedError(f"Unknown loss_args type: {type(loss_args)}")


class ArclinesTask(Task[Shape, ShapeRewriteSpec, StateT]):
    def __init__(self, args: ArclinesArgs, render_args: RenderArgs, device: Union[str, torch.device]):
        super().__init__(render_args)
        self._device = torch.device(device)
        self.args = args
        x = torch.linspace(
            -args.rewrite_args.add_hole_lim,
            args.rewrite_args.add_hole_lim,
            args.rewrite_args.add_hole_grid,
            device=self._device,
        )
        self.grid = torch.stack(
            [
                x.unsqueeze(-1).expand(-1, args.rewrite_args.add_hole_grid),
                x.unsqueeze(0).expand(args.rewrite_args.add_hole_grid, -1),
            ],
            dim=-1,
        ).flatten(
            0, 1
        )  # (add_hole_grid * add_hole_grid, 2)
        self._Collection = ShapeCollection.patch_args(self.args.arclines_args)

    def device(self) -> torch.device:
        return self._device

    def get_collection_constructor(self) -> type[ShapeCollection]:
        return self._Collection

    def initialize_object(self) -> Shape:
        return Shape.create_circle_lines(self.args.n_source_pts, (0, 0), 1, device=self.device())

    def compute_simplicity(self, collection: ObjectCollection[Shape]) -> list[float]:
        """
        Returns: (n,)
        """
        assert isinstance(collection, ShapeCollection)
        metrics: list[float] = []
        for shape_meta in collection.shapes:
            n_arcs = len(shape_meta.arcs_idx)
            n_lines = len(shape_meta.line_idx)
            metrics.append(n_arcs * self.args.arc_weight + n_lines * self.args.line_weight)
        return metrics

    def make_proposals_ex(
        self, obj: Shape, num_proposals: int
    ) -> tuple[ObjectCollection[Shape], list[ShapeRewriteSpec]]:
        args = self.args.rewrite_args
        total_weight = args.add_holes_weight + args.reg_weight
        # regular rewrites
        regular_specs = obj.generate_rewrite_specs(args=args)
        # add holes
        if args.add_holes_weight > 0.0:
            if args.add_hole_random:
                count = (
                    args.add_hole_count
                    if num_proposals == 0
                    else round(num_proposals * args.add_holes_weight / total_weight)
                )
                pos = (
                    torch.rand(count, 2, device=self.device()) * (self.render_args.lim[1] - self.render_args.lim[0])
                    + self.render_args.lim[0]
                )
            else:
                pos = self.grid
            add_holes_specs = obj.gen_add_holes_spec(pos, radius=args.add_hole_radius, n_segments=args.add_hole_segment)
        else:
            add_holes_specs = []

        if num_proposals == 0:
            all_specs = regular_specs + add_holes_specs
        else:
            num_regular = min(round(num_proposals * args.reg_weight / total_weight), len(regular_specs))
            num_add_holes = min(num_proposals - num_regular, len(add_holes_specs))
            all_specs = random.sample(regular_specs, num_regular) + random.sample(add_holes_specs, num_add_holes)

        prop_shapes = [obj.apply_rewrite(spec) for spec in all_specs]
        return self._Collection.from_shapes(prop_shapes), all_specs

    def make_proposals(self, obj: Shape) -> tuple[ObjectCollection[Shape], list[ShapeRewriteSpec]]:
        raise NotImplementedError()

    def combine_proposals(
        self,
        base: Shape,
        proposals: ObjectCollection[Shape],
        base_loss: float,
        proposal_losses: list[float],
        proposal_specs: list[ShapeRewriteSpec],
        accept_parallel: bool = True,
    ) -> tuple[Shape, bool]:
        assert isinstance(proposals, ShapeCollection)
        candidates: list[tuple[float, int]] = []
        for i in range(len(proposals)):
            loss_ = proposal_losses[i]
            # payload = proposals.shape_payloads[i]
            # rewrite_type_ = payload.rewrite_type
            # assert rewrite_type_ is not None
            # assert payload.from_id is not None and payload.from_id == base.id
            # is_simplify_ = rewrite_type_ in [RewriteType.Merge, RewriteType.ToLine, RewriteType.RemoveHole]
            # rel_improvement_ = (base_loss - loss_) / (base_loss + 1e-12)
            abs_improvement_ = base_loss - loss_
            # better_ = rel_improvement_ > self.args.better_rel_eps or abs_improvement_ > self.args.better_abs_eps
            better_ = (
                abs_improvement_ > self.args.better_abs_eps or abs_improvement_ > self.args.better_rel_eps * base_loss
            )
            # not_worse_ = abs_improvement_ > -self.args.better_abs_eps
            if better_:  # (is_simplify_ and not_worse_) or better_:
                candidates.append((-abs_improvement_, i))
            # print(rewrite_type_, prop_shapes[i].payload.rewrite_args, loss_, src_loss_, abs_improvement_)
        candidates.sort()

        # compute conflicts
        conflicts: set[tuple[int, int]] = set()
        for i, j in combinations(list(range(len(proposals))), 2):
            i_spec_ = proposals.shape_payloads[i].rewrite_spec
            j_spec_ = proposals.shape_payloads[j].rewrite_spec
            assert i_spec_ is not None and j_spec_ is not None
            if i_spec_.type != ShapeRewriteType.AddHole and j_spec_.type != ShapeRewriteType.AddHole:
                if set(i_spec_.args) & set(j_spec_.args):
                    conflicts.add((i, j))
                    conflicts.add((j, i))
        # greedily pick candidates without conflicts
        selected: list[ShapeRewriteSpec] = []
        selected_ids: list[int] = []
        for loss_, i in candidates:
            if all((i, j) not in conflicts for j in selected_ids):
                spec_ = proposals.shape_payloads[i].rewrite_spec
                assert spec_ is not None
                selected.append(spec_)
                selected_ids.append(i)
        if not accept_parallel:
            selected = selected[:1]
        if len(selected) > 0:
            return base.do_multiple_rewrites(selected), True
        return base, False

    def cleanup(self, collection: ObjectCollection[Shape]) -> ObjectCollection[Shape]:
        assert isinstance(collection, ShapeCollection)
        return self._Collection.from_shapes(
            [
                shape.remove_all_holes(self.args.cleanup_area).resolve_intersections().canonicalize_loops()
                for shape in collection
            ]
        )


class ArclinesRasterTask(RasterLossMixin[Shape, ShapeRewriteSpec, None], ArclinesTask[None]):
    def __init__(
        self, args: ArclinesArgs, render_args: RenderArgs, raster_args: RasterLossArgs, target_img: torch.Tensor
    ):
        device = target_img.device
        ArclinesTask.__init__(self, args, render_args, device)
        RasterLossMixin.__init__(self, raster_args, target_img)

    def initialize_state(self) -> None:
        return None

    def visualize(self, collection: ObjectCollection[Shape], step: int, loss: float, state: None) -> np.ndarray:
        assert isinstance(collection, ShapeCollection)
        assert len(collection) == 1
        shape = collection[0]
        fig = MPLVisualizer(1, 1, 10.8, 10.8, xlim=self.render_args.lim, ylim=self.render_args.lim, notebook=False)
        ax = fig[0]
        ax.ax.imshow(
            self.target_img.detach().cpu().numpy(),
            extent=(self.render_args.lim[0], self.render_args.lim[1], self.render_args.lim[1], self.render_args.lim[0]),
            cmap="plasma",
            vmin=0,
            vmax=1,
            alpha=0.2,
        )
        ax.ax.set_title(
            f"{self.get_elapsed_time():.0f}s: {shape.id}: {loss:.2e}: L{len(collection.shapes[0].line_idx)}A{len(collection.shapes[0].arcs_idx)}"
        )
        ax.visualize_shape(collection[0], show_text=True)
        return fig.get_image()


class ArclinesSDSTask(SDSLossMixin[Shape, ShapeRewriteSpec, None], ArclinesTask[None]):
    def __init__(
        self, args: ArclinesArgs, render_args: RenderArgs, sds_args: SDSLossArgs, device: Union[str, torch.device]
    ):
        ArclinesTask.__init__(self, args, render_args, device)
        SDSLossMixin.__init__(self, sds_args)

    def initialize_state(self) -> None:
        return None

    def visualize(self, collection: ObjectCollection[Shape], step: int, loss: float, state: None) -> np.ndarray:
        assert isinstance(collection, ShapeCollection)
        assert len(collection) == 1
        shape = collection[0]
        fig = MPLVisualizer(1, 1, 10.8, 10.8, xlim=self.render_args.lim, ylim=self.render_args.lim, notebook=False)
        ax = fig[0]
        ax.ax.set_title(
            f"{self.get_elapsed_time():.0f}s: {shape.id}: {loss:.2e}: L{len(collection.shapes[0].line_idx)}A{len(collection.shapes[0].arcs_idx)}"
        )
        ax.visualize_shape(collection[0])
        return fig.get_image()


class ArclinesTopoptTask(TopoptArclinesMixin, ArclinesTask[TopoptState]):
    def __init__(
        self, args: ArclinesArgs, render_args: RenderArgs, topopt_args: TopoptArgs, device: Union[str, torch.device]
    ):
        assert (
            render_args.size == topopt_args.sens.nelx
        ), f"render_args.size != topopt_args.sens.nelx, got {render_args.size} != {topopt_args.sens.nelx}"
        ArclinesTask.__init__(self, args, render_args, device)
        TopoptArclinesMixin.__init__(self, topopt_args, render_args.lim, device)


# check abstract methods
if __name__ == "__main__":
    from ..third_party.topopt import SensitivityAnalysisArgs

    ArclinesRasterTask(ArclinesArgs(), RenderArgs(), RasterLossArgs(), torch.Tensor(0))
    ArclinesSDSTask(ArclinesArgs(), RenderArgs(), SDSLossArgs(prompt="astronaut"), "cpu")
    ArclinesTopoptTask(ArclinesArgs(), RenderArgs(), TopoptArgs(sens=SensitivityAnalysisArgs(nelx=64, nely=64)), "cpu")
