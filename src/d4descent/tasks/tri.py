import torch
from dataclasses import dataclass, field
from typing import Literal, Optional, Literal, Union
from itertools import combinations
import numpy as np

from ..context import Context
from ..object_collection import ObjectCollection
from ..objects.tri import (
    Tri,
    TriCollection,
    TriRewriteArgs,
    TriRewriteType,
    TriRewrite,
    TriRewriteAddTri,
    TriRewriteRemoveTri,
    TriPayload,
    TriColectionArgs,
    # CleanStrategy,
)
from ._base import Task, ObjectT, TaskArgs, RenderArgs, StateT, ExtraMetrics
from ..visualizer import MPLVisualizer
from ..losses._base import LossArgs
from ..losses.raster import RasterLossMixin, RasterLossArgs
# from ..losses.sds import SDSLossMixin, SDSLossArgs
# from ..losses.topopt import TopoptURMixin, TopoptArgs, TopoptState


@dataclass
class TriArgs(TaskArgs):
    # metrics
    node_weight: float = 1e-6
    size_weight: float = 0
    # rewrite args
    rewrite_args: TriRewriteArgs = field(default_factory=TriRewriteArgs)  # only used if rewrite_algo == "rewrite"
    # cleanup
    # cleanup_len: float = 0.01
    # cleanup_area: float = 0.0005
    # cleanup_strategy: CleanStrategy = "none"
    # cleanup_split_len: float = 0.05
    # cleanup_merge_area_threshold: float = 0.001
    # better
    better_rel_eps: float = 1e-2
    better_abs_eps: float = 1e-8
    # ur
    tri_args: TriColectionArgs = field(default_factory=TriColectionArgs)

    def create(
        self,
        render_args: RenderArgs,
        loss_args: LossArgs,
        device: Union[torch.device, str],
        target_img: Optional[torch.Tensor] = None,
    ) -> "Task":
        if isinstance(loss_args, RasterLossArgs):
            assert target_img is not None, "target_img must be provided for RasterLossArgs"
            return TriRasterTask(self, render_args, loss_args, target_img)
        # elif isinstance(loss_args, SDSLossArgs):
        #     return URSDSTask(self, render_args, loss_args, device)
        # elif isinstance(loss_args, TopoptArgs):
        #     return URTopoptTask(self, render_args, loss_args, device)
        else:
            raise NotImplementedError(f"Unknown loss_args type: {type(loss_args)}")


class TriTask(Task[Tri, TriRewrite, StateT]):
    def __init__(self, args: TriArgs, render_args: RenderArgs, device: Union[str, torch.device]):
        super().__init__(render_args)
        self._device = torch.device(device)
        self.args = args
        self._Collection = TriCollection.patch_args(self.args.tri_args)

    def device(self) -> torch.device:
        return self._device

    def get_collection_constructor(self) -> type[TriCollection]:
        return self._Collection

    def initialize_object(self) -> Tri:
        device = self.device()
        return Tri(
            xs=torch.tensor([[[0.0, 0.0], [0.01, 0.0], [0.0, 0.01]]], device=device),
        )

    def compute_simplicity(self, collection: ObjectCollection[Tri]) -> list[float]:
        """
        Returns: (n,)
        """
        assert isinstance(collection, TriCollection)
        metrics: list[float] = []
        for size_ in collection.get_sizes():
            metrics.append(size_ * self.args.node_weight)
        return metrics

    def make_proposals_ex(self, obj: Tri, num_proposals: int) -> tuple[ObjectCollection[Tri], list[TriRewrite]]:
        assert num_proposals > 0, f"num_proposals must be positive, got {num_proposals}"
        specs = obj.gen_rewrite_specs(
            self.args.rewrite_args, num_rewrites=num_proposals, lim=self.render_args.lim, tri_args=self.args.tri_args
        )

        device = obj.xs.device
        dtype = obj.xs.dtype
        n_base = obj.xs.shape[0]
        xs_base = obj.xs.detach()

        add_specs = [s for s in specs if isinstance(s, TriRewriteAddTri)]
        remove_specs = [s for s in specs if isinstance(s, TriRewriteRemoveTri)]

        sub_collections: list[TriCollection] = []
        ordered_specs: list[TriRewrite] = []
        ctx = Context.get()

        if add_specs:
            n_add = len(add_specs)
            # 全 AddTri の頂点を一括テンソル化（CPU→GPU 転送は1回だけ）
            new_verts = torch.tensor(
                [[[s.x1, s.y1], [s.x2, s.y2], [s.x3, s.y3]] for s in add_specs],
                device=device, dtype=dtype,
            )  # (n_add, 3, 2)
            n_each = n_base + 1
            # 元の三角形群を n_add 回展開して末尾に1枚ずつ追加
            xs_expanded = xs_base.unsqueeze(0).expand(n_add, -1, -1, -1)  # (n_add, n_base, 3, 2)
            xs_all = torch.cat([xs_expanded, new_verts.unsqueeze(1)], dim=1)  # (n_add, n_each, 3, 2)
            xs_flat = xs_all.reshape(-1, 3, 2)  # (n_add * n_each, 3, 2)
            index_of = torch.arange(n_add, device=device).repeat_interleave(n_each)
            sub_collections.append(self._Collection(
                xs=xs_flat,
                index_of=index_of,
                indices=tuple((i * n_each, (i + 1) * n_each) for i in range(n_add)),
                ids=tuple(ctx.gen_id() for _ in range(n_add)),
                payloads=tuple(TriPayload() for _ in range(n_add)),
            ))
            ordered_specs.extend(add_specs)

        if remove_specs:
            n_remove = len(remove_specs)
            n_each = n_base - 1
            # 削除インデックスをベクトル化してマスクを一括生成
            remove_ids_t = torch.tensor([s.id_ for s in remove_specs], device=device)  # (n_remove,)
            all_idx = torch.arange(n_base, device=device)
            keep_mask = all_idx.unsqueeze(0) != remove_ids_t.unsqueeze(1)  # (n_remove, n_base)
            keep_indices = all_idx.unsqueeze(0).expand(n_remove, -1)[keep_mask].reshape(n_remove, n_each)
            # gather で各提案の残存三角形を一括取得
            gather_idx = keep_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 2)
            xs_rem = xs_base.unsqueeze(0).expand(n_remove, -1, -1, -1)
            xs_flat_r = torch.gather(xs_rem, 1, gather_idx).reshape(-1, 3, 2)  # (n_remove * n_each, 3, 2)
            index_of_r = torch.arange(n_remove, device=device).repeat_interleave(n_each)
            sub_collections.append(self._Collection(
                xs=xs_flat_r,
                index_of=index_of_r,
                indices=tuple((i * n_each, (i + 1) * n_each) for i in range(n_remove)),
                ids=tuple(ctx.gen_id() for _ in range(n_remove)),
                payloads=tuple(TriPayload() for _ in range(n_remove)),
            ))
            ordered_specs.extend(remove_specs)

        if not sub_collections:
            return self._Collection.from_objects([obj]), []

        return self._Collection.cat(sub_collections), ordered_specs

    def make_proposals(self, obj: Tri) -> tuple[ObjectCollection[Tri], list[TriRewrite]]:
        raise NotImplementedError()

    def combine_proposals(
        self,
        base: Tri,
        proposals: ObjectCollection[Tri],
        base_loss: float,
        proposal_losses: list[float],
        proposal_specs: list[TriRewrite],
        accept_parallel: bool = True,
    ) -> tuple[Tri, bool]:
        assert isinstance(proposals, TriCollection)
        scores: list[float] = []
        candidates: list[TriRewrite] = []
        for i in range(len(proposals)):
            loss_ = proposal_losses[i]
            abs_improvement_ = base_loss - loss_
            better_ = abs_improvement_ > self.args.better_abs_eps
            if better_:  # (is_simplify_ and not_worse_) or better_:
                scores.append(abs_improvement_)
                candidates.append(proposal_specs[i])

        if not accept_parallel:
            candidates = candidates[:1]
        if len(candidates) > 0:
            return base.apply_all_rewrites(candidates, scores, self.args.tri_args), True
        return base, False

    def compute_losses(self, collection: ObjectCollection[Tri], state: StateT) -> tuple[torch.Tensor, ExtraMetrics]:
        losses, xtra = self._compute_losses(collection, state)
        assert isinstance(collection, TriCollection)
        if self.args.size_weight > 0:
            sizes = collection.get_sum_sizes()
            losses = losses + self.args.size_weight * sizes
        return losses, xtra

    def cleanup(self, collection: ObjectCollection[Tri]) -> ObjectCollection[Tri]:
        return collection
    #     new_collection: list[UR] = []
    #     for node in collection:
    #         cleaned = node.cleanup(
    #             len_eps=self.args.cleanup_len,
    #             area_eps=self.args.cleanup_area,
    #             clean_strategy=self.args.cleanup_strategy,
    #             split_len=self.args.cleanup_split_len,
    #             merge_area_threshold=self.args.cleanup_merge_area_threshold,
    #             lim=self.render_args.lim,
    #             size=self.render_args.size,
    #             ur_args=self.args.ur_args,
    #         )
    #         new_collection.append(cleaned)
    #     return self._Collection.from_objects(new_collection)


class TriRasterTask(RasterLossMixin[Tri, TriRewrite, None], TriTask[None]):
    def __init__(self, args: TriArgs, render_args: RenderArgs, raster_args: RasterLossArgs, target_img: torch.Tensor):
        device = target_img.device
        TriTask.__init__(self, args, render_args, device)
        RasterLossMixin.__init__(self, raster_args, target_img)

    def initialize_state(self) -> None:
        return None

    def visualize(self, collection: ObjectCollection[Tri], step: int, loss: float, state: None) -> np.ndarray:
        assert isinstance(collection, TriCollection)
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
        ax.ax.set_title(f"{self.get_elapsed_time():.0f}s: {shape.id}: {loss:.2e}: R{len(shape.xs)}")
        imgs = collection.render01(
            self.render_args.size,
            self.render_args.lim,
            center_pixel=self.render_args.center_pixel,
            blur=self.render_args.blur,
        )
        ax.ax.imshow(
            imgs[0].detach().cpu().numpy(),
            extent=(self.render_args.lim[0], self.render_args.lim[1], self.render_args.lim[1], self.render_args.lim[0]),
            cmap="winter",
            vmin=0,
            vmax=1,
            alpha=0.2,
        )
        shape.visualize(ax, self.args.tri_args)
        return fig.get_image()

"""
class URSDSTask(SDSLossMixin[UR, URRewrite, None], URTask[None]):
    def __init__(
        self, args: URArgs, render_args: RenderArgs, sds_args: SDSLossArgs, device: Union[str, torch.device]
    ):
        URTask.__init__(self, args, render_args, device)
        SDSLossMixin.__init__(self, sds_args)

    def initialize_state(self) -> None:
        return None

    def visualize(self, collection: ObjectCollection[UR], step: int, loss: float, state: None) -> np.ndarray:
        assert isinstance(collection, URCollection)
        assert len(collection) == 1
        shape = collection[0]
        fig = MPLVisualizer(1, 1, 10.8, 10.8, xlim=self.render_args.lim, ylim=self.render_args.lim, notebook=False)
        ax = fig[0]
        ax.ax.set_title(f"{self.get_elapsed_time():.0f}s: {shape.id}: {loss:.2e}: R{len(shape.xs)}")
        imgs = collection.render01(
            self.render_args.size,
            self.render_args.lim,
            center_pixel=self.render_args.center_pixel,
            blur=self.render_args.blur,
        )
        ax.ax.imshow(
            imgs[0].detach().cpu().numpy(),
            extent=(self.render_args.lim[0], self.render_args.lim[1], self.render_args.lim[1], self.render_args.lim[0]),
            cmap="winter",
            vmin=0,
            vmax=1,
            alpha=0.2,
        )
        shape.visualize(ax, self.args.ur_args)
        return fig.get_image()


class URTopoptTask(TopoptURMixin, URTask[TopoptState]):
    def __init__(
        self, args: URArgs, render_args: RenderArgs, topopt_args: TopoptArgs, device: Union[str, torch.device]
    ):
        assert (
            render_args.size == topopt_args.sens.nelx
        ), f"render_args.size != topopt_args.sens.nelx, got {render_args.size} != {topopt_args.sens.nelx}"
        URTask.__init__(self, args, render_args, device)
        TopoptURMixin.__init__(self, topopt_args, render_args.lim, device)

    def visualize(self, collection: ObjectCollection[UR], step: int, loss: float, state: TopoptState) -> np.ndarray:
        assert isinstance(collection, URCollection)
        assert len(collection) == 1
        shape = collection[0]
        fig = MPLVisualizer(1, 1, 10.8, 10.8, xlim=self.render_args.lim, ylim=self.render_args.lim, notebook=False)
        lambda_ = state.pids[0].last_value or 0.0
        img = collection.render(
            self.render_args.size,
            self.render_args.lim,
            center_pixel=self.render_args.center_pixel,
        )[0]

        area = shape.sizes.abs().prod(dim=-1).sum()

        vlim = self.occ_v
        if vlim > 0:
            occ = (-img.clamp(-vlim, vlim) + vlim) / (2 * vlim)  # (n_shapes, size, size)
        else:
            occ = (-img > 0).float()
        sed, sed_grad = self.sens.get_sensitivity(occ)  # (size, size)
        ax = fig[0]
        ax.ax.set_title(
            f"{self.get_elapsed_time():.0f}s: {shape.id}: {loss:.2e}: R{len(shape.xs)}"
            f"\n$\\lambda$: {lambda_:.2f} occ: {occ.mean():.3f}/{state.pids[0].setpoint:.3f}"
            f"\narea: {area:.2f}/{self.total_area * self.target_vol:.2f}"
        )
        shape.visualize(ax, ur_args=self.args.ur_args)
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
        ax.ax.imshow(
            sed_grad.detach().cpu().numpy() - lambda_,
            extent=(self.lim[0], self.lim[1], self.lim[1], self.lim[0]),
            cmap="RdBu",
            vmin=-3,  # np.abs(grad).max(),
            vmax=3,  # np.abs(grad).max(),
            # alpha=0.2,
        )
        return fig.get_image()
"""

# check abstract methods
if __name__ == "__main__":
    TriRasterTask(TriArgs(), RenderArgs(), RasterLossArgs(), torch.Tensor(0))
