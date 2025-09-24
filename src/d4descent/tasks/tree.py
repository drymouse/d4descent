from itertools import combinations
import torch
from dataclasses import dataclass, field
from typing import Optional, Union, Literal
import numpy as np

from ..object_collection import ObjectCollection
from ..objects.tree import Tree, TreeCollection, TreeRewriteArgs, TreeRewriteType, TreeRewriteSpec, TreeCollectionArgs
from ._base import Task, TaskArgs, RenderArgs, StateT
from ..visualizer import MPLVisualizer
from ..losses._base import LossArgs
from ..losses.raster import RasterLossMixin, RasterLossArgs
from ..losses.sds import SDSLossMixin, SDSLossArgs


@dataclass
class TreeArgs(TaskArgs):
    # metrics
    node_weight: float = 1e-3
    # rewrite args
    rewrite_args: TreeRewriteArgs = field(default_factory=TreeRewriteArgs)
    # better
    better_rel_eps: float = 1e-2
    better_abs_eps: float = 1e-8
    # target_img
    target_img: Optional[torch.Tensor] = None
    # cleanup
    cleanup_iters: int = 0
    cleanup_small_leaves: Optional[float] = None
    # tree args
    tree_args: TreeCollectionArgs = field(default_factory=TreeCollectionArgs)
    # initialization
    init_strategy: Literal["shape-bottom", "frame-bottom"] = "shape-bottom"

    def create(
        self,
        render_args: RenderArgs,
        loss_args: LossArgs,
        device,
        target_img: Optional[torch.Tensor] = None,
    ) -> "Task":
        if isinstance(loss_args, RasterLossArgs):
            assert target_img is not None, "target_img must be provided for RasterLossArgs"
            # Require the image for initialization
            self.target_img = target_img
            return TreeRasterTask(self, render_args, loss_args, target_img)
        elif isinstance(loss_args, SDSLossArgs):
            return TreeSDSTask(self, render_args, loss_args, device=device)
        else:
            raise NotImplementedError(f"Unkown loss_args type: {type(loss_args)}")


class TreeTask(Task[Tree, TreeRewriteSpec, StateT]):
    def __init__(self, args: TreeArgs, render_args: RenderArgs, device: Union[str, torch.device]):
        super().__init__(render_args)
        self._device = torch.device(device)
        self.args = args

    def device(self) -> torch.device:
        return self._device

    def get_collection_constructor(self) -> type[TreeCollection]:
        return TreeCollection

    def initialize_object(self) -> Tree:
        if self.args.init_strategy == "shape-bottom":
            assert self.args.target_img is not None
            R = self.args.target_img.shape[0]
            mask = self.args.target_img > 0.5
            row_ = int(mask.any(dim=-1).nonzero()[0][-1])
            col_ = mask[row_].nonzero()
            col_ = int(col_[len(col_) // 2].item())
            lim0, lim1 = self.render_args.lim
            root = torch.tensor(
                [lim0 + (lim1 - lim0) * (col_ + 0.5) / R, lim0 + (lim1 - lim0) * ((row_ + 0.5) / R)],
                device=self.device(),
            )
        elif self.args.init_strategy == "frame-bottom":
            root = torch.tensor([0.0, self.render_args.lim[0]], device=self.device())
        else:
            raise ValueError(f"Unknown init_strategy: {self.args.init_strategy}")

        return Tree(
            root=root,
            ls=torch.tensor([0.1], device=self.device()),
            thetas=torch.tensor(
                [0.0 if self.args.tree_args.theta_mode == "rel" else torch.pi / 2], device=self.device()
            ),
            parents=(0,),
            rs=torch.tensor([self.args.rewrite_args.default_r, self.args.rewrite_args.default_r], device=self.device()),
            args=self.args.tree_args,
        )

    def make_proposals(self, obj: Tree) -> tuple[ObjectCollection[Tree], list[TreeRewriteSpec]]:
        specs = obj.gen_rewrite_specs(self.args.rewrite_args, self.render_args.lim)
        rewritten = obj.apply_rewrite_each(specs)
        return TreeCollection.from_objects(rewritten), specs

    def combine_proposals(
        self,
        base: Tree,
        proposals: ObjectCollection[Tree],
        base_loss: float,
        proposal_losses: list[float],
        proposal_specs: list[TreeRewriteSpec],
        accept_parallel: bool = True,
    ) -> tuple[Tree, bool]:
        """
        Combinging rewrites for trees requires conflict checking
        If there are multiple AddBranch on the same node, it conflicts
        If there are addbranch and removebranch on the same node it conflicts
        A SplitBranch and addBranch don't conflict, but a SplitBranch and removeBranch do
        """
        assert isinstance(proposals, TreeCollection)
        scores: list[float] = []
        candidates: list[tuple[float, int]] = []
        for i in range(len(proposals)):
            loss_ = proposal_losses[i]
            abs_improvement_ = base_loss - loss_
            better_ = abs_improvement_ > self.args.better_abs_eps
            if better_:  # (is_simplify_ and not_worse_) or better_:
                candidates.append((-abs_improvement_, i))
        candidates.sort()
        # Compute conflicts
        conflicts: set[tuple[int, int]] = set()
        # Loop through all pairs of proposals
        for i, j in combinations(list(range(len(proposal_specs))), 2):
            # Get types and args
            i_type_ = proposal_specs[i].rewrite_type
            i_args_ = proposal_specs[i].rewrite_args
            j_type_ = proposal_specs[j].rewrite_type
            j_args_ = proposal_specs[j].rewrite_args
            assert i_type_ is not None
            assert j_type_ is not None
            assert i_args_ is not None
            assert j_args_ is not None
            # For Add rewrites and Remove rewrites, they conflict if they
            # operate on the same node
            add_remove_rewrites_ = [
                TreeRewriteType.AddBranch,
                TreeRewriteType.AddBranchBoth,
                TreeRewriteType.RemoveBranch,
                TreeRewriteType.SplitBranch,
                TreeRewriteType.AddAnywhere,
            ]
            if i_type_ in add_remove_rewrites_ and j_type_ in add_remove_rewrites_:
                i_nodes = (
                    [int(i_args_[0]), int(i_args_[1])]
                    if i_type_ == TreeRewriteType.RemoveBranch and len(i_args_) == 2
                    else [int(i_args_[0])]
                )
                j_nodes = (
                    [int(j_args_[0]), int(j_args_[1])]
                    if j_type_ == TreeRewriteType.RemoveBranch and len(j_args_) == 2
                    else [int(j_args_[0])]
                )
                if len(set(i_nodes).intersection(set(j_nodes))):
                    conflicts.add((i, j))
                    conflicts.add((j, i))
        # Greedily pick candidates without conflicts
        selected: list[TreeRewriteSpec] = []
        selected_ids: list[int] = []
        for loss_, i in candidates:  # for each proposal (loss, i)
            if all((i, j) not in conflicts for j in selected_ids):  # if no conflicts
                type_ = proposal_specs[i].rewrite_type
                args_ = proposal_specs[i].rewrite_args
                assert type_ is not None
                assert args_ is not None
                selected.append(proposal_specs[i])
                selected_ids.append(i)
        if not accept_parallel:
            selected = selected[:1]
        if len(selected) > 0:
            return base.apply_rewrite_all(selected, []), True
        return base, False

    def cleanup(self, collection: ObjectCollection[Tree]) -> ObjectCollection[Tree]:
        new_collection: list[Tree] = []
        for node in collection:
            cleaned = node.cleanup(
                cleanup_small_leaves=self.args.cleanup_small_leaves, max_iter=self.args.cleanup_iters
            )
            new_collection.append(cleaned)
        return TreeCollection.from_objects(new_collection)

    def simp(self, collection: ObjectCollection[Tree]) -> ObjectCollection[Tree]:
        assert isinstance(collection, TreeCollection)
        return collection

    def compute_simplicity(self, collection: ObjectCollection[Tree]) -> list[float]:
        assert isinstance(collection, TreeCollection)
        return [len(ords) * self.args.node_weight for ords in collection.tree_node_ords]


class TreeRasterTask(RasterLossMixin[Tree, TreeRewriteSpec, None], TreeTask[None]):
    def __init__(self, args: TreeArgs, render_args: RenderArgs, raster_args: RasterLossArgs, target_img: torch.Tensor):
        device = target_img.device
        TreeTask.__init__(self, args, render_args, device)
        RasterLossMixin.__init__(self, raster_args, target_img)

    def initialize_state(self) -> None:
        return None

    def visualize(self, collection: ObjectCollection[Tree], step: int, loss: float, state: None) -> np.ndarray:
        assert isinstance(collection, TreeCollection)
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
        ax.ax.set_title(f"{self.get_elapsed_time():.0f}s: {shape.id}: {loss:.2e}: N{len(shape.rs)}")
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
        shape.visualize(ax)
        return fig.get_image()


class TreeSDSTask(SDSLossMixin[Tree, TreeRewriteSpec, None], TreeTask[None]):
    def __init__(
        self, args: TreeArgs, render_args: RenderArgs, sds_args: SDSLossArgs, device: Union[str, torch.device]
    ):
        TreeTask.__init__(self, args, render_args, device)
        SDSLossMixin.__init__(self, sds_args)

    def initialize_state(self) -> None:
        return None

    def visualize(self, collection: ObjectCollection[Tree], step: int, loss: float, state: None) -> np.ndarray:
        assert isinstance(collection, TreeCollection)
        assert len(collection) == 1
        shape = collection[0]
        fig = MPLVisualizer(1, 1, 10.8, 10.8, xlim=self.render_args.lim, ylim=self.render_args.lim, notebook=False)
        ax = fig[0]
        ax.ax.set_title(f"{self.get_elapsed_time():.0f}s: {shape.id}: {loss:.2e}: N{len(shape.rs)}")
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
        shape.visualize(ax)
        return fig.get_image()


if __name__ == "__main__":
    TreeRasterTask(TreeArgs(), RenderArgs(), RasterLossArgs(), torch.Tensor(0))
    TreeSDSTask(TreeArgs(), RenderArgs(), SDSLossArgs(prompt="astronaut"), "cpu")
