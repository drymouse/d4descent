import math
import random
from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np
import torch

from ..context import Context
from ..object_collection import ObjectCollection
from ..objects.roads import (
    RoadNetwork,
    RoadNetworkCollection,
    RoadPayload,
    RoadRewrite,
    RoadRewriteAdd,
    RoadRewriteAddAnywhere,
    RoadRewriteArgs,
    RoadRewriteMerge,
    RoadRewriteNarrow,
    RoadRewriteRemove,
    RoadRewriteSnap,
    RoadRewriteSplit,
    RoadRewriteUnsnap,
    RoadRewriteWiden,
    RoadCollectionArgs,
)
from ._base import Task, TaskArgs, RenderArgs, StateT, ExtraMetrics
from ..visualizer import MPLVisualizer
from ..losses._base import LossArgs
from ..losses.raster import RasterLossArgs


@dataclass
class RoadArgs(TaskArgs):
    cost_weight: float = 1e-3  # 建設コスト(長さ×幅^cost_width_exponent)の正則化重み。Triのnode_weightに相当
    cost_width_exponent: float = 2.0  # 幅への指数。1より大きいほど幹線道路(幅広)への罰則が超線形に強くなる
    mesh_weight: float = 0.05  # ループ形成(meshedness)への報酬の重み。大きいほどSnapでのループ化を優先する
    # target_img([0,1]の画像やshcのrender01など、入力元によらず同じ扱い)を実際の密度値にアフィン変換する:
    #   effective_target = target_outside_value + (target_inside_value - target_outside_value) * target_img
    # 0/1 に張り付いた値をそのまま目標にはしない。街の外側にも最低限の目標密度(target_outside_value)を
    # 持たせることで、そこにも(幹線道路程度の)道路が伸びる動機を作る。pngs・shcのどちらの入力でも同じ
    # ロジックで解釈する。
    target_inside_value: float = 0.7
    target_outside_value: float = 0.15
    # 交差点の角度が小さすぎる(道路同士がほぼ同じ方向を向いて鋭角に交わる)ことへの罰則。
    # (min_angle - 実際の角度)^angle_penalty_exponent を連続損失に直接加算する
    # (離散書き換えの採否・連続最適化どちらにも効かせるため compute_simplicity ではなく loss 側に入れる)。
    min_angle: float = math.radians(45)
    angle_penalty_exponent: float = 2.0
    angle_weight: float = 0.02
    better_abs_eps: float = 1e-8
    rewrite_args: RoadRewriteArgs = field(default_factory=RoadRewriteArgs)
    road_collection_args: RoadCollectionArgs = field(default_factory=RoadCollectionArgs)

    def create(
        self,
        render_args: RenderArgs,
        loss_args: LossArgs,
        device: Union[torch.device, str],
        target_img: Optional[torch.Tensor] = None,
    ) -> "Task":
        if isinstance(loss_args, RasterLossArgs):
            assert target_img is not None, "target_img (人口密度マップ, [0,1]) must be provided"
            return RoadDensityTask(self, render_args, target_img)
        else:
            raise NotImplementedError(f"Unknown loss_args type: {type(loss_args)}")


class RoadTask(Task[RoadNetwork, RoadRewrite, StateT]):
    def __init__(self, args: RoadArgs, render_args: RenderArgs, device: Union[str, torch.device]):
        super().__init__(render_args)
        self._device = torch.device(device)
        self.args = args
        self._Collection = RoadNetworkCollection.patch_args(self.args.road_collection_args)

    def device(self) -> torch.device:
        return self._device

    def get_collection_constructor(self) -> type[RoadNetworkCollection]:
        return self._Collection

    def initialize_object(self) -> RoadNetwork:
        device = self.device()
        # 幹線道路(最大幅クラス)をシードにする。不変条件B(highwayはhighwayからしか生えない)の
        # 起点となるバックボーンが最初から存在する必要があるため。
        w0 = max(self.args.rewrite_args.width_classes)
        return RoadNetwork(
            nodes=torch.tensor([[-0.01, 0.0], [0.01, 0.0]], device=device),
            edges=torch.tensor([[0, 1]], dtype=torch.long, device=device),
            widths=torch.tensor([w0], device=device),
        )

    def compute_simplicity(self, collection: ObjectCollection[RoadNetwork]) -> list[float]:
        assert isinstance(collection, RoadNetworkCollection)
        costs = collection.get_construction_costs(width_exponent=self.args.cost_width_exponent)
        meshedness = collection.get_meshedness()
        return [
            c * self.args.cost_weight - m * self.args.mesh_weight
            for c, m in zip(costs.tolist(), meshedness.tolist())
        ]

    def make_proposals(self, obj: RoadNetwork) -> tuple[ObjectCollection[RoadNetwork], list[RoadRewrite]]:
        raise NotImplementedError("use make_proposals_ex")

    def make_proposals_ex(
        self, obj: RoadNetwork, num_proposals: int
    ) -> tuple[ObjectCollection[RoadNetwork], list[RoadRewrite]]:
        specs = obj.gen_rewrite_specs(self.args.rewrite_args, lim=self.render_args.lim)
        if num_proposals > 0 and len(specs) > num_proposals:
            specs = random.sample(specs, num_proposals)

        device = obj.nodes.device
        dtype = obj.nodes.dtype
        n_base_nodes = len(obj.nodes)
        n_base_edges = len(obj.edges)
        base_nodes = obj.nodes.detach()
        base_edges = obj.edges
        base_widths = obj.widths.detach()
        all_idx_edges = torch.arange(n_base_edges, device=device)

        add_specs = [s for s in specs if isinstance(s, RoadRewriteAdd)]
        addany_specs = [s for s in specs if isinstance(s, RoadRewriteAddAnywhere)]
        remove_specs = [s for s in specs if isinstance(s, RoadRewriteRemove)]
        split_specs = [s for s in specs if isinstance(s, RoadRewriteSplit)]
        merge_specs = [s for s in specs if isinstance(s, RoadRewriteMerge)]
        snap_specs = [s for s in specs if isinstance(s, RoadRewriteSnap)]
        unsnap_specs = [s for s in specs if isinstance(s, RoadRewriteUnsnap)]
        widen_specs = [s for s in specs if isinstance(s, (RoadRewriteWiden, RoadRewriteNarrow))]

        sub_collections: list[RoadNetworkCollection] = []
        ordered_specs: list[RoadRewrite] = []
        ctx = Context.get()

        def _make_sub(
            n: int,
            n_each_nodes: int,
            n_each_edges: int,
            nodes_all: torch.Tensor,
            edges_all: torch.Tensor,
            widths_all: torch.Tensor,
        ) -> RoadNetworkCollection:
            edge_index_of = torch.arange(n, device=device).repeat_interleave(n_each_edges)
            return self._Collection(
                nodes=nodes_all,
                edges=edges_all,
                widths=widths_all,
                edge_index_of=edge_index_of,
                node_ranges=tuple((i * n_each_nodes, (i + 1) * n_each_nodes) for i in range(n)),
                edge_ranges=tuple((i * n_each_edges, (i + 1) * n_each_edges) for i in range(n)),
                ids=tuple(ctx.gen_id() for _ in range(n)),
                payloads=tuple(RoadPayload() for _ in range(n)),
            )

        # ---- Add ----
        if add_specs:
            n = len(add_specs)
            n_each_nodes = n_base_nodes + 1
            n_each_edges = n_base_edges + 1
            new_pt = torch.tensor([[s.x, s.y] for s in add_specs], device=device, dtype=dtype)
            from_node = torch.tensor([s.from_node for s in add_specs], device=device, dtype=torch.long)
            new_w = torch.tensor([s.width for s in add_specs], device=device, dtype=dtype)

            nodes_exp = base_nodes.unsqueeze(0).expand(n, -1, -1)
            nodes_all = torch.cat([nodes_exp, new_pt.unsqueeze(1)], dim=1).reshape(-1, 2)

            edges_exp = base_edges.unsqueeze(0).expand(n, -1, -1)
            new_edge = torch.stack([from_node, torch.full((n,), n_base_nodes, device=device, dtype=torch.long)], dim=-1)
            edges_local = torch.cat([edges_exp, new_edge.unsqueeze(1)], dim=1)
            offset = (torch.arange(n, device=device) * n_each_nodes).view(n, 1, 1)
            edges_all = (edges_local + offset).reshape(-1, 2)

            widths_exp = base_widths.unsqueeze(0).expand(n, -1)
            widths_all = torch.cat([widths_exp, new_w.unsqueeze(1)], dim=1).reshape(-1)

            sub_collections.append(_make_sub(n, n_each_nodes, n_each_edges, nodes_all, edges_all, widths_all))
            ordered_specs.extend(add_specs)

        # ---- AddAnywhere (鎖の長さkごとにグループ化してバッチ化) ----
        if addany_specs:
            by_k: dict[int, list[RoadRewriteAddAnywhere]] = {}
            for s in addany_specs:
                by_k.setdefault(len(s.pts), []).append(s)
            for k, group in by_k.items():
                n = len(group)
                n_each_nodes = n_base_nodes + k
                n_each_edges = n_base_edges + k
                pts_t = torch.tensor([list(s.pts) for s in group], device=device, dtype=dtype)  # (n,k,2)
                from_node = torch.tensor([s.from_node for s in group], device=device, dtype=torch.long)
                new_w = torch.tensor([s.width for s in group], device=device, dtype=dtype)

                nodes_exp = base_nodes.unsqueeze(0).expand(n, -1, -1)
                nodes_all = torch.cat([nodes_exp, pts_t], dim=1).reshape(-1, 2)

                hop_new_ids = (n_base_nodes + torch.arange(k, device=device)).unsqueeze(0).expand(n, -1)  # (n,k)
                hop_starts = torch.cat([from_node.unsqueeze(1), hop_new_ids[:, :-1]], dim=1)  # (n,k)
                new_edges_local = torch.stack([hop_starts, hop_new_ids], dim=-1)  # (n,k,2)
                edges_exp = base_edges.unsqueeze(0).expand(n, -1, -1)
                edges_local = torch.cat([edges_exp, new_edges_local], dim=1)
                offset = (torch.arange(n, device=device) * n_each_nodes).view(n, 1, 1)
                edges_all = (edges_local + offset).reshape(-1, 2)

                widths_exp = base_widths.unsqueeze(0).expand(n, -1)
                new_w_exp = new_w.unsqueeze(1).expand(-1, k)
                widths_all = torch.cat([widths_exp, new_w_exp], dim=1).reshape(-1)

                sub_collections.append(_make_sub(n, n_each_nodes, n_each_edges, nodes_all, edges_all, widths_all))
                ordered_specs.extend(group)

        # ---- Remove ----
        if remove_specs:
            n = len(remove_specs)
            n_each_nodes = n_base_nodes
            n_each_edges = n_base_edges - 1
            remove_ids = torch.tensor([s.edge_id for s in remove_specs], device=device)
            keep_mask = all_idx_edges.unsqueeze(0) != remove_ids.unsqueeze(1)  # (n, n_base_edges)
            keep_indices = all_idx_edges.unsqueeze(0).expand(n, -1)[keep_mask].reshape(n, n_each_edges)

            nodes_all = base_nodes.unsqueeze(0).expand(n, -1, -1).reshape(-1, 2)

            edges_exp = base_edges.unsqueeze(0).expand(n, -1, -1)
            edges_local = torch.gather(edges_exp, 1, keep_indices.unsqueeze(-1).expand(-1, -1, 2))
            offset = (torch.arange(n, device=device) * n_each_nodes).view(n, 1, 1)
            edges_all = (edges_local + offset).reshape(-1, 2)

            widths_exp = base_widths.unsqueeze(0).expand(n, -1)
            widths_all = torch.gather(widths_exp, 1, keep_indices).reshape(-1)

            sub_collections.append(_make_sub(n, n_each_nodes, n_each_edges, nodes_all, edges_all, widths_all))
            ordered_specs.extend(remove_specs)

        # ---- Split ----
        if split_specs:
            n = len(split_specs)
            n_each_nodes = n_base_nodes + 1
            n_each_edges = n_base_edges + 1
            edge_ids = torch.tensor([s.edge_id for s in split_specs], device=device)
            new_pt = torch.tensor([[s.x, s.y] for s in split_specs], device=device, dtype=dtype)
            ab = base_edges[edge_ids]  # (n,2)
            new_w = base_widths[edge_ids]  # (n,)
            keep_mask = all_idx_edges.unsqueeze(0) != edge_ids.unsqueeze(1)
            keep_indices = all_idx_edges.unsqueeze(0).expand(n, -1)[keep_mask].reshape(n, n_base_edges - 1)

            nodes_exp = base_nodes.unsqueeze(0).expand(n, -1, -1)
            nodes_all = torch.cat([nodes_exp, new_pt.unsqueeze(1)], dim=1).reshape(-1, 2)

            edges_exp = base_edges.unsqueeze(0).expand(n, -1, -1)
            kept_edges = torch.gather(edges_exp, 1, keep_indices.unsqueeze(-1).expand(-1, -1, 2))
            new_node_id = torch.full((n,), n_base_nodes, device=device, dtype=torch.long)
            new_edge1 = torch.stack([ab[:, 0], new_node_id], dim=-1)
            new_edge2 = torch.stack([new_node_id, ab[:, 1]], dim=-1)
            edges_local = torch.cat([kept_edges, new_edge1.unsqueeze(1), new_edge2.unsqueeze(1)], dim=1)
            offset = (torch.arange(n, device=device) * n_each_nodes).view(n, 1, 1)
            edges_all = (edges_local + offset).reshape(-1, 2)

            widths_exp = base_widths.unsqueeze(0).expand(n, -1)
            kept_widths = torch.gather(widths_exp, 1, keep_indices)
            widths_all = torch.cat([kept_widths, new_w.unsqueeze(1), new_w.unsqueeze(1)], dim=1).reshape(-1)

            sub_collections.append(_make_sub(n, n_each_nodes, n_each_edges, nodes_all, edges_all, widths_all))
            ordered_specs.extend(split_specs)

        # ---- Merge ----
        if merge_specs:
            n = len(merge_specs)
            n_each_nodes = n_base_nodes
            n_each_edges = n_base_edges - 1
            ea = torch.tensor([s.edge_id_a for s in merge_specs], device=device)
            eb = torch.tensor([s.edge_id_b for s in merge_specs], device=device)
            outer_a = torch.tensor([s.outer_a for s in merge_specs], device=device, dtype=torch.long)
            outer_b = torch.tensor([s.outer_b for s in merge_specs], device=device, dtype=torch.long)
            new_w = torch.tensor([s.width for s in merge_specs], device=device, dtype=dtype)
            keep_mask = (all_idx_edges.unsqueeze(0) != ea.unsqueeze(1)) & (all_idx_edges.unsqueeze(0) != eb.unsqueeze(1))
            keep_indices = all_idx_edges.unsqueeze(0).expand(n, -1)[keep_mask].reshape(n, n_base_edges - 2)

            nodes_all = base_nodes.unsqueeze(0).expand(n, -1, -1).reshape(-1, 2)

            edges_exp = base_edges.unsqueeze(0).expand(n, -1, -1)
            kept_edges = torch.gather(edges_exp, 1, keep_indices.unsqueeze(-1).expand(-1, -1, 2))
            new_edge = torch.stack([outer_a, outer_b], dim=-1)
            edges_local = torch.cat([kept_edges, new_edge.unsqueeze(1)], dim=1)
            offset = (torch.arange(n, device=device) * n_each_nodes).view(n, 1, 1)
            edges_all = (edges_local + offset).reshape(-1, 2)

            widths_exp = base_widths.unsqueeze(0).expand(n, -1)
            kept_widths = torch.gather(widths_exp, 1, keep_indices)
            widths_all = torch.cat([kept_widths, new_w.unsqueeze(1)], dim=1).reshape(-1)

            sub_collections.append(_make_sub(n, n_each_nodes, n_each_edges, nodes_all, edges_all, widths_all))
            ordered_specs.extend(merge_specs)

        # ---- Snap ----
        if snap_specs:
            n = len(snap_specs)
            n_each_nodes = n_base_nodes
            n_each_edges = n_base_edges
            edge_ids = torch.tensor([s.edge_id for s in snap_specs], device=device)
            ends = torch.tensor([s.end for s in snap_specs], device=device)
            targets = torch.tensor([s.target_node for s in snap_specs], device=device, dtype=torch.long)

            nodes_all = base_nodes.unsqueeze(0).expand(n, -1, -1).reshape(-1, 2)

            edges_local = base_edges.unsqueeze(0).expand(n, -1, -1).clone()
            proposal_idx = torch.arange(n, device=device)
            edges_local[proposal_idx, edge_ids, ends] = targets
            offset = (torch.arange(n, device=device) * n_each_nodes).view(n, 1, 1)
            edges_all = (edges_local + offset).reshape(-1, 2)

            widths_all = base_widths.unsqueeze(0).expand(n, -1).reshape(-1)

            sub_collections.append(_make_sub(n, n_each_nodes, n_each_edges, nodes_all, edges_all, widths_all))
            ordered_specs.extend(snap_specs)

        # ---- Unsnap ----
        if unsnap_specs:
            n = len(unsnap_specs)
            n_each_nodes = n_base_nodes + 1
            n_each_edges = n_base_edges
            edge_ids = torch.tensor([s.edge_id for s in unsnap_specs], device=device)
            ends = torch.tensor([s.end for s in unsnap_specs], device=device)
            proposal_idx = torch.arange(n, device=device)
            old_node = base_edges[edge_ids, ends]  # (n,)
            new_node_pos = base_nodes[old_node]  # (n,2)

            nodes_exp = base_nodes.unsqueeze(0).expand(n, -1, -1)
            nodes_all = torch.cat([nodes_exp, new_node_pos.unsqueeze(1)], dim=1).reshape(-1, 2)

            edges_local = base_edges.unsqueeze(0).expand(n, -1, -1).clone()
            edges_local[proposal_idx, edge_ids, ends] = n_base_nodes
            offset = (torch.arange(n, device=device) * n_each_nodes).view(n, 1, 1)
            edges_all = (edges_local + offset).reshape(-1, 2)

            widths_all = base_widths.unsqueeze(0).expand(n, -1).reshape(-1)

            sub_collections.append(_make_sub(n, n_each_nodes, n_each_edges, nodes_all, edges_all, widths_all))
            ordered_specs.extend(unsnap_specs)

        # ---- Widen / Narrow ----
        if widen_specs:
            n = len(widen_specs)
            n_each_nodes = n_base_nodes
            n_each_edges = n_base_edges
            edge_ids = torch.tensor([s.edge_id for s in widen_specs], device=device)
            new_w = torch.tensor([s.new_width for s in widen_specs], device=device, dtype=dtype)
            proposal_idx = torch.arange(n, device=device)

            nodes_all = base_nodes.unsqueeze(0).expand(n, -1, -1).reshape(-1, 2)
            edges_local = base_edges.unsqueeze(0).expand(n, -1, -1)
            offset = (torch.arange(n, device=device) * n_each_nodes).view(n, 1, 1)
            edges_all = (edges_local + offset).reshape(-1, 2)

            widths_local = base_widths.unsqueeze(0).expand(n, -1).clone()
            widths_local[proposal_idx, edge_ids] = new_w
            widths_all = widths_local.reshape(-1)

            sub_collections.append(_make_sub(n, n_each_nodes, n_each_edges, nodes_all, edges_all, widths_all))
            ordered_specs.extend(widen_specs)

        if not sub_collections:
            return self._Collection.from_objects([obj]), []

        return self._Collection.cat(sub_collections), ordered_specs

    def combine_proposals(
        self,
        base: RoadNetwork,
        proposals: ObjectCollection[RoadNetwork],
        base_loss: float,
        proposal_losses: list[float],
        proposal_specs: list[RoadRewrite],
        accept_parallel: bool = True,
    ) -> tuple[RoadNetwork, bool]:
        assert isinstance(proposals, RoadNetworkCollection)
        scores: list[float] = []
        candidates: list[RoadRewrite] = []
        for i in range(len(proposals)):
            improvement = base_loss - proposal_losses[i]
            if improvement > self.args.better_abs_eps:
                scores.append(improvement)
                candidates.append(proposal_specs[i])

        if not accept_parallel:
            if candidates:
                candidates = candidates[:1]
                scores = scores[:1]
            else:
                candidates = []
                scores = []

        if candidates:
            max_width = max(self.args.rewrite_args.width_classes)
            return base.apply_all_rewrites(candidates, scores, max_width=max_width), True
        return base, False

    def initialize_state(self) -> StateT:
        return None  # type: ignore[return-value]

    def cleanup(self, collection: ObjectCollection[RoadNetwork]) -> ObjectCollection[RoadNetwork]:
        assert isinstance(collection, RoadNetworkCollection)
        return self._Collection.from_objects([net.prune_orphan_nodes().cleanup() for net in collection])


class RoadDensityTask(RoadTask[None]):
    def __init__(self, args: RoadArgs, render_args: RenderArgs, target_img: torch.Tensor):
        RoadTask.__init__(self, args, render_args, target_img.device)
        # target_img(pngs/shcどちらの入力元でも)を target_inside_value/target_outside_value の範囲へ
        # アフィン変換する。以降はこの変換済みの値を target として扱う。
        self.target_img = args.target_outside_value + (args.target_inside_value - args.target_outside_value) * target_img

    def initialize_state(self) -> None:
        return None

    def _compute_losses(
        self, collection: ObjectCollection[RoadNetwork], state: None
    ) -> tuple[torch.Tensor, ExtraMetrics]:
        assert isinstance(collection, RoadNetworkCollection)
        density = collection.compute_density(
            self.render_args.size, self.render_args.lim, center_pixel=self.render_args.center_pixel
        )  # (n_networks, size, size)
        # 最低ラインは compute_density 側で無条件保証済み(density は floor を下回らない)ので、
        # ここでは素直に target とのMSEのみでよい。
        loss = (density - self.target_img).square().flatten(-2).mean(dim=-1)
        angle_penalty = collection.get_angle_penalty(self.args.min_angle, self.args.angle_penalty_exponent)
        loss = loss + self.args.angle_weight * angle_penalty
        return loss, {}

    def visualize(self, collection: ObjectCollection[RoadNetwork], step: int, loss: float, state: None) -> np.ndarray:
        assert isinstance(collection, RoadNetworkCollection)
        assert len(collection) == 1
        net = collection[0]
        fig = MPLVisualizer(1, 1, 10.8, 10.8, xlim=self.render_args.lim, ylim=self.render_args.lim, notebook=False)
        ax = fig[0]
        extent = (self.render_args.lim[0], self.render_args.lim[1], self.render_args.lim[1], self.render_args.lim[0])
        ax.ax.imshow(
            self.target_img.detach().cpu().numpy(), extent=extent, cmap="plasma", vmin=0, vmax=1, alpha=0.25
        )
        density = collection.compute_density(
            self.render_args.size, self.render_args.lim, center_pixel=self.render_args.center_pixel
        )
        ax.ax.imshow(density[0].detach().cpu().numpy(), extent=extent, cmap="magma", vmin=0, vmax=1, alpha=0.55)
        net.visualize(ax)
        ax.ax.set_title(f"{self.get_elapsed_time():.0f}s: {net.id}: {loss:.2e}: E{len(net.edges)}")
        return fig.get_image()


# check abstract methods
if __name__ == "__main__":
    RoadDensityTask(RoadArgs(), RenderArgs(), torch.zeros(256, 256))
