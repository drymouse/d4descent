import math
import random
from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np
import torch

from ..context import Context
from ..object_collection import ObjectCollection
from ..objects.city import (
    CityNetwork,
    CityNetworkCollection,
    CityPayload,
    CityRewrite,
    CityRewriteAdd,
    CityRewriteAddAnywhere,
    CityRewriteArgs,
    CityRewriteBranch,
    CityRewriteMerge,
    CityRewriteRemove,
    CityRewriteSnap,
    CityRewriteSplit,
    CityRewriteUnsnap,
    CityCollectionArgs,
)
from ._base import Task, TaskArgs, RenderArgs, StateT, ExtraMetrics
from ..visualizer import MPLVisualizer
from ..losses._base import LossArgs
from ..losses.raster import RasterLossMixin, RasterLossArgs


@dataclass
class CityArgs(TaskArgs):
    # Road文法から幅クラス階層(highway/street)を取り除いた単純化版。道路は単一種類(street)のみ。
    # 損失は RasterLossMixin(render01 と target_img の素直なMSE)をそのまま使い、道路網が図形の
    # 内部だけを(render01が1に近づくよう)密に埋め尽くすように最適化する。図形の外側は target=0
    # なので、外へ伸びる道路はMSEを悪化させ自然に却下される(専用の内外判定は不要)。
    cost_weight: float = 1e-4  # 建設コストの"離散"正則化重み(compute_simplicity)。Triのnode_weightに相当
    # 建設コストを"連続"損失にも加える重み(Triのsize_weightに相当)。冗長な道路(既に十分カバー
    # 済みの領域を通るだけの道路)を縮める方向の勾配を作る。render01は既に1の場所では追加の道路が
    # 損失を下げないため、0でも動作はするが、経路の無駄な蛇行を抑えたい場合は正の値にする。
    size_weight: float = 0.0
    # 交差点が90度格子 {90°,180°,270°} からずれることへの罰則(原則3: 90度交差を選好)。
    # get_angle_penalty が各ノードの隣接方向間のgapを90度格子からのずれで罰する。90/180/270°
    # (直進・直角カーブ・T字・十字)は無罰、鋭角(→0°)や斜め(45°,135°)は罰する。render01損失だけだと
    # 街路が任意角度で交差するスクリブルになりがちなので、これを連続損失に加えて格子状の街路を促す。
    angle_deadzone: float = math.radians(10)  # 格子まわりの許容幅(ラジアン)。密度カバレッジとの競合を緩和
    angle_penalty_exponent: float = 2.0
    angle_weight: float = 0.05
    # 交差点の次数(接続する道路の本数)が max_degree_threshold 以上になることへの罰則。5叉路以上のような
    # 不自然な交差点を強く抑制する。get_degree_penalty が excess=degree-threshold+1 の
    # exponent乗をノードごとに罰する。Branch書き換え(辺の途中から積極的に新しい枝を伸ばす)と対にして
    # 使う: 既存ノードへのSnap/Addに集中させる代わりに、辺の途中から分岐させてネットワークを広げさせる。
    max_degree_threshold: int = 5
    degree_penalty_exponent: float = 2.0
    degree_penalty_weight: float = 1.0  # render01(スケール~0.1-0.3)やangle_weight(0.05)に対して意図的に大きい
    better_abs_eps: float = 1e-8
    # cleanup で密集地帯のノードと接続道路を間引く(Road文法のdecimate_denseと同型)。
    decimate_dense: bool = True
    decimate_cell_size: float = 0.06
    decimate_max_per_cell: int = 2
    rewrite_args: CityRewriteArgs = field(default_factory=CityRewriteArgs)
    city_collection_args: CityCollectionArgs = field(default_factory=CityCollectionArgs)

    def create(
        self,
        render_args: RenderArgs,
        loss_args: LossArgs,
        device: Union[torch.device, str],
        target_img: Optional[torch.Tensor] = None,
    ) -> "Task":
        if isinstance(loss_args, RasterLossArgs):
            assert target_img is not None, "target_img (図形のシルエット, [0,1]) must be provided"
            return CityRasterTask(self, render_args, loss_args, target_img)
        else:
            raise NotImplementedError(f"Unknown loss_args type: {type(loss_args)}")


class CityTask(Task[CityNetwork, CityRewrite, StateT]):
    def __init__(self, args: CityArgs, render_args: RenderArgs, device: Union[str, torch.device]):
        super().__init__(render_args)
        self._device = torch.device(device)
        self.args = args
        self._Collection = CityNetworkCollection.patch_args(self.args.city_collection_args)

    def device(self) -> torch.device:
        return self._device

    def get_collection_constructor(self) -> type[CityNetworkCollection]:
        return self._Collection

    def initialize_object(self) -> CityNetwork:
        device = self.device()
        return CityNetwork(
            nodes=torch.tensor([[-0.01, 0.0], [0.01, 0.0]], device=device),
            edges=torch.tensor([[0, 1]], dtype=torch.long, device=device),
        )

    def compute_simplicity(self, collection: ObjectCollection[CityNetwork]) -> list[float]:
        assert isinstance(collection, CityNetworkCollection)
        costs = collection.get_construction_costs()
        return [c * self.args.cost_weight for c in costs.tolist()]

    def make_proposals(self, obj: CityNetwork) -> tuple[ObjectCollection[CityNetwork], list[CityRewrite]]:
        raise NotImplementedError("use make_proposals_ex")

    def get_add_anywhere_targets(self) -> Optional[torch.Tensor]:
        """AddAnywhere が狙う候補点プール。基底クラスでは None(lim 全体から一様サンプル)。
        CityRasterTask が図形内部に偏らせた点を返す。"""
        return None

    def _stratified_sample(self, specs: list[CityRewrite], num_proposals: int) -> list[CityRewrite]:
        """
        提案を書き換えタイプごとにグループ化し、num_proposals の予算をタイプ間で公平に配分して
        サンプルする(Road文法と同じラウンドロビン方式)。単純な一様サンプルだと候補数の多いタイプ
        (Add)が予算を独占し、spreading の唯一の手段である AddAnywhere などの少数タイプが評価対象
        から漏れてしまうため。
        """
        if num_proposals <= 0 or len(specs) <= num_proposals:
            return specs
        groups: dict[type, list[CityRewrite]] = {}
        for s in specs:
            groups.setdefault(type(s), []).append(s)
        for pool in groups.values():
            random.shuffle(pool)
        types = list(groups.keys())
        random.shuffle(types)
        chosen: list[CityRewrite] = []
        while len(chosen) < num_proposals:
            progressed = False
            for t in types:
                if groups[t]:
                    chosen.append(groups[t].pop())
                    progressed = True
                    if len(chosen) >= num_proposals:
                        break
            if not progressed:
                break
        return chosen

    def make_proposals_ex(
        self, obj: CityNetwork, num_proposals: int
    ) -> tuple[ObjectCollection[CityNetwork], list[CityRewrite]]:
        specs = obj.gen_rewrite_specs(
            self.args.rewrite_args, lim=self.render_args.lim, add_anywhere_targets=self.get_add_anywhere_targets()
        )
        specs = self._stratified_sample(specs, num_proposals)

        device = obj.nodes.device
        dtype = obj.nodes.dtype
        n_base_nodes = len(obj.nodes)
        n_base_edges = len(obj.edges)
        base_nodes = obj.nodes.detach()
        base_edges = obj.edges
        all_idx_edges = torch.arange(n_base_edges, device=device)

        add_specs = [s for s in specs if isinstance(s, CityRewriteAdd)]
        addany_specs = [s for s in specs if isinstance(s, CityRewriteAddAnywhere)]
        remove_specs = [s for s in specs if isinstance(s, CityRewriteRemove)]
        split_specs = [s for s in specs if isinstance(s, CityRewriteSplit)]
        branch_specs = [s for s in specs if isinstance(s, CityRewriteBranch)]
        merge_specs = [s for s in specs if isinstance(s, CityRewriteMerge)]
        snap_specs = [s for s in specs if isinstance(s, CityRewriteSnap)]
        unsnap_specs = [s for s in specs if isinstance(s, CityRewriteUnsnap)]

        sub_collections: list[CityNetworkCollection] = []
        ordered_specs: list[CityRewrite] = []
        ctx = Context.get()

        def _make_sub(
            n: int, n_each_nodes: int, n_each_edges: int, nodes_all: torch.Tensor, edges_all: torch.Tensor
        ) -> CityNetworkCollection:
            edge_index_of = torch.arange(n, device=device).repeat_interleave(n_each_edges)
            return self._Collection(
                nodes=nodes_all,
                edges=edges_all,
                edge_index_of=edge_index_of,
                node_ranges=tuple((i * n_each_nodes, (i + 1) * n_each_nodes) for i in range(n)),
                edge_ranges=tuple((i * n_each_edges, (i + 1) * n_each_edges) for i in range(n)),
                ids=tuple(ctx.gen_id() for _ in range(n)),
                payloads=tuple(CityPayload() for _ in range(n)),
            )

        # ---- Add ----
        if add_specs:
            n = len(add_specs)
            n_each_nodes = n_base_nodes + 1
            n_each_edges = n_base_edges + 1
            new_pt = torch.tensor([[s.x, s.y] for s in add_specs], device=device, dtype=dtype)
            from_node = torch.tensor([s.from_node for s in add_specs], device=device, dtype=torch.long)

            nodes_exp = base_nodes.unsqueeze(0).expand(n, -1, -1)
            nodes_all = torch.cat([nodes_exp, new_pt.unsqueeze(1)], dim=1).reshape(-1, 2)

            edges_exp = base_edges.unsqueeze(0).expand(n, -1, -1)
            new_edge = torch.stack([from_node, torch.full((n,), n_base_nodes, device=device, dtype=torch.long)], dim=-1)
            edges_local = torch.cat([edges_exp, new_edge.unsqueeze(1)], dim=1)
            offset = (torch.arange(n, device=device) * n_each_nodes).view(n, 1, 1)
            edges_all = (edges_local + offset).reshape(-1, 2)

            sub_collections.append(_make_sub(n, n_each_nodes, n_each_edges, nodes_all, edges_all))
            ordered_specs.extend(add_specs)

        # ---- AddAnywhere (鎖の長さkごとにグループ化してバッチ化) ----
        if addany_specs:
            by_k: dict[int, list[CityRewriteAddAnywhere]] = {}
            for s in addany_specs:
                by_k.setdefault(len(s.pts), []).append(s)
            for k, group in by_k.items():
                n = len(group)
                n_each_nodes = n_base_nodes + k
                n_each_edges = n_base_edges + k
                pts_t = torch.tensor([list(s.pts) for s in group], device=device, dtype=dtype)  # (n,k,2)
                from_node = torch.tensor([s.from_node for s in group], device=device, dtype=torch.long)

                nodes_exp = base_nodes.unsqueeze(0).expand(n, -1, -1)
                nodes_all = torch.cat([nodes_exp, pts_t], dim=1).reshape(-1, 2)

                hop_new_ids = (n_base_nodes + torch.arange(k, device=device)).unsqueeze(0).expand(n, -1)  # (n,k)
                hop_starts = torch.cat([from_node.unsqueeze(1), hop_new_ids[:, :-1]], dim=1)  # (n,k)
                new_edges_local = torch.stack([hop_starts, hop_new_ids], dim=-1)  # (n,k,2)
                edges_exp = base_edges.unsqueeze(0).expand(n, -1, -1)
                edges_local = torch.cat([edges_exp, new_edges_local], dim=1)
                offset = (torch.arange(n, device=device) * n_each_nodes).view(n, 1, 1)
                edges_all = (edges_local + offset).reshape(-1, 2)

                sub_collections.append(_make_sub(n, n_each_nodes, n_each_edges, nodes_all, edges_all))
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

            sub_collections.append(_make_sub(n, n_each_nodes, n_each_edges, nodes_all, edges_all))
            ordered_specs.extend(remove_specs)

        # ---- Split ----
        if split_specs:
            n = len(split_specs)
            n_each_nodes = n_base_nodes + 1
            n_each_edges = n_base_edges + 1
            edge_ids = torch.tensor([s.edge_id for s in split_specs], device=device)
            new_pt = torch.tensor([[s.x, s.y] for s in split_specs], device=device, dtype=dtype)
            ab = base_edges[edge_ids]  # (n,2)
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

            sub_collections.append(_make_sub(n, n_each_nodes, n_each_edges, nodes_all, edges_all))
            ordered_specs.extend(split_specs)

        # ---- Branch (辺の途中を分割し、そこから新しい枝を伸ばす。Split+Addの複合) ----
        if branch_specs:
            n = len(branch_specs)
            n_each_nodes = n_base_nodes + 2  # 分岐点 + 枝の先端
            n_each_edges = n_base_edges + 2  # 元エッジ(-1) + 新規3本(+3)
            edge_ids = torch.tensor([s.edge_id for s in branch_specs], device=device)
            split_pt = torch.tensor([[s.x, s.y] for s in branch_specs], device=device, dtype=dtype)
            branch_pt = torch.tensor([[s.bx, s.by] for s in branch_specs], device=device, dtype=dtype)
            ab = base_edges[edge_ids]  # (n,2)
            keep_mask = all_idx_edges.unsqueeze(0) != edge_ids.unsqueeze(1)
            keep_indices = all_idx_edges.unsqueeze(0).expand(n, -1)[keep_mask].reshape(n, n_base_edges - 1)

            nodes_exp = base_nodes.unsqueeze(0).expand(n, -1, -1)
            nodes_all = torch.cat([nodes_exp, split_pt.unsqueeze(1), branch_pt.unsqueeze(1)], dim=1).reshape(-1, 2)

            edges_exp = base_edges.unsqueeze(0).expand(n, -1, -1)
            kept_edges = torch.gather(edges_exp, 1, keep_indices.unsqueeze(-1).expand(-1, -1, 2))
            split_id = torch.full((n,), n_base_nodes, device=device, dtype=torch.long)
            branch_id = torch.full((n,), n_base_nodes + 1, device=device, dtype=torch.long)
            new_edge1 = torch.stack([ab[:, 0], split_id], dim=-1)
            new_edge2 = torch.stack([split_id, ab[:, 1]], dim=-1)
            new_edge3 = torch.stack([split_id, branch_id], dim=-1)
            edges_local = torch.cat(
                [kept_edges, new_edge1.unsqueeze(1), new_edge2.unsqueeze(1), new_edge3.unsqueeze(1)], dim=1
            )
            offset = (torch.arange(n, device=device) * n_each_nodes).view(n, 1, 1)
            edges_all = (edges_local + offset).reshape(-1, 2)

            sub_collections.append(_make_sub(n, n_each_nodes, n_each_edges, nodes_all, edges_all))
            ordered_specs.extend(branch_specs)

        # ---- Merge ----
        if merge_specs:
            n = len(merge_specs)
            n_each_nodes = n_base_nodes
            n_each_edges = n_base_edges - 1
            ea = torch.tensor([s.edge_id_a for s in merge_specs], device=device)
            eb = torch.tensor([s.edge_id_b for s in merge_specs], device=device)
            outer_a = torch.tensor([s.outer_a for s in merge_specs], device=device, dtype=torch.long)
            outer_b = torch.tensor([s.outer_b for s in merge_specs], device=device, dtype=torch.long)
            keep_mask = (all_idx_edges.unsqueeze(0) != ea.unsqueeze(1)) & (all_idx_edges.unsqueeze(0) != eb.unsqueeze(1))
            keep_indices = all_idx_edges.unsqueeze(0).expand(n, -1)[keep_mask].reshape(n, n_base_edges - 2)

            nodes_all = base_nodes.unsqueeze(0).expand(n, -1, -1).reshape(-1, 2)

            edges_exp = base_edges.unsqueeze(0).expand(n, -1, -1)
            kept_edges = torch.gather(edges_exp, 1, keep_indices.unsqueeze(-1).expand(-1, -1, 2))
            new_edge = torch.stack([outer_a, outer_b], dim=-1)
            edges_local = torch.cat([kept_edges, new_edge.unsqueeze(1)], dim=1)
            offset = (torch.arange(n, device=device) * n_each_nodes).view(n, 1, 1)
            edges_all = (edges_local + offset).reshape(-1, 2)

            sub_collections.append(_make_sub(n, n_each_nodes, n_each_edges, nodes_all, edges_all))
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

            sub_collections.append(_make_sub(n, n_each_nodes, n_each_edges, nodes_all, edges_all))
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

            sub_collections.append(_make_sub(n, n_each_nodes, n_each_edges, nodes_all, edges_all))
            ordered_specs.extend(unsnap_specs)

        if not sub_collections:
            return self._Collection.from_objects([obj]), []

        return self._Collection.cat(sub_collections), ordered_specs

    def combine_proposals(
        self,
        base: CityNetwork,
        proposals: ObjectCollection[CityNetwork],
        base_loss: float,
        proposal_losses: list[float],
        proposal_specs: list[CityRewrite],
        accept_parallel: bool = True,
    ) -> tuple[CityNetwork, bool]:
        assert isinstance(proposals, CityNetworkCollection)
        scores: list[float] = []
        candidates: list[CityRewrite] = []
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
            return base.apply_all_rewrites(candidates, scores), True
        return base, False

    def initialize_state(self) -> StateT:
        return None  # type: ignore[return-value]

    def cleanup(self, collection: ObjectCollection[CityNetwork]) -> ObjectCollection[CityNetwork]:
        assert isinstance(collection, CityNetworkCollection)
        result: list[CityNetwork] = []
        for net in collection:
            net = net.prune_orphan_nodes()
            if self.args.decimate_dense:
                net = net.decimate_dense(cell_size=self.args.decimate_cell_size, max_per_cell=self.args.decimate_max_per_cell)
            result.append(net)
        return self._Collection.from_objects(result)


class CityRasterTask(RasterLossMixin[CityNetwork, CityRewrite, None], CityTask[None]):
    def __init__(self, args: CityArgs, render_args: RenderArgs, raster_args: RasterLossArgs, target_img: torch.Tensor):
        device = target_img.device
        CityTask.__init__(self, args, render_args, device)
        RasterLossMixin.__init__(self, raster_args, target_img)
        self._add_anywhere_targets = self._precompute_add_anywhere_targets(target_img)

    def _precompute_add_anywhere_targets(self, target_img_raw: torch.Tensor, n_points: int = 4096) -> torch.Tensor:
        """
        AddAnywhere が狙う候補点を、target_img(高いほど図形内部)に比例した確率で事前サンプルする。
        こうすることで spreading の提案が背景ではなく図形の内部を狙うようになる。
        """
        size = self.render_args.size
        lim0, lim1 = self.render_args.lim
        probs = target_img_raw.flatten().clamp(min=0).float()
        if float(probs.sum()) <= 0:
            probs = torch.ones_like(probs)
        probs = probs / probs.sum()
        idx = torch.multinomial(probs, num_samples=n_points, replacement=True)
        rows = (idx // size).float()  # y方向(compute_densityと同じ規約: 行=y, 列=x)
        cols = (idx % size).float()  # x方向
        if self.render_args.center_pixel:
            px = (cols + 0.5) / size * (lim1 - lim0) + lim0
            py = (rows + 0.5) / size * (lim1 - lim0) + lim0
        else:
            px = cols / (size - 1) * (lim1 - lim0) + lim0
            py = rows / (size - 1) * (lim1 - lim0) + lim0
        return torch.stack([px, py], dim=-1).cpu()  # (n_points, 2)

    def get_add_anywhere_targets(self) -> Optional[torch.Tensor]:
        return self._add_anywhere_targets

    def initialize_state(self) -> None:
        return None

    def compute_losses(
        self, collection: ObjectCollection[CityNetwork], state: None
    ) -> tuple[torch.Tensor, ExtraMetrics]:
        losses, xtra = self._compute_losses(collection, state)  # RasterLossMixin: render01 と target_img のMSE
        assert isinstance(collection, CityNetworkCollection)
        if self.args.size_weight != 0.0:
            cost = collection.get_construction_costs()
            losses = losses + self.args.size_weight * cost
        if self.args.angle_weight != 0.0:
            angle_penalty = collection.get_angle_penalty(self.args.angle_penalty_exponent, self.args.angle_deadzone)
            losses = losses + self.args.angle_weight * angle_penalty
        if self.args.degree_penalty_weight != 0.0:
            degree_penalty = collection.get_degree_penalty(
                self.args.max_degree_threshold, self.args.degree_penalty_exponent
            )
            losses = losses + self.args.degree_penalty_weight * degree_penalty
        return losses, xtra

    def visualize(self, collection: ObjectCollection[CityNetwork], step: int, loss: float, state: None) -> np.ndarray:
        assert isinstance(collection, CityNetworkCollection)
        assert len(collection) == 1
        net = collection[0]
        fig = MPLVisualizer(1, 1, 10.8, 10.8, xlim=self.render_args.lim, ylim=self.render_args.lim, notebook=False)
        ax = fig[0]
        extent = (self.render_args.lim[0], self.render_args.lim[1], self.render_args.lim[1], self.render_args.lim[0])
        ax.ax.imshow(self.target_img.detach().cpu().numpy(), extent=extent, cmap="plasma", vmin=0, vmax=1, alpha=0.25)
        imgs = collection.render01(
            self.render_args.size, self.render_args.lim, center_pixel=self.render_args.center_pixel, blur=self.render_args.blur
        )
        ax.ax.imshow(imgs[0].detach().cpu().numpy(), extent=extent, cmap="winter", vmin=0, vmax=1, alpha=0.35)
        net.visualize(ax)
        ax.ax.set_title(f"{self.get_elapsed_time():.0f}s: {net.id}: {loss:.2e}: E{len(net.edges)}")
        return fig.get_image()


# check abstract methods
if __name__ == "__main__":
    CityRasterTask(CityArgs(), RenderArgs(), RasterLossArgs(), torch.zeros(256, 256))
