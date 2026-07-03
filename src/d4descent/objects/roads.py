import torch
import math
import random
from dataclasses import dataclass, field
from typing import Optional, Self, Union, Type, cast
from enum import Enum
from functools import partialmethod

from ..context import Context
from ..object_collection import ObjectCollection
from ..util import maybe_detach
from ..visualizer import MPLVisualizerAxes


# region Rewrites
# ================== Rewrites ===========================


class RoadRewriteType(Enum):
    Add = 1
    AddFree = 2
    Remove = 3
    Snap = 4


@dataclass
class RoadRewrite:
    rewrite_type: RoadRewriteType


@dataclass
class RoadRewriteAdd(RoadRewrite):
    """既存ノード from_node から新規ノード (x, y) へ道を伸ばす。分岐も同じ操作（from_node の次数が増えるだけ）。"""

    rewrite_type: RoadRewriteType = field(default=RoadRewriteType.Add, init=False)
    from_node: int
    x: float
    y: float
    width: float


@dataclass
class RoadRewriteAddFree(RoadRewrite):
    """既存構造と繋がらない新規孤立道路（2ノード+1エッジ）を追加する。"""

    rewrite_type: RoadRewriteType = field(default=RoadRewriteType.AddFree, init=False)
    x1: float
    y1: float
    x2: float
    y2: float
    width: float


@dataclass
class RoadRewriteRemove(RoadRewrite):
    rewrite_type: RoadRewriteType = field(default=RoadRewriteType.Remove, init=False)
    edge_id: int


@dataclass
class RoadRewriteSnap(RoadRewrite):
    """edge_id の end側(0 or 1)の端点(次数1のぶら下がりノード)を、既存ノード target_node に張り替える。"""

    rewrite_type: RoadRewriteType = field(default=RoadRewriteType.Snap, init=False)
    edge_id: int
    end: int
    target_node: int


@dataclass
class RoadRewriteArgs:
    width_classes: tuple[float, ...] = (0.02, 0.05)
    length_range: tuple[float, float] = (0.05, 0.15)
    n_add_candidates: int = 32
    n_free_candidates: int = 8
    snap_radius: float = 0.05
    add_weight: float = 1.0
    add_free_weight: float = 1.0
    remove_weight: float = 1.0
    snap_weight: float = 1.0


def find_nearby_nodes(
    nodes: torch.Tensor, query_idx: torch.Tensor, cell_size: float, radius: float
) -> list[list[int]]:
    """
    一様グリッドによる空間ハッシュで近傍探索を行う（全ペア総当たりのO(n^2)を避ける）。
    セルサイズ=radius とすることで、自セル+周囲8近傍だけを調べればよい。

    nodes: (n_nodes, 2)
    query_idx: (k,) 近傍を調べたいノードのインデックス
    returns: 各クエリについて、半径内にある他ノードのインデックス（距離の近い順）
    """
    with torch.no_grad():
        pos = nodes.detach().cpu().numpy()
        cell = max(cell_size, 1e-9)
        buckets: dict[tuple[int, int], list[int]] = {}
        for i, (x, y) in enumerate(pos):
            key = (math.floor(x / cell), math.floor(y / cell))
            buckets.setdefault(key, []).append(i)

        results: list[list[int]] = []
        for qi in query_idx.tolist():
            x, y = pos[qi]
            cx, cy = math.floor(x / cell), math.floor(y / cell)
            cands: list[tuple[float, int]] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in buckets.get((cx + dx, cy + dy), ()):
                        if j == qi:
                            continue
                        d = math.hypot(pos[j][0] - x, pos[j][1] - y)
                        if d <= radius:
                            cands.append((d, j))
            cands.sort(key=lambda t: t[0])
            results.append([j for _, j in cands])
    return results


# endregion
# region RoadNetwork
# ================== RoadNetwork ===========================


@dataclass
class RoadPayload:
    pass


@dataclass
class RoadNetwork:
    nodes: torch.Tensor  # (n_nodes, 2) 座標。勾配対象
    edges: torch.Tensor  # (n_edges, 2) long。ノードindexのペア
    widths: torch.Tensor  # (n_edges,) 道路幅。離散固定値、勾配対象外
    id: int = field(default_factory=lambda: Context.get().gen_id())
    payload: RoadPayload = field(default_factory=RoadPayload)

    def __post_init__(self):
        assert self.nodes.ndim == 2 and self.nodes.shape[-1] == 2, f"nodes must be (n,2), got {self.nodes.shape}"
        assert self.edges.ndim == 2 and self.edges.shape[-1] == 2, f"edges must be (n,2), got {self.edges.shape}"
        assert len(self.edges) == len(self.widths), "len(edges) != len(widths)"

    def device(self) -> torch.device:
        return self.nodes.device

    def _degree(self) -> torch.Tensor:
        n_nodes = len(self.nodes)
        degree = torch.zeros(n_nodes, dtype=torch.long, device=self.nodes.device)
        degree.scatter_add_(0, self.edges.flatten(), torch.ones(2 * len(self.edges), dtype=torch.long, device=self.nodes.device))
        return degree

    def visualize(
        self,
        ax: MPLVisualizerAxes,
        color: str = "white",
        min_lw: float = 1.0,
        max_lw: float = 6.0,
    ) -> None:
        widths = self.widths.tolist()
        if not widths:
            return
        min_w, max_w = min(widths), max(widths)
        span = max(max_w - min_w, 1e-9)
        nodes = self.nodes.tolist()
        for (i, j), w in zip(self.edges.tolist(), widths):
            (x0, y0), (x1, y1) = nodes[i], nodes[j]
            lw = min_lw + (max_lw - min_lw) * (w - min_w) / span
            ax.ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, solid_capstyle="round", zorder=5)

    def prune_orphan_nodes(self) -> "RoadNetwork":
        """どのエッジからも参照されなくなったノードを削除し、エッジのインデックスを詰め直す。"""
        degree = self._degree()
        keep_mask = degree > 0
        if bool(keep_mask.all()):
            return self
        device = self.nodes.device
        new_index = torch.full((len(self.nodes),), -1, dtype=torch.long, device=device)
        new_index[keep_mask] = torch.arange(int(keep_mask.sum().item()), device=device)
        return RoadNetwork(
            nodes=self.nodes[keep_mask],
            edges=new_index[self.edges],
            widths=self.widths,
            id=self.id,
            payload=self.payload,
        )

    def gen_rewrite_specs(self, args: RoadRewriteArgs, lim: tuple[float, float]) -> list[RoadRewrite]:
        device = self.nodes.device
        n_edges = len(self.edges)
        degree = self._degree()
        live_nodes = degree.nonzero().flatten().tolist()
        leaf_nodes = (degree == 1).nonzero().flatten().tolist()

        specs: list[RoadRewrite] = []

        if args.add_weight > 0 and live_nodes:
            n_cand = max(args.n_add_candidates, 1)
            pos = self.nodes.detach().cpu()
            for _ in range(n_cand):
                ni = random.choice(live_nodes)
                ang = random.random() * 2 * math.pi
                length = random.uniform(*args.length_range)
                x0, y0 = pos[ni].tolist()
                x1 = x0 + length * math.cos(ang)
                y1 = y0 + length * math.sin(ang)
                for w in args.width_classes:
                    specs.append(RoadRewriteAdd(from_node=ni, x=x1, y=y1, width=w))

        if args.add_free_weight > 0:
            n_cand = max(args.n_free_candidates, 1)
            lim0, lim1 = lim
            for _ in range(n_cand):
                x0 = random.uniform(lim0, lim1)
                y0 = random.uniform(lim0, lim1)
                ang = random.random() * 2 * math.pi
                length = random.uniform(*args.length_range)
                x1 = x0 + length * math.cos(ang)
                y1 = y0 + length * math.sin(ang)
                for w in args.width_classes:
                    specs.append(RoadRewriteAddFree(x1=x0, y1=y0, x2=x1, y2=y1, width=w))

        if args.remove_weight > 0 and n_edges > 1:
            edges_list = self.edges.tolist()
            for eid, (a, b) in enumerate(edges_list):
                if degree[a].item() == 1 or degree[b].item() == 1:
                    specs.append(RoadRewriteRemove(edge_id=eid))

        if args.snap_weight > 0 and leaf_nodes:
            edges_list = self.edges.tolist()
            incident: dict[int, tuple[int, int]] = {}
            for eid, (a, b) in enumerate(edges_list):
                if degree[a].item() == 1:
                    incident[a] = (eid, 0)
                if degree[b].item() == 1:
                    incident[b] = (eid, 1)
            leaf_idx_t = torch.tensor(leaf_nodes, dtype=torch.long, device=device)
            neighbor_lists = find_nearby_nodes(self.nodes, leaf_idx_t, cell_size=args.snap_radius, radius=args.snap_radius)
            for leaf, neighbors in zip(leaf_nodes, neighbor_lists):
                eid, end = incident[leaf]
                other = edges_list[eid][1 - end]
                for target in neighbors:
                    if target == other or target == leaf:
                        continue
                    specs.append(RoadRewriteSnap(edge_id=eid, end=end, target_node=target))

        return specs

    def apply_rewrite(self, spec: RoadRewrite) -> "RoadNetwork":
        device = self.nodes.device
        dtype = self.nodes.dtype
        if isinstance(spec, RoadRewriteAdd):
            new_node = torch.tensor([[spec.x, spec.y]], dtype=dtype, device=device)
            new_edge = torch.tensor([[spec.from_node, len(self.nodes)]], dtype=torch.long, device=device)
            new_width = torch.tensor([spec.width], dtype=dtype, device=device)
            return RoadNetwork(
                nodes=torch.cat([self.nodes, new_node], dim=0),
                edges=torch.cat([self.edges, new_edge], dim=0),
                widths=torch.cat([self.widths, new_width], dim=0),
            )
        elif isinstance(spec, RoadRewriteAddFree):
            n0 = len(self.nodes)
            new_nodes = torch.tensor([[spec.x1, spec.y1], [spec.x2, spec.y2]], dtype=dtype, device=device)
            new_edge = torch.tensor([[n0, n0 + 1]], dtype=torch.long, device=device)
            new_width = torch.tensor([spec.width], dtype=dtype, device=device)
            return RoadNetwork(
                nodes=torch.cat([self.nodes, new_nodes], dim=0),
                edges=torch.cat([self.edges, new_edge], dim=0),
                widths=torch.cat([self.widths, new_width], dim=0),
            )
        elif isinstance(spec, RoadRewriteRemove):
            keep = torch.ones(len(self.edges), dtype=torch.bool, device=device)
            keep[spec.edge_id] = False
            return RoadNetwork(nodes=self.nodes, edges=self.edges[keep], widths=self.widths[keep])
        elif isinstance(spec, RoadRewriteSnap):
            new_edges = self.edges.clone()
            new_edges[spec.edge_id, spec.end] = spec.target_node
            return RoadNetwork(nodes=self.nodes, edges=new_edges, widths=self.widths)
        else:
            raise ValueError(f"Unknown rewrite {spec}")

    def apply_all_rewrites(self, rewrites: list[RoadRewrite], scores: list[float]) -> "RoadNetwork":
        """スコア降順で適用する。同じエッジを対象とする Remove/Snap は先勝ちとし、以降は無視する。"""
        order = sorted(range(len(rewrites)), key=lambda i: scores[i], reverse=True)
        device = self.nodes.device
        dtype = self.nodes.dtype

        added_nodes: list[torch.Tensor] = []
        added_edges: list[torch.Tensor] = []
        added_widths: list[torch.Tensor] = []
        n_next = len(self.nodes)

        touched_edges: set[int] = set()
        removed: set[int] = set()
        snap_updates: dict[int, tuple[int, int]] = {}
        n_alive = len(self.edges)

        for i in order:
            spec = rewrites[i]
            if isinstance(spec, RoadRewriteAdd):
                added_nodes.append(torch.tensor([[spec.x, spec.y]], dtype=dtype, device=device))
                added_edges.append(torch.tensor([[spec.from_node, n_next]], dtype=torch.long, device=device))
                added_widths.append(torch.tensor([spec.width], dtype=dtype, device=device))
                n_next += 1
            elif isinstance(spec, RoadRewriteAddFree):
                added_nodes.append(torch.tensor([[spec.x1, spec.y1], [spec.x2, spec.y2]], dtype=dtype, device=device))
                added_edges.append(torch.tensor([[n_next, n_next + 1]], dtype=torch.long, device=device))
                added_widths.append(torch.tensor([spec.width], dtype=dtype, device=device))
                n_next += 2
            elif isinstance(spec, RoadRewriteRemove):
                if spec.edge_id in touched_edges or n_alive <= 1:
                    continue
                touched_edges.add(spec.edge_id)
                removed.add(spec.edge_id)
                n_alive -= 1
            elif isinstance(spec, RoadRewriteSnap):
                if spec.edge_id in touched_edges:
                    continue
                touched_edges.add(spec.edge_id)
                snap_updates[spec.edge_id] = (spec.end, spec.target_node)
            else:
                raise ValueError(f"Unknown rewrite {spec}")

        edges = self.edges.clone()
        for eid, (end, target) in snap_updates.items():
            edges[eid, end] = target
        if removed:
            keep = torch.ones(len(edges), dtype=torch.bool, device=device)
            for eid in removed:
                keep[eid] = False
            edges = edges[keep]
            widths = self.widths[keep]
        else:
            widths = self.widths.clone()
        nodes = self.nodes.detach().clone()

        if added_nodes:
            nodes = torch.cat([nodes, *added_nodes], dim=0)
            edges = torch.cat([edges, *added_edges], dim=0)
            widths = torch.cat([widths, *added_widths], dim=0)

        return RoadNetwork(nodes=nodes, edges=edges, widths=widths)


# endregion
# region RoadNetworkCollection
# ================== RoadNetworkCollection ===========================


@dataclass
class RoadCollectionArgs:
    sigma0: float = 0.03  # street相当の最小到達半径
    k_sigma: float = 1.0  # 幅 -> 到達半径の係数
    baseline_density: float = 0.05  # 道路から離れた領域の最低密度


def _always_raise() -> RoadCollectionArgs:
    raise ValueError("RoadCollectionArgs must be set")


def _capsule_sdf(positions: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
    """
    positions: (*shape, 2)
    p0, p1, widths: (n_edges, 2), (n_edges, 2), (n_edges,)
    returns: (n_edges, *shape) 距離 - 幅/2 (負=内側)
    """
    ee = p1 - p0  # (n_edges, 2)
    ep = positions.unsqueeze(-2) - p0  # (*shape, n_edges, 2)
    denom = ee.square().sum(dim=-1).clamp(min=1e-12)  # (n_edges,)
    t = (ep * ee).sum(dim=-1) / denom  # (*shape, n_edges)
    t = t.clamp(0.0, 1.0)
    closest = ep - ee * t.unsqueeze(-1)  # (*shape, n_edges, 2)
    dist = closest.norm(dim=-1)  # (*shape, n_edges)
    sdf = dist - widths / 2  # (*shape, n_edges)
    return sdf.movedim(-1, 0)  # (n_edges, *shape)


@dataclass
class RoadNetworkCollection(ObjectCollection[RoadNetwork]):
    nodes: torch.Tensor  # (total_nodes, 2) 全ネットワーク結合。勾配対象
    edges: torch.Tensor  # (total_edges, 2) long。グローバルnode index
    widths: torch.Tensor  # (total_edges,) 勾配対象外
    edge_index_of: torch.Tensor  # (total_edges,) 各エッジがどのネットワーク(=object)に属するか
    node_ranges: tuple[tuple[int, int], ...]
    edge_ranges: tuple[tuple[int, int], ...]
    ids: tuple[int, ...]
    payloads: tuple[RoadPayload, ...]
    args: RoadCollectionArgs = field(default_factory=_always_raise)

    def __post_init__(self):
        assert self.nodes.ndim == 2 and self.nodes.shape[-1] == 2, f"nodes must be (n,2), got {self.nodes.shape}"
        assert self.edges.ndim == 2 and self.edges.shape[-1] == 2, f"edges must be (n,2), got {self.edges.shape}"
        assert len(self.edges) == len(self.widths) == len(self.edge_index_of)
        assert len(self.node_ranges) == len(self.edge_ranges) == len(self.ids) == len(self.payloads)

    def __len__(self) -> int:
        return len(self.ids)

    def device(self) -> torch.device:
        return self.nodes.device

    def parameters(self) -> list[torch.Tensor]:
        return [self.nodes]

    def parameter_names(self) -> list[str]:
        return ["nodes"]

    def per_object_grads(self) -> list[torch.Tensor]:
        grads: list[torch.Tensor] = []
        nodes_grad = self.nodes.grad
        for s, e in self.node_ranges:
            if nodes_grad is not None:
                grads.append(nodes_grad[s:e, ...].flatten())
            else:
                grads.append(torch.zeros(0, device=self.device()))
        return grads

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        self.nodes = self.nodes.detach().clone().requires_grad_(requires_grad)
        return self

    def clone(self) -> Self:
        return self.__class__(
            nodes=self.nodes.detach().clone(),
            edges=self.edges.clone(),
            widths=self.widths.clone(),
            edge_index_of=self.edge_index_of.clone(),
            node_ranges=self.node_ranges,
            edge_ranges=self.edge_ranges,
            ids=self.ids,
            payloads=self.payloads,
            args=self.args,
        )

    def to(self, device: Union[str, torch.device, None] = None) -> Self:
        return self.__class__(
            nodes=self.nodes.to(device=device),
            edges=self.edges.to(device=device),
            widths=self.widths.to(device=device),
            edge_index_of=self.edge_index_of.to(device=device),
            node_ranges=self.node_ranges,
            edge_ranges=self.edge_ranges,
            ids=self.ids,
            payloads=self.payloads,
            args=self.args,
        )

    def get_object(self, idx: int, detach: bool = True) -> RoadNetwork:
        ns, ne = self.node_ranges[idx]
        es, ee = self.edge_ranges[idx]
        return RoadNetwork(
            nodes=maybe_detach(self.nodes[ns:ne], detach),
            edges=(self.edges[es:ee] - ns).clone(),
            widths=maybe_detach(self.widths[es:ee], detach),
            id=self.ids[idx],
            payload=self.payloads[idx],
        )

    @classmethod
    def from_object(cls, object: RoadNetwork, **kwargs) -> Self:
        device = object.nodes.device
        n_nodes = len(object.nodes)
        n_edges = len(object.edges)
        return cls(
            nodes=object.nodes.detach().clone(),
            edges=object.edges.detach().clone(),
            widths=object.widths.detach().clone(),
            edge_index_of=torch.zeros(n_edges, dtype=torch.long, device=device),
            node_ranges=((0, n_nodes),),
            edge_ranges=((0, n_edges),),
            ids=(object.id,),
            payloads=(object.payload,),
            **kwargs,
        )

    @classmethod
    def cat(cls, collections: list["RoadNetworkCollection"], **kwargs) -> Self:
        collections_ = cast(list[RoadNetworkCollection], collections)
        assert len(collections_) > 0, "collections must not be empty"
        node_offset = 0
        edge_offset = 0
        network_offset = 0
        all_nodes: list[torch.Tensor] = []
        all_edges: list[torch.Tensor] = []
        all_widths: list[torch.Tensor] = []
        all_edge_index_of: list[torch.Tensor] = []
        all_node_ranges: list[tuple[int, int]] = []
        all_edge_ranges: list[tuple[int, int]] = []
        all_ids: tuple[int, ...] = ()
        all_payloads: tuple[RoadPayload, ...] = ()
        for c in collections_:
            n_nodes = len(c.nodes)
            n_edges = len(c.edges)
            all_nodes.append(c.nodes)
            all_edges.append(c.edges + node_offset)
            all_widths.append(c.widths)
            all_edge_index_of.append(c.edge_index_of + network_offset)
            all_node_ranges.extend((s + node_offset, e + node_offset) for s, e in c.node_ranges)
            all_edge_ranges.extend((s + edge_offset, e + edge_offset) for s, e in c.edge_ranges)
            all_ids += c.ids
            all_payloads += c.payloads
            node_offset += n_nodes
            edge_offset += n_edges
            network_offset += len(c.ids)
        return cls(
            nodes=torch.cat(all_nodes, dim=0),
            edges=torch.cat(all_edges, dim=0),
            widths=torch.cat(all_widths, dim=0),
            edge_index_of=torch.cat(all_edge_index_of, dim=0),
            node_ranges=tuple(all_node_ranges),
            edge_ranges=tuple(all_edge_ranges),
            ids=all_ids,
            payloads=all_payloads,
            **kwargs,
        )

    def rasterize(self, positions: torch.Tensor) -> torch.Tensor:
        """
        positions: (*shape, 2)
        returns: (n_networks, *shape) 各ネットワークのSDF（境界判定・可視化用。損失には compute_density を使う）
        """
        *shape, _ = positions.shape
        p0 = self.nodes[self.edges[:, 0]]
        p1 = self.nodes[self.edges[:, 1]]
        sdf = _capsule_sdf(positions, p0, p1, self.widths)  # (total_edges, *shape)
        n_networks = len(self.ids)
        result = torch.full((n_networks, *shape), torch.inf, dtype=sdf.dtype, device=sdf.device)
        index = self.edge_index_of.view(-1, *([1] * len(shape))).expand(-1, *shape)
        return torch.scatter_reduce(result, 0, index, sdf, reduce="amin")

    def compute_density(
        self, size: int, lim: tuple[float, float] = (-1.5, 1.5), center_pixel: bool = True
    ) -> torch.Tensor:
        """
        人口密度予測値を合成する。
        c_e(x)  = exp(-(relu(sdf_e(x)) / sigma_e)^2)
        raw(x)  = 1 - Π_e (1 - c_e(x))   (log1pで数値安定に計算)
        density = baseline + (1 - baseline) * raw(x)

        returns: (n_networks, size, size)
        """
        device = self.device()
        lim0, lim1 = lim
        if center_pixel:
            basis = (torch.arange(size, device=device) + 0.5) / size * (lim1 - lim0) + lim0
        else:
            basis = torch.linspace(lim0, lim1, size, device=device)
        xs = basis.expand(size, -1)
        ys = basis.unsqueeze(-1).expand(-1, size)
        grid = torch.stack([xs, ys], dim=-1)  # (size, size, 2)

        p0 = self.nodes[self.edges[:, 0]]
        p1 = self.nodes[self.edges[:, 1]]
        sdf = _capsule_sdf(grid, p0, p1, self.widths)  # (total_edges, size, size)
        d = sdf.clamp(min=0.0)
        sigma = (self.args.sigma0 + self.args.k_sigma * self.widths).clamp(min=1e-6)  # (total_edges,)
        c = torch.exp(-(d / sigma.view(-1, 1, 1)).square())
        log1mc = torch.log1p(-c.clamp(max=1 - 1e-6))  # (total_edges, size, size)

        n_networks = len(self.ids)
        result = torch.zeros((n_networks, size, size), dtype=log1mc.dtype, device=log1mc.device)
        index = self.edge_index_of.view(-1, 1, 1).expand(-1, size, size)
        sum_log1mc = torch.scatter_reduce(result, 0, index, log1mc, reduce="sum")
        raw = 1 - torch.exp(sum_log1mc)
        baseline = self.args.baseline_density
        return baseline + (1 - baseline) * raw

    def get_construction_costs(self) -> torch.Tensor:
        """長さ×幅 の総和（建設コスト相当）をネットワークごとに集計する。"""
        p0 = self.nodes[self.edges[:, 0]]
        p1 = self.nodes[self.edges[:, 1]]
        cost = (p1 - p0).norm(dim=-1) * self.widths  # (total_edges,)
        n_networks = len(self.ids)
        result = torch.zeros(n_networks, device=cost.device, dtype=cost.dtype)
        return torch.scatter_reduce(result, 0, self.edge_index_of, cost, reduce="sum")

    def get_sizes(self) -> list[int]:
        return [e - s for s, e in self.edge_ranges]

    @classmethod
    def patch_args(cls, args: RoadCollectionArgs) -> Type["RoadNetworkCollection"]:
        return type(
            "RoadNetworkCollectionWithArgs",
            (RoadNetworkCollection,),
            {"__init__": partialmethod(RoadNetworkCollection.__init__, args=args)},
        )

    def to_savable(self) -> "RoadNetworkCollection":
        return RoadNetworkCollection(
            nodes=self.nodes,
            edges=self.edges,
            widths=self.widths,
            edge_index_of=self.edge_index_of,
            node_ranges=self.node_ranges,
            edge_ranges=self.edge_ranges,
            ids=self.ids,
            payloads=self.payloads,
            args=self.args,
        )

    def project_to_valid_(self) -> Self:
        return self
