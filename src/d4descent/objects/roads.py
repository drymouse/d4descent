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
        casing_color: str = "black",
        min_lw: float = 1.8,
        max_lw: float = 7.0,
        casing_extra: float = 1.6,
    ) -> None:
        """
        背景のヒートマップに埋もれないよう、白の道路本体の下に一回り太い黒のケーシングを描く
        （地図でよく使われる技法）。細い道（street）でも min_lw を確保して視認できるようにする。
        """
        widths = self.widths.tolist()
        if not widths:
            return
        min_w, max_w = min(widths), max(widths)
        span = max(max_w - min_w, 1e-9)
        nodes = self.nodes.tolist()
        for (i, j), w in zip(self.edges.tolist(), widths):
            (x0, y0), (x1, y1) = nodes[i], nodes[j]
            lw = min_lw + (max_lw - min_lw) * (w - min_w) / span
            ax.ax.plot([x0, x1], [y0, y1], color=casing_color, linewidth=lw + casing_extra, solid_capstyle="round", zorder=4)
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
            n_cand = max(round(args.n_add_candidates * args.add_weight), 1)
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
            n_cand = max(round(args.n_free_candidates * args.add_free_weight), 1)
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
    k_sigma: float = 1.0  # 幅 -> 到達半径の係数（幅が太いほど広く効く）
    reach_exponent: float = 1.0  # sigma_e = sigma0 + k_sigma * w_e^reach_exponent。1より大きいほど
    # 幅の広い道路(幹線道路)の到達半径だけが不釣り合いに拡大し、幅の狭い道路(街路)はsigma0付近に留まる。
    amp_scale: float = 0.05  # ピーク強度 = amp_scale / sigma_e (<=1)。sigma(=到達半径)が広いほどピークが下がる
    # amp_scale と sigma_e から決まるピーク強度により、幹線道路(幅広->sigma大)は「薄く広く」、
    # 街路(幅狭->sigma小)は「狭く大きく」効くようにする(断面積 amplitude*sigma がほぼ一定になる)。
    min_density_floor: float = 0.15  # 人口密度の最低ライン。道路の有無によらず無条件で保証する（max で合成）。
    # 損失側の非対称罰則（下回った分だけ罰する）にすると、道路を敷かなければ床を満たせない場所では
    # 罰則を払うだけで済んでしまい「保証」にならない。無条件の下駄にすることで確実に保証する。


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
        sigma_e     = sigma0 + k_sigma * w_e^reach_exponent   # 幅が太いほど到達半径(reach)が広い
        amplitude_e = min(amp_scale / sigma_e, 1)             # 到達半径が広いほどピーク強度は下がる
        c_e(x)      = amplitude_e * exp(-(relu(sdf_e(x)) / sigma_e)^2)
        raw(x)      = max_e c_e(x)                             # 重ね合わせではなく最大値を採用
        density(x)  = max(raw(x), min_density_floor)           # 最低ラインを無条件で保証する

        「最大値」にしているのは、近くに幹線道路と街路があるとき、距離が近いというだけで幹線道路を
        優先させず、実際の寄与(c_e)が大きい方（＝街路の方が寄与が強ければ街路）を採用するため。
        幹線道路(幅広->sigma大->amplitude小)は「薄く広く」、街路(幅狭->sigma小->amplitude大)は
        「狭く大きく」効くようになる。

        最低ラインは道路の有無によらず無条件で保証する(道路が無い場所では損失側の罰則を払うだけで
        済んでしまい「保証」にならないため)。

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
        sigma = (self.args.sigma0 + self.args.k_sigma * self.widths.pow(self.args.reach_exponent)).clamp(min=1e-6)
        amplitude = (self.args.amp_scale / sigma).clamp(max=1.0)  # (total_edges,)
        c = amplitude.view(-1, 1, 1) * torch.exp(-(d / sigma.view(-1, 1, 1)).square())  # (total_edges, size, size)

        n_networks = len(self.ids)
        result = torch.zeros((n_networks, size, size), dtype=c.dtype, device=c.device)
        index = self.edge_index_of.view(-1, 1, 1).expand(-1, size, size)
        raw = torch.scatter_reduce(result, 0, index, c, reduce="amax")
        return raw.clamp(min=self.args.min_density_floor)

    def get_construction_costs(self, width_exponent: float = 1.0) -> torch.Tensor:
        """
        長さ×幅^width_exponent の総和（建設コスト相当）をネットワークごとに集計する。
        width_exponent > 1 にすると、幅の広い道路(幹線道路)への罰則が幅に対して超線形に強くなる。
        """
        p0 = self.nodes[self.edges[:, 0]]
        p1 = self.nodes[self.edges[:, 1]]
        cost = (p1 - p0).norm(dim=-1) * self.widths.pow(width_exponent)  # (total_edges,)
        n_networks = len(self.ids)
        result = torch.zeros(n_networks, device=cost.device, dtype=cost.dtype)
        return torch.scatter_reduce(result, 0, self.edge_index_of, cost, reduce="sum")

    def get_sizes(self) -> list[int]:
        return [e - s for s, e in self.edge_ranges]

    def _build_node_index_of(self) -> torch.Tensor:
        """各ノード(グローバルindex)がどのネットワーク(=object)に属するかを返す。 (total_nodes,)"""
        device = self.device()
        node_index_of = torch.empty(len(self.nodes), dtype=torch.long, device=device)
        for i, (s, e) in enumerate(self.node_ranges):
            node_index_of[s:e] = i
        return node_index_of

    def get_meshedness(self) -> torch.Tensor:
        """
        道路網のループ(閉路)の多さを 0〜1 程度で表す指標（meshedness / alpha index）。
        cycles = max(E - V + 1, 0)          # 閉路数（連結成分が1つの場合は厳密。複数ある場合は下限値）
        meshedness = cycles / max(2V - 5, 1)  # 平面グラフが取りうる最大閉路数に対する比

        V は孤立ノード(次数0。Remove直後などでまだ prune されていないもの)を除いた実際に
        道路網を構成するノード数で数える。木構造(閉路なし)では0、Snapでループを作るほど増える。
        """
        device = self.device()
        n_total_nodes = len(self.nodes)
        n_networks = len(self.ids)

        degree = torch.zeros(n_total_nodes, dtype=torch.long, device=device)
        degree.scatter_add_(0, self.edges.flatten(), torch.ones(2 * len(self.edges), dtype=torch.long, device=device))
        live = (degree > 0).float()

        node_index_of = self._build_node_index_of()

        v = torch.scatter_reduce(
            torch.zeros(n_networks, device=device), 0, node_index_of, live, reduce="sum"
        )  # (n_networks,)
        e = torch.tensor(self.get_sizes(), dtype=torch.float32, device=device)  # (n_networks,)

        cycles = (e - v + 1).clamp(min=0.0)
        denom = (2 * v - 5).clamp(min=1.0)
        return cycles / denom

    def get_angle_penalty(self, min_angle: float, exponent: float = 2.0) -> torch.Tensor:
        """
        各ノードにおいて、そこに接続する道路同士の"隣接する"方向間の角度差(gap)が min_angle を
        下回った分だけ (min_angle - gap)^exponent で罰する。鋭角に交わる不自然な交差点を減らし、
        適度に開いた(=都市の道路網らしい)交差点を促す。次数1以下のノードは角度が定義できないため対象外。

        ノード周りの各方向を角度順に並べたときの隣接gap(最後尾から先頭に戻る周回分も含む)を
        すべて求める。次数dのノードには合計d個のgapがあり、その総和は必ず2πになる。

        returns: (n_networks,)
        """
        device = self.device()
        total_nodes = len(self.nodes)
        n_networks = len(self.ids)
        n_edges = len(self.edges)
        if n_edges == 0:
            return torch.zeros(n_networks, device=device)

        center = torch.cat([self.edges[:, 0], self.edges[:, 1]])  # (2E,)
        other = torch.cat([self.edges[:, 1], self.edges[:, 0]])  # (2E,)
        vec = self.nodes[other] - self.nodes[center]  # (2E, 2)
        two_pi = 2 * math.pi
        theta = torch.remainder(torch.atan2(vec[:, 1], vec[:, 0]), two_pi)  # [0, 2pi)

        # (center, theta) の順でソートすると、同一ノードに属する方向が角度順に並ぶ
        key = center.to(theta.dtype) * (two_pi + 1.0) + theta
        order = torch.argsort(key)
        center_sorted = center[order]
        theta_sorted = theta[order]

        diffs = theta_sorted[1:] - theta_sorted[:-1]  # (2E-1,)
        same_group = center_sorted[1:] == center_sorted[:-1]  # 同じノードに属する隣接ペアか
        node_of_gap = center_sorted[:-1]  # (2E-1,)

        # 周回ギャップ = 2π - (そのノードの"内部"gapの総和)
        intra_gap = torch.where(same_group, diffs, torch.zeros_like(diffs))
        sum_intra = torch.zeros(total_nodes, dtype=theta.dtype, device=device)
        sum_intra.scatter_add_(0, node_of_gap, intra_gap)
        wrap_gap = (two_pi - sum_intra).clamp(min=0.0)  # (total_nodes,)

        degree = torch.zeros(total_nodes, dtype=torch.long, device=device)
        degree.scatter_add_(0, self.edges.flatten(), torch.ones(2 * n_edges, dtype=torch.long, device=device))
        has_gaps = degree >= 2

        intra_penalty = torch.where(
            same_group, (min_angle - diffs).clamp(min=0.0).pow(exponent), torch.zeros_like(diffs)
        )
        wrap_penalty = torch.where(
            has_gaps,
            (min_angle - wrap_gap).clamp(min=0.0).pow(exponent),
            torch.zeros(total_nodes, dtype=theta.dtype, device=device),
        )

        node_index_of = self._build_node_index_of()
        result = torch.zeros(n_networks, dtype=theta.dtype, device=device)
        result = torch.scatter_reduce(result, 0, node_index_of[node_of_gap], intra_penalty, reduce="sum")
        result = torch.scatter_reduce(result, 0, node_index_of, wrap_penalty, reduce="sum")
        return result

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
