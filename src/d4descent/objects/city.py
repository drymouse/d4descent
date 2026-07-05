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
from .roads import _capsule_sdf, find_nearby_nodes, _is_reachable_without_edge


# region Rewrites
# ================== Rewrites ===========================
#
# City文法: Road文法(objects/roads.py)から幅クラス階層(highway/street)を完全に取り除いた単純化版。
# 道路は単一種類(street)のみで、幅は定数(CityCollectionArgs.width)。よって不変条件は
# 連結性(A)のみでよく、Road文法にあった「highwayはhighwayからしか生えない」(不変条件B)や
# Widen/Narrowは存在しない。損失は RasterLossMixin(render01とtarget_imgのMSE)をそのまま使う
# ため、道路網は「図形の内部を隙間なく埋め尽くす(render01を1に近づける)」方向に最適化される。
#
# 不変条件A(連結性): 新規ノードは必ず既存ノードから繋がった形でのみ追加する(孤立配置は行わない)。


class CityRewriteType(Enum):
    Add = 1
    AddAnywhere = 2
    Remove = 3
    Split = 4
    Merge = 5
    Snap = 6
    Unsnap = 7
    Branch = 8


@dataclass
class CityRewrite:
    rewrite_type: CityRewriteType


@dataclass
class CityRewriteAdd(CityRewrite):
    """既存ノード from_node から新規ノード (x, y) へ道を伸ばす(epsilon長)。逆操作: Remove。"""

    rewrite_type: CityRewriteType = field(default=CityRewriteType.Add, init=False)
    from_node: int
    x: float
    y: float


@dataclass
class CityRewriteAddAnywhere(CityRewrite):
    """
    既存ノード from_node から、空間中の任意の点まで新規ノードの鎖(pts)で繋がったまま到達する
    (Tree.AddAnywhere と同型)。逆操作: 鎖の末端から Remove を繰り返す。
    """

    rewrite_type: CityRewriteType = field(default=CityRewriteType.AddAnywhere, init=False)
    from_node: int
    pts: tuple[tuple[float, float], ...]


@dataclass
class CityRewriteRemove(CityRewrite):
    """次数1(末端)のエッジを削除する。末端なので常に連結性を壊さない。逆操作: Add/AddAnywhere。"""

    rewrite_type: CityRewriteType = field(default=CityRewriteType.Remove, init=False)
    edge_id: int


@dataclass
class CityRewriteSplit(CityRewrite):
    """エッジを (x,y) の位置で2本に分割する。分割前と全く同じ位置なので形状は変化しない。逆操作: Merge。"""

    rewrite_type: CityRewriteType = field(default=CityRewriteType.Split, init=False)
    edge_id: int
    x: float
    y: float


@dataclass
class CityRewriteBranch(CityRewrite):
    """
    既存エッジ edge_id を (x,y) で分割して新規ノードを作り、そこから新規ノード (bx,by) へ
    短い枝を伸ばす(Split+Addの複合操作)。単独のSplitは幾何形状を変えないため損失を改善せず
    採択されないが、この複合操作は新しい枝の分だけ損失を即座に改善しうる。5叉路以上の交差点への
    罰則(degree_penalty)の代わりに、既存ノードへのSnap/Addに集中させず辺の途中から積極的に
    新しい分岐点を作ってネットワークを広げるために導入する。逆操作: 枝をRemoveしてMerge。
    """

    rewrite_type: CityRewriteType = field(default=CityRewriteType.Branch, init=False)
    edge_id: int
    x: float
    y: float
    bx: float
    by: float


@dataclass
class CityRewriteMerge(CityRewrite):
    """次数2のノードで、両側のエッジがほぼ共線の場合に1本へ統合する。逆操作: Split。"""

    rewrite_type: CityRewriteType = field(default=CityRewriteType.Merge, init=False)
    node_id: int
    edge_id_a: int
    edge_id_b: int
    outer_a: int
    outer_b: int


@dataclass
class CityRewriteSnap(CityRewrite):
    """edge_id の end側(0 or 1)の端点(次数1のぶら下がりノード)を、既存ノード target_node に張り替える。逆操作: Unsnap。"""

    rewrite_type: CityRewriteType = field(default=CityRewriteType.Snap, init=False)
    edge_id: int
    end: int
    target_node: int


@dataclass
class CityRewriteUnsnap(CityRewrite):
    """サイクル上のエッジの一端を、同じ座標に複製した新規ノードへ付け替え、ぶら下がった端点に戻す。Snapの逆操作。"""

    rewrite_type: CityRewriteType = field(default=CityRewriteType.Unsnap, init=False)
    edge_id: int
    end: int


@dataclass
class CityRewriteArgs:
    length_range: tuple[float, float] = (0.05, 0.15)
    n_add_candidates: int = 32
    n_add_anywhere_candidates: int = 32
    n_split_candidates: int = 16
    n_branch_candidates: int = 32
    branch_t_range: tuple[float, float] = (0.15, 0.85)  # エッジ上の分岐点の位置(両端の近くは避ける)
    snap_radius: float = 0.05
    max_add_anywhere_hops: int = 6  # AddAnywhereの鎖の最大エッジ数。1提案で長大な橋を作らせない
    merge_angle_eps: float = math.radians(5.0)  # 反対方向(=直線)からのずれがこれ以内ならMerge可
    unsnap_bfs_budget: int = 256  # Unsnap候補の連結性チェック(BFS)で訪問するノード数の上限
    add_weight: float = 1.0
    add_anywhere_weight: float = 1.0
    remove_weight: float = 1.0
    split_weight: float = 1.0
    merge_weight: float = 1.0
    snap_weight: float = 1.0
    unsnap_weight: float = 1.0
    # 辺の途中から新しい分岐を積極的に伸ばす(Branch)。5叉路以上への罰則(degree_penalty)の代わりに、
    # 既存ノードへのSnap/Addで交差点を密集させず、辺の途中から新しい分岐点を作ってネットワークを
    # 広げる経路を確保する。既定でAddより強めにして"積極的に"分岐を試みるようにする。
    branch_weight: float = 2.0


# endregion
# region CityNetwork
# ================== CityNetwork ===========================


@dataclass
class CityPayload:
    pass


@dataclass
class CityNetwork:
    nodes: torch.Tensor  # (n_nodes, 2) 座標。勾配対象
    edges: torch.Tensor  # (n_edges, 2) long。ノードindexのペア
    id: int = field(default_factory=lambda: Context.get().gen_id())
    payload: CityPayload = field(default_factory=CityPayload)

    def __post_init__(self):
        assert self.nodes.ndim == 2 and self.nodes.shape[-1] == 2, f"nodes must be (n,2), got {self.nodes.shape}"
        assert self.edges.ndim == 2 and self.edges.shape[-1] == 2, f"edges must be (n,2), got {self.edges.shape}"

    def device(self) -> torch.device:
        return self.nodes.device

    def _degree(self) -> torch.Tensor:
        n_nodes = len(self.nodes)
        degree = torch.zeros(n_nodes, dtype=torch.long, device=self.nodes.device)
        degree.scatter_add_(
            0, self.edges.flatten(), torch.ones(2 * len(self.edges), dtype=torch.long, device=self.nodes.device)
        )
        return degree

    def visualize(self, ax: MPLVisualizerAxes, color: str = "white", casing_color: str = "black", lw: float = 2.5) -> None:
        nodes = self.nodes.tolist()
        for i, j in self.edges.tolist():
            (x0, y0), (x1, y1) = nodes[i], nodes[j]
            ax.ax.plot([x0, x1], [y0, y1], color=casing_color, linewidth=lw + 1.4, solid_capstyle="round", zorder=4)
            ax.ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, solid_capstyle="round", zorder=5)

    def prune_orphan_nodes(self) -> "CityNetwork":
        """どのエッジからも参照されなくなったノードを削除し、エッジのインデックスを詰め直す。"""
        degree = self._degree()
        keep_mask = degree > 0
        if bool(keep_mask.all()):
            return self
        device = self.nodes.device
        new_index = torch.full((len(self.nodes),), -1, dtype=torch.long, device=device)
        new_index[keep_mask] = torch.arange(int(keep_mask.sum().item()), device=device)
        return CityNetwork(nodes=self.nodes[keep_mask], edges=new_index[self.edges], id=self.id, payload=self.payload)

    @torch.no_grad()
    def decimate_dense(self, cell_size: float, max_per_cell: int) -> "CityNetwork":
        """
        密集地帯を検出し、そこのノード(交差点)と接続道路を間引く(Road文法のdecimate_denseと同型)。
        cell_size のグリッドに区切り、1セル内のノード数が max_per_cell を超える密集セルでは、
        そのセル内のノードを1つの代表ノード(最高次数)へ統合(collapse)する。自己ループは除去、
        同一ノード対の重複エッジは1本に集約する。統合はノードをまとめるだけなので連結性は保たれる。
        """
        n_nodes = len(self.nodes)
        if n_nodes == 0 or len(self.edges) == 0:
            return self
        device = self.nodes.device
        pos = self.nodes.detach().cpu().tolist()
        edges_list = self.edges.tolist()
        degree = [0] * n_nodes
        for a, b in edges_list:
            degree[a] += 1
            degree[b] += 1

        cell = max(cell_size, 1e-9)
        buckets: dict[tuple[int, int], list[int]] = {}
        for i, (x, y) in enumerate(pos):
            buckets.setdefault((math.floor(x / cell), math.floor(y / cell)), []).append(i)

        remap = list(range(n_nodes))
        changed = False
        for members in buckets.values():
            if len(members) <= max_per_cell:
                continue
            rep = max(members, key=lambda m: degree[m])
            for m in members:
                if m == rep:
                    continue
                remap[m] = rep
                changed = True
        if not changed:
            return self

        edge_set: set[tuple[int, int]] = set()
        new_edges_list: list[list[int]] = []
        for a, b in edges_list:
            u, v = remap[a], remap[b]
            if u == v:
                continue
            key = (u, v) if u < v else (v, u)
            if key in edge_set:
                continue
            edge_set.add(key)
            new_edges_list.append([u, v])
        if not new_edges_list:
            return self
        new_edges = torch.tensor(new_edges_list, dtype=torch.long, device=device)
        return CityNetwork(nodes=self.nodes, edges=new_edges, id=self.id, payload=self.payload).prune_orphan_nodes()

    def gen_rewrite_specs(
        self,
        args: CityRewriteArgs,
        lim: tuple[float, float],
        add_anywhere_targets: Optional[torch.Tensor] = None,
    ) -> list[CityRewrite]:
        """
        add_anywhere_targets: (M, 2) の候補点プール。渡された場合、AddAnywhere はこの中から
        ランダムに狙う点を選ぶ(図形内部に偏らせるために Task 側が渡す)。
        None の場合は lim 全体から一様サンプルする。
        """
        device = self.nodes.device
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)
        degree = self._degree()
        live_nodes = degree.nonzero().flatten().tolist()
        leaf_nodes = (degree == 1).nonzero().flatten().tolist()
        pos = self.nodes.detach().cpu()
        edges_list = self.edges.tolist()

        adj: list[list[tuple[int, int]]] = [[] for _ in range(n_nodes)]
        for eid, (a, b) in enumerate(edges_list):
            adj[a].append((b, eid))
            adj[b].append((a, eid))

        specs: list[CityRewrite] = []

        # ---- Add: 既存ノードから短い新規エッジを伸ばす(epsilon長) ----
        if args.add_weight > 0 and live_nodes:
            n_cand = max(round(args.n_add_candidates * args.add_weight), 1)
            for _ in range(n_cand):
                ni = random.choice(live_nodes)
                ang = random.random() * 2 * math.pi
                length = random.uniform(*args.length_range)
                x0, y0 = pos[ni].tolist()
                x1 = x0 + length * math.cos(ang)
                y1 = y0 + length * math.sin(ang)
                specs.append(CityRewriteAdd(from_node=ni, x=x1, y=y1))

        # ---- AddAnywhere: 図形内部の点へ、最も近いノードから鎖で繋がったまま到達する ----
        if args.add_anywhere_weight > 0 and live_nodes:
            n_cand = max(round(args.n_add_anywhere_candidates * args.add_anywhere_weight), 1)
            lim0, lim1 = lim
            max_len = args.length_range[1]
            for _ in range(n_cand):
                if add_anywhere_targets is not None and len(add_anywhere_targets) > 0:
                    ti = random.randrange(len(add_anywhere_targets))
                    tx, ty = add_anywhere_targets[ti].tolist()
                else:
                    tx = random.uniform(lim0, lim1)
                    ty = random.uniform(lim0, lim1)
                target = torch.tensor([tx, ty])
                cand_pos = pos[live_nodes]  # (k, 2)
                d = (cand_pos - target).norm(dim=-1)
                best = int(d.argmin().item())
                from_node = live_nodes[best]
                dist = float(d[best].item())
                if dist < 1e-6:
                    continue
                x0, y0 = pos[from_node].tolist()
                n_hops = max(1, math.ceil(dist / max_len))
                n_hops = min(n_hops, args.max_add_anywhere_hops)
                reach = min(1.0, n_hops * max_len / dist)  # 目標点まで届かない場合はその手前まで
                ex, ey = x0 + (tx - x0) * reach, y0 + (ty - y0) * reach
                pts = tuple(
                    (x0 + (ex - x0) * (k / n_hops), y0 + (ey - y0) * (k / n_hops)) for k in range(1, n_hops + 1)
                )
                specs.append(CityRewriteAddAnywhere(from_node=from_node, pts=pts))

        # ---- Remove: 末端エッジの削除(常に連結性を壊さない) ----
        if args.remove_weight > 0 and n_edges > 1:
            for eid, (a, b) in enumerate(edges_list):
                if degree[a].item() == 1 or degree[b].item() == 1:
                    specs.append(CityRewriteRemove(edge_id=eid))

        # ---- Split: エッジをその場で2分割(形状不変) ----
        if args.split_weight > 0 and n_edges > 0:
            n_cand = max(round(args.n_split_candidates * args.split_weight), 1)
            for _ in range(n_cand):
                eid = random.randrange(n_edges)
                a, b = edges_list[eid]
                t = random.uniform(0.05, 0.95)
                x0, y0 = pos[a].tolist()
                x1, y1 = pos[b].tolist()
                specs.append(CityRewriteSplit(edge_id=eid, x=x0 + (x1 - x0) * t, y=y0 + (y1 - y0) * t))

        # ---- Branch: 辺の途中から新しい枝を積極的に伸ばす(Split+Addの複合) ----
        # 既存ノードへのSnap/Addが集中して5叉路以上を作ってしまう(degree_penaltyで罰される)代わりに、
        # 辺の途中から新しい分岐点を作ってネットワークを広げる経路を提供する。
        if args.branch_weight > 0 and n_edges > 0:
            n_cand = max(round(args.n_branch_candidates * args.branch_weight), 1)
            t0, t1 = args.branch_t_range
            for _ in range(n_cand):
                eid = random.randrange(n_edges)
                a, b = edges_list[eid]
                t = random.uniform(t0, t1)
                x0, y0 = pos[a].tolist()
                x1, y1 = pos[b].tolist()
                sx, sy = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
                ang = random.random() * 2 * math.pi
                length = random.uniform(*args.length_range)
                bx, by = sx + length * math.cos(ang), sy + length * math.sin(ang)
                specs.append(CityRewriteBranch(edge_id=eid, x=sx, y=sy, bx=bx, by=by))

        # ---- Merge: ほぼ共線の次数2ノードを統合(Splitの逆操作) ----
        if args.merge_weight > 0:
            for ni in live_nodes:
                if degree[ni].item() != 2:
                    continue
                (n0, e0), (n1, e1) = adj[ni]
                if n0 == n1:
                    continue  # 2重辺は対象外
                p_c, p0, p1 = pos[ni], pos[n0], pos[n1]
                v0, v1 = p0 - p_c, p1 - p_c
                ang0 = math.atan2(v0[1].item(), v0[0].item())
                ang1 = math.atan2(v1[1].item(), v1[0].item())
                diff = abs((ang0 - ang1 + math.pi) % (2 * math.pi) - math.pi)
                if abs(diff - math.pi) > args.merge_angle_eps:
                    continue  # 直線からずれすぎている(実際の交差点/曲がり角)ので統合不可
                specs.append(CityRewriteMerge(node_id=ni, edge_id_a=e0, edge_id_b=e1, outer_a=n0, outer_b=n1))

        # ---- Snap: ぶら下がった端点を既存ノードに接続してループ/交差点を作る ----
        if args.snap_weight > 0 and leaf_nodes:
            incident_leaf: dict[int, tuple[int, int]] = {}
            for eid, (a, b) in enumerate(edges_list):
                if degree[a].item() == 1:
                    incident_leaf[a] = (eid, 0)
                if degree[b].item() == 1:
                    incident_leaf[b] = (eid, 1)
            leaf_idx_t = torch.tensor(leaf_nodes, dtype=torch.long, device=device)
            neighbor_lists = find_nearby_nodes(self.nodes, leaf_idx_t, cell_size=args.snap_radius, radius=args.snap_radius)
            for leaf, neighbors in zip(leaf_nodes, neighbor_lists):
                eid, end = incident_leaf[leaf]
                other = edges_list[eid][1 - end]
                for target in neighbors:
                    if target == other or target == leaf:
                        continue
                    specs.append(CityRewriteSnap(edge_id=eid, end=end, target_node=target))

        # ---- Unsnap: サイクル上のエッジを開放してぶら下がり端点に戻す(Snapの逆操作) ----
        if args.unsnap_weight > 0:
            for eid, (a, b) in enumerate(edges_list):
                if degree[a].item() < 2 or degree[b].item() < 2:
                    continue  # 片方が次数1(leaf)ならブリッジ確定なので対象外
                if not _is_reachable_without_edge(adj, eid, a, b, args.unsnap_bfs_budget):
                    continue  # ブリッジ(削除すると非連結)なので対象外(不変条件A)
                for end in (0, 1):
                    specs.append(CityRewriteUnsnap(edge_id=eid, end=end))

        return specs

    def apply_rewrite(self, spec: CityRewrite) -> "CityNetwork":
        device = self.nodes.device
        dtype = self.nodes.dtype
        if isinstance(spec, CityRewriteAdd):
            new_node = torch.tensor([[spec.x, spec.y]], dtype=dtype, device=device)
            new_edge = torch.tensor([[spec.from_node, len(self.nodes)]], dtype=torch.long, device=device)
            return CityNetwork(
                nodes=torch.cat([self.nodes, new_node], dim=0), edges=torch.cat([self.edges, new_edge], dim=0)
            )
        elif isinstance(spec, CityRewriteAddAnywhere):
            n0 = len(self.nodes)
            k = len(spec.pts)
            new_nodes = torch.tensor(list(spec.pts), dtype=dtype, device=device)  # (k,2)
            starts = torch.tensor([spec.from_node] + list(range(n0, n0 + k - 1)), dtype=torch.long, device=device)
            ends = torch.arange(n0, n0 + k, dtype=torch.long, device=device)
            new_edges = torch.stack([starts, ends], dim=-1)
            return CityNetwork(
                nodes=torch.cat([self.nodes, new_nodes], dim=0), edges=torch.cat([self.edges, new_edges], dim=0)
            )
        elif isinstance(spec, CityRewriteRemove):
            keep = torch.ones(len(self.edges), dtype=torch.bool, device=device)
            keep[spec.edge_id] = False
            return CityNetwork(nodes=self.nodes, edges=self.edges[keep])
        elif isinstance(spec, CityRewriteSplit):
            n0 = len(self.nodes)
            a, b = self.edges[spec.edge_id].tolist()
            new_node = torch.tensor([[spec.x, spec.y]], dtype=dtype, device=device)
            keep = torch.ones(len(self.edges), dtype=torch.bool, device=device)
            keep[spec.edge_id] = False
            new_edges = torch.tensor([[a, n0], [n0, b]], dtype=torch.long, device=device)
            return CityNetwork(
                nodes=torch.cat([self.nodes, new_node], dim=0), edges=torch.cat([self.edges[keep], new_edges], dim=0)
            )
        elif isinstance(spec, CityRewriteBranch):
            n0 = len(self.nodes)  # 分岐点(Splitで生じる新規ノード)
            n1 = n0 + 1  # 枝の先端
            a, b = self.edges[spec.edge_id].tolist()
            split_node = torch.tensor([[spec.x, spec.y]], dtype=dtype, device=device)
            branch_node = torch.tensor([[spec.bx, spec.by]], dtype=dtype, device=device)
            keep = torch.ones(len(self.edges), dtype=torch.bool, device=device)
            keep[spec.edge_id] = False
            new_edges = torch.tensor([[a, n0], [n0, b], [n0, n1]], dtype=torch.long, device=device)
            return CityNetwork(
                nodes=torch.cat([self.nodes, split_node, branch_node], dim=0),
                edges=torch.cat([self.edges[keep], new_edges], dim=0),
            )
        elif isinstance(spec, CityRewriteMerge):
            keep = torch.ones(len(self.edges), dtype=torch.bool, device=device)
            keep[spec.edge_id_a] = False
            keep[spec.edge_id_b] = False
            new_edge = torch.tensor([[spec.outer_a, spec.outer_b]], dtype=torch.long, device=device)
            return CityNetwork(nodes=self.nodes, edges=torch.cat([self.edges[keep], new_edge], dim=0))
        elif isinstance(spec, CityRewriteSnap):
            new_edges = self.edges.clone()
            new_edges[spec.edge_id, spec.end] = spec.target_node
            return CityNetwork(nodes=self.nodes, edges=new_edges)
        elif isinstance(spec, CityRewriteUnsnap):
            n0 = len(self.nodes)
            old_node = int(self.edges[spec.edge_id, spec.end].item())
            new_node = self.nodes[old_node : old_node + 1].clone()
            new_edges = self.edges.clone()
            new_edges[spec.edge_id, spec.end] = n0
            return CityNetwork(nodes=torch.cat([self.nodes, new_node], dim=0), edges=new_edges)
        else:
            raise ValueError(f"Unknown rewrite {spec}")

    def apply_rewrite_each(self, specs: list[CityRewrite]) -> list["CityNetwork"]:
        return [self.apply_rewrite(spec) for spec in specs]

    def apply_all_rewrites(self, rewrites: list[CityRewrite], scores: list[float]) -> "CityNetwork":
        """
        スコア降順で適用する。同じエッジを対象とする書き換えは先勝ちとし、以降は無視する。

        Unsnap は「削除しても連結性を壊さないか」を gen_rewrite_specs 時点のスナップショットに対して
        チェック済みだが、同じバッチ内の他のUnsnapと組み合わせた結果、初めて非連結になるケースが
        あるため、このバッチ内で既に確定した変更を反映した最新のグラフ状態に対して都度再検証する。

        Merge/Remove は対象ノードを孤立させる操作なので、同じバッチ内の別のAdd/AddAnywhere/Snapが
        "孤立する側"に新しい枝を付けてしまうと、その枝ごと本体から切り離される(不連結バグ)。これを
        防ぐため、ノードごとに"orphaned"/"attached"を記録し、孤立操作は既に何らかの記録があるノード
        には適用せず、接続操作は既に孤立確定済みのノードには接続しない。

        ノード削除・再インデックスはここでは行わない(orphanは Task.cleanup が prune_orphan_nodes で処理)。
        """
        order = sorted(range(len(rewrites)), key=lambda i: scores[i], reverse=True)
        device = self.nodes.device
        dtype = self.nodes.dtype
        base_degree = self._degree().tolist()

        live_edges: dict[int, tuple[int, int]] = {eid: (a, b) for eid, (a, b) in enumerate(self.edges.tolist())}
        next_edge_id = len(self.edges)
        node_status: dict[int, str] = {}  # node_id -> "orphaned" | "attached"

        def _reachable(exclude_eid: int, start: int, target: int) -> bool:
            if start == target:
                return True
            adj: dict[int, list[tuple[int, int]]] = {}
            for eid, (a, b) in live_edges.items():
                if eid == exclude_eid:
                    continue
                adj.setdefault(a, []).append((b, eid))
                adj.setdefault(b, []).append((a, eid))
            visited = {start}
            stack = [start]
            while stack:
                cur = stack.pop()
                for nb, eid2 in adj.get(cur, []):
                    if nb == target:
                        return True
                    if nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            return False

        added_nodes: list[torch.Tensor] = []
        added_edges: list[torch.Tensor] = []
        n_next = len(self.nodes)

        touched_edges: set[int] = set()
        removed: set[int] = set()
        snap_updates: dict[int, tuple[int, int]] = {}
        n_alive = len(self.edges)

        for i in order:
            spec = rewrites[i]
            if isinstance(spec, CityRewriteAdd):
                if node_status.get(spec.from_node) == "orphaned":
                    continue
                added_nodes.append(torch.tensor([[spec.x, spec.y]], dtype=dtype, device=device))
                added_edges.append(torch.tensor([[spec.from_node, n_next]], dtype=torch.long, device=device))
                live_edges[next_edge_id] = (spec.from_node, n_next)
                next_edge_id += 1
                n_next += 1
                node_status.setdefault(spec.from_node, "attached")
            elif isinstance(spec, CityRewriteAddAnywhere):
                if node_status.get(spec.from_node) == "orphaned":
                    continue
                k = len(spec.pts)
                added_nodes.append(torch.tensor(list(spec.pts), dtype=dtype, device=device))
                starts = torch.tensor(
                    [spec.from_node] + list(range(n_next, n_next + k - 1)), dtype=torch.long, device=device
                )
                ends = torch.arange(n_next, n_next + k, dtype=torch.long, device=device)
                added_edges.append(torch.stack([starts, ends], dim=-1))
                for s, e in zip(starts.tolist(), ends.tolist()):
                    live_edges[next_edge_id] = (s, e)
                    next_edge_id += 1
                n_next += k
                node_status.setdefault(spec.from_node, "attached")
            elif isinstance(spec, CityRewriteRemove):
                if spec.edge_id in touched_edges or spec.edge_id not in live_edges or n_alive <= 1:
                    continue
                a, b = live_edges[spec.edge_id]
                leaf = a if base_degree[a] == 1 else b
                if node_status.get(leaf) is not None:
                    continue
                touched_edges.add(spec.edge_id)
                removed.add(spec.edge_id)
                del live_edges[spec.edge_id]
                n_alive -= 1
                node_status[leaf] = "orphaned"
            elif isinstance(spec, CityRewriteSplit):
                if spec.edge_id in touched_edges or spec.edge_id not in live_edges:
                    continue
                touched_edges.add(spec.edge_id)
                removed.add(spec.edge_id)
                a, b = live_edges.pop(spec.edge_id)
                added_nodes.append(torch.tensor([[spec.x, spec.y]], dtype=dtype, device=device))
                added_edges.append(torch.tensor([[a, n_next], [n_next, b]], dtype=torch.long, device=device))
                live_edges[next_edge_id] = (a, n_next)
                live_edges[next_edge_id + 1] = (n_next, b)
                next_edge_id += 2
                n_next += 1
                n_alive += 1
            elif isinstance(spec, CityRewriteBranch):
                if spec.edge_id in touched_edges or spec.edge_id not in live_edges:
                    continue
                touched_edges.add(spec.edge_id)
                removed.add(spec.edge_id)
                a, b = live_edges.pop(spec.edge_id)
                split_id = n_next
                branch_id = n_next + 1
                added_nodes.append(torch.tensor([[spec.x, spec.y]], dtype=dtype, device=device))
                added_nodes.append(torch.tensor([[spec.bx, spec.by]], dtype=dtype, device=device))
                added_edges.append(
                    torch.tensor([[a, split_id], [split_id, b], [split_id, branch_id]], dtype=torch.long, device=device)
                )
                live_edges[next_edge_id] = (a, split_id)
                live_edges[next_edge_id + 1] = (split_id, b)
                live_edges[next_edge_id + 2] = (split_id, branch_id)
                next_edge_id += 3
                n_next += 2
                n_alive += 2
            elif isinstance(spec, CityRewriteMerge):
                if (
                    spec.edge_id_a in touched_edges
                    or spec.edge_id_b in touched_edges
                    or node_status.get(spec.node_id) is not None
                    or spec.edge_id_a not in live_edges
                    or spec.edge_id_b not in live_edges
                ):
                    continue
                touched_edges.add(spec.edge_id_a)
                touched_edges.add(spec.edge_id_b)
                removed.add(spec.edge_id_a)
                removed.add(spec.edge_id_b)
                del live_edges[spec.edge_id_a]
                del live_edges[spec.edge_id_b]
                added_edges.append(torch.tensor([[spec.outer_a, spec.outer_b]], dtype=torch.long, device=device))
                live_edges[next_edge_id] = (spec.outer_a, spec.outer_b)
                next_edge_id += 1
                n_alive -= 1
                node_status[spec.node_id] = "orphaned"
            elif isinstance(spec, CityRewriteSnap):
                if spec.edge_id in touched_edges or spec.edge_id not in live_edges:
                    continue
                if node_status.get(spec.target_node) == "orphaned":
                    continue
                a, b = live_edges[spec.edge_id]
                dangling = a if spec.end == 0 else b
                if node_status.get(dangling) is not None:
                    continue
                touched_edges.add(spec.edge_id)
                if spec.end == 0:
                    live_edges[spec.edge_id] = (spec.target_node, b)
                else:
                    live_edges[spec.edge_id] = (a, spec.target_node)
                snap_updates[spec.edge_id] = (spec.end, spec.target_node)
                node_status[dangling] = "orphaned"
                node_status.setdefault(spec.target_node, "attached")
            elif isinstance(spec, CityRewriteUnsnap):
                if spec.edge_id in touched_edges or spec.edge_id not in live_edges:
                    continue
                a, b = live_edges[spec.edge_id]
                old_node = a if spec.end == 0 else b
                other = b if spec.end == 0 else a
                if not _reachable(spec.edge_id, old_node, other):
                    continue  # このバッチ内の他の変更と合わせるとブリッジになってしまう(不変条件A)
                touched_edges.add(spec.edge_id)
                if spec.end == 0:
                    live_edges[spec.edge_id] = (n_next, b)
                else:
                    live_edges[spec.edge_id] = (a, n_next)
                added_nodes.append(self.nodes[old_node : old_node + 1].detach().clone())
                snap_updates[spec.edge_id] = (spec.end, n_next)
                n_next += 1
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
        nodes = self.nodes.detach().clone()

        if added_nodes:
            nodes = torch.cat([nodes, *added_nodes], dim=0)
        if added_edges:
            edges = torch.cat([edges, *added_edges], dim=0)

        return CityNetwork(nodes=nodes, edges=edges)


# endregion
# region CityNetworkCollection
# ================== CityNetworkCollection ===========================


@dataclass
class CityCollectionArgs:
    width: float = 0.02  # 道路(街路)の幅。City文法では全道路が単一のこの幅を持つ


def _always_raise() -> CityCollectionArgs:
    raise ValueError("CityCollectionArgs must be set")


@dataclass
class CityNetworkCollection(ObjectCollection[CityNetwork]):
    nodes: torch.Tensor  # (total_nodes, 2) 全ネットワーク結合。勾配対象
    edges: torch.Tensor  # (total_edges, 2) long。グローバルnode index
    edge_index_of: torch.Tensor  # (total_edges,) 各エッジがどのネットワーク(=object)に属するか
    node_ranges: tuple[tuple[int, int], ...]
    edge_ranges: tuple[tuple[int, int], ...]
    ids: tuple[int, ...]
    payloads: tuple[CityPayload, ...]
    args: CityCollectionArgs = field(default_factory=_always_raise)

    def __post_init__(self):
        assert self.nodes.ndim == 2 and self.nodes.shape[-1] == 2, f"nodes must be (n,2), got {self.nodes.shape}"
        assert self.edges.ndim == 2 and self.edges.shape[-1] == 2, f"edges must be (n,2), got {self.edges.shape}"
        assert len(self.edges) == len(self.edge_index_of)
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
            edge_index_of=self.edge_index_of.to(device=device),
            node_ranges=self.node_ranges,
            edge_ranges=self.edge_ranges,
            ids=self.ids,
            payloads=self.payloads,
            args=self.args,
        )

    def get_object(self, idx: int, detach: bool = True) -> CityNetwork:
        ns, ne = self.node_ranges[idx]
        es, ee = self.edge_ranges[idx]
        return CityNetwork(
            nodes=maybe_detach(self.nodes[ns:ne], detach),
            edges=(self.edges[es:ee] - ns).clone(),
            id=self.ids[idx],
            payload=self.payloads[idx],
        )

    @classmethod
    def from_object(cls, object: CityNetwork, **kwargs) -> Self:
        device = object.nodes.device
        n_nodes = len(object.nodes)
        n_edges = len(object.edges)
        return cls(
            nodes=object.nodes.detach().clone(),
            edges=object.edges.detach().clone(),
            edge_index_of=torch.zeros(n_edges, dtype=torch.long, device=device),
            node_ranges=((0, n_nodes),),
            edge_ranges=((0, n_edges),),
            ids=(object.id,),
            payloads=(object.payload,),
            **kwargs,
        )

    @classmethod
    def cat(cls, collections: list["CityNetworkCollection"], **kwargs) -> Self:
        collections_ = cast(list[CityNetworkCollection], collections)
        assert len(collections_) > 0, "collections must not be empty"
        node_offset = 0
        edge_offset = 0
        network_offset = 0
        all_nodes: list[torch.Tensor] = []
        all_edges: list[torch.Tensor] = []
        all_edge_index_of: list[torch.Tensor] = []
        all_node_ranges: list[tuple[int, int]] = []
        all_edge_ranges: list[tuple[int, int]] = []
        all_ids: tuple[int, ...] = ()
        all_payloads: tuple[CityPayload, ...] = ()
        for c in collections_:
            n_nodes = len(c.nodes)
            n_edges = len(c.edges)
            all_nodes.append(c.nodes)
            all_edges.append(c.edges + node_offset)
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
        returns: (n_networks, *shape) 各ネットワークのSDF(render01がこれをそのまま使う)
        """
        *shape, _ = positions.shape
        p0 = self.nodes[self.edges[:, 0]]
        p1 = self.nodes[self.edges[:, 1]]
        n_edges = len(self.edges)
        widths = torch.full((n_edges,), self.args.width, dtype=p0.dtype, device=p0.device)
        sdf = _capsule_sdf(positions, p0, p1, widths)  # (total_edges, *shape)
        n_networks = len(self.ids)
        result = torch.full((n_networks, *shape), torch.inf, dtype=sdf.dtype, device=sdf.device)
        index = self.edge_index_of.view(-1, *([1] * len(shape))).expand(-1, *shape)
        return torch.scatter_reduce(result, 0, index, sdf, reduce="amin")

    def get_construction_costs(self) -> torch.Tensor:
        """長さの総和(建設コスト相当)をネットワークごとに集計する。全道路が同一幅なので幅は考慮しない。"""
        p0 = self.nodes[self.edges[:, 0]]
        p1 = self.nodes[self.edges[:, 1]]
        cost = (p1 - p0).norm(dim=-1)  # (total_edges,)
        n_networks = len(self.ids)
        result = torch.zeros(n_networks, device=cost.device, dtype=cost.dtype)
        return torch.scatter_reduce(result, 0, self.edge_index_of, cost, reduce="sum")

    def get_sizes(self) -> list[int]:
        return [e - s for s, e in self.edge_ranges]

    def _build_node_index_of(self) -> torch.Tensor:
        device = self.device()
        node_index_of = torch.empty(len(self.nodes), dtype=torch.long, device=device)
        for i, (s, e) in enumerate(self.node_ranges):
            node_index_of[s:e] = i
        return node_index_of

    def get_meshedness(self) -> torch.Tensor:
        """
        道路網のループ(閉路)の多さを 0〜1 程度で表す指標(meshedness / alpha index)。
        cycles = max(E - V + 1, 0)、meshedness = cycles / max(2V - 5, 1)。
        """
        device = self.device()
        n_total_nodes = len(self.nodes)
        n_networks = len(self.ids)

        degree = torch.zeros(n_total_nodes, dtype=torch.long, device=device)
        degree.scatter_add_(0, self.edges.flatten(), torch.ones(2 * len(self.edges), dtype=torch.long, device=device))
        live = (degree > 0).float()

        node_index_of = self._build_node_index_of()

        v = torch.scatter_reduce(torch.zeros(n_networks, device=device), 0, node_index_of, live, reduce="sum")
        e = torch.tensor(self.get_sizes(), dtype=torch.float32, device=device)

        cycles = (e - v + 1).clamp(min=0.0)
        denom = (2 * v - 5).clamp(min=1.0)
        return cycles / denom

    def get_angle_penalty(self, exponent: float = 2.0, deadzone: float = 0.0) -> torch.Tensor:
        """
        各ノードにおいて、接続する道路同士の"隣接する"方向間の角度差(gap)が 90度の格子
        {90°,180°,270°} からどれだけずれているかを罰する(Road文法のget_angle_penaltyと同型)。
        gapが 90/180/270°(=直進・直角カーブ・T字・十字)なら罰則0、鋭角(→0°)や斜め(45°,135°)は
        罰される。次数1以下のノードは角度が定義できないため対象外。

        gap g に対し dev(g) = min(|g-90°|, |g-180°|, |g-270°|)、罰則 = relu(dev - deadzone)^exponent。

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
        half_pi = math.pi / 2
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

        def _lattice_penalty(g: torch.Tensor) -> torch.Tensor:
            dev = torch.minimum(
                torch.minimum((g - half_pi).abs(), (g - math.pi).abs()), (g - 3 * half_pi).abs()
            )
            return (dev - deadzone).clamp(min=0.0).pow(exponent)

        intra_penalty = torch.where(same_group, _lattice_penalty(diffs), torch.zeros_like(diffs))
        wrap_penalty = torch.where(
            has_gaps, _lattice_penalty(wrap_gap), torch.zeros(total_nodes, dtype=theta.dtype, device=device)
        )

        node_index_of = self._build_node_index_of()
        result = torch.zeros(n_networks, dtype=theta.dtype, device=device)
        result = torch.scatter_reduce(result, 0, node_index_of[node_of_gap], intra_penalty, reduce="sum")
        result = torch.scatter_reduce(result, 0, node_index_of, wrap_penalty, reduce="sum")
        return result

    def get_degree_penalty(self, threshold: int = 5, exponent: float = 2.0) -> torch.Tensor:
        """
        次数(接続する辺の数)が threshold 以上のノード(=threshold本以上の道路が集まる交差点)に
        罰則を課す。不自然な多叉路(5叉路以上)の形成を抑制する。
        excess = degree - threshold + 1 (threshold以上のときのみ正) として、罰則 = excess^exponent の
        ネットワークごとの総和を返す。

        returns: (n_networks,)
        """
        device = self.device()
        total_nodes = len(self.nodes)
        n_networks = len(self.ids)
        if len(self.edges) == 0:
            return torch.zeros(n_networks, device=device)

        degree = torch.zeros(total_nodes, dtype=torch.long, device=device)
        degree.scatter_add_(0, self.edges.flatten(), torch.ones(2 * len(self.edges), dtype=torch.long, device=device))
        excess = (degree - threshold + 1).clamp(min=0).to(dtype=self.nodes.dtype)
        penalty = excess.pow(exponent)

        node_index_of = self._build_node_index_of()
        result = torch.zeros(n_networks, dtype=self.nodes.dtype, device=device)
        return torch.scatter_reduce(result, 0, node_index_of, penalty, reduce="sum")

    @classmethod
    def patch_args(cls, args: CityCollectionArgs) -> Type["CityNetworkCollection"]:
        return type(
            "CityNetworkCollectionWithArgs", (CityNetworkCollection,), {"__init__": partialmethod(CityNetworkCollection.__init__, args=args)}
        )

    def to_savable(self) -> "CityNetworkCollection":
        return CityNetworkCollection(
            nodes=self.nodes,
            edges=self.edges,
            edge_index_of=self.edge_index_of,
            node_ranges=self.node_ranges,
            edge_ranges=self.edge_ranges,
            ids=self.ids,
            payloads=self.payloads,
            args=self.args,
        )

    def project_to_valid_(self) -> Self:
        return self


# endregion
