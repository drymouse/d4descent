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
#
# 4性質(Reversibility/Jump Continuity/Local Geometric Control/Repairability, design-for-descent論文Table1)
# に基づく再設計(docs/road_grammar_plan.md 9節)。連結性・幹線道路の階層性は罰則ではなく
# 「そもそも違反する提案を生成しない」という書き換えの生成条件で保証する。
#
# 不変条件A(連結性): AddFree(孤立道路の追加)を廃止。新規ノードは必ず既存ノードから繋がった形でのみ追加する。
# 不変条件B(幹線道路の階層性): highway(width_classesの最大値)エッジは、既にhighwayエッジに
#   接しているノード(=highway適格ノード)からしか生えない。streetはhighway/street問わずどこからでも生える。


class RoadRewriteType(Enum):
    Add = 1
    AddAnywhere = 2
    Remove = 3
    Split = 4
    Merge = 5
    Snap = 6
    Unsnap = 7
    Widen = 8
    Narrow = 9


@dataclass
class RoadRewrite:
    rewrite_type: RoadRewriteType


@dataclass
class RoadRewriteAdd(RoadRewrite):
    """既存ノード from_node から新規ノード (x, y) へ道を伸ばす(epsilon長)。逆操作: Remove。"""

    rewrite_type: RoadRewriteType = field(default=RoadRewriteType.Add, init=False)
    from_node: int
    x: float
    y: float
    width: float


@dataclass
class RoadRewriteAddAnywhere(RoadRewrite):
    """
    既存の適格ノード from_node から、空間中の任意の点まで新規ノードの鎖(pts)で繋がったまま到達する。
    Tree.AddAnywhere と同型(Local Geometric Control: 遠方への局所変更)。鎖の長さは距離に応じて可変。
    逆操作: 鎖の末端から Remove を繰り返す。
    """

    rewrite_type: RoadRewriteType = field(default=RoadRewriteType.AddAnywhere, init=False)
    from_node: int
    pts: tuple[tuple[float, float], ...]  # 1個以上。各hopの終点座標(最後の要素がtarget点そのもの)
    width: float


@dataclass
class RoadRewriteRemove(RoadRewrite):
    """次数1(末端)のエッジを削除する。末端なので常に連結性を壊さない。逆操作: Add/AddAnywhere。"""

    rewrite_type: RoadRewriteType = field(default=RoadRewriteType.Remove, init=False)
    edge_id: int


@dataclass
class RoadRewriteSplit(RoadRewrite):
    """
    エッジを (x,y) の位置で同じ幅クラスの2本に分割する。分割前と全く同じ位置なので形状は変化しない
    (Jump Continuity、論文の"局所的なセグメント分割"の例そのもの)。逆操作: Merge。
    """

    rewrite_type: RoadRewriteType = field(default=RoadRewriteType.Split, init=False)
    edge_id: int
    x: float
    y: float


@dataclass
class RoadRewriteMerge(RoadRewrite):
    """
    次数2のノードで、両側のエッジが同じ幅クラスかつほぼ共線の場合に1本へ統合する。逆操作: Split。
    outer_a/outer_b はノード削除側(node_id)の外側にある2つの隣接ノード。
    """

    rewrite_type: RoadRewriteType = field(default=RoadRewriteType.Merge, init=False)
    node_id: int
    edge_id_a: int
    edge_id_b: int
    outer_a: int
    outer_b: int
    width: float


@dataclass
class RoadRewriteSnap(RoadRewrite):
    """edge_id の end側(0 or 1)の端点(次数1のぶら下がりノード)を、既存ノード target_node に張り替える。逆操作: Unsnap。"""

    rewrite_type: RoadRewriteType = field(default=RoadRewriteType.Snap, init=False)
    edge_id: int
    end: int
    target_node: int


@dataclass
class RoadRewriteUnsnap(RoadRewrite):
    """
    サイクル上のエッジ(=削除しても連結性を壊さないエッジ)の一端を、同じ座標に複製した新規ノードへ
    付け替え、ぶら下がった端点に戻す(形状は瞬間的に不変)。Snapの逆操作(論文のAdd/Remove-Loop例)。
    """

    rewrite_type: RoadRewriteType = field(default=RoadRewriteType.Unsnap, init=False)
    edge_id: int
    end: int


@dataclass
class RoadRewriteWiden(RoadRewrite):
    """
    エッジの幅クラスを格上げする(例: street -> highway)。少なくとも片方の端点が既にhighwayに
    接している場合のみ許可(不変条件Bを維持するため)。逆操作: Narrow。
    """

    rewrite_type: RoadRewriteType = field(default=RoadRewriteType.Widen, init=False)
    edge_id: int
    new_width: float


@dataclass
class RoadRewriteNarrow(RoadRewrite):
    """エッジの幅クラスを格下げする(例: highway -> street)。逆操作: Widen。"""

    rewrite_type: RoadRewriteType = field(default=RoadRewriteType.Narrow, init=False)
    edge_id: int
    new_width: float


@dataclass
class RoadRewriteArgs:
    width_classes: tuple[float, ...] = (0.02, 0.05)  # 昇順。最後の要素をhighway(幹線道路)とみなす
    length_range: tuple[float, float] = (0.05, 0.15)
    n_add_candidates: int = 32
    n_add_anywhere_candidates: int = 32
    n_split_candidates: int = 16
    snap_radius: float = 0.05
    max_add_anywhere_hops: int = 6  # AddAnywhereの鎖の最大エッジ数。背景を貫く長大な橋を1提案で作らせない
    merge_angle_eps: float = math.radians(5.0)  # 反対方向(=直線)からのずれがこれ以内ならMerge可
    unsnap_bfs_budget: int = 256  # Unsnap候補の連結性チェック(BFS)で訪問するノード数の上限
    add_weight: float = 1.0
    add_anywhere_weight: float = 1.0
    remove_weight: float = 1.0
    split_weight: float = 1.0
    merge_weight: float = 1.0
    snap_weight: float = 1.0
    unsnap_weight: float = 1.0
    widen_weight: float = 1.0
    narrow_weight: float = 1.0

    def __post_init__(self):
        assert list(self.width_classes) == sorted(self.width_classes), "width_classes must be ascending"


def find_nearby_nodes(nodes: torch.Tensor, query_idx: torch.Tensor, cell_size: float, radius: float) -> list[list[int]]:
    """
    一様グリッドによる空間ハッシュで近傍探索を行う(全ペア総当たりのO(n^2)を避ける)。
    セルサイズ=radius とすることで、自セル+周囲8近傍だけを調べればよい。

    nodes: (n_nodes, 2)
    query_idx: (k,) 近傍を調べたいノードのインデックス
    returns: 各クエリについて、半径内にある他ノードのインデックス(距離の近い順)
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


def _is_reachable_without_edge(
    adj: list[list[tuple[int, int]]],
    exclude_edge: int,
    start: int,
    target: int,
    budget: int,
    edge_mask: Optional[list[bool]] = None,
) -> bool:
    """
    exclude_edge を除いたグラフ(edge_maskが渡された場合はさらに edge_mask[eid]==True のエッジのみ)で
    start から target に到達できるかをBFSで判定する。budgetを超えたら安全側(=False, 到達不可とみなす)
    を返す(Unsnapが誤って連結性を壊す側には倒れない)。
    """
    if start == target:
        return True
    visited = {start}
    queue = [start]
    steps = 0
    while queue:
        cur = queue.pop()
        for nbr, eid in adj[cur]:
            if eid == exclude_edge:
                continue
            if edge_mask is not None and not edge_mask[eid]:
                continue
            if nbr == target:
                return True
            if nbr in visited:
                continue
            visited.add(nbr)
            queue.append(nbr)
            steps += 1
            if steps > budget:
                return False
    return False


def _seg_intersect_2d(
    p: torch.Tensor, q: torch.Tensor, r: torch.Tensor, s: torch.Tensor
) -> tuple[bool, torch.Tensor]:
    """p->q, r->s の2線分が(端点近傍を除いて)交差するか判定する。交差する場合は交点も返す。"""
    v = q - p
    w = s - r
    denom = v[0] * w[1] - v[1] * w[0]
    if denom.abs() < 1e-9:
        return False, p
    diff = r - p
    t = (diff[0] * w[1] - diff[1] * w[0]) / denom
    u = (diff[0] * v[1] - diff[1] * v[0]) / denom
    if 0.02 < t.item() < 0.98 and 0.02 < u.item() < 0.98:
        return True, p + v * t
    return False, p


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
        degree.scatter_add_(
            0, self.edges.flatten(), torch.ones(2 * len(self.edges), dtype=torch.long, device=self.nodes.device)
        )
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
        (地図でよく使われる技法)。細い道(street)でも min_lw を確保して視認できるようにする。
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
            ax.ax.plot(
                [x0, x1], [y0, y1], color=casing_color, linewidth=lw + casing_extra, solid_capstyle="round", zorder=4
            )
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

    @torch.no_grad()
    def cleanup(self, max_iter: int = 4) -> "RoadNetwork":
        """
        共有ノードを持たずに幾何的に交差してしまった2辺(勾配降下でノード位置が動いた結果生じうる)
        を検出し、交点に新規ノードを1つ挿入して両辺をそこで分割する(Repairability、9.3節)。
        制約「交差する道路は必ずノードを共有する」に対する修復操作。
        """
        net = self
        for _ in range(max_iter):
            n_edges = len(net.edges)
            if n_edges < 2:
                break
            pos = net.nodes.detach()
            edges_list = net.edges.tolist()
            found: Optional[tuple[int, int, torch.Tensor]] = None
            for i in range(n_edges):
                a, b = edges_list[i]
                for j in range(i + 1, n_edges):
                    c, d = edges_list[j]
                    if len({a, b, c, d}) < 4:
                        continue  # ノードを共有している(隣接エッジ)ので交差ではない
                    hit, pt = _seg_intersect_2d(pos[a], pos[b], pos[c], pos[d])
                    if hit:
                        found = (i, j, pt)
                        break
                if found is not None:
                    break
            if found is None:
                break
            i, j, pt = found
            a, b = edges_list[i]
            c, d = edges_list[j]
            n0 = len(net.nodes)
            w_i = net.widths[i : i + 1]
            w_j = net.widths[j : j + 1]
            keep = torch.ones(n_edges, dtype=torch.bool, device=net.nodes.device)
            keep[i] = False
            keep[j] = False
            new_node = pt.unsqueeze(0)
            new_edges = torch.tensor([[a, n0], [n0, b], [c, n0], [n0, d]], dtype=torch.long, device=net.nodes.device)
            net = RoadNetwork(
                nodes=torch.cat([net.nodes, new_node], dim=0),
                edges=torch.cat([net.edges[keep], new_edges], dim=0),
                widths=torch.cat([net.widths[keep], w_i, w_i, w_j, w_j], dim=0),
                id=net.id,
                payload=net.payload,
            )
        return net

    def gen_rewrite_specs(
        self,
        args: RoadRewriteArgs,
        lim: tuple[float, float],
        add_anywhere_targets: Optional[torch.Tensor] = None,
    ) -> list[RoadRewrite]:
        """
        add_anywhere_targets: (M, 2) の候補点プール。渡された場合、AddAnywhere はこの中から
        ランダムに狙う点を選ぶ(ターゲット人口密度の高い領域に偏らせるために Task 側が渡す)。
        None の場合は lim 全体から一様サンプルする(従来動作)。
        """
        device = self.nodes.device
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)
        degree = self._degree()
        live_nodes = degree.nonzero().flatten().tolist()
        leaf_nodes = (degree == 1).nonzero().flatten().tolist()
        pos = self.nodes.detach().cpu()
        edges_list = self.edges.tolist()
        widths_list = self.widths.tolist()
        max_width = max(args.width_classes)
        min_width = min(args.width_classes)

        is_highway_edge = [w >= max_width - 1e-9 for w in widths_list]
        n_highway_edges = sum(is_highway_edge)

        highway_incident = [0] * n_nodes
        for (a, b), hw in zip(edges_list, is_highway_edge):
            if hw:
                highway_incident[a] += 1
                highway_incident[b] += 1
        highway_eligible = [c > 0 for c in highway_incident]  # 不変条件B: highwayはここからしか生えない

        adj: list[list[tuple[int, int]]] = [[] for _ in range(n_nodes)]
        for eid, (a, b) in enumerate(edges_list):
            adj[a].append((b, eid))
            adj[b].append((a, eid))

        specs: list[RoadRewrite] = []

        # ---- Add: 既存ノードから短い新規エッジを伸ばす(epsilon長, Jump Continuity) ----
        if args.add_weight > 0 and live_nodes:
            n_cand = max(round(args.n_add_candidates * args.add_weight), 1)
            for _ in range(n_cand):
                ni = random.choice(live_nodes)
                ang = random.random() * 2 * math.pi
                length = random.uniform(*args.length_range)
                x0, y0 = pos[ni].tolist()
                x1 = x0 + length * math.cos(ang)
                y1 = y0 + length * math.sin(ang)
                for w in args.width_classes:
                    if w >= max_width - 1e-9 and not highway_eligible[ni]:
                        continue  # 不変条件B
                    specs.append(RoadRewriteAdd(from_node=ni, x=x1, y=y1, width=w))

        # ---- AddAnywhere: ターゲット領域内の点へ、最も近い適格ノードから鎖で繋がったまま到達する ----
        # (Local Geometric Control。不変条件Aを満たすため、AddFreeの代替として孤立配置は行わない)
        # add_anywhere_targets が渡されればそこから狙う点を選ぶ(人口密度の高い領域に偏らせる)。
        # spreading の唯一の手段なので、狙う点を有効領域に集中させることが収束に効く。
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
                for w in args.width_classes:
                    is_hw = w >= max_width - 1e-9
                    eligible = [ni for ni in live_nodes if (highway_eligible[ni] if is_hw else True)]
                    if not eligible:
                        continue
                    cand_pos = pos[eligible]  # (k, 2)
                    d = (cand_pos - target).norm(dim=-1)
                    best = int(d.argmin().item())
                    from_node = eligible[best]
                    dist = float(d[best].item())
                    if dist < 1e-6:
                        continue
                    x0, y0 = pos[from_node].tolist()
                    # 鎖の長さ(hop数)に上限を設ける。上限を超える遠方点へは、その方向へ max_hops 分だけ
                    # 伸ばす(背景を貫く長大な橋を1提案で作らない。連結を保ったまま徐々に伸ばす)。
                    n_hops = max(1, math.ceil(dist / max_len))
                    n_hops = min(n_hops, args.max_add_anywhere_hops)
                    reach = min(1.0, n_hops * max_len / dist)  # 目標点まで届かない場合はその手前まで
                    ex, ey = x0 + (tx - x0) * reach, y0 + (ty - y0) * reach
                    pts = tuple(
                        (x0 + (ex - x0) * (k / n_hops), y0 + (ey - y0) * (k / n_hops)) for k in range(1, n_hops + 1)
                    )
                    specs.append(RoadRewriteAddAnywhere(from_node=from_node, pts=pts, width=w))

        # ---- Remove: 末端エッジの削除(常に連結性を壊さない) ----
        if args.remove_weight > 0 and n_edges > 1:
            for eid, (a, b) in enumerate(edges_list):
                if degree[a].item() == 1 or degree[b].item() == 1:
                    if is_highway_edge[eid] and n_highway_edges <= 1:
                        continue  # 最後の幹線道路は消せない(不変条件B: highwayサブグラフを空にしない)
                    specs.append(RoadRewriteRemove(edge_id=eid))

        # ---- Split: エッジをその場で2分割(形状不変, Jump Continuityの直接例) ----
        if args.split_weight > 0 and n_edges > 0:
            n_cand = max(round(args.n_split_candidates * args.split_weight), 1)
            for _ in range(n_cand):
                eid = random.randrange(n_edges)
                a, b = edges_list[eid]
                t = random.uniform(0.05, 0.95)
                x0, y0 = pos[a].tolist()
                x1, y1 = pos[b].tolist()
                specs.append(RoadRewriteSplit(edge_id=eid, x=x0 + (x1 - x0) * t, y=y0 + (y1 - y0) * t))

        # ---- Merge: 同幅・ほぼ共線の次数2ノードを統合(Splitの逆操作) ----
        if args.merge_weight > 0:
            for ni in live_nodes:
                if degree[ni].item() != 2:
                    continue
                (n0, e0), (n1, e1) = adj[ni]
                if n0 == n1:
                    continue  # 2重辺は対象外
                if widths_list[e0] != widths_list[e1]:
                    continue
                p_c, p0, p1 = pos[ni], pos[n0], pos[n1]
                v0, v1 = p0 - p_c, p1 - p_c
                ang0 = math.atan2(v0[1].item(), v0[0].item())
                ang1 = math.atan2(v1[1].item(), v1[0].item())
                diff = abs((ang0 - ang1 + math.pi) % (2 * math.pi) - math.pi)
                if abs(diff - math.pi) > args.merge_angle_eps:
                    continue  # 直線からずれすぎている(実際の交差点/曲がり角)ので統合不可
                specs.append(
                    RoadRewriteMerge(
                        node_id=ni, edge_id_a=e0, edge_id_b=e1, outer_a=n0, outer_b=n1, width=widths_list[e0]
                    )
                )

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
                edge_is_hw = is_highway_edge[eid]
                for target in neighbors:
                    if target == other or target == leaf:
                        continue
                    if edge_is_hw and not highway_eligible[target]:
                        continue  # 不変条件B: highwayはhighway適格ノードにしかスナップできない
                    specs.append(RoadRewriteSnap(edge_id=eid, end=end, target_node=target))

        # ---- Unsnap: サイクル上のエッジを開放してぶら下がり端点に戻す(Snapの逆操作) ----
        if args.unsnap_weight > 0:
            for eid, (a, b) in enumerate(edges_list):
                if degree[a].item() < 2 or degree[b].item() < 2:
                    continue  # 片方が次数1(leaf)ならブリッジ確定なので対象外
                if not _is_reachable_without_edge(adj, eid, a, b, args.unsnap_bfs_budget):
                    continue  # ブリッジ(削除すると非連結)なので対象外(不変条件A)
                if is_highway_edge[eid] and not _is_reachable_without_edge(
                    adj, eid, a, b, args.unsnap_bfs_budget, edge_mask=is_highway_edge
                ):
                    continue  # highwayサブグラフ限定でもサイクル上であることを要求(不変条件B)
                for end in (0, 1):
                    specs.append(RoadRewriteUnsnap(edge_id=eid, end=end))

        # ---- Widen / Narrow: 幅クラスの格上げ/格下げ(互いに逆操作) ----
        if len(args.width_classes) >= 2:
            for eid, w in enumerate(widths_list):
                a, b = edges_list[eid]
                if args.widen_weight > 0 and w < max_width - 1e-9:
                    # 昇格後もhighwayサブグラフの連結性を保てるのは、どちらかの端点が既にhighwayに
                    # 接している場合のみ(不変条件B)
                    if highway_incident[a] > 0 or highway_incident[b] > 0:
                        for nw in args.width_classes:
                            if nw > w:
                                specs.append(RoadRewriteWiden(edge_id=eid, new_width=nw))
                if args.narrow_weight > 0 and w > min_width:
                    if is_highway_edge[eid]:
                        if n_highway_edges <= 1:
                            continue  # 最後のhighwayエッジは空にできない(不変条件B)
                        if not _is_reachable_without_edge(
                            adj, eid, a, b, args.unsnap_bfs_budget, edge_mask=is_highway_edge
                        ):
                            continue  # highwayサブグラフ内のブリッジ辺。格下げするとhighway網が分断される(不変条件B)
                    for nw in args.width_classes:
                        if nw < w:
                            specs.append(RoadRewriteNarrow(edge_id=eid, new_width=nw))

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
        elif isinstance(spec, RoadRewriteAddAnywhere):
            n0 = len(self.nodes)
            k = len(spec.pts)
            new_nodes = torch.tensor(list(spec.pts), dtype=dtype, device=device)  # (k,2)
            starts = torch.tensor([spec.from_node] + list(range(n0, n0 + k - 1)), dtype=torch.long, device=device)
            ends = torch.arange(n0, n0 + k, dtype=torch.long, device=device)
            new_edges = torch.stack([starts, ends], dim=-1)
            new_widths = torch.full((k,), spec.width, dtype=dtype, device=device)
            return RoadNetwork(
                nodes=torch.cat([self.nodes, new_nodes], dim=0),
                edges=torch.cat([self.edges, new_edges], dim=0),
                widths=torch.cat([self.widths, new_widths], dim=0),
            )
        elif isinstance(spec, RoadRewriteRemove):
            keep = torch.ones(len(self.edges), dtype=torch.bool, device=device)
            keep[spec.edge_id] = False
            return RoadNetwork(nodes=self.nodes, edges=self.edges[keep], widths=self.widths[keep])
        elif isinstance(spec, RoadRewriteSplit):
            n0 = len(self.nodes)
            a, b = self.edges[spec.edge_id].tolist()
            w = self.widths[spec.edge_id : spec.edge_id + 1]
            new_node = torch.tensor([[spec.x, spec.y]], dtype=dtype, device=device)
            keep = torch.ones(len(self.edges), dtype=torch.bool, device=device)
            keep[spec.edge_id] = False
            new_edges = torch.tensor([[a, n0], [n0, b]], dtype=torch.long, device=device)
            return RoadNetwork(
                nodes=torch.cat([self.nodes, new_node], dim=0),
                edges=torch.cat([self.edges[keep], new_edges], dim=0),
                widths=torch.cat([self.widths[keep], w, w], dim=0),
            )
        elif isinstance(spec, RoadRewriteMerge):
            keep = torch.ones(len(self.edges), dtype=torch.bool, device=device)
            keep[spec.edge_id_a] = False
            keep[spec.edge_id_b] = False
            new_edge = torch.tensor([[spec.outer_a, spec.outer_b]], dtype=torch.long, device=device)
            new_width = torch.tensor([spec.width], dtype=dtype, device=device)
            return RoadNetwork(
                nodes=self.nodes,
                edges=torch.cat([self.edges[keep], new_edge], dim=0),
                widths=torch.cat([self.widths[keep], new_width], dim=0),
            )
        elif isinstance(spec, RoadRewriteSnap):
            new_edges = self.edges.clone()
            new_edges[spec.edge_id, spec.end] = spec.target_node
            return RoadNetwork(nodes=self.nodes, edges=new_edges, widths=self.widths)
        elif isinstance(spec, RoadRewriteUnsnap):
            n0 = len(self.nodes)
            old_node = int(self.edges[spec.edge_id, spec.end].item())
            new_node = self.nodes[old_node : old_node + 1].clone()
            new_edges = self.edges.clone()
            new_edges[spec.edge_id, spec.end] = n0
            return RoadNetwork(nodes=torch.cat([self.nodes, new_node], dim=0), edges=new_edges, widths=self.widths)
        elif isinstance(spec, (RoadRewriteWiden, RoadRewriteNarrow)):
            new_widths = self.widths.clone()
            new_widths[spec.edge_id] = spec.new_width
            return RoadNetwork(nodes=self.nodes, edges=self.edges, widths=new_widths)
        else:
            raise ValueError(f"Unknown rewrite {spec}")

    def apply_rewrite_each(self, specs: list[RoadRewrite]) -> list["RoadNetwork"]:
        return [self.apply_rewrite(spec) for spec in specs]

    def apply_all_rewrites(
        self, rewrites: list[RoadRewrite], scores: list[float], max_width: Optional[float] = None
    ) -> "RoadNetwork":
        """
        スコア降順で適用する。同じエッジを対象とする書き換えは先勝ちとし、以降は無視する。

        Unsnap/Narrow は「削除/格下げしても連結性(該当ならhighwayサブグラフの連結性も)を壊さないか」を
        gen_rewrite_specs 時点のスナップショットに対してチェック済みだが、**同じバッチ内の他の書き換え**
        (別のUnsnap/Narrowなど)と組み合わせた結果、初めて非連結になるケースがある。これを防ぐため、
        このバッチ内で既に確定した変更を反映した最新のグラフ状態に対して都度再検証する。

        また Merge/Remove は対象ノード(node_id/末端ノード)を孤立させる操作であるため、同じバッチ内の
        別の Add/AddAnywhere/Snap が"孤立する側"に新しい枝を付けてしまうと、その枝ごと本体から切り離される
        (見た目上は消えていないのに実体は非連結、というバグになる)。これを防ぐため、ノードごとに
        "orphaned"(孤立させる操作が確定済み)/"attached"(新しい接続が確定済み)を記録し、
        孤立操作は既に何らかの記録があるノードには適用せず、接続操作は既に孤立確定済みのノードには
        接続しないようにする(複数のAdd/Snapを同じノードから同時に生やすことは引き続き許可する)。

        ノード削除・再インデックスはここでは行わない(orphanは Task.cleanup が prune_orphan_nodes で処理)。
        max_width が渡された場合はhighwayサブグラフの連結性チェックも行う(不変条件B)。
        """
        order = sorted(range(len(rewrites)), key=lambda i: scores[i], reverse=True)
        device = self.nodes.device
        dtype = self.nodes.dtype
        base_degree = self._degree().tolist()

        # このバッチ内で確定した変更を反映した「現在のグラフ」を軽量なPython辞書で追跡する
        # (Unsnap/Narrowの動的な安全性チェックにのみ使う。最終的なテンソルはこの関数の最後で1回だけ構築する)
        live_edges: dict[int, tuple[int, int, float]] = {
            eid: (a, b, w) for eid, ((a, b), w) in enumerate(zip(self.edges.tolist(), self.widths.tolist()))
        }
        next_edge_id = len(self.edges)
        node_status: dict[int, str] = {}  # node_id -> "orphaned" | "attached"

        def _reachable(exclude_eid: int, start: int, target: int, highway_only: bool = False) -> bool:
            if start == target:
                return True
            adj: dict[int, list[tuple[int, int]]] = {}
            for eid, (a, b, w) in live_edges.items():
                if eid == exclude_eid:
                    continue
                if highway_only and (max_width is None or w < max_width - 1e-9):
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
        added_widths: list[torch.Tensor] = []
        n_next = len(self.nodes)

        touched_edges: set[int] = set()
        removed: set[int] = set()
        snap_updates: dict[int, tuple[int, int]] = {}
        width_updates: dict[int, float] = {}
        n_alive = len(self.edges)

        for i in order:
            spec = rewrites[i]
            if isinstance(spec, RoadRewriteAdd):
                if node_status.get(spec.from_node) == "orphaned":
                    continue  # このバッチ内の他の変更(Merge/Remove)で from_node が孤立する予定
                added_nodes.append(torch.tensor([[spec.x, spec.y]], dtype=dtype, device=device))
                added_edges.append(torch.tensor([[spec.from_node, n_next]], dtype=torch.long, device=device))
                added_widths.append(torch.tensor([spec.width], dtype=dtype, device=device))
                live_edges[next_edge_id] = (spec.from_node, n_next, spec.width)
                next_edge_id += 1
                n_next += 1
                node_status.setdefault(spec.from_node, "attached")
            elif isinstance(spec, RoadRewriteAddAnywhere):
                if node_status.get(spec.from_node) == "orphaned":
                    continue
                k = len(spec.pts)
                added_nodes.append(torch.tensor(list(spec.pts), dtype=dtype, device=device))
                starts = torch.tensor(
                    [spec.from_node] + list(range(n_next, n_next + k - 1)), dtype=torch.long, device=device
                )
                ends = torch.arange(n_next, n_next + k, dtype=torch.long, device=device)
                added_edges.append(torch.stack([starts, ends], dim=-1))
                added_widths.append(torch.full((k,), spec.width, dtype=dtype, device=device))
                for s, e in zip(starts.tolist(), ends.tolist()):
                    live_edges[next_edge_id] = (s, e, spec.width)
                    next_edge_id += 1
                n_next += k
                node_status.setdefault(spec.from_node, "attached")
            elif isinstance(spec, RoadRewriteRemove):
                if spec.edge_id in touched_edges or spec.edge_id not in live_edges or n_alive <= 1:
                    continue
                a, b, _ = live_edges[spec.edge_id]
                leaf = a if base_degree[a] == 1 else b
                if node_status.get(leaf) is not None:
                    continue  # このバッチ内の他の変更(Add/AddAnywhere/Snap)で leaf に新しい枝が付いた
                touched_edges.add(spec.edge_id)
                removed.add(spec.edge_id)
                del live_edges[spec.edge_id]
                n_alive -= 1
                node_status[leaf] = "orphaned"
            elif isinstance(spec, RoadRewriteSplit):
                if spec.edge_id in touched_edges or spec.edge_id not in live_edges:
                    continue
                touched_edges.add(spec.edge_id)
                removed.add(spec.edge_id)
                a, b, w = live_edges.pop(spec.edge_id)
                added_nodes.append(torch.tensor([[spec.x, spec.y]], dtype=dtype, device=device))
                added_edges.append(torch.tensor([[a, n_next], [n_next, b]], dtype=torch.long, device=device))
                added_widths.append(torch.tensor([w, w], dtype=dtype, device=device))
                live_edges[next_edge_id] = (a, n_next, w)
                live_edges[next_edge_id + 1] = (n_next, b, w)
                next_edge_id += 2
                n_next += 1
                n_alive += 1
            elif isinstance(spec, RoadRewriteMerge):
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
                added_widths.append(torch.tensor([spec.width], dtype=dtype, device=device))
                live_edges[next_edge_id] = (spec.outer_a, spec.outer_b, spec.width)
                next_edge_id += 1
                n_alive -= 1
                node_status[spec.node_id] = "orphaned"
            elif isinstance(spec, RoadRewriteSnap):
                if spec.edge_id in touched_edges or spec.edge_id not in live_edges:
                    continue
                if node_status.get(spec.target_node) == "orphaned":
                    continue  # このバッチ内の他の変更(Merge/Remove)で target_node が孤立する予定
                a, b, w = live_edges[spec.edge_id]
                # Snap は「ぶら下がった端点(=このエッジのみを持つ次数1のノード)」を target_node へ
                # 付け替える操作。付け替え元の端点はこのエッジしか持たないため、付け替えると孤立する
                # (Removeの葉ノードと同じ扱いが必要)
                dangling = a if spec.end == 0 else b
                if node_status.get(dangling) is not None:
                    continue  # このバッチ内の他の変更(Add/AddAnywhere/Snap)で付け替え元に新しい枝が付いた
                if max_width is not None and w >= max_width - 1e-9:
                    target_has_hw = any(
                        ww >= max_width - 1e-9 and (aa == spec.target_node or bb == spec.target_node)
                        for aa, bb, ww in live_edges.values()
                    )
                    if not target_has_hw:
                        continue  # このバッチ内の他の変更でtargetがhighway適格でなくなった(不変条件B)
                touched_edges.add(spec.edge_id)
                if spec.end == 0:
                    live_edges[spec.edge_id] = (spec.target_node, b, w)
                else:
                    live_edges[spec.edge_id] = (a, spec.target_node, w)
                snap_updates[spec.edge_id] = (spec.end, spec.target_node)
                node_status[dangling] = "orphaned"
                node_status.setdefault(spec.target_node, "attached")
            elif isinstance(spec, RoadRewriteUnsnap):
                if spec.edge_id in touched_edges or spec.edge_id not in live_edges:
                    continue
                a, b, w = live_edges[spec.edge_id]
                old_node = a if spec.end == 0 else b
                other = b if spec.end == 0 else a
                is_hw = max_width is not None and w >= max_width - 1e-9
                if not _reachable(spec.edge_id, old_node, other) or (
                    is_hw and not _reachable(spec.edge_id, old_node, other, highway_only=True)
                ):
                    continue  # このバッチ内の他の変更と合わせるとブリッジになってしまう(不変条件A/B)
                touched_edges.add(spec.edge_id)
                if spec.end == 0:
                    live_edges[spec.edge_id] = (n_next, b, w)
                else:
                    live_edges[spec.edge_id] = (a, n_next, w)
                added_nodes.append(self.nodes[old_node : old_node + 1].detach().clone())
                snap_updates[spec.edge_id] = (spec.end, n_next)
                n_next += 1
            elif isinstance(spec, (RoadRewriteWiden, RoadRewriteNarrow)):
                if spec.edge_id in touched_edges or spec.edge_id not in live_edges:
                    continue
                a, b, w = live_edges[spec.edge_id]
                if max_width is not None:
                    if isinstance(spec, RoadRewriteWiden) and spec.new_width >= max_width - 1e-9:
                        has_hw_neighbor = any(
                            eid2 != spec.edge_id and ww >= max_width - 1e-9 and (a in (aa, bb) or b in (aa, bb))
                            for eid2, (aa, bb, ww) in live_edges.items()
                        )
                        if not has_hw_neighbor:
                            continue  # このバッチ内の他の変更でhighwayに接していなくなった(不変条件B)
                    if isinstance(spec, RoadRewriteNarrow) and w >= max_width - 1e-9:
                        if not _reachable(spec.edge_id, a, b, highway_only=True):
                            continue  # highwayサブグラフのブリッジ辺。格下げすると分断される(不変条件B)
                touched_edges.add(spec.edge_id)
                live_edges[spec.edge_id] = (a, b, spec.new_width)
                width_updates[spec.edge_id] = spec.new_width
            else:
                raise ValueError(f"Unknown rewrite {spec}")

        edges = self.edges.clone()
        widths = self.widths.clone()
        for eid, (end, target) in snap_updates.items():
            edges[eid, end] = target
        for eid, w in width_updates.items():
            widths[eid] = w
        if removed:
            keep = torch.ones(len(edges), dtype=torch.bool, device=device)
            for eid in removed:
                keep[eid] = False
            edges = edges[keep]
            widths = widths[keep]
        nodes = self.nodes.detach().clone()

        if added_nodes:
            nodes = torch.cat([nodes, *added_nodes], dim=0)
        if added_edges:
            edges = torch.cat([edges, *added_edges], dim=0)
            widths = torch.cat([widths, *added_widths], dim=0)

        return RoadNetwork(nodes=nodes, edges=edges, widths=widths)


# endregion
# region RoadNetworkCollection
# ================== RoadNetworkCollection ===========================


@dataclass
class RoadCollectionArgs:
    sigma0: float = 0.03  # street相当の最小到達半径
    k_sigma: float = 1.0  # 幅 -> 到達半径の係数(幅が太いほど広く効く)
    reach_exponent: float = 1.0  # sigma_e = sigma0 + k_sigma * w_e^reach_exponent。1より大きいほど
    # 幅の広い道路(幹線道路)の到達半径だけが不釣り合いに拡大し、幅の狭い道路(街路)はsigma0付近に留まる。
    amp_scale: float = 0.05  # ピーク強度 = amp_scale / sigma_e (<=1)。sigma(=到達半径)が広いほどピークが下がる
    # amp_scale と sigma_e から決まるピーク強度により、幹線道路(幅広->sigma大)は「薄く広く」、
    # 街路(幅狭->sigma小)は「狭く大きく」効くようにする(断面積 amplitude*sigma がほぼ一定になる)。
    min_density_floor: float = 0.15  # 人口密度の最低ライン。道路の有無によらず無条件で保証する(max で合成)。
    # 損失側の非対称罰則(下回った分だけ罰する)にすると、道路を敷かなければ床を満たせない場所では
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
        returns: (n_networks, *shape) 各ネットワークのSDF(境界判定・可視化用。損失には compute_density を使う)
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
        長さ×幅^width_exponent の総和(建設コスト相当)をネットワークごとに集計する。
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
        道路網のループ(閉路)の多さを 0〜1 程度で表す指標(meshedness / alpha index)。
        cycles = max(E - V + 1, 0)          # 閉路数(連結成分が1つの場合は厳密。複数ある場合は下限値)
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
