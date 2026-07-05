"""
道路網(RoadNetwork/CityNetworkなど、`.nodes: (N,2)` と `.edges: (E,2)` を持つグラフ状オブジェクト)の
「輸送効率性」を評価するユーティリティ。

指標は circuity(迂回率) と呼ばれる、交通網解析で標準的に使われるもの:
    ratio = (道路網上の最短経路長) / (2点間の直線距離)
1.0 が理想(完全に直線で結ばれている)。値が大きいほど道路がグネグネ迂回していることを意味する。
人口密度の高い地点ほど「速く移動できるべき」重要度が高いため、target_img(人口密度マップ)に
比例した確率でサンプルした2点間で評価する。
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Protocol
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class _GraphLike(Protocol):
    nodes: torch.Tensor
    edges: torch.Tensor


@dataclass
class TransportEfficiencyResult:
    ratios: list[float]  # 各サンプル対の (網内最短経路長 / 直線距離)。1.0が理想
    mean_ratio: float
    median_ratio: float
    max_ratio: float
    n_pairs: int  # 評価に使えた点対の数
    n_unreachable: int  # 経路が存在しなかった点対の数(不変条件Aが保たれていれば通常0)

    def __str__(self) -> str:
        return (
            f"transport efficiency (circuity, 1.0=ideal): "
            f"mean={self.mean_ratio:.3f} median={self.median_ratio:.3f} max={self.max_ratio:.3f} "
            f"(n={self.n_pairs}, unreachable={self.n_unreachable})"
        )


def _node_weights_from_density(
    nodes: np.ndarray, target_img: torch.Tensor, lim: tuple[float, float]
) -> np.ndarray:
    """各ノード位置における target_img の値(最近傍ピクセル)を正規化して重みにする。
    render01/compute_density と同じグリッド規約(次元0=y, 次元1=x)を使う。"""
    size = target_img.shape[-1]
    lim0, lim1 = lim
    img = target_img.detach().cpu().numpy()
    cols = np.clip(((nodes[:, 0] - lim0) / (lim1 - lim0) * size).astype(int), 0, size - 1)
    rows = np.clip(((nodes[:, 1] - lim0) / (lim1 - lim0) * size).astype(int), 0, size - 1)
    w = np.clip(img[rows, cols], a_min=0, a_max=None)
    total = w.sum()
    if total <= 0:
        return np.ones(len(nodes)) / len(nodes)
    return w / total


@torch.no_grad()
def compute_transport_efficiency(
    net: _GraphLike,
    target_img: Optional[torch.Tensor] = None,
    lim: tuple[float, float] = (-1.5, 1.5),
    n_pairs: int = 200,
    min_euclidean_dist: float = 0.05,
    seed: Optional[int] = None,
) -> TransportEfficiencyResult:
    """
    人口密度の高い任意の2点間の輸送効率性(circuity比)を評価する。

    - target_img が与えられた場合、2点は target_img の値に比例した確率でノードから重み付きサンプルする
      (人口密度の高い場所ほど選ばれやすい)。None の場合は全ノードから一様にサンプルする。
    - 最短経路はエッジ長(ユークリッド長)を重みとしたDijkstra法で計算する。
    - min_euclidean_dist 未満の近すぎる点対は比が不安定になるため除外する。
    """
    nodes = net.nodes.detach().cpu().numpy()
    edges = net.edges.detach().cpu().numpy()
    n = len(nodes)
    if n < 2 or len(edges) == 0:
        return TransportEfficiencyResult([], float("nan"), float("nan"), float("nan"), 0, 0)

    lengths = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=-1)
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([lengths, lengths])
    graph = csr_matrix((data, (row, col)), shape=(n, n))

    rng = np.random.default_rng(seed)
    weights = _node_weights_from_density(nodes, target_img, lim) if target_img is not None else np.ones(n) / n

    src_idx = rng.choice(n, size=n_pairs, p=weights)
    dst_idx = rng.choice(n, size=n_pairs, p=weights)

    ratios: list[float] = []
    n_unreachable = 0
    dist_cache: dict[int, np.ndarray] = {}
    for s, d in zip(src_idx.tolist(), dst_idx.tolist()):
        if s == d:
            continue
        euclid = float(np.linalg.norm(nodes[s] - nodes[d]))
        if euclid < min_euclidean_dist:
            continue
        if s not in dist_cache:
            dist_cache[s] = dijkstra(graph, directed=False, indices=s)
        network_dist = dist_cache[s][d]
        if not np.isfinite(network_dist):
            n_unreachable += 1
            continue
        ratios.append(network_dist / euclid)

    if not ratios:
        return TransportEfficiencyResult([], float("nan"), float("nan"), float("nan"), 0, n_unreachable)

    arr = np.array(ratios)
    return TransportEfficiencyResult(
        ratios=ratios,
        mean_ratio=float(arr.mean()),
        median_ratio=float(np.median(arr)),
        max_ratio=float(arr.max()),
        n_pairs=len(ratios),
        n_unreachable=n_unreachable,
    )
