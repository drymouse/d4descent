import torch
from dataclasses import dataclass

from .objects.arclines import Shape, Arc, Line, ShapeCollection


@dataclass
class ShapeMetricSummary:
    pct_matches: float  # number of matches
    pct_perfect_matches: float  # number of perfect matches
    pct_false_line_matches: float  # number of false line matches
    pct_false_arc_matches: float  # number of false arc matches
    match_score: float  # sum of percentage of samples that match

    @classmethod
    def mean(cls, summaries: list["ShapeMetricSummary"]) -> "ShapeMetricSummary":
        return ShapeMetricSummary(
            pct_matches=sum([s.pct_matches for s in summaries]) / len(summaries),
            pct_perfect_matches=sum([s.pct_perfect_matches for s in summaries]) / len(summaries),
            pct_false_line_matches=sum([s.pct_false_line_matches for s in summaries]) / len(summaries),
            pct_false_arc_matches=sum([s.pct_false_arc_matches for s in summaries]) / len(summaries),
            match_score=sum([s.match_score for s in summaries]) / len(summaries),
        )


@dataclass
class ShapeMetric:
    source_prim_types: list[int]  # (n,)
    target_prim_types: list[int]  # (m,)
    matches: list[tuple[int, int]]  # (min(n, m),)
    match_scores: list[float]  # (min(n, m),) percentage of samples that match (higher is better)

    def summarize(self, threshold: float = 0.8) -> ShapeMetricSummary:
        working_idx = [i for i, score in enumerate(self.match_scores) if score >= threshold]
        working = [self.matches[i] for i in working_idx]

        m = len(self.target_prim_types)

        return ShapeMetricSummary(
            pct_matches=len(working) / m,
            pct_perfect_matches=len([0 for i, j in working if self.source_prim_types[i] == self.target_prim_types[j]])
            / m,
            pct_false_line_matches=len(
                [0 for i, j in working if self.source_prim_types[i] == 0 and self.target_prim_types[j] != 0]
            )
            / m,
            pct_false_arc_matches=len(
                [0 for i, j in working if self.source_prim_types[i] == 1 and self.target_prim_types[j] != 1]
            )
            / m,
            match_score=sum([self.match_scores[i] for i in working_idx]) / m,
        )


@torch.no_grad
def compute_metric(
    source_shape: Shape,
    target_shape: Shape,
    n_samples: int = 32,
) -> ShapeMetric:
    n = len(source_shape.primitives)
    m = len(target_shape.primitives)

    # Do chamfer dist between pairwise of source primitives and target primitives.
    # Two primitives are a candidate match if the number of point samples from the source (target) primitve that matches
    # that of the target (source) primitive is greater than n_threshold.
    # A source (target) point sample is said to match with a target (source) primitive if the distance to the closest
    # target (source) point samples is less than the threshold (target (source) primitive length / n_samples).

    src_pts, _, src_lens = ShapeCollection.from_shape(source_shape).sample_points_length_fast(n_samples)
    tgt_pts, _, tgt_lens = ShapeCollection.from_shape(target_shape).sample_points_length_fast(n_samples)

    src_thresh = src_lens.squeeze(0) / (n_samples * 2) + 1e-6  # (n,)
    tgt_thresh = tgt_lens.squeeze(0) / (n_samples * 2) + 1e-6  # (m,)

    src_pts = src_pts.reshape(n, 1, n_samples, 1, 2)
    tgt_pts = tgt_pts.reshape(m, 1, n_samples, 2)

    dists = (src_pts - tgt_pts).norm(dim=-1)  # (n, m, n_samples (src), n_samples (tgt))

    dists_from_src = dists.min(dim=-1).values  # (n, m, n_samples (src))
    n_matched_from_src = (dists_from_src < tgt_thresh.unsqueeze(-1)).sum(dim=-1)  # (n, m)
    dists_from_tgt = dists.min(dim=-2).values  # (n, m, n_samples (tgt))
    n_matched_from_tgt = (dists_from_tgt < src_thresh.unsqueeze(-1).unsqueeze(-1)).sum(dim=-1)  # (n, m)

    match_scores = torch.min(n_matched_from_src, n_matched_from_tgt) / n_samples  # (n, m) 0 to 1

    # Use hungarian algorithm to find matchings

    swapped = False
    A = match_scores.detach().cpu()
    if n > m:
        A = A.transpose(0, 1)
        swapped = True

    ind, _ = hungarian(-A)  # (min(n, m),)
    inds = [(j, i) if swapped else (i, j) for i, j in enumerate(ind.tolist())]

    return ShapeMetric(
        source_prim_types=[src.type_id() for src in source_shape.primitives],
        target_prim_types=[tgt.type_id() for tgt in target_shape.primitives],
        matches=inds,
        match_scores=[match_scores[i, j].item() for i, j in inds],
    )


def make_A(source_shape: Shape, target_shape: Shape) -> torch.Tensor:
    """
    Construct's a cost matrix A, where A_ij is the distance
    between prim_i and prim_j in respective shapes. Swaps
    shapes if needed to make cost matrix wide.

    Returns: (n, m), n <= m
    """
    n, m = len(source_shape.primitives), len(target_shape.primitives)
    # Cost matrix A must be wide, otherwise the O(n^3) algo can run forever
    if n > m:
        source_shape, target_shape = target_shape, source_shape
        n, m = m, n
    A = torch.full((n, m), -1.0)
    for i in range(n):
        for j in range(m):
            a = source_shape.primitives[i]
            b = target_shape.primitives[j]
            if isinstance(a, Line):
                if isinstance(b, Line):
                    A[i, j] = torch.norm(a.start - b.start) + torch.norm(a.end - b.end)
                elif isinstance(b, Arc):
                    A[i, j] = torch.norm(a.start - b.start) + torch.norm(a.end - b.end) + torch.norm(b.k)
            elif isinstance(a, Arc):
                if isinstance(b, Line):
                    A[i, j] = torch.norm(a.start - b.start) + torch.norm(a.end - b.end) + torch.norm(a.k)
                elif isinstance(b, Arc):
                    A[i, j] = torch.norm(a.start - b.start) + torch.norm(a.end - b.end) + torch.norm(a.k - b.k)
    assert torch.all(A >= 0)
    return A


def hungarian(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    From https://cp-algorithms.com/graph/hungarian-algorithm.html
    Input: Cost matrix A (n, m), n <= m
    Finds a permutation p of length n such that sum A[i, p[i]] is minimized.

    Returns:
    (n,) col indexes of match for each row (p[i])
    () sum of cost along matching indexes
    """
    device = A.device
    assert len(A.shape) == 2
    n, m = A.shape
    assert n <= m
    A = A.detach().cpu()
    # u, v are potentials, start at all zeros
    # p is the matching
    u = torch.full((n + 1,), 0.0)
    v = torch.full((m + 1,), 0.0)
    p = torch.full((m + 1,), 0)
    way = torch.full((m + 1,), 0)  # way[j] has num in prev col in the aug path

    for i in range(1, n + 1):
        p[0] = i  # curr row
        j0 = 0  # init free column
        j1 = 0
        minv = torch.full((m + 1,), torch.inf)  # aux min
        used = torch.full((m + 1,), False)
        while True:
            used[j0] = True
            i0 = p[j0]
            Delta = torch.inf
            for j in range(1, m + 1):
                if not used[j]:
                    # Calculate potential adjustment
                    cur = A[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < Delta:
                        Delta = minv[j]
                        j1 = j
            for j in range(0, m + 1):
                # Potential adjustment
                if used[j]:
                    u[p[j]] += Delta
                    v[j] -= Delta
                else:
                    minv[j] -= Delta
            j0 = j1
            if not p[j0] != 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if not j0:
                break

    ans = torch.full((n + 1,), 0)
    for j in range(1, m + 1):
        ans[p[j]] = j - 1
    cost = -v[0]
    return ans[1:].to(device), cost.to(device)
