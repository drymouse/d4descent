import torch
from dataclasses import dataclass, field
from typing import Optional, Self, Union, overload, Literal, cast, Type
from collections.abc import Sequence
from abc import ABC, abstractmethod
from enum import Enum
from itertools import combinations, product
from functools import partial, partialmethod
import random
import numpy as np

from ..context import Context
from ..object_collection import ObjectCollection
from ..util import torch_interp, safe_cat, safe_stack, maybe_detach

# region Primitives
# ================== Primitives ===========================


@dataclass
class Primitive(ABC):
    start: torch.Tensor  # (2,)
    end: torch.Tensor  # (2,)

    def __post_init__(self):
        assert self.start.shape == (2,), f"start must be (2,), got {self.start.shape}"
        assert self.end.shape == (2,), f"end must be (2,), got {self.end.shape}"
        assert self.start is not self.end, "degenerate primitive"
        assert self.start.isnan().any() == False, f"start must not be NaN, got {self.start}"
        assert self.end.isnan().any() == False, f"end must not be NaN, got {self.end}"

    def device(self) -> torch.device:
        return self.start.device

    @abstractmethod
    def reverse(self) -> Self: ...

    @abstractmethod
    def type_id(self) -> int: ...

    @abstractmethod
    def get_fast_bbox(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a bounding box, not necessarily smallest.
        """
        ...

    @abstractmethod
    def clone(self) -> Self: ...


def _format_tensor(x: torch.Tensor) -> str:
    ss = [f"{x_.item():.2f}" for x_ in x]
    return f"({', '.join(ss)})"


@dataclass
class Line(Primitive):
    def __repr__(self) -> str:
        return f"Line({_format_tensor(self.start)}, {_format_tensor(self.end)})"

    def reverse(self) -> "Line":
        return Line(self.end, self.start)

    def type_id(self) -> int:
        return 0

    def get_fast_bbox(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.stack([self.start, self.end], dim=0)
        return x.min(dim=0).values, x.max(dim=0).values

    def clone(self) -> "Line":
        return Line(self.start, self.end)


@dataclass
class Arc(Primitive):
    k: torch.Tensor  # ()

    def __repr__(self) -> str:
        return f" Arc({_format_tensor(self.start)}, {_format_tensor(self.end)}, {self.k.item():.2f})"

    def __post_init__(self):
        super().__post_init__()
        assert self.k.shape == (), f"k must be (), got {self.k.shape}"
        assert self.k.isnan().any() == False, f"k must not be NaN, got {self.k}"

    def reverse(self) -> "Arc":
        return Arc(self.end, self.start, -self.k)

    def clone(self) -> "Arc":
        return Arc(self.start, self.end, self.k)

    def type_id(self) -> int:
        return 1

    def get_fast_bbox(self) -> tuple[torch.Tensor, torch.Tensor]:
        # circle of radius k on the midpoint
        mid = (self.start + self.end) / 2
        emin = torch.minimum(self.start, self.end)
        emax = torch.maximum(self.start, self.end)
        return torch.minimum(emin, mid - self.k.abs()), torch.maximum(emax, mid + self.k.abs())

    def compute_o_r_thetas(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
        - o: (2,)
        - r: (,)
        - theta0: (,)
        - theta1: (,)
        If k >= 0, theta0 < theta1. Otherwise, theta0 > theta1
        """
        with torch.no_grad():
            midpoint = (self.start + self.end) / 2
            dir = self.end - self.start
            norm = (dir**2).sum().sqrt() + 1e-9
            perp = torch.stack([-dir[1], dir[0]], dim=0) / norm  # (2,)
            r = (self.k**2 + (norm / 2) ** 2) / (2 * self.k - 1e-4 + 2e-4 * (self.k >= 0))
            o = midpoint + (r - self.k) * perp  # (2,)
            theta0 = torch.atan2(self.start[1] - o[1], self.start[0] - o[0])
            theta1 = torch.atan2(self.end[1] - o[1], self.end[0] - o[0])
            if self.k >= 0 and theta0 > theta1:
                theta1 += 2 * torch.pi
            elif self.k < 0 and theta0 < theta1:
                theta0 += 2 * torch.pi
        return o, r.abs(), theta0, theta1

    def sample(self, t: Union[torch.Tensor, float]) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=self.start.dtype, device=self.start.device)
        return vectorized_sample_arc(self.start, self.end, self.k, t)


_Vec2Like = Union[torch.Tensor, tuple[float, float]]
_ScalarLike = Union[torch.Tensor, float]


class PrimitiveHelper:
    @classmethod
    def create_circle_arcs(
        cls,
        num_points: int,
        center: _Vec2Like,
        radius: _ScalarLike,
        device: Union[str, torch.device, None] = None,
        reversed: bool = False,
    ) -> list[Primitive]:
        """
        center: (2,)
        radius: ()
        returns: (num_points, 2)
        """
        center = torch.as_tensor(center, dtype=torch.float32, device=device)
        radius = torch.as_tensor(radius, dtype=torch.float32, device=device)
        # k = r * (1 - cos(theta)), where  2*theta is the span of the arc
        theta = torch.as_tensor((2 * torch.pi / num_points) / 2, dtype=torch.float32, device=device)
        k = torch.as_tensor(radius * (1 - torch.cos(theta)), dtype=torch.float32, device=device)
        phi = 2.0 * torch.pi * torch.arange(num_points, device=device) / num_points
        circle_points = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)  # (num_points, 2)
        points_ = radius * circle_points + center  # (num_points * n, 2)
        points = [p for p in points_]
        sgn = -1 if reversed else 1

        return [
            Arc(start=points[sgn * i], end=points[(sgn * (i + 1)) % len(points)], k=k.clone() * sgn)
            for i in range(len(points))
        ]

    @classmethod
    def create_circle_lines(
        cls,
        num_points: int,
        center: _Vec2Like,
        radius: _ScalarLike,
        device: Union[str, torch.device, None] = None,
        phase: float = 0.0,
    ) -> list[Primitive]:
        """
        center: (2,)
        radius: ()
        returns: (num_points, 2)
        """
        center = torch.as_tensor(center, dtype=torch.float32, device=device)
        radius = torch.as_tensor(radius, dtype=torch.float32, device=device)
        phi = 2.0 * torch.pi * torch.arange(num_points, device=device) / num_points + phase
        circle_points = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)  # (num_points, 2)
        points_ = radius * circle_points + center  # (num_points * n, 2)
        points = [p for p in points_]
        return [Line(start=points[i], end=points[(i + 1) % len(points)]) for i in range(len(points))]

    @classmethod
    def create_rectangle(
        cls,
        top_left: _Vec2Like,
        bottom_right: _Vec2Like,
        device: Union[str, torch.device, None] = None,
    ) -> list[Primitive]:
        """
        top_left: (2,)
        bottom_right: (2,)
        returns: (2,)
        """
        top_left = torch.as_tensor(top_left, dtype=torch.float32, device=device)
        bottom_right = torch.as_tensor(bottom_right, dtype=torch.float32, device=device)
        p1 = top_left
        p2 = torch.stack([top_left[0], bottom_right[1]], dim=-1)
        p3 = bottom_right
        p4 = torch.stack([bottom_right[0], top_left[1]], dim=-1)

        return [Line(start=p1, end=p2), Line(start=p2, end=p3), Line(start=p3, end=p4), Line(start=p4, end=p1)]

    @classmethod
    def create_polygon(
        cls,
        points: list[_Vec2Like],
        device: Union[str, torch.device, None] = None,
    ) -> list[Primitive]:
        n = len(points)
        assert n >= 3, f"Polygon must have at least 3 points, got {n}"
        tensors = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in points]
        return [Line(start=tensors[i], end=tensors[(i + 1) % n]) for i in range(n)]


# endregion
# region Rewrites
# ================== Rewrites ===========================


def _safe_sign(x: torch.Tensor) -> torch.Tensor:
    """returns 1 if x >= 0, else -1"""
    return torch.where(x >= 0, torch.tensor(1.0, device=x.device), torch.tensor(-1.0, device=x.device))


class ShapeRewriteType(Enum):
    Split = 0
    ToLine = 1
    ToArc = 2
    Merge = 3
    NoOp = 4
    ResolveIntersections = 5
    RandomSubdivision = 6
    Simplify = 7
    AddHole = 8
    RemoveHole = 9
    CanonicalizeLoops = 10
    MergeClose = 11
    Multiple = 100

    @classmethod
    def inverse(cls, rewrite_type: "ShapeRewriteType") -> "ShapeRewriteType":
        if rewrite_type == ShapeRewriteType.Split:
            return ShapeRewriteType.Merge
        elif rewrite_type == ShapeRewriteType.ToLine:
            return ShapeRewriteType.ToArc
        elif rewrite_type == ShapeRewriteType.ToArc:
            return ShapeRewriteType.ToLine
        elif rewrite_type == ShapeRewriteType.Merge:
            return ShapeRewriteType.Split
        elif rewrite_type == ShapeRewriteType.NoOp:
            return ShapeRewriteType.NoOp
        else:
            raise ValueError(f"Inverse of rewrite type: {rewrite_type} is not supported")


@dataclass
class ShapeRewriteSpec:
    type: ShapeRewriteType
    args: tuple[float, ...] = field(default_factory=tuple)


# Splits


def split_line(line: Line, t: float = 0.5) -> tuple[Line, Line]:
    assert 0 <= t <= 1, "t must be between 0 and 1"
    midpoint = t * (line.start + line.end)
    first_line = Line(line.start, midpoint)
    second_line = Line(midpoint, line.end)
    return first_line, second_line


def split_arc(arc: Arc, t: float = 0.5) -> tuple[Arc, Arc]:
    assert 0 <= t <= 1, "t must be between 0 and 1"
    with torch.no_grad():
        p1, p2, p3 = vectorized_sample_arc(
            arc.start, arc.end, arc.k, torch.tensor([t / 2, t, (t + 1) / 2], device=arc.start.device)
        )
        p1_ = (arc.start + p2) / 2
        p3_ = (p2 + arc.end) / 2
        sgn = torch.sign(arc.k)
        first_arc = Arc(arc.start, p2, sgn * (p1 - p1_).norm())
        second_arc = Arc(p2, arc.end, sgn * (p3 - p3_).norm())
    return first_arc, second_arc


# Merges


def merge_line(line1: Line, line2: Line) -> tuple[Line, float]:
    """
    line1.start --> line1.end --> line2.end
    returns
    - merged line
    - max distance between before and after
    """
    if id(line1.end) != id(line2.start):
        raise ValueError("line1.end must be equal to line2.start")
    if line1.start is line2.end:
        return line1, torch.inf
    with torch.no_grad():
        dir = line2.end - line1.start
        dir /= dir.norm() + 1e-9
        p = line1.start + ((line1.end - line1.start) * dir).sum(dim=-1) * dir
        cost = (p - line1.end).norm().item()
    return Line(line1.start, line2.end), cost


def merge_arc(prim1: Primitive, prim2: Primitive) -> tuple[Arc, float]:
    """
    prim1.start --> prim1.end --> prim2.end
    returns:
    - merged arc. The arc passes through start, prim1.end, and prim2.end
    - max distance between before and after
    """
    if id(prim1.end) != id(prim2.start):
        raise ValueError("prim1.end must be equal to prim2.start")
    if prim1.start is prim2.end:
        return Arc(prim1.start, prim1.end, torch.tensor(0.0, device=prim1.start.device)), torch.inf
    with torch.no_grad():
        p1 = prim1.start
        p2 = prim1.end
        p3 = prim2.end
        x = (p1 + p3) / 2
        vx1 = p1 - x
        vx2 = p2 - x
        d = torch.stack([vx1[1], -vx1[0]], dim=0)
        d /= d.norm()  # p1 != p3
        num = (vx2**2).sum() - (vx1**2).sum()
        denom = 2 * (vx2 * d).sum()
        denom = _safe_sign(denom) * torch.max(denom.abs(), torch.tensor(1e-3, device=prim1.device()))
        t = num / denom
        o = x + t * d
        r = (p1 - o).norm()
        v13 = p3 - p1
        v12 = p2 - p1
        ori = _safe_sign(v13[0] * v12[1] - v13[1] * v12[0])
        k = -(r * ori + (x - o).norm() * _safe_sign(t))

        merged_arc = Arc(p1, p3, k)

        # Compute cost
        theta1 = torch.atan2(p1[1] - o[1], p1[0] - o[0])
        theta2 = torch.atan2(p2[1] - o[1], p2[0] - o[0])
        theta3 = torch.atan2(p3[1] - o[1], p3[0] - o[0])
        if k > 0:
            if theta1 > theta2:
                theta2 += 2 * torch.pi
            if theta1 > theta3:
                theta3 += 2 * torch.pi
        else:
            if theta1 < theta3:
                theta1 += 2 * torch.pi
            if theta2 < theta3:
                theta2 += 2 * torch.pi

        thetas = torch.stack([(theta1 + theta2) / 2, (theta2 + theta3) / 2], dim=0)
        m1, m2 = o + r.abs() * torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=-1)  # (num_samples, 2)
        half = torch.tensor([0.5], device=prim1.device())
        k1 = prim1.k if isinstance(prim1, Arc) else torch.tensor(0.0, device=prim1.device())
        k2 = prim2.k if isinstance(prim2, Arc) else torch.tensor(0.0, device=prim2.device())
        m1_ = vectorized_sample_arc(prim1.start, prim1.end, k1, half)[0]
        m2_ = vectorized_sample_arc(prim2.start, prim2.end, k2, half)[0]
        cost = max((m1 - m1_).norm().item(), (m2 - m2_).norm().item())
        return merged_arc, cost


def merge_close(prim1: Primitive, prim2: Primitive) -> tuple[Primitive, float]:
    if prim1.start is prim2.end:
        return prim1, torch.inf
    d1 = (prim1.end - prim1.start).norm().item()
    d2 = (prim2.end - prim2.start).norm().item()
    d3 = (prim1.start - prim2.end).norm().item()
    mn = min(d1, d2, d3)
    if d1 == mn:
        prim2 = prim2.clone()
        prim2.start = prim1.start
        return prim2, d1
    elif d2 == mn:
        prim1 = prim1.clone()
        prim1.end = prim2.end
        return prim1, d2
    else:
        return Line(prim1.start, prim2.end), d3


# Change Type


def arc2_to_line(arc: Arc) -> tuple[Line, float]:
    return Line(arc.start, arc.end), abs(arc.k.item())


def line_to_arc2(line: Line) -> tuple[Arc, float]:
    return Arc(line.start, line.end, torch.tensor(0.0, dtype=line.end.dtype, device=line.end.device)), 0.0


# endregion
# region Intersection
# ================== Intersection ===========================

__X_EPS = 1e-4
__X_EPS2 = __X_EPS**2
__allclose = partial(torch.allclose, atol=__X_EPS, rtol=1e-4)


def _bbox_intersect(bbox1: tuple[torch.Tensor, torch.Tensor], bbox2: tuple[torch.Tensor, torch.Tensor]) -> bool:
    """
    Assume bbox is in the form (min, max). No checks are performed
    """
    return not ((bbox1[0] > bbox2[1]).any().item() or (bbox2[0] > bbox1[1]).any().item())


def _is_adjacent_primitives(p1: Primitive, p2: Primitive) -> bool:
    return (
        __allclose(p1.start, p2.start)
        or __allclose(p1.start, p2.end)
        or __allclose(p1.end, p2.start)
        or __allclose(p1.end, p2.end)
    )


def line_line_intersection(line1: Line, line2: Line) -> Optional[torch.Tensor]:
    """
    line1: (start, end)
    line2: (start, end)
    returns: (2,)
    """
    if _is_adjacent_primitives(line1, line2):
        return None
    with torch.no_grad():
        dir1 = line1.end - line1.start
        dir2 = line2.end - line2.start
        denom = dir1[0] * dir2[1] - dir1[1] * dir2[0]
        if denom.abs() < __X_EPS2:
            return None
        t1 = (line1.start[1] - line2.start[1]) * dir2[0] - (line1.start[0] - line2.start[0]) * dir2[1]
        t2 = (line1.start[1] - line2.start[1]) * dir1[0] - (line1.start[0] - line2.start[0]) * dir1[1]
        t1 = t1 / denom
        t2 = t2 / denom
        if t1 < __X_EPS or t1 > 1 - __X_EPS or t2 < __X_EPS or t2 > 1 - __X_EPS:
            return None
        c_ = line1.start + dir1 * t1
        if __allclose(c_, line2.start) or __allclose(c_, line2.end):
            return None
        return c_


def line_arc_intersection(line: Line, arc: Arc) -> tuple[list[torch.Tensor], list[Arc], bool]:
    """
    Returns:
    - points: (num_points,) in the order of line (num_points = 0, 1, or 2)
    - arcs: [num_points + 1], split arcs, arcs don't share endpoints
    - rev: whether the order of points is reversed for arcs
    """
    with torch.no_grad():
        o, r, t1, t2 = arc.compute_o_r_thetas()
        o = o.cpu().numpy()
        r = r.item()
        t1 = t1.item()
        t2 = t2.item()
        p1 = line.start.cpu().numpy()
        p2 = line.end.cpu().numpy()

    v12 = p2 - p1
    v1o = o - p1
    d12 = np.linalg.norm(v12) + 1e-12
    n12 = v12 / d12

    u_oproj = np.dot(v1o, n12)
    oproj = u_oproj * n12 + p1
    d = np.linalg.norm(oproj - o)
    disc = r**2 - d**2
    if disc < __X_EPS2:
        return [], [arc], False
    disc_sqrt = np.sqrt(disc)

    candidates: list[np.ndarray] = []
    line_eps_ = __X_EPS / d12
    for u_ in [u_oproj - disc_sqrt, u_oproj + disc_sqrt]:
        if u_ >= line_eps_ and u_ <= d12 - line_eps_:
            candidates.append(u_ * n12 + p1)

    rev_ = False
    if t1 > t2:
        t1, t2 = t2, t1
        rev_ = True
    t2 = t1 + (t2 - t1) % (2 * np.pi)

    intersections: list[tuple[torch.Tensor, float]] = []

    arc_eps_ = __X_EPS / r
    for c in candidates:
        cdir = c - o
        angle = np.arctan2(cdir[1], cdir[0])
        angle = t1 + (angle - t1) % (2 * np.pi)
        if angle >= t1 + arc_eps_ and angle <= t2 - arc_eps_:
            c_ = torch.from_numpy(c).to(dtype=line.start.dtype, device=line.start.device)
            if not (
                __allclose(c_, line.start)
                or __allclose(c_, line.end)
                or __allclose(c_, arc.start)
                or __allclose(c_, arc.end)
            ):
                # import pdb; pdb.set_trace()
                intersections.append((c_, angle))

    if len(intersections) == 0:
        return [], [arc], False

    xs = [x[0] for x in intersections]

    rev_1 = False
    if len(intersections) == 2:
        a1 = intersections[0][1]
        a2 = intersections[1][1]
        if a1 > a2:
            rev_1 = True
            intersections.reverse()

    endpoints: list[tuple[torch.Tensor, float]] = [
        (arc.start if not rev_ else arc.end, float(t1)),
        *intersections,
        (arc.end if not rev_ else arc.start, float(t2)),
    ]
    arcs: list[Arc] = []
    for i in range(len(endpoints) - 1):
        p1_, t1_ = endpoints[i]
        p3_, t3_ = endpoints[i + 1]
        if i > 0:
            p1_ = p1_.clone()
        if i + 1 < len(endpoints) - 1:
            p3_ = p3_.clone()
        k = torch.tensor(r * (1 - np.cos((t3_ - t1_) / 2)), dtype=p1_.dtype, device=p1_.device)
        arcs.append(Arc(p1_, p3_, k))

    if rev_:
        arcs = [arc.reverse() for arc in arcs]
        arcs.reverse()

    return xs, arcs, rev_ != rev_1


def arc_arc_intersection(arc1: Arc, arc2: Arc) -> tuple[list[Arc], list[Arc], bool]:
    with torch.no_grad():
        o1, r1, t11, t12 = arc1.compute_o_r_thetas()
        o2, r2, t21, t22 = arc2.compute_o_r_thetas()
        o1 = o1.cpu().numpy().astype(np.float64)
        r1 = r1.item()
        t11 = t11.item()
        t12 = t12.item()
        o2 = o2.cpu().numpy().astype(np.float64)
        r2 = r2.item()
        t21 = t21.item()
        t22 = t22.item()
        p11 = arc1.start
        p12 = arc1.end
        p21 = arc2.start
        p22 = arc2.end

    o12 = o2 - o1
    d = np.linalg.norm(o12)
    if d < __X_EPS:
        return [arc1], [arc2], False
    x = (d**2 + r1**2 - r2**2) / (2 * d)
    if x < -r1 or x > r1:
        return [arc1], [arc2], False

    rev1 = False
    if t11 > t12:
        rev1 = True
        t11, t12 = t12, t11
        p11, p12 = p12, p11
    rev2 = False
    if t21 > t22:
        rev2 = True
        t21, t22 = t22, t21
        p21, p22 = p22, p21

    t12 = t11 + (t12 - t11) % (2 * np.pi)
    t22 = t21 + (t22 - t21) % (2 * np.pi)

    n12 = o12 / d
    perp = np.array([-n12[1], n12[0]])
    diff = np.sqrt(r1**2 - x**2)
    xs: list[tuple[torch.Tensor, float, float]] = []
    arc1_eps_ = __X_EPS / r1
    arc2_eps_ = __X_EPS / r2
    for c in [o1 + x * n12 + diff * perp, o1 + x * n12 - diff * perp]:
        vo1c = c - o1
        vo2c = c - o2
        ang1 = np.arctan2(vo1c[1], vo1c[0])
        ang2 = np.arctan2(vo2c[1], vo2c[0])
        ang1 = t11 + (ang1 - t11) % (2 * np.pi)
        ang2 = t21 + (ang2 - t21) % (2 * np.pi)
        if ang1 >= t11 + arc1_eps_ and ang1 <= t12 - arc1_eps_ and ang2 >= t21 + arc2_eps_ and ang2 <= t22 - arc2_eps_:
            c_ = torch.from_numpy(c).to(dtype=arc1.start.dtype, device=arc1.start.device)
            if not (
                __allclose(c_, arc1.start)
                or __allclose(c_, arc1.end)
                or __allclose(c_, arc2.start)
                or __allclose(c_, arc2.end)
            ):
                xs.append((c_, ang1, ang2))

    if len(xs) == 0:
        return [arc1], [arc2], False

    swap1 = False
    swap2 = False
    xs1 = [(x[0], x[1]) for x in xs]
    xs2 = [(x[0], x[2]) for x in xs]
    if len(xs) == 2:
        _, a11, a21 = xs[0]
        _, a12, a22 = xs[1]
        if a11 > a12:
            swap1 = True
            xs1.reverse()
        if a21 > a22:
            swap2 = True
            xs2.reverse()
    xs1: list[tuple[torch.Tensor, float]] = [(p11, t11), *xs1, (p12, t12)]
    xs2: list[tuple[torch.Tensor, float]] = [(p21, t21), *xs2, (p22, t22)]
    arcs1: list[Arc] = []
    arcs2: list[Arc] = []
    for xs_, arcs_, k_, r_ in zip([xs1, xs2], [arcs1, arcs2], [arc1.k, arc2.k], [r1, r2]):
        for i in range(len(xs_) - 1):
            p1_, t1_ = xs_[i]
            p3_, t3_ = xs_[i + 1]
            if i > 0:
                p1_ = p1_.clone()
            if i + 1 < len(xs_) - 1:
                p3_ = p3_.clone()
            k = torch.tensor(r_ * (1 - np.cos((t3_ - t1_) / 2)), dtype=p1_.dtype, device=p1_.device)
            arcs_.append(Arc(p1_, p3_, k))

    if rev1:
        arcs1 = [arc.reverse() for arc in arcs1]
        arcs1.reverse()
    if rev2:
        arcs2 = [arc.reverse() for arc in arcs2]
        arcs2.reverse()
    return arcs1, arcs2, (rev1 != rev2) != (swap1 != swap2)


@dataclass
class ShapeRewriteArgs:
    split_line: bool = True
    split_arc: bool = True
    merge_line: bool = True
    merge_arc: bool = True
    merge_close: bool = True
    to_arc: bool = True
    to_line: bool = True
    lossy_threshold: float = 0.03
    remove_holes: bool = True
    add_holes: bool = True
    add_hole_random: bool = False
    add_hole_count: int = 32  # if add_hole_random
    add_hole_lim: float = 0.8
    add_hole_grid: int = 5
    add_hole_radius: float = 0.01
    add_hole_segment: int = 4
    remove_hole_area: float = 0.001
    # random
    reg_weight: float = 0.75
    add_holes_weight: float = 0.25


@dataclass
class RandomSubdivisionArgs:
    # split_weight: float = 0.5
    pass


@dataclass
class SimplifyArgs:
    merge_threshold: float = 0.005
    to_line_threshold: float = 0.005


# endregion
# region Shape
# ================== Shape ===========================


@dataclass
class ShapePayload:
    from_id: Optional[int] = None
    rewrite_spec: Optional[ShapeRewriteSpec] = None
    rewrite_prim_ids: Optional[tuple[int, ...]] = None
    multiple_rewrites: Optional[list[ShapeRewriteSpec]] = None
    loss: Optional[float] = None
    duplicates: list[int] = field(default_factory=list)


@dataclass
class Shape:
    primitives: list[Primitive]
    id: int = field(default_factory=lambda: Context.get().gen_id())
    payload: ShapePayload = field(default_factory=ShapePayload)

    def __post_init__(self):
        Context().get().register(self.id, self)

    def __repr__(self) -> str:
        inner = ",\n      ".join([repr(p) for p in self.primitives])
        return f"Shape({inner})"

    def clone(self) -> "Shape":
        """
        Create a shallow copy of Shape
        """
        return Shape(list(self.primitives), id=self.id, payload=self.payload)

    def device(self) -> torch.device:
        return self.primitives[0].device()

    def apply_rewrite(self, spec: ShapeRewriteSpec) -> "Shape":
        type_ = spec.type
        args = spec.args

        if type_ == ShapeRewriteType.Split:
            assert len(args) == 1, f"Split args must be (id)"
            i = int(args[0])
            prim = self.primitives[i]
            if isinstance(prim, Line):
                l0, l1 = split_line(prim)
            elif isinstance(prim, Arc):
                l0, l1 = split_arc(prim)
            else:
                raise ValueError(f"Unknown primitive type: {type(prim)}")
            return Shape(
                self.primitives[:i] + [l0, l1] + self.primitives[i + 1 :],
                payload=ShapePayload(
                    from_id=self.id,
                    rewrite_spec=spec,
                    rewrite_prim_ids=(i, i + 1),
                ),
            )
        elif type_ == ShapeRewriteType.ToLine:
            assert len(args) == 1, f"ToLine args must be (id)"
            i = int(args[0])
            prim = self.primitives[i]
            assert isinstance(prim, Arc), f"ToLine only works on Arcs, got {type(prim)}"
            l, _ = arc2_to_line(prim)
            return Shape(
                self.primitives[:i] + [l] + self.primitives[i + 1 :],
                payload=ShapePayload(
                    from_id=self.id,
                    rewrite_spec=spec,
                    rewrite_prim_ids=(i,),
                ),
            )
        elif type_ == ShapeRewriteType.ToArc:
            assert len(args) == 1, f"ToArc args must be (id)"
            i = int(args[0])
            prim = self.primitives[i]
            assert isinstance(prim, Line), f"ToArc only works on Lines, got {type(prim)}"
            a, _ = line_to_arc2(prim)
            return Shape(
                self.primitives[:i] + [a] + self.primitives[i + 1 :],
                payload=ShapePayload(
                    from_id=self.id,
                    rewrite_spec=spec,
                    rewrite_prim_ids=(i,),
                ),
            )
        elif type_ == ShapeRewriteType.Merge or type_ == ShapeRewriteType.MergeClose:
            assert len(args) == 2, f"Merge args must be (id1, id2)"
            i1, i2 = int(args[0]), int(args[1])
            prim1 = self.primitives[i1]
            prim2 = self.primitives[i2]
            if type_ == ShapeRewriteType.MergeClose:
                l, _ = merge_close(prim1, prim2)
            else:
                if isinstance(prim1, Line) and isinstance(prim2, Line):
                    l, _ = merge_line(prim1, prim2)
                else:
                    l, _ = merge_arc(prim1, prim2)
            i1_, i2_ = i1, i2
            if i1_ > i2_:
                i1_, i2_ = i2_, i1_
            return Shape(
                self.primitives[:i1_] + [l] + self.primitives[i1_ + 1 : i2_] + self.primitives[i2_ + 1 :],
                payload=ShapePayload(
                    from_id=self.id,
                    rewrite_spec=spec,
                    rewrite_prim_ids=(i1_,),
                ),
            )
        elif type_ == ShapeRewriteType.RemoveHole:
            ids_ = set(int(x) for x in args)
            new_prims: list[Primitive] = []
            for i, prim in enumerate(self.primitives):
                if i not in ids_:
                    new_prims.append(prim)
            return Shape(
                new_prims,
                payload=ShapePayload(
                    from_id=self.id,
                    rewrite_spec=spec,
                ),
            )
        elif type_ == ShapeRewriteType.AddHole:
            assert len(args) == 5, f"AddHole args must be (x, y, radius)"
            x, y, radius, reversed_, n_segments = args
            reversed_ = bool(reversed_)
            n_segments = int(n_segments)
            return Shape(
                [
                    *self.primitives,
                    *PrimitiveHelper.create_circle_arcs(
                        n_segments, (x, y), radius, device=self.device(), reversed=reversed_
                    ),
                ],
                payload=ShapePayload(
                    from_id=self.id,
                    rewrite_spec=spec,
                ),
            )
        else:
            raise ValueError(f"Unknown rewrite type {type_}")

    def generate_rewrite_specs(
        self,
        args: ShapeRewriteArgs = ShapeRewriteArgs(),
    ) -> list[ShapeRewriteSpec]:
        """
        Returns list of shapes after performing one rewrite on each primitive.
        Discards if the rewrite cost is greater than lossy_threshold.
        Note: the returned shapes may share underlying tensors
        Note: the order of primitives will remain unchanged.
        For merge rewrites, the merged primitive will take the order of the first primitive.
        """
        lossy_threshold = args.lossy_threshold
        res: list[ShapeRewriteSpec] = []
        for i, prim in enumerate(self.primitives):
            if isinstance(prim, Line):
                # split
                if args.split_line:
                    l0, l1 = split_line(prim)
                    res.append(ShapeRewriteSpec(ShapeRewriteType.Split, (i,)))
                # change to arc
                if args.to_arc:
                    a, cost = line_to_arc2(prim)
                    if cost <= lossy_threshold:
                        res.append(ShapeRewriteSpec(ShapeRewriteType.ToArc, (i,)))
            elif isinstance(prim, Arc):
                # split
                if args.split_arc:
                    a0, a1 = split_arc(prim)
                    res.append(ShapeRewriteSpec(ShapeRewriteType.Split, (i,)))
                # change to line
                if args.to_line:
                    l, cost = arc2_to_line(prim)
                    if cost <= lossy_threshold:
                        res.append(ShapeRewriteSpec(ShapeRewriteType.ToLine, (i,)))

        if args.merge_arc or args.merge_line or args.merge_close:
            # find common node for merges
            ins_: dict[int, list[int]] = {}  # list of primitives ending at node
            outs_: dict[int, list[int]] = {}  # list of primitives starting at node
            for i, prim in enumerate(self.primitives):
                ins__ = ins_.get(id(prim.end), [])
                outs__ = outs_.get(id(prim.start), [])
                ins__.append(i)
                outs__.append(i)
                ins_[id(prim.end)] = ins__
                outs_[id(prim.start)] = outs__

            for k, ins in ins_.items():
                outs = outs_.get(k, [])
                if len(ins) == 1 and len(outs) == 1:
                    in_ = self.primitives[ins[0]]
                    out_ = self.primitives[outs[0]]
                    cost = float("inf")
                    l: Optional[Primitive] = None
                    if args.merge_line and isinstance(in_, Line) and isinstance(out_, Line):
                        # merge
                        l, cost = merge_line(in_, out_)
                    elif args.merge_arc and isinstance(in_, Arc) or isinstance(out_, Arc):
                        # merge
                        l, cost = merge_arc(in_, out_)

                    if l is not None and cost <= lossy_threshold:
                        res.append(ShapeRewriteSpec(ShapeRewriteType.Merge, (ins[0], outs[0])))

                    if args.merge_close:
                        l, cost = merge_close(in_, out_)
                        if cost <= lossy_threshold:
                            res.append(ShapeRewriteSpec(ShapeRewriteType.MergeClose, (ins[0], outs[0])))

        # remove holes
        if args.remove_holes:
            loops = self.find_loops()
            if len(loops) > 1:
                for ids_, area in loops:
                    if abs(area) <= args.remove_hole_area:
                        res.append(ShapeRewriteSpec(ShapeRewriteType.RemoveHole, tuple(ids_)))

        return res

    def do_multiple_rewrites(
        self,
        rewrites: list[ShapeRewriteSpec],
    ) -> "Shape":
        """
        Assumes all rewrites do not conflict. Preserves ordering of primitives.
        Merges take the order of the lower indexed primitive.
        """
        rewrites_ = [(r.type, r.args) for r in rewrites]
        if not all(
            type_
            in [
                ShapeRewriteType.Merge,
                ShapeRewriteType.MergeClose,
                ShapeRewriteType.Split,
                ShapeRewriteType.ToLine,
                ShapeRewriteType.ToArc,
                ShapeRewriteType.AddHole,
                ShapeRewriteType.RemoveHole,
            ]
            for type_, _ in rewrites_
        ):
            raise ValueError(f"Unsupported rewrite type")

        # sort by primitive id
        # save AddHole for last
        f_rewrites = sorted(
            [(type_, tuple(sorted(args_)), args_) for type_, args_ in rewrites_ if type_ != ShapeRewriteType.AddHole],
            key=lambda x: x[1],
        )
        add_holes = [x[1] for x in rewrites_ if x[0] == ShapeRewriteType.AddHole]

        seen: set[int] = set()
        new_prims: list[Primitive] = []
        cur_ = 0
        for type_, sorted_, args_ in f_rewrites:
            args_ = cast(tuple[int, ...], args_)
            sargs_ = set(args_)
            if sargs_ & seen:
                raise ValueError(f"Conflicting rewrites: {type_}, {args_}")
            seen.update(sargs_)
            while cur_ < sorted_[0]:
                if cur_ not in seen:
                    new_prims.append(self.primitives[cur_])
                cur_ += 1
            if type_ == ShapeRewriteType.Merge:
                assert len(args_) == 2
                in_ = self.primitives[args_[0]]
                out_ = self.primitives[args_[1]]
                if isinstance(in_, Line) and isinstance(out_, Line):
                    l, cost = merge_line(in_, out_)
                else:
                    l, cost = merge_arc(in_, out_)
                new_prims.append(l)
            elif type_ == ShapeRewriteType.MergeClose:
                assert len(args_) == 2
                in_ = self.primitives[args_[0]]
                out_ = self.primitives[args_[1]]
                l, cost = merge_close(in_, out_)
                new_prims.append(l)
            elif type_ == ShapeRewriteType.Split:
                assert len(args_) == 1
                in_ = self.primitives[args_[0]]
                if isinstance(in_, Line):
                    l0, l1 = split_line(in_)
                    new_prims.append(l0)
                    new_prims.append(l1)
                elif isinstance(in_, Arc):
                    a0, a1 = split_arc(in_)
                    new_prims.append(a0)
                    new_prims.append(a1)
                else:
                    raise ValueError(f"Invalid split: {args_}")
            elif type_ == ShapeRewriteType.ToLine:
                assert len(args_) == 1
                in_ = self.primitives[args_[0]]
                if isinstance(in_, Arc):
                    # change to line
                    l, cost = arc2_to_line(in_)
                    new_prims.append(l)
                else:
                    raise ValueError(f"Invalid ToLine: {args_}")
            elif type_ == ShapeRewriteType.ToArc:
                assert len(args_) == 1
                in_ = self.primitives[args_[0]]
                if isinstance(in_, Line):
                    # change to arc
                    a, cost = line_to_arc2(in_)
                    new_prims.append(a)
                else:
                    raise ValueError(f"Invalid ToArc: {args_}")
            elif type_ == ShapeRewriteType.RemoveHole:
                pass
            else:
                raise ValueError(f"Unknown rewrite type: {type_}")
        while cur_ < len(self.primitives):
            if cur_ not in seen:
                new_prims.append(self.primitives[cur_])
            cur_ += 1

        if len(add_holes) > 0:
            positions = torch.tensor(add_holes, device=self.device())[..., :2]
            _, wn = ShapeCollection.from_shape(self).rasterize(positions)  # (num_holes,)

            for x, y, r, reversed, n_segments in add_holes:
                new_prims.extend(
                    PrimitiveHelper.create_circle_arcs(
                        int(n_segments),
                        (x, y),
                        r,
                        device=self.device(),
                        reversed=bool(reversed),
                    )
                )

        if len(new_prims) == 0:
            print("Multiple rewrites yielded no primitives")
            return self.clone()

        return Shape(
            primitives=new_prims,
            payload=ShapePayload(
                from_id=self.id, rewrite_spec=ShapeRewriteSpec(ShapeRewriteType.Multiple), multiple_rewrites=rewrites
            ),
        )

    def do_random_subdivision(self, args: RandomSubdivisionArgs = RandomSubdivisionArgs()) -> "Shape":
        """Apply split or to_arc to random primitives"""
        new_prims: list[Primitive] = []
        for i, prim in enumerate(self.primitives):
            if isinstance(prim, Line):
                # split
                l0, l1 = split_line(prim)
                new_prims.append(line_to_arc2(l0)[0])
                new_prims.append(line_to_arc2(l1)[0])
            elif isinstance(prim, Arc):
                # split
                a0, a1 = split_arc(prim)
                new_prims.append(a0)
                new_prims.append(a1)
            else:
                raise ValueError(f"Unknown primitive type: {type(prim)}")
        return Shape(
            new_prims,
            payload=ShapePayload(from_id=self.id, rewrite_spec=ShapeRewriteSpec(ShapeRewriteType.RandomSubdivision)),
        )

    def do_simplify(self, args: SimplifyArgs) -> tuple["Shape", bool]:
        """Apply to_line then merge_arc and merge_line if possible

        Returns:
        - new shape or self if no changes
        - whether the shape was changed
        """
        # # compute thresholds
        # merge_threshold = args.merge_threshold
        # to_line_threshold = args.to_line_threshold
        # if loss is not None:
        #     sqrt_loss = loss ** 0.5
        #     merge_threshold = min(merge_threshold, args.merge_rel_threhold * sqrt_loss)
        #     to_line_threshold = min(to_line_threshold, args.to_line_rel_threhold * sqrt_loss)

        # find common node for merges
        ins_: dict[int, list[int]] = {}  # list of primitives ending at node
        outs_: dict[int, list[int]] = {}  # list of primitives starting at node
        for i, prim in enumerate(self.primitives):
            ins__ = ins_.get(id(prim.end), [])
            outs__ = outs_.get(id(prim.start), [])
            ins__.append(i)
            outs__.append(i)
            ins_[id(prim.end)] = ins__
            outs_[id(prim.start)] = outs__

        deleted: set[int] = set()
        new_prim_ords: list[tuple[Primitive, int]] = [(p, i) for i, p in enumerate(self.primitives)]

        has_changed = False
        changed = True
        while changed:
            changed = False
            for k, ins in ins_.items():
                outs = outs_.get(k, [])
                if len(ins) == 1 and len(outs) == 1:
                    in_id = ins[0]
                    out_id = outs[0]
                    in_ = new_prim_ords[in_id][0]
                    out_ = new_prim_ords[out_id][0]

                    # merge
                    l, cost = merge_arc(in_, out_)
                    # print(f"{in_id},{out_id}: {cost}")

                    if cost <= args.merge_threshold:
                        outs_[id(in_.start)].remove(in_id)
                        ins.remove(in_id)
                        outs.remove(out_id)
                        ins_[id(out_.end)].remove(out_id)
                        deleted.add(in_id)
                        deleted.add(out_id)
                        merged_id = len(new_prim_ords)
                        new_prim_ords.append((l, new_prim_ords[in_id][1]))
                        outs_[id(in_.start)].append(merged_id)
                        ins_[id(out_.end)].append(merged_id)
                        changed = True
                        has_changed = True
        new_prim_ords = [p for i, p in enumerate(new_prim_ords) if i not in deleted]
        new_prim_ords.sort(key=lambda x: x[1])

        # Turn unnecessary arcs into lines
        new_prims: list[Primitive] = []
        for prim, i in new_prim_ords:
            if isinstance(prim, Arc):
                a, cost = arc2_to_line(prim)
                # print(f"to_line: {i}: {cost}")
                if cost <= args.to_line_threshold:
                    new_prims.append(a)
                    has_changed = True
                else:
                    new_prims.append(prim)
            else:
                new_prims.append(prim)

        if not has_changed:
            return self, False
        return (
            Shape(
                new_prims,
                payload=ShapePayload(from_id=self.id, rewrite_spec=ShapeRewriteSpec(ShapeRewriteType.Simplify)),
            ),
            True,
        )

    def resolve_intersections(self) -> "Shape":
        """
        Resolves intersections between primitives. Returns a new shape.

        TODO: Support more intersections
        """
        primitives = list(self.primitives)
        pairs = list(combinations(primitives, 2))
        i_ = 0
        changed_once = False
        deleted: set[int] = set()
        og_pairs = len(pairs)
        while i_ < min(len(pairs), og_pairs * 3):
            pi, pj = pairs[i_]
            if id(pi) in deleted or id(pj) in deleted:
                i_ += 1
                continue
            if _bbox_intersect(pi.get_fast_bbox(), pj.get_fast_bbox()):
                new_prims = None
                if isinstance(pi, Line) and isinstance(pj, Line):
                    x1 = line_line_intersection(pi, pj)
                    if x1 is not None:
                        x2 = x1.detach().clone()
                        # pi.start -> x1 -> pj.end
                        ix = Line(pi.start, x1)
                        xj = Line(x1, pj.end)
                        # pj.start -> x2 -> pi.end
                        jx = Line(pj.start, x2)
                        xi = Line(x2, pi.end)
                        new_prims = [ix, xj, jx, xi]
                elif (isinstance(pi, Line) and isinstance(pj, Arc)) or (isinstance(pj, Line) and isinstance(pi, Arc)):
                    if isinstance(pi, Arc) and isinstance(pj, Line):
                        pi, pj = pj, pi
                    assert isinstance(pi, Line) and isinstance(pj, Arc)
                    xs, arcs, rev = line_arc_intersection(pi, pj)
                    if len(xs) == 1:
                        assert len(arcs) == 2
                        a1, a2 = arcs
                        # a1 -> line.end
                        l1 = Line(a1.end, pi.end)
                        # line.start -> a2
                        l2 = Line(pi.start, a2.start)
                        new_prims = [a1, a2, l1, l2]
                    elif len(xs) == 2:
                        assert len(arcs) == 3
                        a1, a2, a3 = arcs
                        if rev:
                            l1 = Line(pi.start, a3.start)
                            l2 = Line(a2.end, a2.start)
                            l3 = Line(a1.end, pi.end)
                        else:
                            l1 = Line(pi.start, a2.start)
                            l2 = Line(a1.end, a3.start)
                            l3 = Line(a2.end, pi.end)
                        new_prims = [a1, a2, a3, l1, l2, l3]
                elif isinstance(pi, Arc) and isinstance(pj, Arc):
                    arcs1, arcs2, rev = arc_arc_intersection(pi, pj)
                    assert len(arcs1) == len(arcs2)
                    if len(arcs1) == 2:
                        a11, a12 = arcs1
                        a21, a22 = arcs2
                        a22.start = a11.end
                        a12.start = a21.end
                        new_prims = [a11, a12, a21, a22]
                    elif len(arcs1) == 3:
                        a11, a12, a13 = arcs1
                        a21, a22, a23 = arcs2
                        if rev:
                            a13.start = a21.end
                            a12.start = a22.end
                            a22.start = a12.end
                            a23.start = a11.end
                        else:
                            a22.start = a11.end
                            a13.start = a22.end
                            a12.start = a21.end
                            a23.start = a12.end
                        new_prims = [a11, a12, a13, a21, a22, a23]

                if new_prims is not None:
                    primitives = [p for p in primitives if p is not pi and p is not pj]
                    deleted.add(id(pi))
                    deleted.add(id(pj))
                    pairs.extend(list(product(primitives, new_prims)))
                    primitives.extend(new_prims)
                    changed_once = True
            i_ += 1

        return (
            Shape(
                primitives,
                payload=ShapePayload(
                    from_id=self.id, rewrite_spec=ShapeRewriteSpec(ShapeRewriteType.ResolveIntersections)
                ),
            )
            if changed_once
            else self
        )

    def canonicalize_loops(self) -> "Shape":
        """
        Re-orient loops so that its CCW -> CW -> CCW -> ... from outside to inside.
        Behavior is undefined if loops intersect.
        Order of primitives may be changed.
        """
        loops = self.find_loops()
        loops.sort(key=lambda x: abs(x[1]), reverse=True)
        in_loops = set(sum([x[0] for x in loops], []))
        cur_prims = [self.primitives[i] for i in range(len(self.primitives)) if i not in in_loops]
        # _, wn = ShapeCollection.from_shape(self).rasterize(positions)  # (num_holes,)
        for loop, area in loops:
            positions = torch.stack([self.primitives[i].start for i in loop], dim=0)  # (loop_size, 2)
            if len(cur_prims) > 0:
                _, wn = ShapeCollection.from_shape(Shape(cur_prims, id=-1)).rasterize(positions)
                rev = (wn.abs() > torch.pi).sum().item() >= (wn.numel() // 2)
            else:
                rev = False
            new_prims = [self.primitives[i] for i in loop]
            if rev != (area < 0):
                new_prims = [p.reverse() for p in new_prims]
                new_prims.reverse()
            cur_prims.extend(new_prims)
        return Shape(
            cur_prims,
            payload=ShapePayload(from_id=self.id, rewrite_spec=ShapeRewriteSpec(ShapeRewriteType.CanonicalizeLoops)),
        )

    def remove_all_holes(self, threshold: float) -> "Shape":
        loops = self.find_loops()
        loops.sort(key=lambda x: abs(x[1]))
        del_ids: list[int] = []
        n_loops = len(loops)
        for ids_, area in loops:
            if abs(area) <= threshold and n_loops > 1:
                del_ids.extend(ids_)
                n_loops -= 1
        del_ids_ = set(del_ids)
        return Shape(
            [self.primitives[i] for i in range(len(self.primitives)) if i not in del_ids_],
            payload=ShapePayload(
                from_id=self.id, rewrite_spec=ShapeRewriteSpec(ShapeRewriteType.RemoveHole, tuple(del_ids))
            ),
        )

    def gen_add_holes_spec(
        self, positions: torch.Tensor, radius: float = 0.01, n_segments: int = 4
    ) -> list[ShapeRewriteSpec]:
        """
        positions: (num_holes, 2)
        returns: list of shapes (num_holes,) each with each hole
        """
        assert positions.ndim == 2, f"positions must be (num_holes, 2), got {positions.shape}"
        _, wn = ShapeCollection.from_shape(self).rasterize(positions)  # (num_holes,)
        res: list[ShapeRewriteSpec] = []
        for wn_, pos_ in zip(wn[0].tolist(), positions):
            res.append(
                ShapeRewriteSpec(
                    type=ShapeRewriteType.AddHole,
                    args=(pos_[0].item(), pos_[1].item(), radius, abs(wn_) >= torch.pi and wn_ > 0, n_segments),
                )
            )
        return res

    def find_loops(self) -> list[tuple[list[int], float]]:
        """
        Returns a list of (prim_ids, loop_area) for each loop in the shape. The areas are signed. Positive means CCW.
        Doesn't account for self-intersections.
        """
        prims_of_: dict[int, int] = {}
        degs_: dict[int, int] = {}  # primitives starting/ending at node
        nexts_: dict[int, int] = {}
        all_points: set[int] = set()  # all ids of points
        for i, prim in enumerate(self.primitives):
            start_id_ = id(prim.start)
            end_id_ = id(prim.end)
            all_points.add(start_id_)
            all_points.add(end_id_)
            degs_[start_id_] = degs_.get(start_id_, 0) + 1
            degs_[end_id_] = degs_.get(end_id_, 0) + 1
            nexts_[start_id_] = end_id_
            prims_of_[start_id_] = i

        if any(deg > 2 for deg in degs_.values()):
            raise ValueError("Cannot reorder a shape with junctions")
        seen: set[int] = set()
        all_loops: list[list[int]] = []
        for i in all_points:
            if i in seen or degs_.get(i, 0) <= 1:
                continue
            cur_ = i
            seen.add(cur_)
            cur_loop: list[int] = [cur_]
            while True:
                next_ = nexts_[cur_]
                if next_ in seen:
                    break
                cur_ = next_
                seen.add(next_)
                cur_loop.append(next_)
            all_loops.append(cur_loop)

        result: list[tuple[list[int], float]] = []
        for loop in all_loops:
            prim_ids_ = [prims_of_[i_] for i_ in loop]  # list of prim_ids in this loop
            all_points_ = torch.stack([self.primitives[i_].start for i_ in prim_ids_], dim=0)  # (loop_len, 2)
            x_, y_ = all_points_.unbind(dim=-1)  # (loop_len,) (loop_len,)
            area = (x_ * y_.roll(-1, 0) - x_.roll(-1, 0) * y_).sum().item() / 2
            for prim_id_ in prim_ids_:
                prim = self.primitives[prim_id_]
                if isinstance(prim, Arc):
                    _, r_, theta0_, theta1_ = prim.compute_o_r_thetas()
                    dt = theta1_ - theta0_
                    area += (dt.item() - dt.sin().item()) * r_.square().item() / 2
            result.append((prim_ids_, area))
        return result

    @overload
    def reorder_primitives(self, return_stats: Literal[True]) -> tuple["Shape", int, int]: ...
    @overload
    def reorder_primitives(self, return_stats: Literal[False] = False) -> "Shape": ...

    def reorder_primitives(self, return_stats: bool = False):
        """
        If the shape only contains directed (closed or unclosed) segments, order the primitives
        according to the segments. Raises ValueError if the shape contains junctions.
        Returns a Shape with the same id as self.
        """
        degs_: dict[int, list[int]] = {}  # primitives starting/ending at node
        all_points: set[int] = set()  # all ids of points
        for i, prim in enumerate(self.primitives):
            all_points.add(id(prim.start))
            all_points.add(id(prim.end))
            ins = degs_.get(id(prim.end), [])
            outs = degs_.get(id(prim.start), [])
            ins.append(i)
            outs.append(i)
            degs_[id(prim.end)] = ins
            degs_[id(prim.start)] = outs
        if any(len(deg) > 2 for deg in degs_.values()):
            raise ValueError("Cannot reorder a shape with junctions")
        seen: set[int] = set()
        ones = [k for k, v in degs_.items() if len(v) == 1]  # node with no incoming edges
        prims: list[Primitive] = []
        unclosed = 0

        def _get_next(x: list[int], y: int) -> Optional[int]:
            x = list(x)
            try:
                x.remove(y)
            except ValueError:
                pass
            if len(x) == 0:
                return None
            return x[0]

        for i in ones:
            if i in seen:
                continue
            cur = i  # node id
            prev = -1  # primitive id
            seen.add(cur)
            unclosed += 1
            while (out_prim_id := _get_next(degs_[cur], prev)) is not None:
                prim = self.primitives[out_prim_id]
                if id(prim.start) != cur:
                    prim = prim.reverse()
                assert id(prim.start) == cur
                nxt_id = id(prim.end)
                seen.add(nxt_id)
                prims.append(prim)
                cur = nxt_id
                prev = out_prim_id
        closed = 0
        for i in all_points:
            if i in seen:
                continue
            cur = i  # node id
            prev = -1  # primitive id
            seen.add(cur)
            closed += 1
            while (out_prim_id := _get_next(degs_[cur], prev)) is not None:
                prim = self.primitives[out_prim_id]
                if id(prim.start) != cur:
                    prim = prim.reverse()
                assert id(prim.start) == cur
                nxt_id = id(prim.end)
                prims.append(prim)
                if nxt_id in seen:
                    break
                seen.add(nxt_id)
                cur = nxt_id
                prev = out_prim_id
        new_shape = Shape(prims, id=self.id)
        if return_stats:
            return new_shape, unclosed, closed
        return new_shape

    @classmethod
    def create_rectangle(
        cls,
        top_left: _Vec2Like,
        bottom_right: _Vec2Like,
        device: Union[str, torch.device, None] = None,
    ) -> "Shape":
        """
        top_left: (2,)
        bottom_right: (2,)
        returns: (2,)
        """
        return Shape(
            PrimitiveHelper.create_rectangle(top_left, bottom_right, device),
        )

    @classmethod
    def create_circle_lines(
        cls,
        num_points: int,
        center: _Vec2Like,
        radius: _ScalarLike,
        device: Union[str, torch.device, None] = None,
        phase: float = 0.0,
    ) -> "Shape":
        """
        center: (2,)
        radius: ()
        returns: (num_points, 2)
        """
        return Shape(PrimitiveHelper.create_circle_lines(num_points, center, radius, device, phase))

    @classmethod
    def create_circle_arcs(
        cls,
        num_points: int,
        center: _Vec2Like,
        radius: _ScalarLike,
        device: Union[str, torch.device, None] = None,
        reversed: bool = False,
    ) -> "Shape":
        """
        center: (2,)
        radius: ()
        returns: (num_points, 2)
        """
        return Shape(
            PrimitiveHelper.create_circle_arcs(num_points, center, radius, device, reversed),
        )


# endregion
# region ShapeMeta
# ================== ShapeMeta ===========================


class PrimitiveType(Enum):
    Line = 0
    Arc = 1


@dataclass
class ShapeMeta:
    line_idx: torch.Tensor  # (n_lines,) indices to lines
    arcs_idx: torch.Tensor  # (n_arcs,) indices to arcs
    order: list[tuple[PrimitiveType, int]]  # (n_primitives,) order of primitives, map to line_dix or arcs_idx
    # derived fields
    _ordered_ids: Optional[list[int]] = field(init=False, default=None)

    def to(self, device: Union[str, torch.device, None] = None) -> "ShapeMeta":
        return ShapeMeta(
            line_idx=self.line_idx.to(device=device),
            arcs_idx=self.arcs_idx.to(device=device),
            order=self.order.copy(),
        )

    def device(self) -> torch.device:
        return self.line_idx.device

    def n_prims(self) -> int:
        return len(self.order)

    def n_prims_arcs(self) -> tuple[int, int]:
        # for ordering. fewer primitives first then fewer arcs
        return len(self.order), len(self.arcs_idx)

    def get_ordered_ids(self) -> list[int]:
        if self._ordered_ids is not None:
            return self._ordered_ids
        self._ordered_ids = []
        n_lines = len(self.line_idx)
        for type_, idx_ in self.order:
            if type_ == PrimitiveType.Line:
                self._ordered_ids.append(idx_)
            elif type_ == PrimitiveType.Arc:
                self._ordered_ids.append(n_lines + idx_)
        return self._ordered_ids


def _insert_ids(ids: dict[int, int], objs: list[torch.Tensor], obj: torch.Tensor) -> int:
    id_ = id(obj)
    res = ids.get(id_, None)
    if res is None:
        res = len(ids)
        ids[id_] = res
        objs.append(obj.detach().clone())
    return res


# endregion
# region Vectorized sampling
# ================== Vectorized sampling functions ===========================
def vectorized_sample_line(starts: torch.Tensor, ends: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    starts: (..., 2)
    ends: (..., 2)
    t: (num_samples,)
    Returns: (..., num_samples, 2)
    """
    t = t.unsqueeze(-1)  # (num_samples, 1)
    starts = starts.unsqueeze(-2)  # (..., 1, 2)
    ends = ends.unsqueeze(-2)  # (..., 1, 2)
    sampled_points = (1 - t) * starts + t * ends  # (..., num_samples, 2)
    return sampled_points


def vectorized_line_length(starts: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    """
    starts: (..., 2)
    ends: (..., 2)
    returns: (...)
    """
    return torch.norm(ends - starts, dim=-1)


def vectorized_arc_sample_length(
    starts: torch.Tensor, ends: torch.Tensor, ks: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized arc samples and computes the length of the arc.

    Params:
    - starts: (..., 2)
    - ends: (..., 2)
    - ks: (...,)
    - t: (num_samples,) or (..., num_samples)

    Returns:
    - point samples: (..., num_samples, 2)
    - arc lengths: (...)
    """
    # Compute midpoints: (..., 2)
    midpoints = (starts + ends) / 2
    # Compute dir: (..., 2)
    dirs = ends - starts
    # Compute norm: (..., 1)
    norms = torch.norm(dirs, dim=-1, keepdim=True) + 1e-9
    # Compute perp: (..., 2)
    perps = torch.stack([-dirs[..., 1], dirs[..., 0]], dim=-1) / norms  # (..., 2)
    # Compute r: (..., 1)
    ks = ks.unsqueeze(-1)  # (..., 1)
    ks_positive = ks >= 0  # (..., 1)
    r = (ks**2 + (norms / 2) ** 2) / (2 * ks - 1e-4 + 2e-4 * ks_positive)  # (..., 1)
    # Compute o: (..., 2)
    o = midpoints + (r - ks) * perps  # (..., 2)
    # Compute theta0 and theta1: (...,)
    sx, sy = (starts - o).unbind(dim=-1)  # (...,) (...,)
    ex, ey = (ends - o).unbind(dim=-1)  # (...,) (...,)
    theta0 = torch.atan2(sy, sx)  # (...,)
    theta1 = torch.atan2(ey, ex)  # (...,)
    # Adjust theta0 and theta1 based on ks
    ks_positive = ks_positive.squeeze(-1)  # (...,)
    mask1 = ks_positive & (theta0 > theta1)
    mask2 = (~ks_positive) & (theta0 < theta1)
    theta1 = torch.where(mask1, theta1 + 2 * torch.pi, theta1)
    theta0 = torch.where(mask2, theta0 + 2 * torch.pi, theta0)
    # Interpolate theta:
    theta0 = theta0.unsqueeze(-1)  # (..., 1)
    theta1 = theta1.unsqueeze(-1)  # (..., 1)
    theta = theta0 * (1 - t) + theta1 * t  # (..., num_samples)
    # Compute points: (..., num_samples, 2)
    point_samples = o.unsqueeze(-2) + r.abs().unsqueeze(-2) * torch.stack(
        [torch.cos(theta), torch.sin(theta)], dim=-1
    )  # (..., num_samples, 2)
    # Compute arc lengths
    arc_lengths = (r.abs() * (theta1 - theta0).abs()).squeeze(-1)  # (...)
    return point_samples, arc_lengths


# TODO: rewrite everything to use this
def vectorized_o_r_thetas(
    starts: torch.Tensor, ends: torch.Tensor, ks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized sample arcs.
    - starts: (..., 2)
    - ends: (..., 2)
    - ks: (...,)

    Returns:
    - o: (..., 2)
    - r: (...,)  signed
    - theta0: (...,)
    - theta1: (...,)
      If k >= 0, theta0 < theta1. Otherwise, theta0 > theta1
    """
    # Compute midpoints: (..., 2)
    midpoints = (starts + ends) / 2
    # Compute dir: (..., 2)
    dirs = ends - starts
    # Compute norm: (..., 1)
    norms = torch.norm(dirs, dim=-1, keepdim=True) + 1e-9
    # Compute perp: (..., 2)
    perps = torch.stack([-dirs[..., 1], dirs[..., 0]], dim=-1) / norms  # (..., 2)
    # Compute r: (..., 1)
    ks = ks.unsqueeze(-1)  # (..., 1)
    ks_positive = ks >= 0  # (..., 1)
    r = (ks**2 + (norms / 2) ** 2) / (2 * ks - 1e-4 + 2e-4 * ks_positive)  # (..., 1)
    # Compute o: (..., 2)
    o = midpoints + (r - ks) * perps  # (..., 2)
    # Compute theta0 and theta1: (...,)
    sx, sy = (starts - o).unbind(dim=-1)  # (...,) (...,)
    ex, ey = (ends - o).unbind(dim=-1)  # (...,) (...,)
    theta0 = torch.atan2(sy, sx)  # (...,)
    theta1 = torch.atan2(ey, ex)  # (...,)
    # Adjust theta0 and theta1 based on ks
    ks_positive = ks_positive.squeeze(-1)  # (...,)
    mask1 = ks_positive & (theta0 > theta1)
    mask2 = (~ks_positive) & (theta0 < theta1)
    theta1 = torch.where(mask1, theta1 + 2 * torch.pi, theta1)
    theta0 = torch.where(mask2, theta0 + 2 * torch.pi, theta0)
    return o, r.squeeze(-1), theta0, theta1


def vectorized_sample_arc(starts: torch.Tensor, ends: torch.Tensor, ks: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Vectorized sample arcs.
    starts: (..., 2)
    ends: (..., 2)
    ks: (...,)
    t: (num_samples,) or (..., num_samples)
    Returns: (..., num_samples, 2)
    """
    # Compute midpoints: (..., 2)
    midpoints = (starts + ends) / 2
    # Compute dir: (..., 2)
    dirs = ends - starts
    # Compute norm: (..., 1)
    norms = torch.norm(dirs, dim=-1, keepdim=True) + 1e-9
    # Compute perp: (..., 2)
    perps = torch.stack([-dirs[..., 1], dirs[..., 0]], dim=-1) / norms  # (..., 2)
    # Compute r: (..., 1)
    ks = ks.unsqueeze(-1)  # (..., 1)
    ks_positive = ks >= 0  # (..., 1)
    r = (ks**2 + (norms / 2) ** 2) / (2 * ks - 1e-4 + 2e-4 * ks_positive)  # (..., 1)
    # Compute o: (..., 2)
    o = midpoints + (r - ks) * perps  # (..., 2)
    # Compute theta0 and theta1: (...,)
    sx, sy = (starts - o).unbind(dim=-1)  # (...,) (...,)
    ex, ey = (ends - o).unbind(dim=-1)  # (...,) (...,)
    theta0 = torch.atan2(sy, sx)  # (...,)
    theta1 = torch.atan2(ey, ex)  # (...,)
    # Adjust theta0 and theta1 based on ks
    ks_positive = ks_positive.squeeze(-1)  # (...,)
    mask1 = ks_positive & (theta0 > theta1)
    mask2 = (~ks_positive) & (theta0 < theta1)
    theta1 = torch.where(mask1, theta1 + 2 * torch.pi, theta1)
    theta0 = torch.where(mask2, theta0 + 2 * torch.pi, theta0)
    # Interpolate theta:
    theta0 = theta0.unsqueeze(-1)  # (..., 1)
    theta1 = theta1.unsqueeze(-1)  # (..., 1)
    theta = theta0 * (1 - t) + theta1 * t  # (..., num_samples)
    # Compute points: (..., num_samples, 2)
    return o.unsqueeze(-2) + r.abs().unsqueeze(-2) * torch.stack(
        [torch.cos(theta), torch.sin(theta)], dim=-1
    )  # (..., num_samples, 2)


def _get_index(src: torch.Tensor, cache: dict[int, torch.Tensor], idx: int, detach: bool) -> torch.Tensor:
    """
    Returns src[idx]. Ensures that id(_get_index(...)) is the same after subsequent calls.
    """
    res = cache.get(idx, None)
    if res is None:
        res = maybe_detach(src[idx], detach)
        cache[idx] = res
    return res


# endregion
# region ShapeCollection
# ================== ShapeCollection ===========================


@dataclass
class ShapeCollectionArgs:
    ks_scale: float = 1.0


def _always_raise() -> ShapeCollectionArgs:
    raise ValueError("ShapeCollectionArgs must be set")


@dataclass
class ShapeCollection(ObjectCollection[Shape]):
    control_points: torch.Tensor  # (_, 2)
    ks: torch.Tensor  # (_,)
    lines: torch.Tensor  # (n_lines, 2) (start_idx, end_idx) indicies to control_points
    arcs: torch.Tensor  # (n_arcs, 3) (start_idx, end_idx, k_idx) indicies to control_points, ks
    shapes: list[ShapeMeta]  # each shape does not share control_points and ks
    shape_ids: list[int]
    shape_payloads: list[ShapePayload]
    args: ShapeCollectionArgs = field(default_factory=ShapeCollectionArgs)
    rescale_ks: bool = False
    _batched_ids: Optional[torch.Tensor] = field(init=False, default=None)

    def __post_init__(self):
        assert self.control_points.ndim == 2 and self.control_points.shape[-1] == 2
        assert self.ks.ndim == 1
        assert self.lines.ndim == 2 and self.lines.shape[-1] == 2
        assert self.arcs.ndim == 2 and self.arcs.shape[-1] == 3
        assert len(self.shapes) > 0
        if self.rescale_ks:
            self.ks = self.ks / self.args.ks_scale

    def get_batched_ids(self) -> torch.Tensor:
        """
        During point sampling, we concatenate point samples to
            P = (1 + n_lines + n_arcs, n_samples, 2)
            Note: the first point is PAD
        Precompute indexing (to P) for each shape: (n_shapes, max_prims)
        """
        if self._batched_ids is not None:
            return self._batched_ids
        max_prims = max([len(shape.order) for shape in self.shapes])
        idxs: list[torch.Tensor] = []
        n_lines = self.lines.shape[0]
        for shape in self.shapes:
            prim_idxs = torch.cat([shape.line_idx + 1, shape.arcs_idx + n_lines + 1], dim=0)
            cur_idxs = prim_idxs[shape.get_ordered_ids()]
            # padding
            if len(cur_idxs) < max_prims:
                cur_idxs = torch.cat(
                    [cur_idxs, torch.zeros(max_prims - len(cur_idxs), dtype=torch.long, device=cur_idxs.device)], dim=0
                )
            idxs.append(cur_idxs)
        self._batched_ids = torch.stack(idxs, dim=0)  # (n_shapes, max_prims)
        return self._batched_ids

    def __len__(self) -> int:
        return len(self.shapes)

    def parameters(self) -> list[torch.Tensor]:
        return [self.control_points, self.ks]

    def parameter_names(self) -> list[str]:
        return ["control_points", "ks"]

    def per_object_grads(self) -> list[torch.Tensor]:
        batched_ids: list[list[int]] = self.get_batched_ids().tolist()

        assert self.control_points.grad is not None, "grad must be set for control_points"
        assert self.ks.grad is not None, "grad must be set for ks"

        # Lines
        line_params = self.control_points.grad[self.lines].flatten(-2)  # (n_lines, 4)

        # Arcs
        arc_params = torch.cat(
            [
                self.control_points.grad[self.arcs[:, :2]].flatten(-2),  # (n_arcs, 4)
                self.ks.grad[self.arcs[:, 2]].unsqueeze(-1),  # (n_arcs, 1)
            ],
            dim=-1,
        )  # (n_arcs, 5)

        prim_grads: list[torch.Tensor] = [torch.empty(0), *line_params.unbind(0), *arc_params.unbind(0)]

        res: list[torch.Tensor] = []
        for ids_ in batched_ids:
            params_: list[torch.Tensor] = []
            for id_ in ids_:
                if id_ != 0:
                    params_.append(prim_grads[id_])
            res.append(torch.cat(params_, dim=0))
        return res

    def device(self) -> torch.device:
        return self.control_points.device

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        self.control_points.requires_grad_(requires_grad)
        self.ks.requires_grad_(requires_grad)
        return self

    def clone(self) -> "ShapeCollection":
        """
        Detach and clone
        """
        return ShapeCollection(
            control_points=self.control_points.detach().clone(),
            ks=self.ks.detach().clone(),
            lines=self.lines.detach().clone(),
            arcs=self.arcs.detach().clone(),
            shapes=self.shapes.copy(),
            shape_ids=self.shape_ids.copy(),
            shape_payloads=self.shape_payloads.copy(),
            args=self.args,
        )

    def to(self, device: Union[str, torch.device, None] = None) -> "ShapeCollection":
        """
        Detach and clone
        """
        return ShapeCollection(
            control_points=self.control_points.to(device=device),
            ks=self.ks.to(device=device),
            lines=self.lines.to(device=device),
            arcs=self.arcs.to(device=device),
            shapes=[s.to(device=device) for s in self.shapes],
            shape_ids=self.shape_ids.copy(),
            shape_payloads=self.shape_payloads.copy(),
            args=self.args,
        )

    def get_shape(self, idx: int, detach: bool = True) -> Shape:
        shape = self.shapes[idx]
        prims_: Sequence[Primitive] = []

        # cache of params used in this shape
        control_points_: dict[int, torch.Tensor] = {}
        ks_: dict[int, torch.Tensor] = {}

        scaled_ks = self.ks * self.args.ks_scale

        for type_, idx_ in shape.order:
            if type_ == PrimitiveType.Line:
                line_ = self.lines[shape.line_idx[idx_]].detach().cpu().tolist()
                prims_.append(
                    Line(
                        start=_get_index(self.control_points, control_points_, line_[0], detach),
                        end=_get_index(self.control_points, control_points_, line_[1], detach),
                    )
                )
            elif type_ == PrimitiveType.Arc:
                arc_ = self.arcs[shape.arcs_idx[idx_]].detach().cpu().tolist()
                prims_.append(
                    Arc(
                        start=_get_index(self.control_points, control_points_, arc_[0], detach),
                        end=_get_index(self.control_points, control_points_, arc_[1], detach),
                        k=_get_index(scaled_ks, ks_, arc_[2], detach),
                    )
                )
            else:
                raise ValueError(f"Unknown primitive type: {type_}")
        return Shape(prims_, payload=self.shape_payloads[idx], id=self.shape_ids[idx])

    def get_object(self, idx: int, detach: bool = True) -> Shape:
        return self.get_shape(idx, detach)

    @classmethod
    def from_shape(cls, shape: Shape, **kwargs) -> "ShapeCollection":
        device = shape.device()
        control_points_: list[torch.Tensor] = []
        control_points_ids_: dict[int, int] = {}
        ks_: list[torch.Tensor] = []
        ks_ids_: dict[int, int] = {}

        lines_: list[torch.Tensor] = []
        arcs_: list[torch.Tensor] = []

        line_ids_: list[int] = []
        arc_ids_: list[int] = []
        order_: list[tuple[PrimitiveType, int]] = []

        for prim in shape.primitives:
            if isinstance(prim, Line):
                start_id = _insert_ids(control_points_ids_, control_points_, prim.start)
                end_id = _insert_ids(control_points_ids_, control_points_, prim.end)
                line_id_ = len(lines_)
                lines_.append(torch.tensor([start_id, end_id], dtype=torch.long, device=device))
                line_ids_.append(line_id_)
                order_.append((PrimitiveType.Line, line_id_))
            elif isinstance(prim, Arc):
                start_id = _insert_ids(control_points_ids_, control_points_, prim.start)
                end_id = _insert_ids(control_points_ids_, control_points_, prim.end)
                k_id = _insert_ids(ks_ids_, ks_, prim.k)
                arc_id_ = len(arcs_)
                arcs_.append(torch.tensor([start_id, end_id, k_id], dtype=torch.long, device=device))
                arc_ids_.append(arc_id_)
                order_.append((PrimitiveType.Arc, arc_id_))
            else:
                raise NotImplementedError()

        control_points = safe_stack(control_points_, (2,), device)
        ks = safe_stack(ks_, (), device)
        lines = safe_stack(lines_, (2,), device, dtype=torch.long)
        arcs = safe_stack(arcs_, (3,), device, dtype=torch.long)

        return cls(
            control_points=control_points,
            ks=ks,
            rescale_ks=True,
            lines=lines,
            arcs=arcs,
            shapes=[
                ShapeMeta(
                    line_idx=torch.tensor(line_ids_, dtype=torch.long, device=device),
                    arcs_idx=torch.tensor(arc_ids_, dtype=torch.long, device=device),
                    order=order_,
                )
            ],
            shape_ids=[shape.id],
            shape_payloads=[shape.payload],
            **kwargs,
        )

    @classmethod
    def from_object(cls, object: Shape, **kwargs) -> "ShapeCollection":
        return cls.from_shape(object, **kwargs)

    @classmethod
    def from_shapes(cls, shapes: list[Shape]) -> "ShapeCollection":
        return cls.cat([cls.from_shape(shape) for shape in shapes])

    @classmethod
    def cat(cls, collections: list["ShapeCollection"], **kwargs) -> "ShapeCollection":
        assert len(collections) > 0, "collections must not be empty"
        device = collections[0].device()

        control_points_cnt = 0
        ks_cnt = 0
        lines_cnt = 0
        arcs_cnt = 0
        new_control_points: list[torch.Tensor] = []
        new_ks: list[torch.Tensor] = []
        new_lines: list[torch.Tensor] = []
        new_arcs: list[torch.Tensor] = []
        new_shapes: list[ShapeMeta] = []
        new_shape_ids: list[int] = []
        new_shape_payloads: list[ShapePayload] = []
        for collection in collections:
            new_control_points.append(collection.control_points)
            new_ks.append(collection.ks)
            new_lines.append(collection.lines + control_points_cnt)
            new_arcs.append(
                collection.arcs
                + torch.tensor([control_points_cnt, control_points_cnt, ks_cnt], dtype=torch.long, device=device)
            )

            for shape in collection.shapes:
                new_shapes.append(
                    ShapeMeta(
                        line_idx=shape.line_idx + lines_cnt,
                        arcs_idx=shape.arcs_idx + arcs_cnt,
                        order=list(shape.order),
                    )
                )
            new_shape_ids.extend(collection.shape_ids)
            new_shape_payloads.extend(collection.shape_payloads)

            control_points_cnt += collection.control_points.shape[0]
            ks_cnt += collection.ks.shape[0]
            lines_cnt += collection.lines.shape[0]
            arcs_cnt += collection.arcs.shape[0]

        return cls(
            control_points=safe_cat(new_control_points, (2,), device),
            ks=safe_cat(new_ks, (), device),
            lines=safe_cat(new_lines, (2,), device, dtype=torch.long),
            arcs=safe_cat(new_arcs, (3,), device, dtype=torch.long),
            shapes=new_shapes,
            shape_ids=new_shape_ids,
            shape_payloads=new_shape_payloads,
            **kwargs,
        )

    def sample_points(
        self,
        num_samples: int,
        order: bool = False,
        shape_ids: Optional[list[int]] = None,
        centered: bool = True,
    ) -> list[torch.Tensor]:
        """
        returns a list of (num_samples * num_prims, 2) for each shape.
        """
        device = self.device()
        t = (
            torch.arange(num_samples, device=device, dtype=torch.get_default_dtype()) + (0.5 if centered else 0)
        ) / num_samples

        # Lines
        start_points = self.control_points[self.lines[:, 0]]
        end_points = self.control_points[self.lines[:, 1]]
        line_sampled_points = vectorized_sample_line(start_points, end_points, t)  # (num_lines, num_samples, 2)

        # Arcs
        start_points = self.control_points[self.arcs[:, 0]]
        end_points = self.control_points[self.arcs[:, 1]]
        ks = self.ks[self.arcs[:, 2]] * self.args.ks_scale
        arc_sampled_points = vectorized_sample_arc(start_points, end_points, ks, t)  # (num_arcs, num_samples, 2)

        res: list[torch.Tensor] = []
        if shape_ids is None:
            shape_ids = list(range(len(self.shapes)))
        for i_ in shape_ids:
            shape = self.shapes[i_]
            lines__ = line_sampled_points[shape.line_idx]  # (shape.n_lines, num_samples, 2)
            arcs__ = arc_sampled_points[shape.arcs_idx]  # (shape.n_arcs, num_samples, 2)
            if not order:
                res.append(torch.cat([lines__.flatten(0, -2), arcs__.flatten(0, -2)], dim=0))
            else:
                prims__ = torch.cat([lines__, arcs__], dim=0)
                ids__ = shape.get_ordered_ids()
                res.append(prims__[ids__].flatten(0, -2))
        return res

    def sample_points_fast(self, num_samples: int, centered: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample `num_samples` points for each primitive.
        - centered: if True, shift points by half the period

        Returns:
        - point samples: (num_shapes, max_prims * num_samples, 2)
        - num samples: (num_shapes,) long; num samples for each shape
        """
        device = self.device()
        t = (
            torch.arange(num_samples, device=device, dtype=torch.get_default_dtype()) + (0.5 if centered else 0)
        ) / num_samples

        # Lines
        start_points = self.control_points[self.lines[:, 0]]
        end_points = self.control_points[self.lines[:, 1]]
        line_sampled_points = vectorized_sample_line(start_points, end_points, t)  # (num_lines, num_samples, 2)

        # Arcs
        start_points = self.control_points[self.arcs[:, 0]]
        end_points = self.control_points[self.arcs[:, 1]]
        ks = self.ks[self.arcs[:, 2]] * self.args.ks_scale
        arc_sampled_points = vectorized_sample_arc(start_points, end_points, ks, t)  # (num_arcs, num_samples, 2)

        all_sampled_points = torch.cat(
            [
                torch.zeros((1, num_samples, 2), device=device, dtype=line_sampled_points.dtype),
                line_sampled_points,
                arc_sampled_points,
            ],
            dim=0,
        )  # (1 + num_lines + num_arcs, num_samples, 2)
        batched_ids = self.get_batched_ids()  # (n_shapes, max_prims)

        batched_sampled_points = all_sampled_points[batched_ids].flatten(
            -3, -2
        )  # (n_shapes, max_prims * num_samples, 2)
        batched_n_samples = (batched_ids != 0).sum(dim=-1) * num_samples  # (n_shapes,)
        return batched_sampled_points, batched_n_samples

    def sample_points_length_fast(
        self, num_samples: int, centered: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample `num_samples` points for each primitive.
        - centered: if True, shift points by half the period

        Returns:
        - point samples: (num_shapes, max_prims * num_samples, 2)
        - num samples: (num_shapes,) long; num samples for each shape (= num_prims * num_samples)
        - prim lengths: (num_shapes, max_prims) primitive lengths
        """
        device = self.device()
        t = (
            torch.arange(num_samples, device=device, dtype=torch.get_default_dtype()) + (0.5 if centered else 0)
        ) / num_samples

        # Lines
        line_start_points = self.control_points[self.lines[:, 0]]
        line_end_points = self.control_points[self.lines[:, 1]]
        line_sampled_points = vectorized_sample_line(
            line_start_points, line_end_points, t
        )  # (num_lines, num_samples, 2)
        line_lengths = vectorized_line_length(line_start_points, line_end_points).detach()  # (num_lines,)

        # Arcs
        arc_start_points = self.control_points[self.arcs[:, 0]]
        arc_end_points = self.control_points[self.arcs[:, 1]]
        ks = self.ks[self.arcs[:, 2]] * self.args.ks_scale
        arc_sampled_points, arc_lengths = vectorized_arc_sample_length(
            arc_start_points, arc_end_points, ks, t
        )  # (num_arcs, num_samples, 2), (num_arcs,)
        arc_lengths = arc_lengths.detach()  # (num_arcs,)

        all_sampled_points = torch.cat(
            [
                torch.zeros((1, num_samples, 2), device=device, dtype=line_sampled_points.dtype),
                line_sampled_points,
                arc_sampled_points,
            ],
            dim=0,
        )  # (1 + num_lines + num_arcs, num_samples, 2)
        all_lengths = torch.cat(
            [torch.zeros((1,), device=device, dtype=line_lengths.dtype), line_lengths, arc_lengths], dim=0
        )  # (1 + num_lines + num_arcs,)
        batched_ids = self.get_batched_ids()  # (n_shapes, max_prims)

        batched_sampled_points = all_sampled_points[batched_ids].flatten(
            -3, -2
        )  # (n_shapes, max_prims * num_samples, 2)
        batched_n_samples = (batched_ids != 0).sum(dim=-1) * num_samples  # (n_shapes,)
        batched_lengths = all_lengths[batched_ids]  # (n_shapes, max_prims)

        return batched_sampled_points, batched_n_samples, batched_lengths

    def sample_uniform_points(self, num_out_samples: int, num_samples: int) -> torch.Tensor:
        """
        Samples `num_out_samples` points from each shape that are distributed uniformly along the curve.
        Assumes each shape contains one closed contour and its primitives are ordered accordingly.

        Implementation:
        - Sample `num_samples` points for each primitive
        - Find points that are uniformly distributed along the curve, linearly interpolate between two points

        Returns:
        - point samples: (num_shapes, num_out_samples, 2)
        """
        device = self.device()
        t = (torch.arange(0, num_samples + 1, device=device, dtype=torch.get_default_dtype())) / num_samples

        # Lines
        line_start_points = self.control_points[self.lines[:, 0]]
        line_end_points = self.control_points[self.lines[:, 1]]
        line_sampled_points = vectorized_sample_line(
            line_start_points, line_end_points, t
        )  # (num_lines, num_samples + 1, 2)
        line_lengths = vectorized_line_length(line_start_points, line_end_points).detach()  # (num_lines,)

        # Arcs
        arc_start_points = self.control_points[self.arcs[:, 0]]
        arc_end_points = self.control_points[self.arcs[:, 1]]
        ks = self.ks[self.arcs[:, 2]] * self.args.ks_scale
        arc_sampled_points, arc_lengths = vectorized_arc_sample_length(
            arc_start_points, arc_end_points, ks, t
        )  # (num_arcs, num_samples + 1, 2), (num_arcs,)
        arc_lengths = arc_lengths.detach()  # (num_arcs,)

        all_sampled_points = torch.cat(
            [
                torch.zeros((1, num_samples + 1, 2), device=device, dtype=line_sampled_points.dtype),
                line_sampled_points,
                arc_sampled_points,
            ],
            dim=0,
        )  # (1 + num_lines + num_arcs, num_samples + 1, 2)
        all_lengths = torch.cat(
            [torch.zeros((1,), device=device, dtype=line_lengths.dtype), line_lengths, arc_lengths], dim=0
        )  # (1 + num_lines + num_arcs,)

        batched_ids = self.get_batched_ids()  # (n_shapes, max_prims)

        batched_sampled_points = all_sampled_points[batched_ids].flatten(
            -3, -2
        )  # (n_shapes, max_prims * (num_samples + 1), 2)
        # assumes point samples are uniform along the primitive
        batched_lengths = (
            torch.cat(
                [
                    torch.zeros(*batched_ids.shape, 1, device=device),
                    (all_lengths[batched_ids].unsqueeze(-1) / num_samples).expand(-1, -1, num_samples),
                ],
                dim=-1,
            )
            .flatten(-2)
            .cumsum(dim=-1)
        )  # (n_shapes, max_prims * (num_samples + 1))
        batched_lengths = batched_lengths / batched_lengths[:, -1:]  # (n_shapes, max_prims * (num_samples + 1))

        out_t = torch.arange(num_out_samples, device=device) / num_out_samples  # (num_out_samples,)

        return torch_interp(out_t, batched_lengths, batched_sampled_points, dim=-2)  # (n_shapes, num_out_samples, 2)

    def render_with_wn(
        self,
        size: int,
        lim: tuple[float, float] = (-1.5, 1.5),
        center_pixel: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render as a signed distance field.
        returns:
        - img: (num_shapes, size, size)
        - wn: (num_shapes, size, size)
        """
        device = self.device()
        lim0, lim1 = lim

        if center_pixel:
            basis = (torch.arange(size, device=device) + 0.5) / size * (lim1 - lim0) + lim0  # (size,)
        else:
            basis = torch.linspace(lim0, lim1, size, device=device)  # (size,)
        xs = basis.expand(size, -1)  # (size, size)
        ys = basis.unsqueeze(-1).expand(-1, size)  # (size, size)
        grid = torch.stack([xs, ys], dim=-1)  # (size, size, 2)
        return self.rasterize(grid)

    def render(
        self,
        size: int,
        lim: tuple[float, float] = (-1.5, 1.5),
        center_pixel: bool = True,
    ) -> torch.Tensor:
        """
        Render as a signed distance field.
        returns:
        - img: (num_shapes, size, size)
        - wn: (num_shapes, size, size)
        """
        return self.render_with_wn(size, lim, center_pixel)[0]

    def rasterize(
        self,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render as a signed distance field.
        - positions: (..., 2)
        returns:
        - img: (num_shapes, ...)
        - wn: (num_shapes, ...)
        """
        device = self.device()
        *shape, _ = positions.shape
        assert _ == 2, f"positions must be (..., 2), got {positions.shape}"

        # Lines
        p1 = self.control_points[self.lines[:, 0]]  # (n_lines, 2)
        p2 = self.control_points[self.lines[:, 1]]  # (n_lines, 2)
        d = p2 - p1  # (n_lines, 2)
        norm2 = d.pow(2).sum(dim=-1) + 1e-10  # (n_lines,)
        u = ((positions.unsqueeze(-2) - p1) * d).sum(dim=-1) / norm2  # (..., n_lines)
        u = u.clamp(0, 1)  # (..., n_lines)
        p = p1 + u.unsqueeze(-1).detach() * d  # (..., n_lines, 2) closest point on a line to grid
        line_dist = (p - positions.unsqueeze(-2)).norm(dim=-1)  # (..., n_lines)
        gp1x, gp1y = (p1 - positions.unsqueeze(-2)).unbind(-1)  # (..., n_lines)
        gp2x, gp2y = (p2 - positions.unsqueeze(-2)).unbind(-1)  # (..., n_lines)
        theta1 = torch.atan2(gp1y, gp1x)  # (..., n_lines)
        theta2 = torch.atan2(gp2y, gp2x)  # (..., n_lines)
        line_wn = (theta2 - theta1 + torch.pi) % (2 * torch.pi) - torch.pi  # (..., n_lines)

        # Arcs
        p1 = self.control_points[self.arcs[:, 0]]  # (n_arcs, 2)
        p2 = self.control_points[self.arcs[:, 1]]  # (n_arcs, 2)
        ks = self.ks[self.arcs[:, 2]] * self.args.ks_scale  # (n_arcs,)
        # Compute midpoints: (n_arcs, 2)
        midpoints = (p1 + p2) / 2
        # Compute dir: (n_arcs, 2)
        dirs = p2 - p1
        # Compute norm: (n_arcs, 1)
        norms = torch.norm(dirs, dim=-1, keepdim=True) + 1e-9
        # Compute perp: (n_arcs, 2)
        perps = torch.stack([-dirs[..., 1], dirs[..., 0]], dim=-1) / norms  # (n_arcs, 2)
        # Compute r: (n_arcs, 1)
        ks = ks.unsqueeze(-1)  # (n_arcs, 1)
        ks_positive = ks >= 0  # (n_arcs, 1)
        r = (ks**2 + (norms / 2) ** 2) / (2 * ks - 1e-4 + 2e-4 * ks_positive)  # (n_arcs, 1)
        # Compute o: (n_arcs, 2)
        o = midpoints + (r - ks) * perps  # (n_arcs, 2)

        gp1x, gp1y = (p1 - positions.unsqueeze(-2)).unbind(-1)  # (..., n_arcs)
        gp2x, gp2y = (p2 - positions.unsqueeze(-2)).unbind(-1)  # (..., n_arcs)
        theta0 = torch.atan2(gp1y, gp1x)  # (..., n_arcs)
        theta1 = torch.atan2(gp2y, gp2x)  # (..., n_arcs)
        wn = (theta1 - theta0 + torch.pi) % (2 * torch.pi) - torch.pi  # (..., n_arcs)

        out_chord = (((positions.unsqueeze(-2) - p1) * perps).sum(dim=-1) > 0) ^ ~ks_positive.squeeze(
            -1
        )  # (..., n_arcs,)

        gx, gy = (positions.unsqueeze(-2) - o).unbind(dim=-1)  # (..., n_arcs,)
        out_circle = torch.stack([gx, gy], dim=-1).norm(dim=-1) > r.abs().squeeze(-1)  # (..., n_arcs,)

        # Adjust theta0 and theta1 based on ks
        ks_positive = ks_positive.squeeze(-1)  # (...,)
        mask1 = ks_positive & (theta0 > theta1)
        mask2 = (~ks_positive) & (theta0 < theta1)
        theta1 = torch.where(mask1, theta1 + 2 * torch.pi, theta1)
        theta0 = torch.where(mask2, theta0 + 2 * torch.pi, theta0)

        arc_wn = torch.where(~out_chord & ~out_circle, theta1 - theta0, wn)  # (..., n_arcs,)

        # Compute theta0 and theta1: (n_arcs,)
        sx, sy = (p1 - o).unbind(dim=-1)  # (n_arcs,) (n_arcs,)
        ex, ey = (p2 - o).unbind(dim=-1)  # (n_arcs,) (n_arcs,)
        theta0 = torch.atan2(sy, sx)  # (n_arcs,)
        theta1 = torch.atan2(ey, ex)  # (n_arcs,)
        # if not ks_positive: theta0, theta1 = theta1, theta0
        ks_positive = ks_positive.squeeze(-1)  # (n_arcs,)
        theta0_ = torch.where(ks_positive, theta0, theta1)  # (n_arcs,)
        theta1_ = torch.where(ks_positive, theta1, theta0)  # (n_arcs,)
        theta0, theta1 = theta0_, theta1_
        # Adjust so that theta0 < theta1
        theta1 = torch.where(theta1 < theta0, theta1 + 2 * torch.pi, theta1)  # (n_arcs,)

        with torch.no_grad():
            thetag = torch.atan2(gy, gx)  # (..., n_arcs,)

            # midpoint between theta0 and theta1 (outside)
            thetac = (theta0 + theta1) / 2 - torch.pi  # (n_arcs,)

            # Adjust so that thetac < thetag and clamp to [theta0, theta1]
            thetag = thetac + (thetag - thetac).remainder(2 * torch.pi)  # (..., n_arcs,)
            # thetag = thetag.clamp(theta0, theta1)  # (..., n_arcs)
            u = (thetag - theta0) / (theta1 - theta0)  # (..., n_arcs,)
            u = u.clamp(0, 1)  # (..., n_arcs,)
        # Reparameterize
        thetag = theta0 + u.detach() * (theta1 - theta0)  # (..., n_arcs,)

        # Closest point on arc to grid
        p = o + r.abs() * torch.stack([torch.cos(thetag), torch.sin(thetag)], dim=-1)  # (..., n_arcs, 2)
        arc_dist = (p - positions.unsqueeze(-2)).norm(dim=-1)  # (..., n_arcs)

        all_dist = torch.cat(
            [torch.tensor(torch.inf, device=device).expand(*shape, 1), line_dist, arc_dist], dim=-1
        )  # (..., 1 + n_lines + n_arcs)
        all_wn = torch.cat(
            [torch.tensor(0.0, device=device).expand(*shape, 1), line_wn, arc_wn], dim=-1
        )  # (..., 1 + n_lines + n_arcs)

        batched_ids = self.get_batched_ids()  # (n_shapes, max_n_prims)
        b_all_dist = all_dist.permute(-1, *range(len(shape)))[batched_ids]  # (n_shapes, max_n_prims, ...)
        b_all_wn = all_wn.permute(-1, *range(len(shape)))[batched_ids]  # (n_shapes, max_n_prims, ...)

        b_dist = b_all_dist.amin(dim=1)  # (n_shapes, ...)
        b_wn = b_all_wn.sum(dim=1)  # (n_shapes, ...)
        b_sgn = (b_wn.abs() > torch.pi) * -2 + 1  # (n_shapes, ...) sign of wn
        return b_dist * b_sgn, b_wn

    def render01(
        self, size: int, lim: tuple[float, float] = (-1.5, 1.5), center_pixel: bool = True, blur: float = 1.0
    ) -> torch.Tensor:
        vlim = blur * (lim[1] - lim[0]) / size
        imgs = self.render(size, lim, center_pixel=center_pixel)  # (n_shapes, size, size)
        imgs = (-imgs.clamp(-vlim, vlim) + vlim) / (2 * vlim)  # (n_shapes, size, size)
        return imgs

    def compute_area(self) -> torch.Tensor:
        """
        Compute the area of each shape. Assumes shapes only contain closed loops.

        returns (n_shapes,)
        """
        # Lines
        p1 = self.control_points[self.lines[:, 0]]  # (n_lines, 2)
        p2 = self.control_points[self.lines[:, 1]]  # (n_lines, 2)

        p1x, p1y = p1.unbind(dim=-1)  # (n_lines,) (n_lines,)
        p2x, p2y = p2.unbind(dim=-1)  # (n_lines,) (n_lines,)
        vol_lines = (p1x * p2y - p2x * p1y) / 2  # (n_lines,)

        # Arcs
        p1 = self.control_points[self.arcs[:, 0]]  # (n_arcs, 2)
        p2 = self.control_points[self.arcs[:, 1]]  # (n_arcs, 2)
        ks = self.ks[self.arcs[:, 2]] * self.args.ks_scale  # (n_arcs,)

        # endpoint component
        p1x, p1y = p1.unbind(dim=-1)  # (n_arcs,) (n_arcs,)
        p2x, p2y = p2.unbind(dim=-1)  # (n_arcs,) (n_arcs,)
        vol_arcs_ = (p1x * p2y - p2x * p1y) / 2  # (n_arcs,)

        # curvature component
        _, r, theta0, theta1 = vectorized_o_r_thetas(p1, p2, ks)  # (n_arcs, 2), (n_arcs,), (n_arcs,), (n_arcs,)
        dt = theta1 - theta0  # (n_arcs,)
        vol_arcs = vol_arcs_ + (dt - dt.sin()) * r.square() / 2  # (n_arcs,)

        all_vols = torch.cat(
            [torch.tensor([0.0], device=self.device()), vol_lines, vol_arcs], dim=-1
        )  # (1 + n_lines + n_arcs)

        batched_ids = self.get_batched_ids()  # (n_shapes, max_n_prims)

        return all_vols[batched_ids].sum(dim=-1)  # (n_shapes,)

    def deduplicate(self, eps: float) -> "ShapeCollection":
        """
        Removes duplicate shapes. Does not detach from self. Does not prune unused parameters.
        """
        # Lines
        start_points = self.control_points[self.lines[:, 0]]  # (num_lines, 2)
        end_points = self.control_points[self.lines[:, 1]]  # (num_lines, 2)
        line_params = torch.cat([start_points, end_points], dim=-1)  # (num_lines, 4)

        # Arcs
        start_points = self.control_points[self.arcs[:, 0]]
        end_points = self.control_points[self.arcs[:, 1]]
        ks = self.ks[self.arcs[:, 2]] * self.args.ks_scale
        arc_params = torch.cat([start_points, end_points, ks.unsqueeze(-1)], dim=-1)  # (num_arcs, 5)

        og_ids: list[int] = []
        new_shapes: list[ShapeMeta] = []
        new_shape_ids: list[int] = []
        new_shape_payloads: list[ShapePayload] = []

        for i, shape in enumerate(self.shapes):
            dup_id = -1
            for j in range(len(og_ids)):
                if self._is_same_shape(i, og_ids[j], line_params, arc_params, eps):
                    dup_id = j
                    break
            if dup_id == -1:
                new_shapes.append(shape)
                new_shape_ids.append(self.shape_ids[i])
                new_shape_payloads.append(self.shape_payloads[i])
                og_ids.append(i)
            else:
                new_shape_payloads[dup_id].duplicates.append(self.shape_ids[i])
        return ShapeCollection(
            control_points=self.control_points,
            ks=self.ks,
            lines=self.lines,
            arcs=self.arcs,
            shapes=new_shapes,
            shape_ids=new_shape_ids,
            shape_payloads=new_shape_payloads,
        )

    def _is_same_shape(
        self, idx1: int, idx2: int, line_params: torch.Tensor, arc_params: torch.Tensor, eps: float
    ) -> bool:
        """
        line_params: (n_lines, 4) start_points, end_points
        arc_params: (n_arcs, 5) start_points, end_points, ks
        Returns True if shapes at idx1 and idx2 are the same shape
        """
        shape1 = self.shapes[idx1]
        shape2 = self.shapes[idx2]
        if len(shape1.line_idx) != len(shape2.line_idx) or len(shape1.arcs_idx) != len(shape2.arcs_idx):
            return False
        for ids1_, ids2_, params_ in [
            (shape1.line_idx, shape2.line_idx, line_params),
            (shape1.arcs_idx, shape2.arcs_idx, arc_params),
        ]:
            n_prim_ = len(ids1_)
            if n_prim_ == 0:
                continue
            params1_ = params_[ids1_]  # (n_prim_, *)
            params2_ = params_[ids2_]  # (n_prim_, *)
            losses = (params1_.unsqueeze(-2) - params2_).abs().max(dim=-1).values  # (n_prim_, n_prim_)
            min1, amin1 = losses.min(dim=-1)  # i -> argmin_j (losses[i, j])
            min2, amin2 = losses.min(dim=0)  # j -> argmin_i (losses[i, j])
            if min1.max() > eps or min2.max() > eps:
                return False

            # the minimum in both directions must be the same
            ids = torch.arange(n_prim_, device=losses.device)
            if (amin2[amin1[ids]] != ids).any():
                return False
        return True

    @classmethod
    def patch_args(cls, args: ShapeCollectionArgs) -> Type["ShapeCollection"]:
        return type(
            "ShapeCollectionWithArgs",
            (ShapeCollection,),
            {"__init__": partialmethod(ShapeCollection.__init__, args=args)},
        )

    def to_savable(self) -> "ShapeCollection":
        return ShapeCollection(
            control_points=self.control_points,
            ks=self.ks,
            lines=self.lines,
            arcs=self.arcs,
            shapes=self.shapes,
            shape_ids=self.shape_ids,
            shape_payloads=self.shape_payloads,
            args=self.args,
        )


# endregion
