import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Self, Union, Type, Literal, Sequence, cast
from matplotlib.patches import Rectangle
import math
from enum import Enum
import random
import itertools
from functools import partialmethod

from ..util import maybe_detach
from ..context import Context
from ..visualizer import LineStyle, MPLVisualizerAxes
from ..object_collection import ObjectCollection


@dataclass
class URRewriteArgs:
    split_h_weight: float = 1.0
    split_v_weight: float = 1.0
    add_rect_weight: float = 1.0
    remove_rect_weight: float = 1.0
    merge_weight: float = 1.0
    add_hole_weight: float = 1.0
    rect_scale: float = 0.03
    hole_scale: float = 0.05
    merge_threshold: float = 0.01  # area
    remove_threshold: float = 0.005  # area
    enable_is_sub: bool = False


@dataclass
class URColectionArgs:
    theta_scale: float = torch.pi
    offset_scale: float = 1.0


CleanStrategy = Literal["none", "smart", "smarter"]


class URRewriteType(Enum):
    SplitH = 0  # (id, t)
    SplitV = 1  # (id, t)
    AddHole = 2  # (x, y)
    AddRect = 3  # (x, y)
    RemoveRect = 4  # (id)
    Merge = 5  # (id1, id2)


@dataclass
class URRewrite:
    rewrite_type: URRewriteType


@dataclass
class URRewriteSplitH(URRewrite):
    rewrite_type: URRewriteType = field(default=URRewriteType.SplitH, init=False)
    id_: int
    t: float


@dataclass
class URRewriteSplitV(URRewrite):
    rewrite_type: URRewriteType = field(default=URRewriteType.SplitV, init=False)
    id_: int
    t: float


@dataclass
class URRewriteAddRect(URRewrite):
    rewrite_type: URRewriteType = field(default=URRewriteType.AddRect, init=False)
    x: float
    y: float
    s: float
    r: float
    is_sub: bool


@dataclass
class URRewriteAddHole(URRewrite):
    rewrite_type: URRewriteType = field(default=URRewriteType.AddHole, init=False)
    x: float
    y: float
    s: float
    is_sub: bool


@dataclass
class URRewriteRemoveRect(URRewrite):
    rewrite_type: URRewriteType = field(default=URRewriteType.RemoveRect, init=False)
    id_: int


@dataclass
class URRewriteMerge(URRewrite):
    rewrite_type: URRewriteType = field(default=URRewriteType.Merge, init=False)
    id1: int
    id2: int
    center: torch.Tensor  # (2,)
    size: torch.Tensor  # (2,)
    rot: torch.Tensor  # (,)


@dataclass
class URPayload:
    pass


@dataclass
class UR:
    xs: torch.Tensor  # (n_shapes, 2)
    sizes: torch.Tensor  # (n_shapes, 2)
    rots: torch.Tensor  # (n_shapes,)
    is_subs: torch.Tensor  # (n_shapes,)
    id: int = field(default_factory=lambda: Context.get().gen_id())
    payload: URPayload = field(default_factory=URPayload)

    def __post_init__(self):
        assert self.xs.ndim == 2, f"xs must be a 2D tensor, got {self.xs.ndim}"
        assert self.sizes.ndim == 2, f"sizes must be a 2D tensor, got {self.sizes.ndim}"
        assert self.rots.ndim == 1, f"rots must be a 1D tensor, got {self.rots.ndim}"
        assert self.is_subs.ndim == 1, f"is_subs must be a 1D tensor, got {self.is_subs.ndim}"
        assert (
            len(self.xs) == len(self.sizes) == len(self.rots) == len(self.is_subs)
        ), f"len(xs) != len(sizes) != len(rots) != len(is_subs). Got {len(self.xs)} != {len(self.sizes)} != {len(self.rots)} != {len(self.is_subs)}"

    def visualize(
        self,
        ax: MPLVisualizerAxes,
        ur_args: URColectionArgs,
        line_style: LineStyle = LineStyle(),
        sub_line_style: LineStyle = LineStyle(color="red"),
    ) -> None:
        for i, ((cx, cy), (sx, sy), rot, is_sub) in enumerate(
            zip(self.xs.tolist(), self.sizes.tolist(), self.rots.tolist(), self.is_subs.tolist())
        ):
            line_style_ = sub_line_style if is_sub else line_style
            ax.ax.add_patch(
                Rectangle(
                    (cx - sx, cy - sy),
                    sx * 2,
                    sy * 2,
                    angle=rot * 180 / math.pi * ur_args.theta_scale,
                    rotation_point="center",
                    color=line_style_.color,
                    linewidth=line_style_.linewidth,
                    fill=False,
                )
            )
            ax.ax.text(cx, cy, str(i), color="black", fontsize=12, ha="center", va="center")

    def cleanup(
        self,
        len_eps: float,
        area_eps: float,
        clean_strategy: CleanStrategy,
        split_len: float,
        merge_area_threshold: float,
        lim: tuple[float, float],
        size: int,
        ur_args: URColectionArgs,
    ) -> "UR":
        xs = self.xs
        sizes = self.sizes
        rots = self.rots
        is_subs = self.is_subs

        with torch.no_grad():
            s = self.sizes.abs()
            mask = (s.min(dim=-1).values > len_eps) & (s.prod(dim=-1) > area_eps)
        if not mask.any():
            # don't do anything
            return UR(
                xs=xs,
                sizes=sizes,
                rots=rots,
                is_subs=is_subs,
            )
        else:
            xs = xs[mask]
            sizes = sizes[mask]
            rots = rots[mask]
            is_subs = is_subs[mask]

        if clean_strategy == "none":
            return UR(
                xs=xs,
                sizes=sizes,
                rots=rots,
                is_subs=is_subs,
            )
        elif clean_strategy == "smart":
            # split so that each rectable side is at most split_len
            new_xs_: list[torch.Tensor] = []
            new_sizes_: list[torch.Tensor] = []
            new_rots_: list[torch.Tensor] = []
            new_is_subs_: list[torch.Tensor] = []
            for xs_, sizes_, rots_, is_subs_ in zip(xs, sizes, rots, is_subs):
                nxs = math.ceil(sizes_[0].abs().item() / split_len)
                nys = math.ceil(sizes_[1].abs().item() / split_len)
                new_xs__, new_sizes__, new_rots__ = _split_rect(
                    xs_, sizes_, rots_, [i / nxs for i in range(1, nxs)], [i / nys for i in range(1, nys)], ur_args
                )
                new_xs_.append(new_xs__)
                new_sizes_.append(new_sizes__)
                new_rots_.append(new_rots__)
                new_is_subs_.append(is_subs_.expand(len(new_xs__)))
            new_xs = torch.cat(new_xs_, dim=0)
            new_sizes = torch.cat(new_sizes_, dim=0)
            new_rots = torch.cat(new_rots_, dim=0)
            new_is_subs = torch.cat(new_is_subs_, dim=0)

            # merge back
            while True:
                ijs, m_xs, m_sizes, m_rots, m_area_gain = _gen_merge_candidates(
                    new_xs, new_sizes, new_rots, new_is_subs, merge_area_threshold, ur_args
                )

                used = set()
                new_xs_: list[torch.Tensor] = []
                new_sizes_: list[torch.Tensor] = []
                new_rots_: list[torch.Tensor] = []
                new_is_subs_: list[torch.Tensor] = []
                candidates = list(zip(ijs, m_xs, m_sizes, m_rots, m_area_gain.tolist()))
                candidates.sort(key=lambda x: x[4])
                for (i_, j_), m_x, m_size, m_rot, m_area_gain_ in zip(ijs, m_xs, m_sizes, m_rots, m_area_gain):
                    if i_ in used or j_ in used:
                        continue
                    if m_area_gain_ >= merge_area_threshold:
                        break
                    used.add(i_)
                    used.add(j_)
                    new_xs_.append(m_x)
                    new_sizes_.append(m_size)
                    new_rots_.append(m_rot)
                    new_is_subs_.append(new_is_subs[i_])
                for i_ in range(len(new_xs)):
                    if i_ in used:
                        continue
                    new_xs_.append(new_xs[i_])
                    new_sizes_.append(new_sizes[i_])
                    new_rots_.append(new_rots[i_])
                    new_is_subs_.append(new_is_subs[i_])

                new_xs = torch.stack(new_xs_, dim=0)
                new_sizes = torch.stack(new_sizes_, dim=0)
                new_rots = torch.stack(new_rots_, dim=0)
                new_is_subs = torch.stack(new_is_subs_, dim=0)
                if len(used) == 0:
                    break

            return UR(
                xs=new_xs,
                sizes=new_sizes,
                rots=new_rots,
                is_subs=new_is_subs,
            )
        elif clean_strategy == "smarter":
            # don't support is_subs for now
            assert not is_subs.any(), "is_subs must be all False"
            device = self.xs.device

            # # sort by size
            ids_ = sizes.abs().prod(dim=-1).argsort(descending=True)
            xs = xs[ids_]
            sizes = sizes[ids_]
            rots = rots[ids_]
            is_subs = is_subs[ids_]

            # split so that each rectable side is at most split_len
            new_xs_: list[torch.Tensor] = []
            new_sizes_: list[torch.Tensor] = []
            new_rots_: list[torch.Tensor] = []
            nxys: list[tuple[int, int]] = []
            for xs_, sizes_, rots_, is_subs_ in zip(xs, sizes, rots, is_subs):
                nxs = math.ceil(sizes_[0].abs().item() / split_len)
                nys = math.ceil(sizes_[1].abs().item() / split_len)
                new_xs__, new_sizes__, new_rots__ = _split_rect(
                    xs_, sizes_, rots_, [i / nxs for i in range(1, nxs)], [i / nys for i in range(1, nys)], ur_args
                )
                new_xs_.append(new_xs__)
                new_sizes_.append(new_sizes__)
                new_rots_.append(new_rots__)
                nxys.append((nxs, nys))
            new_xs = torch.cat(new_xs_, dim=0)  # (n_subrects, 2)
            new_sizes = torch.cat(new_sizes_, dim=0)  # (n_subrects, 2)
            new_rots = torch.cat(new_rots_, dim=0)  # (n_subrects,)

            # rasterize
            lim0, lim1 = lim
            basis = (torch.arange(size, device=device) + 0.5) / size * (lim1 - lim0) + lim0  # (size,)
            xs__ = basis.expand(size, -1)  # (size, size)
            ys__ = basis.unsqueeze(-1).expand(-1, size)  # (size, size)
            positions = torch.stack([xs__, ys__], dim=-1).flatten(0, 1)  # (size * size, 2)

            cos_ = torch.cos(new_rots * ur_args.theta_scale)  # (n_subrects,)
            sin_ = torch.sin(new_rots * ur_args.theta_scale)  # (n_subrects,)
            irot = torch.stack([cos_, sin_, -sin_, cos_], dim=-1).reshape(-1, 2, 2)  # (total_nodes, 2, 2)
            diff = (irot @ (positions.unsqueeze(-2) - new_xs).unsqueeze(-1)).squeeze(
                -1
            )  # (size * size, total_nodes, 2)
            d = diff.abs() - new_sizes.abs()  # (size * size, n_subrects, 2)
            sdf = d.clamp(min=0).norm(dim=-1) + d.max(dim=-1).values.clamp(max=0)  # (size * size, n_subrects)
            sdf = (sdf.permute(1, 0) <= 0).long()  # (n_subrects, size * size)
            og = sdf.any(dim=0)  # (size * size,)
            painted = torch.zeros(size * size, dtype=torch.bool, device=device)
            selected = torch.zeros(len(new_xs), dtype=torch.bool, device=device)  # (n_subrects,)
            while True:
                # find needed rects
                rect_cnt = sdf.sum(dim=0)  # (size * size,)
                one_pixels = (rect_cnt == 1) & ~painted  # (size * size,)
                if not one_pixels.any():
                    one_pixels = (rect_cnt > 1) & ~painted  # (size * size,)
                    if not one_pixels.any():
                        break
                needed_rects = sdf.argmax(dim=0)[one_pixels].unique()
                selected[needed_rects] = True
                painted = painted | sdf[needed_rects].any(dim=0)
                # remove fully covered rects
                contained = ((1 - sdf) | painted).all(dim=-1)  # (n_subrects,)
                sdf[contained] = 0
            assert (og == painted).all(), "og != painted"

            offset = 0
            new_xs_: list[torch.Tensor] = []
            new_sizes_: list[torch.Tensor] = []
            new_rots_: list[torch.Tensor] = []
            for i_, (nx, ny) in enumerate(nxys):
                cur = selected[offset : offset + nx * ny].reshape((nx, ny))
                if not cur.any():
                    offset += nx * ny
                    continue
                curx = cur.any(dim=1).tolist()  # (nx,)
                cury = cur.any(dim=0).tolist()  # (ny,)
                trim_left = 0
                trim_right = 0
                trim_bot = 0
                trim_top = 0
                while trim_left < len(curx) and not curx[trim_left]:
                    trim_left += 1
                while trim_right < len(curx) and not curx[-trim_right - 1]:
                    trim_right += 1
                while trim_bot < len(cury) and not cury[trim_bot]:
                    trim_bot += 1
                while trim_top < len(cury) and not cury[-trim_top - 1]:
                    trim_top += 1
                txmin = trim_left / nx
                txmax = (nx - trim_right) / nx
                tymin = trim_bot / ny
                tymax = (ny - trim_top) / ny
                # trim the rects
                rot_ = rots[i_]
                cos_ = (rot_ * ur_args.theta_scale).cos()
                sin_ = (rot_ * ur_args.theta_scale).sin()
                dx = torch.stack([cos_, sin_])  # (2,)
                dy = torch.stack([-sin_, cos_])  # (2,)
                size_ = sizes[i_].abs()  # (2,)
                center_ = xs[i_] + (txmin + txmax - 1) * size_[0] * dx + (tymin + tymax - 1) * size_[1] * dy  # (2,)
                size_ = size_ * torch.tensor([txmax - txmin, tymax - tymin], device=device)  # (2,)
                new_xs_.append(center_)
                new_sizes_.append(size_)
                new_rots_.append(rot_)
                offset += nx * ny
            new_xs = torch.stack(new_xs_, dim=0)  # (n_subrects, 2)
            new_sizes = torch.stack(new_sizes_, dim=0)  # (n_subrects, 2)
            new_rots = torch.stack(new_rots_, dim=0)  # (n_subrects,)
            return UR(
                xs=new_xs,
                sizes=new_sizes,
                rots=new_rots,
                is_subs=is_subs[0].expand_as(new_rots),
            )
        else:
            raise ValueError(f"Unknown clean_strategy {clean_strategy}")

    def gen_rewrite_specs(
        self, args: URRewriteArgs, num_rewrites: int, lim: tuple[float, float], ur_args: URColectionArgs
    ) -> list[URRewrite]:
        # TODO: fix the distribution according to weights

        # merge
        merge_rewrites: list[URRewrite] = []
        if args.merge_weight > 0.0:
            merge_ij, merge_center, merge_size, merge_rot, _ = self._gen_merge_candidates(
                lossy_threshold=args.merge_threshold, ur_args=ur_args
            )
            merge_rewrites = [
                URRewriteMerge(i, j, center, size, rot)
                for (i, j), center, size, rot in zip(merge_ij, merge_center, merge_size, merge_rot)
            ]

        # remove
        remove_rewrites: list[URRewrite] = []
        if args.remove_rect_weight > 0.0 and len(self.xs) > 1:
            area = self.sizes.abs().prod(dim=-1)  # (n_rects,)
            removable: list[int] = (area <= args.remove_threshold).nonzero().flatten().tolist()
            remove_rewrites = [URRewriteRemoveRect(i) for i in removable]

        # split
        split_h_rewrites: list[URRewrite] = []
        split_v_rewrites: list[URRewrite] = []
        if args.split_h_weight > 0.0:
            for i in range(len(self.xs)):
                split_h_rewrites.append(URRewriteSplitH(i, random.random()))
        if args.split_v_weight > 0.0:
            for i in range(len(self.xs)):
                split_v_rewrites.append(URRewriteSplitV(i, random.random()))

        # add rect
        lim0, lim1 = lim
        add_rect_rewrites: list[URRewrite] = []
        if args.add_rect_weight > 0.0 or args.add_hole_weight > 0.0:
            n_ = max(len(self.xs), 32)
            pos = torch.rand((n_, 2), device=self.xs.device) * (lim1 - lim0) + lim0  # (n_, 2)
            rs = torch.rand((n_,), device=self.xs.device) * torch.pi / 2 / ur_args.theta_scale  # (n_,)
            inside = (URCollection.from_object(self, args=ur_args).rasterize(pos)[0] < 0).tolist()  # (n_,)
            for (x, y), inside_, r_ in zip(pos.tolist(), inside, rs.tolist()):
                if not inside_:
                    if args.add_rect_weight > 0.0:
                        add_rect_rewrites.append(URRewriteAddRect(x, y, args.rect_scale, r_, False))
                else:
                    if args.add_hole_weight > 0.0:
                        add_rect_rewrites.append(URRewriteAddHole(x, y, args.hole_scale, False))

        rewrites = merge_rewrites + remove_rewrites + split_h_rewrites + split_v_rewrites + add_rect_rewrites
        return random.sample(rewrites, min(num_rewrites, len(rewrites)))

    def apply_rewrite(self, spec: URRewrite, ur_args: URColectionArgs) -> "UR":
        if isinstance(spec, URRewriteSplitH) or isinstance(spec, URRewriteSplitV):
            id_, t = spec.id_, spec.t
            rtype = spec.rewrite_type
            new_xs, new_sizes, new_rots = _split_rect(
                self.xs[id_],
                self.sizes[id_],
                self.rots[id_],
                [t] if rtype == URRewriteType.SplitH else [],
                [t] if rtype == URRewriteType.SplitV else [],
                ur_args,
            )
            new_is_subs = torch.full((len(new_xs),), self.is_subs[id_].item(), device=self.is_subs.device)
            return UR(
                xs=torch.cat([self.xs[:id_, ...], new_xs, self.xs[id_ + 1 :, ...]], dim=0),
                sizes=torch.cat([self.sizes[:id_, ...], new_sizes, self.sizes[id_ + 1 :, ...]], dim=0),
                rots=torch.cat([self.rots[:id_, ...], new_rots, self.rots[id_ + 1 :, ...]], dim=0),
                is_subs=torch.cat([self.is_subs[:id_, ...], new_is_subs, self.is_subs[id_ + 1 :, ...]], dim=0),
            )
        elif isinstance(spec, URRewriteAddRect):
            x, y, s = spec.x, spec.y, spec.s
            r, is_sub = spec.r, spec.is_sub
            new_xs = torch.tensor([[x, y]], device=self.xs.device)
            new_sizes = torch.tensor([[s, s]], device=self.xs.device)
            new_rots = torch.tensor([r], device=self.xs.device)
            new_is_subs = torch.tensor([is_sub], device=self.xs.device)
            return UR(
                xs=torch.cat([self.xs, new_xs], dim=0),
                sizes=torch.cat([self.sizes, new_sizes], dim=0),
                rots=torch.cat([self.rots, new_rots], dim=0),
                is_subs=torch.cat([self.is_subs, new_is_subs], dim=0),
            )
        elif isinstance(spec, URRewriteRemoveRect):
            id_ = spec.id_
            new_xs = torch.cat([self.xs[:id_, ...], self.xs[id_ + 1 :, ...]], dim=0)
            new_sizes = torch.cat([self.sizes[:id_, ...], self.sizes[id_ + 1 :, ...]], dim=0)
            new_rots = torch.cat([self.rots[:id_, ...], self.rots[id_ + 1 :, ...]], dim=0)
            new_is_subs = torch.cat([self.is_subs[:id_, ...], self.is_subs[id_ + 1 :, ...]], dim=0)
            return UR(
                xs=new_xs,
                sizes=new_sizes,
                rots=new_rots,
                is_subs=new_is_subs,
            )
        elif isinstance(spec, URRewriteMerge):
            id1, id2, center, size, rot = spec.id1, spec.id2, spec.center, spec.size, spec.rot
            assert id1 < id2, f"id1 must be less than id2, got {id1} >= {id2}"
            assert (
                self.is_subs[id1] == self.is_subs[id2]
            ), f"id1 and id2 must have the same is_subs, got {self.is_subs[id1]} != {self.is_subs[id2]}"
            new_xs = torch.cat(
                [self.xs[:id1, ...], center.unsqueeze(0), self.xs[id1 + 1 : id2, ...], self.xs[id2 + 1 :, ...]], dim=0
            )
            new_sizes = torch.cat(
                [self.sizes[:id1, ...], size.unsqueeze(0), self.sizes[id1 + 1 : id2, ...], self.sizes[id2 + 1 :, ...]],
                dim=0,
            )
            new_rots = torch.cat(
                [self.rots[:id1, ...], rot.unsqueeze(0), self.rots[id1 + 1 : id2, ...], self.rots[id2 + 1 :, ...]],
                dim=0,
            )
            new_is_subs = torch.cat(
                [
                    self.is_subs[:id1, ...],
                    self.is_subs[id1].unsqueeze(0),
                    self.is_subs[id1 + 1 : id2, ...],
                    self.is_subs[id2 + 1 :, ...],
                ],
                dim=0,
            )
            return UR(
                xs=new_xs,
                sizes=new_sizes,
                rots=new_rots,
                is_subs=new_is_subs,
            )
        elif isinstance(spec, URRewriteAddHole):
            x, y, s, is_sub = spec.x, spec.y, spec.s, spec.is_sub
            is_sub_mask = self.is_subs == is_sub
            device = self.xs.device
            new_xs, new_sizes, new_rots = _punch_hole(
                self.xs[is_sub_mask],
                self.sizes[is_sub_mask],
                self.rots[is_sub_mask],
                torch.tensor([x, y], device=device),
                s,
                ur_args,
            )
            return UR(
                xs=torch.cat([self.xs[~is_sub_mask], new_xs], dim=0),
                sizes=torch.cat([self.sizes[~is_sub_mask], new_sizes], dim=0),
                rots=torch.cat([self.rots[~is_sub_mask], new_rots], dim=0),
                is_subs=torch.cat([self.is_subs[~is_sub_mask], torch.full((len(new_xs),), is_sub, device=device)]),
            )
        else:
            raise ValueError(f"Unknown rewrite {spec}")

    def apply_all_rewrites(
        self, rewrites: list[URRewrite], scores: list[float], ur_args: URColectionArgs
    ) -> "UR":
        # sort rewrites
        rewrites, scores = zip(*sorted(zip(rewrites, scores), key=lambda x: x[1], reverse=True))  # type: ignore
        # print("\n".join(map(str, list(zip(rewrites, scores)))))

        n_delete = 0
        deleted: set[int] = set()
        splits: dict[int, tuple[list[float], list[float]]] = {}
        merges: dict[int, URRewriteMerge] = {}
        add_rects: list[tuple[float, float, float, float, bool]] = []
        add_holes: list[URRewriteAddHole] = []
        for spec in rewrites:
            if isinstance(spec, URRewriteSplitH) or isinstance(spec, URRewriteSplitV):
                id_, t = spec.id_, spec.t
                id_ = int(id_)
                if id_ in deleted:
                    continue
                if id_ not in splits:
                    splits[id_] = ([], [])
                k_ = 0 if spec.rewrite_type == URRewriteType.SplitH else 1
                # only one split axis is allowed
                if len(splits[id_][1 - k_]) > 0:
                    continue
                splits[id_][k_].append(t)
                # print(spec.rewrite_type, id_, t)
            elif isinstance(spec, URRewriteAddRect):
                x, y, s = spec.x, spec.y, spec.s
                r, is_sub = spec.r, spec.is_sub
                add_rects.append((x, y, s, r, is_sub))  # (x, y, s, r, is_sub)
                # print(spec.rewrite_type, x, y, s, r, is_sub)
            elif isinstance(spec, URRewriteRemoveRect):
                id_ = spec.id_
                if id_ in splits:
                    continue
                if n_delete + 1 == len(self.xs):
                    continue
                deleted.add(id_)
                n_delete += 1
                # print(spec.rewrite_type, id_)
            elif isinstance(spec, URRewriteMerge):
                id1, id2 = spec.id1, spec.id2
                if id1 in deleted or id2 in deleted or id1 in splits or id2 in splits:
                    continue
                deleted.add(id1)
                deleted.add(id2)
                merges[id1] = spec
                # print(spec.rewrite_type, id1, id2)
            elif isinstance(spec, URRewriteAddHole):
                add_holes.append(spec)
            else:
                raise ValueError(f"Unknown rewrite {spec}")

        new_xs: list[torch.Tensor] = []
        new_sizes: list[torch.Tensor] = []
        new_rots: list[torch.Tensor] = []
        new_is_subs: list[torch.Tensor] = []
        for i, (xs, size, rot, is_sub) in enumerate(zip(self.xs, self.sizes, self.rots, self.is_subs)):
            if i in merges:
                merge = merges[i]
                new_xs.append(merge.center.unsqueeze(0))
                new_sizes.append(merge.size.unsqueeze(0))
                new_rots.append(merge.rot.unsqueeze(0))
                new_is_subs.append(is_sub.unsqueeze(0))
                continue
            if i in deleted:
                continue
            if i not in splits:
                new_xs.append(xs.unsqueeze(0))
                new_sizes.append(size.unsqueeze(0))
                new_rots.append(rot.unsqueeze(0))
                new_is_subs.append(is_sub.unsqueeze(0))
                continue
            txs, tys = splits[i]
            new_xs_, new_sizes_, new_rots_ = _split_rect(xs, size, rot, txs, tys, ur_args)
            new_xs.append(new_xs_)
            new_sizes.append(new_sizes_)
            new_rots.append(new_rots_)
            new_is_subs.append(is_sub.expand(len(new_xs_)))

        if len(add_holes) > 0:
            new_xs_ = torch.cat(new_xs, dim=0)
            new_sizes_ = torch.cat(new_sizes, dim=0)
            new_rots_ = torch.cat(new_rots, dim=0)
            new_is_subs_ = torch.cat(new_is_subs, dim=0)

            new_xs__ = [new_xs_[~new_is_subs_], new_xs_[new_is_subs_]]
            new_sizes__ = [new_sizes_[~new_is_subs_], new_sizes_[new_is_subs_]]
            new_rots__ = [new_rots_[~new_is_subs_], new_rots_[new_is_subs_]]

            device = self.xs.device

            for add_hole in add_holes:
                x, y, s, is_sub = add_hole.x, add_hole.y, add_hole.s, add_hole.is_sub
                new_xs__[is_sub], new_sizes__[is_sub], new_rots__[is_sub] = _punch_hole(
                    new_xs__[is_sub],
                    new_sizes__[is_sub],
                    new_rots__[is_sub],
                    torch.tensor([x, y], device=device),
                    s,
                    ur_args,
                )
            new_xs = new_xs__
            new_sizes = new_sizes__
            new_rots = new_rots__
            new_is_subs = [
                torch.full((len(new_xs[0]),), False, device=device),
                torch.full((len(new_xs[1]),), True, device=device),
            ]

        for x, y, s, r, is_sub in add_rects:
            new_xs.append(torch.tensor([[x, y]], device=self.xs.device))
            new_sizes.append(torch.tensor([[s, s]], device=self.xs.device))
            new_rots.append(torch.tensor([r], device=self.xs.device))
            new_is_subs.append(torch.tensor([is_sub], device=self.xs.device))

        return UR(
            xs=torch.cat(new_xs, dim=0),
            sizes=torch.cat(new_sizes, dim=0),
            rots=torch.cat(new_rots, dim=0),
            is_subs=torch.cat(new_is_subs, dim=0),
        )

    def _gen_merge_candidates(
        self, lossy_threshold: float, ur_args: URColectionArgs
    ) -> tuple[list[tuple[int, int]], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns:
        - ij: list of pairs of indices
        - center: (n_combs, 2)
        - size: (n_combs, 2)
        - rot: (n_combs,)
        """
        return _gen_merge_candidates(self.xs, self.sizes, self.rots, self.is_subs, lossy_threshold, ur_args)


def _split_rect(
    center: torch.Tensor,
    size: torch.Tensor,
    rot: torch.Tensor,
    txs: list[float],
    tys: list[float],
    ur_args: URColectionArgs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    - center: (2,)
    - size: (2,)
    - rot: ()
    - txs: (nx,) sorted
    - tys: (ny,) sorted

    returns:
    - xs: ((nx + 1) * (ny + 1), 2)
    - sizes: ((nx + 1) * (ny + 1), 2)
    - rots: ((nx + 1) * (ny + 1),)
    """
    nx = len(txs)
    ny = len(tys)
    cos_ = (rot * ur_args.theta_scale).cos()
    sin_ = (rot * ur_args.theta_scale).sin()
    dx = torch.stack([cos_, sin_])
    dy = torch.stack([-sin_, cos_])
    txs.sort()
    tys.sort()
    txs_ = torch.tensor([0.0, *txs, 1.0], device=center.device) * (size[0] * 2)
    tys_ = torch.tensor([0.0, *tys, 1.0], device=center.device) * (size[1] * 2)
    ctxs = (txs_[:-1] + txs_[1:]) / 2  # (nx + 1,)
    ctys = (tys_[:-1] + tys_[1:]) / 2  # (ny + 1,)
    sxs = txs_[1:] - txs_[:-1]  # (nx + 1,)
    sys = tys_[1:] - tys_[:-1]  # (ny + 1,)
    sizes = torch.stack([sxs.unsqueeze(-1).expand(-1, ny + 1), sys.expand(nx + 1, -1)], dim=-1)  # (nx + 1, ny + 1, 2)
    ll = center - dx * size[0] - dy * size[1]  # (2,)
    pos = ll + dx * ctxs.reshape(-1, 1, 1) + dy * ctys.unsqueeze(-1)  # (nx + 1, ny + 1, 2)
    return pos.flatten(0, 1), sizes.flatten(0, 1) / 2, rot.expand((nx + 1) * (ny + 1))


def _gen_merge_candidates(
    xs: torch.Tensor,
    sizes: torch.Tensor,
    rots: torch.Tensor,
    is_subs: torch.Tensor,
    lossy_threshold: float,
    ur_args: URColectionArgs,
) -> tuple[list[tuple[int, int]], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    xs: (n_rects, 2)
    sizes: (n_rects, 2)
    rots: (n_rects,)
    is_subs: (n_rects,)

    returns:
    - ij: list of pairs of indices
    - center: (n_combs, 2)
    - size: (n_combs, 2)
    - rot: (n_combs,)
    - area_gain: (n_combs,)
    """
    device = xs.device
    ids__ = ([], [])
    for i, is_sub in enumerate(is_subs):
        ids__[int(is_sub.item())].append(i)
    ij = list(itertools.combinations(ids__[0], 2)) + list(itertools.combinations(ids__[1], 2))
    if len(ij) == 0:
        empty = torch.empty((0, 2), dtype=torch.long, device=device)
        empty1 = torch.empty((0,), dtype=torch.long, device=device)
        return [], empty, empty, empty1, empty1
    is_, js_ = zip(*ij)
    is_ = list(is_)
    js_ = list(js_)
    cos = (rots * ur_args.theta_scale).cos()  # (n_rects,)
    sin = (rots * ur_args.theta_scale).sin()  # (n_rects,)
    dirx = torch.stack([cos, sin], dim=-1)  # (n_rects, 2)
    diry = torch.stack([-sin, cos], dim=-1)  # (n_rects, 2)
    dx = torch.tensor([-1, -1, 1, 1], device=device)  # (4,)
    dy = torch.tensor([-1, 1, -1, 1], device=device)  # (4,)
    endpoints = (
        xs.unsqueeze(-2)
        + dx.unsqueeze(-1) * dirx.unsqueeze(-2) * sizes[:, 0].reshape(-1, 1, 1)
        + dy.unsqueeze(-1) * diry.unsqueeze(-2) * sizes[:, 1].reshape(-1, 1, 1)
    )  # (n_rects, 4, 2)

    pi_ = xs[is_, ...]  # (n_combs, 2)
    pj_ = xs[js_, ...]  # (n_combs, 2)
    ns_ = sizes.norm(dim=-1)  # (n_rects,)
    si_ = ns_[is_]  # (n_combs,)
    sj_ = ns_[js_]  # (n_combs,)
    mask = (pj_ - pi_).norm(dim=-1) <= si_ + sj_  # (n_combs,)
    is__ = []
    js__ = []
    for id_ in mask.nonzero().flatten().tolist():
        is__.append(is_[id_])
        js__.append(js_[id_])
    is_ = is__
    js_ = js__

    if len(is_) == 0:
        empty = torch.empty((0, 2), dtype=torch.long, device=device)
        empty1 = torch.empty((0,), dtype=torch.long, device=device)
        return [], empty, empty, empty1, empty1

    pi = endpoints[is_, ...]  # (n_combs, 4, 2)
    pj = endpoints[js_, ...]  # (n_combs, 4, 2)

    pts = torch.cat([pi, pj], dim=-2)  # (n_combs, 8, 2)

    dij = pi.unsqueeze(-2) - pj.unsqueeze(-3)  # (n_combs, 4, 4, 2)
    test_angs1 = torch.atan2(dij[..., 1], dij[..., 0]).flatten(-2)  # (n_combs, 16)
    test_angs2 = torch.stack([rots[is_], rots[js_]], dim=-1) * ur_args.theta_scale  # (n_combs, 2)
    test_angs = torch.cat([test_angs1, test_angs2], dim=-1)  # (n_combs, 18)

    test_cos = torch.cos(test_angs)  # (n_combs, 18)
    test_sin = torch.sin(test_angs)  # (n_combs, 18)

    irot = torch.stack([test_cos, test_sin, -test_sin, test_cos], dim=-1).reshape(-1, 18, 2, 2)  # (n_combs, 18, 2, 2)

    rot_pts = (irot.unsqueeze(-3) @ pts.reshape(-1, 1, 8, 2, 1)).squeeze(-1)  # (n_combs, 18, 8, 2)
    min_ = rot_pts.min(dim=-2).values  # (n_combs, 18, 2)
    max_ = rot_pts.max(dim=-2).values  # (n_combs, 18, 2)
    size_ = max_ - min_  # (n_combs, 18, 2)
    center_ = (max_ + min_) / 2  # (n_combs, 18, 2)
    area_ = size_.prod(dim=-1)  # (n_combs, 18)

    min_area_, amin_ = area_.min(dim=-1)  # (n_combs,)

    size = torch.gather(size_, dim=-2, index=amin_.reshape(-1, 1, 1).expand(-1, -1, 2)).squeeze(-2) / 2  # (n_combs, 2)

    si_ = sizes[is_, :].abs().prod(dim=-1)  # (n_combs,)
    sj_ = sizes[js_, :].abs().prod(dim=-1)  # (n_combs,)
    area_gain = size.prod(dim=-1) - si_ - sj_  # (n_combs,)
    mask = area_gain <= lossy_threshold  # (n_combs,)
    # print(si_ + sj_ - size.prod(dim=-1))

    size = size[mask]  # (n_masked,)
    amin_ = amin_[mask]  # (n_masked,)

    rot = torch.gather(test_angs[mask], dim=-1, index=amin_.unsqueeze(-1)).squeeze(-1)  # (n_masked,)
    center = torch.gather(center_[mask], dim=-2, index=amin_.reshape(-1, 1, 1).expand(-1, -1, 2)).squeeze(
        -2
    )  # (n_masked, 2)
    c_ = rot.cos()  # (n_masked,)
    s_ = rot.sin()  # (n_masked,)
    rot_mat = torch.stack([c_, -s_, s_, c_], dim=-1).reshape(-1, 2, 2)  # (n_masked, 2, 2)
    center = (rot_mat @ center.unsqueeze(-1)).squeeze(-1)  # (n_masked, 2)
    area_gain = area_gain[mask]  # (n_masked,)
    mask = mask.tolist()
    ijs = [(is_[i], js_[i]) for i in range(len(is_)) if mask[i]]
    return ijs, center, size, rot / ur_args.theta_scale, area_gain


def _punch_hole(
    xs: torch.Tensor,
    sizes: torch.Tensor,
    rots: torch.Tensor,
    pos: torch.Tensor,
    hole_r: float,
    ur_args: URColectionArgs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    xs: (n_rects, 2)
    sizes: (n_rects, 2)
    rots: (n_rects,)
    pos: (2,)

    returns:
    - xs: (n_rects, 2)
    - sizes: (n_rects, 2)
    - rots: (n_rects,)
    """
    diff = pos - xs  # (n_rects, 2)
    cos_ = (rots * ur_args.theta_scale).cos()  # (n_rects,)
    sin_ = (rots * ur_args.theta_scale).sin()  # (n_rects,)
    irot = torch.stack([cos_, sin_, -sin_, cos_], dim=-1).reshape(-1, 2, 2)  # (n_rects, 2, 2)
    diff_ = (irot @ diff.unsqueeze(-1)).squeeze(-1)  # (n_rects, 2)
    t_ = (diff_ + sizes.abs()) / (sizes.abs() * 2)  # (n_rects, 2)
    hole_t_ = hole_r / (sizes.abs() * 2)  # (n_rects,)
    mask = ((t_ > 0.0) & (t_ < 1.0)).all(dim=-1).tolist()  # (n_rects,)
    t_min: list[tuple[float, float]] = (t_ - hole_t_).tolist()
    t_max: list[tuple[float, float]] = (t_ + hole_t_).tolist()
    xs_: list[torch.Tensor] = []
    sizes_: list[torch.Tensor] = []
    rots_: list[torch.Tensor] = []
    for i_ in range(len(xs)):
        if not mask[i_]:
            # no hole
            xs_.append(xs[i_])
            sizes_.append(sizes[i_])
            rots_.append(rots[i_])
            continue
        (tminx, tminy), (tmaxx, tmaxy) = t_min[i_], t_max[i_]
        tx = [x for x in [tminx, tmaxx] if 0 < x < 1]
        ty = [y for y in [tminy, tmaxy] if 0 < y < 1]
        if len(tx) == 0 and len(ty) == 0:
            continue
        c_xs, c_sizes, c_rots = _split_rect(xs[i_], sizes[i_], rots[i_], tx, ty, ur_args)
        # HEURISTIC: remove the region closest to the hole
        amin = int((c_xs - pos).norm(dim=-1).min(dim=0).indices.item())
        for j_ in range(len(c_xs)):
            if j_ == amin:
                continue
            xs_.append(c_xs[j_])
            sizes_.append(c_sizes[j_])
            rots_.append(c_rots[j_])

    return torch.stack(xs_, dim=0), torch.stack(sizes_, dim=0), torch.stack(rots_, dim=0)


def _always_raise() -> URColectionArgs:
    raise ValueError("URCollectionArgs must be set")


@dataclass
class URCollection(ObjectCollection[UR]):
    xs: torch.Tensor  # (tot_rects, 2)
    sizes: torch.Tensor  # (tot_rects, 2)
    rots: torch.Tensor  # (tot_rects,)
    is_subs: torch.Tensor  # (tot_rects,)
    index_of: torch.Tensor  # (tot_rects,) -> index of shapes
    indices: tuple[tuple[int, int], ...]  # [(n_rects, 2), ...] start, end for each shape
    ids: tuple[int, ...]
    payloads: tuple[URPayload, ...]
    args: URColectionArgs = field(default_factory=_always_raise)

    def __post_init__(self):
        assert self.xs.ndim == 2, f"xs must be a 2D tensor, got {self.xs.ndim}"
        assert self.sizes.ndim == 2, f"sizes must be a 2D tensor, got {self.sizes.ndim}"
        assert self.rots.ndim == 1, f"rots must be a 1D tensor, got {self.rots.ndim}"
        assert self.index_of.ndim == 1, f"index_of must be a 1D tensor, got {self.index_of.ndim}"
        assert (
            len(self.xs) == len(self.sizes) == len(self.rots) == len(self.index_of) == len(self.is_subs)
        ), f"len(xs) != len(sizes) != len(rots) != len(index_of). Got {len(self.xs)} != {len(self.sizes)} != {len(self.rots)} != {len(self.index_of)}"
        assert (
            len(self.indices) == len(self.ids) == len(self.payloads)
        ), f"len(indices) != len(ids) != len(payloads). Got {len(self.indices)} != {len(self.ids)} != {len(self.payloads)}"

    def __len__(self) -> int:
        return len(self.ids)

    def parameters(self) -> list[torch.Tensor]:
        return [self.xs, self.sizes, self.rots]

    def parameter_names(self) -> list[str]:
        return ["xs", "sizes", "rots"]

    def per_object_grads(self) -> list[torch.Tensor]:
        grads: list[torch.Tensor] = []
        xs_grad = self.xs.grad
        sizes_grad = self.sizes.grad
        rots_grad = self.rots.grad
        for s, e in self.indices:
            grads_: list[torch.Tensor] = []
            if xs_grad is not None:
                grads_.append(xs_grad[s:e, ...].flatten())
            if sizes_grad is not None:
                grads_.append(sizes_grad[s:e, ...].flatten())
            if rots_grad is not None:
                grads_.append(rots_grad[s:e, ...].flatten())
            grads.append(torch.cat(grads_, dim=0))
        return grads

    def device(self) -> torch.device:
        return self.xs.device

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        self.xs = self.xs.requires_grad_(requires_grad)
        self.sizes = self.sizes.requires_grad_(requires_grad)
        self.rots = self.rots.requires_grad_(requires_grad)
        return self

    def clone(self) -> Self:
        """
        Detach and clone
        """
        return self.__class__(
            xs=self.xs.detach().clone(),
            sizes=self.sizes.detach().clone(),
            rots=self.rots.detach().clone(),
            is_subs=self.is_subs.clone(),
            index_of=self.index_of.clone(),
            indices=self.indices,
            ids=self.ids,
            payloads=self.payloads,
            args=self.args,
        )

    def to(self, device: Union[str, torch.device, None] = None) -> "URCollection":
        """
        Detach and clone
        """
        return URCollection(
            xs=self.xs.to(device=device),
            sizes=self.sizes.to(device=device),
            rots=self.rots.to(device=device),
            is_subs=self.is_subs.to(device=device),
            index_of=self.index_of.to(device=device),
            indices=self.indices,
            ids=self.ids,
            payloads=self.payloads,
            args=self.args,
        )

    def get_object(self, idx: int, detach: bool = True) -> UR:
        s, e = self.indices[idx]
        return UR(
            xs=maybe_detach(self.xs[s:e, ...], detach),
            sizes=maybe_detach(self.sizes[s:e, ...], detach),
            rots=maybe_detach(self.rots[s:e], detach),
            is_subs=self.is_subs[s:e],
            id=self.ids[idx],
            payload=self.payloads[idx],
        )

    @classmethod
    def from_object(cls, object: UR, **kwargs) -> "URCollection":
        device = object.xs.device
        return cls(
            xs=object.xs.detach().clone(),
            sizes=object.sizes.detach().clone(),
            rots=object.rots.detach().clone(),
            is_subs=object.is_subs.clone(),
            index_of=torch.full((len(object.xs),), 0, dtype=torch.long, device=device),
            indices=((0, len(object.xs)),),
            ids=(object.id,),
            payloads=(object.payload,),
            **kwargs,
        )

    @classmethod
    def cat(cls, collections: list[ObjectCollection[UR]], **kwargs) -> Self:
        collections_ = cast(list[URCollection], collections)
        assert len(collections_) > 0, "collections must not be empty"
        n_shapes = 0
        n_rects = 0
        index_of: list[torch.Tensor] = []
        indices: list[tuple[int, int]] = []
        for collection in collections_:
            index_of.append(collection.index_of + n_shapes)
            indices.extend((x + n_rects, y + n_rects) for x, y in collection.indices)
            n_rects += len(collection.xs)
            n_shapes += len(collection.ids)
        return cls(
            xs=torch.cat([x.xs for x in collections_], dim=0),
            sizes=torch.cat([x.sizes for x in collections_], dim=0),
            rots=torch.cat([x.rots for x in collections_], dim=0),
            is_subs=torch.cat([x.is_subs for x in collections_], dim=0),
            index_of=torch.cat(index_of, dim=0),
            indices=tuple(indices),
            ids=sum(tuple(x.ids for x in collections_), ()),
            payloads=sum(tuple(x.payloads for x in collections_), ()),
            **kwargs,
        )

    def rasterize(self, positions: torch.Tensor, offset: float = 0.0) -> torch.Tensor:
        """
        Render as a signed distance field.
        - positions: (..., 2)

        returns:
        - img: (num_shapes, ...)
        """
        *shape, _ = positions.shape
        assert _ == 2, f"positions must be (..., 2), got {positions.shape}"
        cos_ = torch.cos(self.rots * self.args.theta_scale)  # (n_rects,)
        sin_ = torch.sin(self.rots * self.args.theta_scale)  # (n_rects,)
        irot = torch.stack([cos_, sin_, -sin_, cos_], dim=-1).reshape(-1, 2, 2)  # (total_nodes, 2, 2)
        diff = (irot @ (positions.unsqueeze(-2) - self.xs).unsqueeze(-1)).squeeze(-1)  # (..., total_nodes, 2)
        d = diff.abs() - self.sizes.abs()  # (..., n_rects, 2)
        sdf = d.clamp(min=0).norm(dim=-1) + d.max(dim=-1).values.clamp(max=0)  # (..., n_rects)
        sdf = sdf.permute(-1, *range(len(shape)))  # (n_rects, ...)
        result = torch.full(
            (len(self.ids) * 2, *shape), torch.inf, dtype=sdf.dtype, device=sdf.device
        )  # (n_shapes * 2, ...)
        result = torch.scatter_reduce(
            result,
            0,
            (self.index_of * 2 + self.is_subs).reshape(-1, *[1] * len(shape)).expand(-1, *shape),
            sdf,
            reduce="amin",
        )  # (n_shapes * 2, ...)
        res1, res2 = result.reshape(len(self.ids), 2, *shape).unbind(1)  # (n_shapes, ...)
        return torch.maximum(res1 - offset, -res2 + offset)  # res1 - res2

    def render(
        self,
        size: int,
        lim: tuple[float, float] = (-1.5, 1.5),
        center_pixel: bool = True,
        offset: float = 0.0,
    ) -> torch.Tensor:
        """
        Render as a signed distance field.
        returns:
        - img: (num_shapes, size, size)
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
        return self.rasterize(grid, offset=offset)

    def render01(
        self, size: int, lim: tuple[float, float] = (-1.5, 1.5), center_pixel: bool = True, blur: float = 1.0
    ) -> torch.Tensor:
        """Render the collection as a signed distance field.

        Returns:
        - img: (n_shapes, size, size)
        """
        vlim = blur * (lim[1] - lim[0]) / size
        imgs = self.render(
            size, lim, center_pixel=center_pixel, offset=vlim * self.args.offset_scale
        )  #  - vlim / 2  # (n_shapes, size, size)
        imgs = (-imgs.clamp(-vlim, vlim) + vlim) / (2 * vlim)  # (n_shapes, size, size)
        return imgs

    def get_sizes(self) -> list[int]:
        """Get number of rects in each shape"""
        return [y - x for x, y in self.indices]

    def get_sum_sizes(self) -> torch.Tensor:
        """Get sum of sizes of all rects"""
        device = self.sizes.device
        result = torch.full((len(self.ids),), 0.0, device=device)  # (n_shapes, ...)
        return torch.scatter_reduce(result, 0, self.index_of, self.sizes.abs().sum(dim=-1), reduce="sum")  # (n_shapes,)

    @classmethod
    def patch_args(cls, args: URColectionArgs) -> Type["URCollection"]:
        return type(
            "URCollectionWithArgs", (URCollection,), {"__init__": partialmethod(URCollection.__init__, args=args)}
        )

    def to_savable(self) -> "URCollection":
        return URCollection(
            xs=self.xs,
            sizes=self.sizes,
            rots=self.rots,
            is_subs=self.is_subs,
            index_of=self.index_of,
            indices=self.indices,
            ids=self.ids,
            payloads=self.payloads,
            args=self.args,
        )

    def project_to_valid_(self) -> Self:
        self.sizes.data.clamp_(min=0)
        return self
