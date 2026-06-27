import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Self, Union, Type, Literal, Sequence, cast
from matplotlib.patches import Polygon
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
class TriRewriteArgs:
    add_rect_weight: float = 1.0
    remove_rect_weight: float = 1.0
    rect_scale: float = 0.05
    remove_threshold: float = 0.005  # area
    pass


@dataclass
class TriColectionArgs:
    theta_scale: float = torch.pi
    offset_scale: float = 1.0


# CleanStrategy = Literal["none", "smart", "smarter"]


class TriRewriteType(Enum):
    AddTri = 1  # (x, y)
    RemoveTri = 2  # (id)


@dataclass
class TriRewrite:
    rewrite_type: TriRewriteType


@dataclass
class TriRewriteAddTri(TriRewrite):
    rewrite_type: TriRewriteType = field(default=TriRewriteType.AddTri, init=False)
    x1: float
    y1: float
    x2: float
    y2: float
    x3: float
    y3: float


@dataclass
class TriRewriteRemoveTri(TriRewrite):
    rewrite_type: TriRewriteType = field(default=TriRewriteType.RemoveTri, init=False)
    id_: int


@dataclass
class TriPayload:
    pass


@dataclass
class Tri:
    xs: torch.Tensor  # (n_shapes, 3, 2)
    id: int = field(default_factory=lambda: Context.get().gen_id())
    payload: TriPayload = field(default_factory=TriPayload)

    def __post_init__(self):
        assert self.xs.ndim == 3, f"xs must be a 3D tensor, got {self.xs.ndim}"

    def visualize(
        self,
        ax: MPLVisualizerAxes,
        ur_args: TriColectionArgs,
        line_style: LineStyle = LineStyle(),
        sub_line_style: LineStyle = LineStyle(color="red"),
    ) -> None:
        for i, vs in enumerate(self.xs.tolist()):
            line_style_ = line_style
            ax.ax.add_patch(
                Polygon(
                    vs,
                    color=line_style_.color,
                    linewidth=line_style_.linewidth,
                    fill=False,
                )
            )
            # ax.ax.text(cx, cy, str(i), color="black", fontsize=12, ha="center", va="center")

    def gen_rewrite_specs(
        self,
        args: TriRewriteArgs,
        num_rewrites: int,
        lim: tuple[float, float],
        tri_args: TriColectionArgs,
    ) -> list[TriRewrite]:
        # TODO: implement
        # remove
        remove_rewrites: list[TriRewrite] = []
        if args.remove_rect_weight > 0.0 and len(self.xs) > 1:
            # 面積を求めたい
            a, b, c = self.xs.unbind(dim=1)
            e1 = b - a
            e2 = c - a
            area = (e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]) / 2
            area.abs_()
            removable: list[int] = (
                (area <= args.remove_threshold).nonzero().flatten().tolist()
            )
            remove_rewrites = [TriRewriteRemoveTri(i) for i in removable]
        # add rect
        lim0, lim1 = lim
        add_tri_rewrites: list[TriRewrite] = []
        if args.add_rect_weight > 0.0:
            n_ = max(len(self.xs), 32)
            pos = (
                torch.rand((n_, 2), device=self.xs.device) * (lim1 - lim0) + lim0
            )  # (n_, 2)
            rel = (
                torch.rand((n_, 3, 2), device=self.xs.device) - 0.5
            ) * args.rect_scale
            triangles = pos.unsqueeze(1) + rel
            for vs in triangles.to("cpu").detach().numpy():
                # print(f"vs: {vs}")
                add_tri_rewrites.append(TriRewriteAddTri(*vs.ravel()))

        rewrites = remove_rewrites + add_tri_rewrites
        return random.sample(rewrites, min(num_rewrites, len(rewrites)))

    def apply_rewrite(self, spec: TriRewrite, tri_args: TriColectionArgs) -> "Tri":
        if isinstance(spec, TriRewriteAddTri):
            new_xs = torch.tensor(
                [[[spec.x1, spec.y1], [spec.x2, spec.y2], [spec.x3, spec.y3]]],
                device=self.xs.device,
            )
            return Tri(
                xs=torch.cat([self.xs, new_xs], dim=0),
            )
        elif isinstance(spec, TriRewriteRemoveTri):
            id_ = spec.id_
            new_xs = torch.cat([self.xs[:id_, ...], self.xs[id_ + 1 :, ...]], dim=0)
            return Tri(
                xs=new_xs,
            )
        else:
            raise ValueError(f"Unknown rewrite {spec}")

    def apply_all_rewrites(
        self,
        rewrites: list[TriRewrite],
        scores: list[float],
        tri_args: TriColectionArgs,
    ) -> "Tri":
        # sort rewrites
        rewrites, scores = zip(
            *sorted(zip(rewrites, scores), key=lambda x: x[1], reverse=True)
        )  # type: ignore
        # print("\n".join(map(str, list(zip(rewrites, scores)))))

        n_delete = 0
        deleted: set[int] = set()
        add_tris: list[tuple[float, float, float, float, bool]] = []
        for spec in rewrites:
            # print(spec.rewrite_type, id_, t)
            if isinstance(spec, TriRewriteAddTri):
                new_x = torch.tensor(
                    [[[spec.x1, spec.y1], [spec.x2, spec.y2], [spec.x3, spec.y3]]],
                    device=self.xs.device,
                )
                add_tris.append(new_x)  # (x, y, s, r, is_sub)
                # print(spec.rewrite_type, new_x)
            elif isinstance(spec, TriRewriteRemoveTri):
                id_ = spec.id_
                if n_delete + 1 == len(self.xs):
                    continue
                deleted.add(id_)
                n_delete += 1
                # print(spec.rewrite_type, id_)
            else:
                raise ValueError(f"Unknown rewrite {spec}")

        if len(add_tris) > 0:
            new_xs = torch.cat([self.xs, torch.cat(add_tris, dim=0)])
        else:
            new_xs = self.xs.detach().clone()

        for i in sorted(deleted, reverse=True):
            new_xs = torch.cat([new_xs[:i], new_xs[i + 1 :]])

        return Tri(xs=new_xs)


def _always_raise() -> TriColectionArgs:
    raise ValueError("TriCollectionArgs must be set")


@dataclass
class TriCollection(ObjectCollection[Tri]):
    xs: torch.Tensor  # (tot_rects, 2)
    index_of: torch.Tensor  # (tot_rects,) -> index of shapes
    indices: tuple[
        tuple[int, int], ...
    ]  # [(n_rects, 2), ...] start, end for each shape
    ids: tuple[int, ...]
    payloads: tuple[TriPayload, ...]
    args: TriColectionArgs = field(default_factory=_always_raise)

    def __post_init__(self):
        assert self.xs.ndim == 3, f"xs must be a 3D tensor, got {self.xs.ndim}"
        assert len(self.indices) == len(self.ids) == len(self.payloads), (
            f"len(indices) != len(ids) != len(payloads). Got {len(self.indices)} != {len(self.ids)} != {len(self.payloads)}"
        )

    def __len__(self) -> int:
        return len(self.ids)

    def parameters(self) -> list[torch.Tensor]:
        return [self.xs]

    def parameter_names(self) -> list[str]:
        return ["xs"]

    def per_object_grads(self) -> list[torch.Tensor]:
        grads: list[torch.Tensor] = []
        xs_grad = self.xs.grad
        for s, e in self.indices:
            grads_: list[torch.Tensor] = []
            if xs_grad is not None:
                grads_.append(xs_grad[s:e, ...].flatten())
            grads.append(torch.cat(grads_, dim=0))
        return grads

    def device(self) -> torch.device:
        return self.xs.device

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        self.xs = self.xs.requires_grad_(requires_grad)
        return self

    def clone(self) -> Self:
        """
        Detach and clone
        """
        return self.__class__(
            xs=self.xs.detach().clone(),
            index_of=self.index_of.clone(),
            indices=self.indices,
            ids=self.ids,
            payloads=self.payloads,
            args=self.args,
        )

    def to(self, device: Union[str, torch.device, None] = None) -> "TriCollection":
        """
        Detach and clone
        """
        return TriCollection(
            xs=self.xs.to(device=device),
            index_of=self.index_of.to(device=device),
            indices=self.indices,
            ids=self.ids,
            payloads=self.payloads,
            args=self.args,
        )

    def get_object(self, idx: int, detach: bool = True) -> Tri:
        s, e = self.indices[idx]
        return Tri(
            xs=maybe_detach(self.xs[s:e, ...], detach),
            id=self.ids[idx],
            payload=self.payloads[idx],
        )

    @classmethod
    def from_object(cls, object: Tri, **kwargs) -> "TriCollection":
        device = object.xs.device
        return cls(
            xs=object.xs.detach().clone(),
            index_of=torch.full((len(object.xs),), 0, dtype=torch.long, device=device),
            indices=((0, len(object.xs)),),
            ids=(object.id,),
            payloads=(object.payload,),
            **kwargs,
        )

    @classmethod
    def cat(cls, collections: list[ObjectCollection[Tri]], **kwargs) -> Self:
        collections_ = cast(list[TriCollection], collections)
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

        a, b, c = self.xs.unbind(dim=1)
        assert a.shape == b.shape == c.shape

        def judge(p, v1, v2):
            ep = p - v1
            ee = v2 - v1

            t = (ep * ee).sum(dim=-1) / ee.square().sum(dim=-1)
            t.clamp_(0, 1)

            dist = (ep - ee * t.unsqueeze(dim=-1)).norm(dim=-1)
            cross = ep[..., 0] * ee[..., 1] - ep[..., 1] * ee[..., 0]

            return dist, cross > 0

        da, ca = judge(positions.unsqueeze(dim=-2), a, b)
        db, cb = judge(positions.unsqueeze(dim=-2), b, c)
        dc, cc = judge(positions.unsqueeze(dim=-2), c, a)

        inside = (ca == cb) & (cb == cc)

        dist = torch.stack([da, db, dc], dim=-1).min(dim=-1).values
        sdf = torch.where(inside, -dist, dist)
        sdf = sdf.permute(2, 0, 1)  # (n_tris, *shape)

        n_shapes = len(self.ids)
        result = torch.full(
            (n_shapes, *shape), torch.inf, dtype=sdf.dtype, device=sdf.device
        )
        index = self.index_of.view(-1, *([1] * len(shape))).expand(-1, *shape)
        return torch.scatter_reduce(
            result, 0, index, sdf, reduce="amin"
        )  # (n_shapes, *shape)

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
            basis = (torch.arange(size, device=device) + 0.5) / size * (
                lim1 - lim0
            ) + lim0  # (size,)
        else:
            basis = torch.linspace(lim0, lim1, size, device=device)  # (size,)
        xs = basis.expand(size, -1)  # (size, size)
        ys = basis.unsqueeze(-1).expand(-1, size)  # (size, size)
        grid = torch.stack([xs, ys], dim=-1)  # (size, size, 2)
        return self.rasterize(grid, offset=offset)

    def render01(
        self,
        size: int,
        lim: tuple[float, float] = (-1.5, 1.5),
        center_pixel: bool = True,
        blur: float = 1.0,
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
        device = self.xs.device
        result = torch.full((len(self.ids),), 0.0, device=device)  # (n_shapes, ...)

        a, b, c = self.xs.unbind(dim=1)
        e1 = b - a
        e2 = c - a
        area = (e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]) / 2
        area.abs_()

        return torch.scatter_reduce(
            result, 0, self.index_of, area, reduce="sum"
        )  # (n_shapes,)

    @classmethod
    def patch_args(cls, args: TriColectionArgs) -> Type["TriCollection"]:
        return type(
            "TriCollectionWithArgs",
            (TriCollection,),
            {"__init__": partialmethod(TriCollection.__init__, args=args)},
        )

    def to_savable(self) -> "TriCollection":
        return TriCollection(
            xs=self.xs,
            index_of=self.index_of,
            indices=self.indices,
            ids=self.ids,
            payloads=self.payloads,
            args=self.args,
        )

    def project_to_valid_(self) -> Self:
        return self
