import torch
from ipycanvas import Canvas
from typing import Optional, Union, overload, Iterable, TypedDict, Callable, TypeVar, Literal, Generic
from IPython.display import display, Image
from ipywidgets import Output
from dataclasses import dataclass
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Arc as MPLArc, Rectangle
from matplotlib.colors import to_rgb
import io
from abc import ABC, abstractmethod
from collections.abc import Sequence
import math


from .objects.prim2 import Shape, vectorized_sample_arc, Line, Arc


@dataclass
class LineStyle:
    color: str = "black"
    linewidth: float = 2
    arrowwidth: Optional[float] = 1


@dataclass
class PointStyle:
    color: str = "red"
    radius: float = 2


class VisualizerAxes(ABC):
    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def render_points(self, points: Iterable[torch.Tensor], style: PointStyle = PointStyle()) -> None: ...

    @abstractmethod
    def render_lines(
        self,
        start_points: torch.Tensor,
        end_points: torch.Tensor,
        style: LineStyle = LineStyle(),
    ) -> None: ...

    @abstractmethod
    def render_arcs(
        self,
        start_points: torch.Tensor,
        end_points: torch.Tensor,
        ks: torch.Tensor,
        style: LineStyle = LineStyle(color="green"),
    ) -> None: ...

    @abstractmethod
    def render_text(self, text: str, position: tuple[float, float], color: str = "black", size: float = 10) -> None: ...

    @abstractmethod
    def invalidate(self) -> None: ...

    def visualize_shape(
        self,
        shape: Shape,
        line_style: LineStyle = LineStyle("black", 2),
        arc_style: LineStyle = LineStyle("green", 2),
        point_style: PointStyle = PointStyle("red", 4),
        show_text: bool = False,
    ):
        for i, prim in enumerate(shape.primitives):
            if isinstance(prim, Line):
                self.render_lines(prim.start.unsqueeze(0), prim.end.unsqueeze(0), line_style)
            elif isinstance(prim, Arc):
                self.render_arcs(prim.start.unsqueeze(0), prim.end.unsqueeze(0), prim.k.unsqueeze(0), arc_style)
            else:
                raise ValueError(f"Unknown primitive type: {type(prim)}")
            if show_text:
                self.render_text(f"{i}", tuple(((prim.start + prim.end) / 2).tolist()), size=10)
        points_: dict[int, torch.Tensor] = {}
        for prim in shape.primitives:
            if isinstance(prim, (Line, Arc)):
                points_[id(prim.start)] = prim.start
                points_[id(prim.end)] = prim.end
        self.render_points(points_.values(), point_style)


VisualizerAxesT = TypeVar("VisualizerAxesT", bound=VisualizerAxes)


class Visualizer(ABC, Generic[VisualizerAxesT]):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: Union[int, tuple[int, int]]) -> VisualizerAxesT: ...

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @abstractmethod
    def display(self) -> None: ...

    @abstractmethod
    def invalidate(self) -> None: ...

    @abstractmethod
    def get_image(self) -> np.ndarray: ...

    def close(self) -> None:
        pass


class MPLVisualizerAxes(VisualizerAxes):
    def __init__(
        self,
        ax: Axes,
        /,
        skip_init: bool = False,
    ):
        self.ax = ax
        self._zorder = 100
        self._skip_init = skip_init

    def clear(self):
        self._zorder = 0
        if self._skip_init:
            self.ax.clear()
            return
        else:
            old_xlim = self.ax.get_xlim()
            old_ylim = self.ax.get_ylim()
            self.ax.clear()
            self.ax.set_xlim(old_xlim)
            self.ax.set_ylim(old_ylim)
            self.ax.set_aspect("equal")
            self.ax.axis("off")

    def render_points(self, points: Iterable[torch.Tensor], style: PointStyle = PointStyle()):
        points = list(points)
        if len(points) == 0:
            return
        x, y = zip(*[p.tolist() for p in points])
        self.ax.scatter(x, y, color=style.color, s=style.radius, zorder=self._zorder)
        self._zorder += 1

    def render_lines(
        self,
        start_points: torch.Tensor,
        end_points: torch.Tensor,
        style: LineStyle = LineStyle(),
    ):
        assert start_points.shape == end_points.shape
        trans = self.ax.transData.transform
        assert self.ax.figure is not None
        ppd = 72.0 / self.ax.figure.dpi
        scale = np.linalg.norm(trans((0, 1)) - trans((0, 0))) * ppd
        width = style.linewidth * (style.arrowwidth or 1) / scale
        head_width = 3 * width
        head_length = 1.5 * head_width
        for start, end in zip(start_points.detach().cpu().numpy(), end_points.detach().cpu().numpy()):
            self.ax.add_line(
                Line2D(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    color=style.color,
                    linewidth=style.linewidth,
                    zorder=self._zorder,
                )
            )
            if style.arrowwidth is not None:
                d = end - start
                norm = np.linalg.norm(d)

                if norm > head_length:
                    d = d / norm * head_length
                    dx, dy = d
                    self.ax.arrow(
                        end[0] - dx,
                        end[1] - dy,
                        dx,
                        dy,
                        width=width,
                        head_width=head_width,
                        head_length=head_length,
                        length_includes_head=True,
                        color=style.color,
                        zorder=self._zorder,
                    )
            self._zorder += 1

    def render_rects(
        self,
        centers: torch.Tensor,
        sizes: torch.Tensor,
        rots: torch.Tensor,
        style: LineStyle = LineStyle(color="black"),
    ):
        for (cx, cy), (sx, sy), rot in zip(centers.tolist(), sizes.tolist(), rots.tolist()):
            self.ax.add_patch(
                Rectangle(
                    (cx - sx, cy - sy),
                    sx * 2,
                    sy * 2,
                    angle=rot * 180 / math.pi,
                    rotation_point="center",
                    color=style.color,
                    linewidth=style.linewidth,
                    fill=False,
                    zorder=self._zorder,
                )
            )
            self._zorder += 1

    def render_arcs(
        self,
        start_points: torch.Tensor,
        end_points: torch.Tensor,
        ks: torch.Tensor,
        style: LineStyle = LineStyle(color="green"),
    ):
        """
        start_points: (B, 2)
        end_points: (B, 2)
        ks: (B,)
        """
        assert start_points.shape == end_points.shape
        assert start_points.shape[0] == ks.shape[0]

        with torch.no_grad():
            midpoints = (start_points + end_points) / 2  # (B, 2)
            dirs = end_points - start_points  # (B, 2)
            norms = torch.norm(dirs, dim=-1, keepdim=True) + 1e-9  # (B ,1)
            perps = torch.stack([-dirs[:, 1], dirs[:, 0]], dim=-1) / norms  # (B, 2)
            ks = ks.unsqueeze(-1)  # (B, 1)
            ks_positive = ks >= 0  # (B, 1)
            r = (ks**2 + (norms / 2) ** 2) / (2 * ks - 1e-4 + 2e-4 * ks_positive)  # (B, 1)
            o = midpoints + (r - ks) * perps  # (B, 2)
            theta0 = torch.atan2(start_points[:, 1] - o[:, 1], start_points[:, 0] - o[:, 0])  # (B,)
            theta1 = torch.atan2(end_points[:, 1] - o[:, 1], end_points[:, 0] - o[:, 0])  # (B,)
            # Adjust theta0 and theta1 based on ks
            ks_positive = ks_positive.squeeze(-1)  # (batch_size,)
            mask1 = ks_positive & (theta0 > theta1)
            mask2 = (~ks_positive) & (theta0 < theta1)
            theta1 = torch.where(mask1, theta1 + 2 * torch.pi, theta1)
            theta0 = torch.where(mask2, theta0 + 2 * torch.pi, theta0)

            o = o.detach().cpu().numpy()
            r = r.abs().detach().cpu().flatten().numpy()
            theta0 = theta0.detach().cpu().flatten().numpy()
            theta1 = theta1.detach().cpu().flatten().numpy()
            ks_ = ks.detach().cpu().flatten().numpy()

        trans = self.ax.transData.transform
        assert self.ax.figure is not None
        ppd = 72.0 / self.ax.figure.dpi
        scale = np.linalg.norm(trans((0, 1)) - trans((0, 0))) * ppd
        width = style.linewidth * (style.arrowwidth or 1) / scale
        head_width = 3 * width
        head_length = 1.5 * head_width

        for o_, r_, theta0_, theta1_, k_, end_point_ in zip(
            o, r, theta0, theta1, ks_, end_points.detach().cpu().numpy()
        ):
            og_theta1 = theta1_
            if k_ < 0:
                theta0_, theta1_ = theta1_, theta0_
            norm = abs(theta1_ - theta0_) * abs(r_)
            self.ax.add_patch(
                MPLArc(
                    (o_[0], o_[1]),
                    width=r_ * 2,
                    height=r_ * 2,
                    theta1=np.rad2deg(theta0_),
                    theta2=np.rad2deg(theta1_),
                    color=style.color,
                    linewidth=style.linewidth,
                    zorder=self._zorder,
                )
            )
            if style.arrowwidth is not None and norm > head_length:
                if head_length > 2 * r_ or abs(k_) < 0.001:
                    dx = -np.sin(og_theta1) * head_length
                    dy = np.cos(og_theta1) * head_length
                    if k_ < 0:
                        dx = -dx
                        dy = -dy
                else:
                    ang = np.arccos(1 - head_length**2 / (2 * r_**2))
                    if k_ < 0:
                        ang = -ang
                    ang = og_theta1 - ang
                    p = o_ + r_ * np.array([np.cos(ang), np.sin(ang)])
                    d = end_point_ - p
                    dx = d[0]
                    dy = d[1]

                self.ax.arrow(
                    end_point_[0] - dx,
                    end_point_[1] - dy,
                    dx,
                    dy,
                    width=width,
                    head_width=head_width,
                    head_length=head_length,
                    length_includes_head=True,
                    color=style.color,
                    zorder=self._zorder,
                )
            self._zorder += 1

    def render_text(self, text: str, position: tuple[float, float], color: str = "black", size: float = 10) -> None:
        self.ax.text(
            position[0],
            position[1],
            text,
            color=color,
            size=size,
            zorder=self._zorder,
            horizontalalignment="center",
            verticalalignment="center",
        )
        self._zorder += 1

    def invalidate(self):
        pass


class MPLVisualizer(Visualizer[MPLVisualizerAxes]):
    def __init__(
        self,
        nrows: int,
        ncols: int,
        width: float,
        height: float,
        xlim: tuple[float, float] = (-1.25, 1.25),
        ylim: tuple[float, float] = (-1.25, 1.25),
        notebook: bool = True,
        skip_init: list[int] = [],
        layout: Literal["constrained", "tight"] = "constrained",
    ):
        fig = Figure(figsize=(width, height), layout=layout)
        axs = fig.subplots(nrows, ncols, squeeze=False)
        axs_: list[Axes] = axs.flatten().tolist()
        skip_init_ = set(skip_init)
        for i, ax in enumerate(axs_):
            if i not in skip_init_:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_aspect("equal")
                ax.axis("off")
        self.nrows = nrows
        self.ncols = ncols
        self.fig = fig
        self.axs = [MPLVisualizerAxes(ax, skip_init=i in skip_init_) for i, ax in enumerate(axs_)]
        self.out = Output() if notebook else None
        self.invalidate()

    def __len__(self) -> int:
        return self.nrows * self.ncols

    def __getitem__(self, idx: Union[int, tuple[int, int]]) -> MPLVisualizerAxes:
        if isinstance(idx, tuple):
            a, b = idx
            idx = a * self.ncols + b
        return self.axs[idx]

    def display(self):
        display(self.out)

    def invalidate(self):
        if self.out is not None:
            self.out.clear_output(wait=True)
            with self.out:
                buf = io.BytesIO()
                self.fig.savefig(buf, format="png")
                display(Image(buf.getvalue()))

    def get_image(self, alpha: bool = False) -> np.ndarray:
        """
        returns: (H, W, 3) uint8 array
        """
        buf = io.BytesIO()
        self.fig.savefig(buf, format="raw", transparent=alpha)
        data = np.frombuffer(buf.getvalue(), dtype=np.uint8)  # type: ignore
        data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
        return _rgba_to_rgb(data) if not alpha else data


def _rgba_to_rgb(rgba: np.ndarray) -> np.ndarray:
    rgba = rgba.astype(np.float32)
    image, alpha = rgba[..., :3], rgba[..., 3:] / 255.0
    image = image * alpha + (1.0 - alpha) * 255.0
    return image.astype(np.uint8)


# for type checking
if __name__ == "__main__":
    viz = MPLVisualizer(1, 1, 4, 4, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
