import torch
from dataclasses import dataclass
from typing import Generic, TypeVar, Optional

from ._base import LossArgs
from ..object_collection import ObjectCollection
from ..tasks._base import Task, ObjectT, RewriteSpecT, StateT, ExtraMetrics


@dataclass
class RasterLossArgs(LossArgs):
    pass


class RasterLossMixin(Task[ObjectT, RewriteSpecT, StateT]):
    def __init__(self, args: RasterLossArgs, target_img: torch.Tensor):
        self.loss_args = args
        self.target_img = target_img

    def _compute_losses(
        self, collection: ObjectCollection[ObjectT], state: StateT
    ) -> tuple[torch.Tensor, ExtraMetrics]:
        imgs = collection.render01(
            self.render_args.size,
            self.render_args.lim,
            center_pixel=self.render_args.center_pixel,
            blur=self.render_args.blur,
        )
        return torch.mean((imgs - self.target_img).square().flatten(-2), dim=-1), {}  # (n_shapes,)
