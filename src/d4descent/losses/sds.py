import torch
from dataclasses import dataclass, field
from typing import Optional, Generic, TypeVar, Optional

from ._base import LossArgs
from ..object_collection import ObjectCollection
from ..tasks._base import Task, ObjectT, RewriteSpecT, StateT, ExtraMetrics
from ..third_party.sds import StableDiffusion, SdConfig


@dataclass
class SDSLossArgs(LossArgs):
    prompt: str  # prompt if using SDS loss
    neg_prompt: str = ""  # negative prompt if using SDS loss
    sd: SdConfig = field(default_factory=SdConfig)


class SDSLossMixin(Task[ObjectT, RewriteSpecT, StateT]):
    def __init__(self, args: SDSLossArgs):
        self.loss_args = args
        self.sd = StableDiffusion(args.sd)  # init SD
        # Compute text embedding
        self.text_embedding = torch.cat(
            [self.sd.get_text_embeds(args.prompt), self.sd.get_text_embeds(args.neg_prompt)]
        )

    def _compute_losses(
        self, collection: ObjectCollection[ObjectT], state: StateT
    ) -> tuple[torch.Tensor, ExtraMetrics]:
        imgs = collection.render01(
            self.render_args.size,
            self.render_args.lim,
            center_pixel=self.render_args.center_pixel,
            blur=self.render_args.blur,
        )
        imgs = (1 - imgs).unsqueeze(-3).flip(-2)  # black on white; vertical flip
        text_embedding = self.text_embedding.repeat_interleave(imgs.shape[0], dim=0)
        loss = self.sd.compute_sds_loss(imgs, text_embedding)
        return loss, {}
