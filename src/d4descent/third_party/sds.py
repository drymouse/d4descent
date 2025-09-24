"""
Updated from GenerativeEscherMeshes /escher/guidance/sd.py, which is from
ThreeStudio

https://github.com/Shiriluz/Word-As-Image/blob/ed72b2b33f7b2fecc5aecc610700973af754b2b7/code/losses.py
"""

from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from typing import Union
import kornia.augmentation as K

from ..util import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


@dataclass
class SdConfig:
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
    guidance_scale: float = 100.0
    half_precision_weights: bool = True

    min_step_percent: float = 0.02
    max_step_percent: float = 0.98


class StableDiffusion(nn.Module):
    def __init__(self, config: SdConfig = SdConfig(), device: Union[torch.device, str, None] = None):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading Stable Diffusion ... from {self.config.pretrained_model_name_or_path}")

        self.weights_dtype = torch.float16 if self.config.half_precision_weights else torch.float32
        logger.debug(f"{self.config.half_precision_weights=}")

        # Create SD pipeline
        self.pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=self.weights_dtype,
        ).to(self.device)

        # SD components
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        self.scheduler = self.pipe.scheduler

        # Compute actual min/max step
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.config.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.config.max_step_percent)

        self.alphas: torch.Tensor = self.scheduler.alphas_cumprod.to(self.device)
        self.sigmas = 1 - self.alphas
        self.alphas_sqrt = self.alphas**0.5

        logger.info(f"Loaded Stable Diffusion!")

        self.aug = nn.Sequential(
            K.RandomPerspective(distortion_scale=0.5, p=0.7),
            K.RandomCrop(size=(512, 512), pad_if_needed=True, padding_mode="reflect", p=1.0),
        )

    @torch.amp.autocast(device_type="cuda", enabled=False)  # type: ignore
    def encode_images(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (B, 3, H, W)
        """
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.amp.autocast(device_type="cuda", enabled=False)  # type: ignore
    def decode_latents(
        self,
        latents: torch.Tensor,
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> torch.Tensor:
        input_dtype = latents.dtype
        latents = F.interpolate(latents, (latent_height, latent_width), mode="bilinear", align_corners=False)
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @torch.no_grad()
    def get_text_embeds(self, prompt: str) -> torch.Tensor:
        """
        Computes text embedding for prompt

        Returns: (1, 77, 1024)
        """
        logger.debug(f"{self.tokenizer.model_max_length=}")
        inputs = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
        )
        logger.debug(f"{inputs=}")
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        logger.debug(f"{embeddings.shape=}")
        return embeddings

    @torch.amp.autocast(device_type="cuda", enabled=False)  # type: ignore
    def forward_unet(self, latents: torch.Tensor, t: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype), t.to(self.weights_dtype), encoder_hidden_states.to(self.weights_dtype)
        ).sample.to(input_dtype)

    def compute_sds_loss(self, imgs: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Computes sds loss for images wrt a text embedding

        - imgs: (B, C, H, W)
        - text_embedding (2, 77, 1024)

        Returns: (B,)
        """
        rgb = imgs.expand(-1, 3, -1, -1)
        rgb = self.aug(rgb)
        # 1) Embed images
        batch_size = rgb.shape[0]
        latents = self.encode_images(rgb)
        logger.debug(f"{latents.shape=}")
        # 2) Sample t (t ~ Unif(0.01, 0.98) to avoid very high/low noise level)
        t = torch.randint(self.min_step, self.max_step + 1, [batch_size], dtype=torch.long, device=self.device)
        # 3) Push through UNet
        with torch.no_grad():
            noise = torch.randn_like(latents)
            noisy_latents = self.scheduler.add_noise(latents, noise, t)
            noise_pred = self.forward_unet(
                torch.cat([noisy_latents] * 2, dim=0), torch.cat([t] * 2), encoder_hidden_states=text_embedding
            )

        # Guidance (high scale from paper)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_uncond)

        # TODO: Read derivation from VectorFusion
        grad: torch.Tensor = (self.alphas_sqrt[t] * self.sigmas[t]).reshape(-1, 1, 1, 1) * (noise_pred - noise)
        loss = (grad.detach() * latents).sum(dim=1).mean((-1, -2))
        return loss

        # # Using SDS weighting
        # w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        # grad = w * (noise_pred - noise)
        # grad = torch.nan_to_num(grad)
        # logger.debug(f"{grad.shape=}")

        # # 4) Compute grad
        # target = (latents - grad).detach()
        # loss = 0.5 * F.mse_loss(latents, target, reduction="none").mean(dim=(1, 2, 3))
        # logger.debug(f"{loss.shape=}")
        # return loss
