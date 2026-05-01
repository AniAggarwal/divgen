import logging
import os
from typing import Any
import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    Transformer2DModel,
)

from models.RewardPixart import RewardPixartPipeline
from models.RewardStableDiffusionXL import RewardStableDiffusionXL
from models.RewardFlux import RewardFluxPipeline
from models.RewardFlux2Klein import RewardFlux2KleinPipeline
from models.utils import freeze_params


def get_model(
    model_name: str,
    dtype: torch.dtype,
    device: torch.device,
    cache_dir: str,
    enable_sequential_cpu_offload: bool = False,
):
    logging.info(f"Loading model: {model_name}")
    if model_name == "sdxl-turbo":
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
        )
        pipe = RewardStableDiffusionXL.from_pretrained(
            "stabilityai/sdxl-turbo",
            vae=vae,
            torch_dtype=dtype,
            variant="fp16",
            use_safetensors=True,
            cache_dir=cache_dir,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
        pipe = pipe.to(device, dtype)
    elif model_name == "pixart":
        pipe = RewardPixartPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            torch_dtype=dtype,
            cache_dir=cache_dir,
        )
        pipe.transformer = Transformer2DModel.from_pretrained(
            "PixArt-alpha/PixArt-Alpha-DMD-XL-2-512x512",
            subfolder="transformer",
            torch_dtype=dtype,
            cache_dir=cache_dir,
        )
        pipe.scheduler = DDPMScheduler.from_pretrained(
            "PixArt-alpha/PixArt-Alpha-DMD-XL-2-512x512",
            subfolder="scheduler",
            cache_dir=cache_dir,
        )

        # speed-up T5 (optional optimization)
        #pipe.text_encoder.to_bettertransformer()
        
        pipe.transformer.eval()
        freeze_params(pipe.transformer.parameters())
        pipe.transformer.enable_gradient_checkpointing()
        pipe = pipe.to(device)
    elif model_name == "flux-schnell":
        # Get HuggingFace token from environment variable
        hf_token = os.environ.get("HF_TOKEN", None)
        pipe = RewardFluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            token=hf_token,
        )
        pipe.to(device, torch.bfloat16)
    elif model_name == "flux-klein":
        hf_token = os.environ.get("HF_TOKEN", None)
        pipe = RewardFlux2KleinPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B",
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            token=hf_token,
        )
        pipe.to(device, torch.bfloat16)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    if enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    return pipe


def get_multi_apply_fn(seed: int, pipe: Any):
    """Multi-step rendering helper for flux schnell. Renders the initial and
    best images at 4 inference steps even when optimization runs at 1 step."""
    generator = torch.Generator("cuda").manual_seed(seed)

    @torch.no_grad()
    def _multi_apply(latents, prompt):
        return pipe.apply(
            latents=latents,
            prompt=prompt,
            num_inference_steps=4,
            generator=generator,
            num_images_per_prompt=latents.shape[0],
        )

    return _multi_apply
