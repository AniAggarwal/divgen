from typing import Any, List

import torch
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, Resize)
from transformers import AutoProcessor

from rewards.base_reward import BaseRewardLoss
from rewards.clip import CLIPLoss
from rewards.clip_b32 import CLIPB32Loss
from rewards.hps import HPSLoss
from rewards.imagereward import ImageRewardLoss
from rewards.pickscore import PickScoreLoss


def get_reward_losses(
    cfg: Any, dtype: torch.dtype, device: torch.device, cache_dir: str
) -> List[BaseRewardLoss]:
    # Load tokenizer for CLIP H/14 (used by CLIP and PickScore)
    if cfg.rewards.clip.enable or cfg.rewards.pickscore.enable:
        tokenizer = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=cache_dir
        )

    # Load tokenizer for CLIP B/32 (used by CLIP_b32)
    if cfg.rewards.clip_b32.enable:
        tokenizer_b32 = AutoProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir=cache_dir
        )

    reward_losses = []
    if cfg.rewards.hps.enable:
        reward_losses.append(
            HPSLoss(cfg.rewards.hps.weighting, dtype, device, cache_dir)
        )
    if cfg.rewards.imagereward.enable:
        reward_losses.append(
            ImageRewardLoss(
                cfg.rewards.imagereward.weighting,
                dtype,
                device,
                cache_dir,
            )
        )
    if cfg.rewards.clip.enable:
        reward_losses.append(
            CLIPLoss(
                cfg.rewards.clip.weighting,
                dtype,
                device,
                cache_dir,
                tokenizer,
            )
        )
    if cfg.rewards.clip_b32.enable:
        reward_losses.append(
            CLIPB32Loss(
                cfg.rewards.clip_b32.weighting,
                dtype,
                device,
                cache_dir,
                tokenizer_b32,
            )
        )
    if cfg.rewards.pickscore.enable:
        reward_losses.append(
            PickScoreLoss(
                cfg.rewards.pickscore.weighting,
                dtype,
                device,
                cache_dir,
                tokenizer,
            )
        )
    return reward_losses


def get_all_reward_losses(
    dtype: torch.dtype, device: torch.device, cache_dir: str
) -> List[BaseRewardLoss]:
    """
    Create instances of ALL available reward losses for comprehensive metric logging.
    These are used for logging purposes only (computed without gradients).

    Includes: HPS, CLIP H/14, and CLIP B/32
    (ImageReward and PickScore are excluded to save computation)

    Args:
        dtype: Data type for models
        device: Device for computation
        cache_dir: Cache directory for model weights

    Returns:
        List of reward loss instances (with weighting=0 since not used for optimization)
    """
    # Load tokenizer for CLIP H/14
    tokenizer = AutoProcessor.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=cache_dir
    )

    # Load tokenizer for CLIP B/32
    tokenizer_b32 = AutoProcessor.from_pretrained(
        "openai/clip-vit-base-patch32", cache_dir=cache_dir
    )

    all_losses = []

    # Create reward losses (weighting=0 since only for logging)
    all_losses.append(HPSLoss(0.0, dtype, device, cache_dir))
    all_losses.append(CLIPLoss(0.0, dtype, device, cache_dir, tokenizer))
    all_losses.append(CLIPB32Loss(0.0, dtype, device, cache_dir, tokenizer_b32))

    return all_losses


def get_all_reward_losses_shared(
    enabled_losses: List[BaseRewardLoss],
    dtype: torch.dtype,
    device: torch.device,
    cache_dir: str,
) -> List[BaseRewardLoss]:
    """
    Create instances of reward losses for comprehensive metric logging,
    reusing already-loaded models when possible.

    Args:
        enabled_losses: List of reward losses already loaded for optimization
        dtype: Data type for models
        device: Device for computation
        cache_dir: Cache directory for model weights

    Returns:
        List of reward loss instances for comprehensive logging
    """
    import logging

    # Get names of already-enabled losses
    enabled_names = {loss.name for loss in enabled_losses}

    # Required losses for logging: only HPS (CLIP models are too heavy for benchmarks)
    required = {'HPS'}
    needed = required - enabled_names

    if not needed:
        logging.info(f"All reward losses already loaded ({len(enabled_losses)}), reusing for logging")
        return enabled_losses

    # Start with enabled losses (reuse them)
    all_losses = list(enabled_losses)

    logging.info(f"Reusing {len(enabled_losses)} enabled losses, loading {len(needed)} additional: {needed}")

    # Load tokenizers only if needed
    tokenizer = None
    tokenizer_b32 = None

    if 'CLIP' in needed:
        tokenizer = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=cache_dir
        )
        all_losses.append(CLIPLoss(0.0, dtype, device, cache_dir, tokenizer))

    if 'CLIP_b32' in needed:
        tokenizer_b32 = AutoProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir=cache_dir
        )
        all_losses.append(CLIPB32Loss(0.0, dtype, device, cache_dir, tokenizer_b32))

    if 'HPS' in needed:
        all_losses.append(HPSLoss(0.0, dtype, device, cache_dir))

    return all_losses


def clip_img_transform(size: int = 224):
    return Compose(
        [
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
