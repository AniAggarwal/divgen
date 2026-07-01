"""Image saving helpers for EvoDiv (grid + per-image, matching divgen's style)."""

from __future__ import annotations

import math
import os
from typing import List

import torch
from diffusers import DiffusionPipeline
from PIL import Image


def tensor_set_to_pil(images: torch.Tensor) -> List[Image.Image]:
    """(B, 3, H, W) in [0,1] -> list of PIL images."""
    arr = images.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    return DiffusionPipeline.numpy_to_pil(arr)


def save_image_set(images: List[Image.Image], path: str) -> None:
    """Save a set as a square-ish grid plus individual frames (like main.save_image_batch)."""
    if not images:
        return
    root, ext = os.path.splitext(path)
    ext = ext or ".jpg"
    if len(images) == 1:
        images[0].save(root + ext)
        return
    cols = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)
    w, h = images[0].size
    grid = Image.new(images[0].mode, (cols * w, rows * h))
    for idx, im in enumerate(images):
        grid.paste(im, ((idx % cols) * w, (idx // cols) * h))
    grid.save(root + ext)
    for idx, im in enumerate(images):
        im.save(f"{root}_{idx:02d}{ext}")
