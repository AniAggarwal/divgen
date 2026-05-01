"""Helpers shared by the two sequential-generation entry points
(`div_utils.sequential.run_single_sequential` and
`training.task_utils._run_sequential_round`).

Both build the DPP early-stop callback the same way, then post-process the
optimized image to compute the sequence-aware DPP score against the running
reference set. Keeping these in one module avoids drift between the two call
sites.
"""

from typing import Callable, Dict, List, Optional, Set

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from training.early_stopping import make_sequential_dpp_stop


# DPP-style sequential objectives share a normalization (B*log(2)) and thus
# the same stop/score logic; they only differ in which trainer objective
# provides the score.
_SEQUENTIAL_DPP_OBJECTIVES = ("diversity_dpp", "diversity_dpp_patch")


def build_sequential_stop_fn(
    ref_objective: str,
    n_images: int,
    available_diversity: Set[str],
    threshold: float = 0.02,
) -> Optional[Callable[[Dict[str, float], int], Optional[str]]]:
    """Build the per-iteration stop callback for sequential generation.

    Returns a sequential-DPP stop function (keyed on the chosen objective)
    when `ref_objective` is one of the DPP-style objectives and that
    objective is enabled; otherwise None.
    """
    if ref_objective in _SEQUENTIAL_DPP_OBJECTIVES and ref_objective in available_diversity:
        return make_sequential_dpp_stop(n_images, threshold=threshold, key=ref_objective)
    return None


def compute_sequential_dpp_reward(
    trainer,
    ref_objective: str,
    reference_tensors: List[torch.Tensor],
    best_image: Image.Image,
) -> Optional[float]:
    """Score the best image against the running reference set using the
    selected DPP-style objective.

    Returns None if the trainer has no matching objective configured.
    """
    target_obj = next(
        (obj for obj in trainer._base_diversity_objectives if obj.name == ref_objective),
        None,
    )
    if target_obj is None:
        return None
    combined = reference_tensors + [TF.to_tensor(best_image)]
    stacked = torch.stack(combined, dim=0).to(target_obj.device)
    with torch.no_grad():
        score = -target_obj(stacked).item()
    return score


def apply_sequential_dpp_correction(
    trainer,
    ref_objective: str,
    reference_tensors: List[torch.Tensor],
    best_image: Image.Image,
    rewards: Dict[str, float],
) -> None:
    """For sequential DPP-style rounds: overwrite `rewards[ref_objective]`
    with the score computed against the full reference set + best_image.
    No-op when `ref_objective` is not a DPP-style objective.
    """
    if ref_objective not in _SEQUENTIAL_DPP_OBJECTIVES:
        return
    score = compute_sequential_dpp_reward(trainer, ref_objective, reference_tensors, best_image)
    if score is not None:
        rewards[ref_objective] = score
