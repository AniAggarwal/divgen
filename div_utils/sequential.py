"""Helpers for sequential single-image optimization."""
import json
import os
import shutil
from typing import Dict, List, Optional, Set, Tuple

import logging
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from training import (
    disable_diversity_context,
    apply_reference_diversity_context,
    reset_diversity_context,
    get_optimizer,
)
from training.noise_utils import generate_latents
from training.sequential_utils import (
    apply_sequential_dpp_correction,
    build_sequential_stop_fn,
)


def run_single_sequential(
    cfg,
    trainer,
    device: torch.device,
    dtype: torch.dtype,
    single_sample_shape: Tuple[int, ...],
    multi_apply_fn,
    save_image_batch_fn,
    save_dir: str,
) -> Tuple[List[Image.Image], List[Image.Image], Dict[str, float], Dict[str, float]]:
    os.makedirs(save_dir, exist_ok=True)
    sequential_count = max(0, cfg.optimization.sequential_diverse_samples)
    if sequential_count <= 0:
        raise ValueError("run_single_sequential requires sequential_diverse_samples > 0")

    available_diversity = {obj.name for obj in trainer.diversity_objectives}
    ref_objective = cfg.optimization.sequential_diversity_objective
    if sequential_count > 1 and ref_objective not in available_diversity:
        raise ValueError(
            f"Sequential diversity objective '{ref_objective}' is not enabled. "
            "Please enable the corresponding diversity flag."
        )

    reference_objectives: Optional[Set[str]] = {ref_objective} if ref_objective else None
    reference_tensors: List[torch.Tensor] = []
    all_init_images: List[Image.Image] = []
    all_best_images: List[Image.Image] = []
    init_reward_list: List[Dict[str, float]] = []
    best_reward_list: List[Dict[str, float]] = []
    sample_records: List[Dict] = []

    for seq_idx in range(sequential_count):
        seed_offset = cfg.optimization.seed + seq_idx
        seq_latents = generate_latents(
            single_sample_shape,
            device,
            dtype,
            noise_type=cfg.optimization.noise_type,
            seed=seed_offset,
            noise_exponent=cfg.optimization.noise_exponent,
            flux_schnell_pack=(cfg.model.name == "flux-schnell"),
        )
        latents = torch.nn.Parameter(seq_latents, requires_grad=True)
        if seq_idx == 0:
            disable_diversity_context(trainer)
            generator_device = trainer.device if hasattr(trainer, "device") else "cuda"
            generator = torch.Generator(device=str(generator_device)).manual_seed(seed_offset)
            with torch.no_grad():
                initial_tensor = trainer.model.apply(
                    latents=seq_latents,
                    prompt=cfg.task.prompt,
                    generator=generator,
                    num_inference_steps=trainer.n_inference_steps,
                    num_images_per_prompt=seq_latents.shape[0],
                )
            if multi_apply_fn is not None:
                init_batch = multi_apply_fn(seq_latents.detach(), cfg.task.prompt)
            else:
                init_batch = initial_tensor
            seq_init_images = trainer._tensor_to_pil(init_batch)
            seq_best_images = seq_init_images
            seq_init_rewards = trainer._compute_all_metrics(init_batch, cfg.task.prompt)
            seq_best_rewards = seq_init_rewards
            if ref_objective:
                seq_best_rewards[ref_objective] = 0.0
        else:
            optimizer = get_optimizer(cfg.optimization.optim, latents, cfg.optimization.lr)
            dpp_stop_fn = build_sequential_stop_fn(
                ref_objective, len(reference_tensors) + 1, available_diversity
            )
            apply_reference_diversity_context(
                trainer, reference_tensors, reference_objectives
            )
            (
                seq_init_images,
                seq_best_images,
                seq_init_rewards,
                seq_best_rewards,
                _,
            ) = trainer.train(
                latents,
                cfg.task.prompt,
                optimizer,
                save_dir,
                multi_apply_fn,
                extra_stop_fn=dpp_stop_fn,
                skip_final_metrics_log=True,
                hps_warmup_iters=cfg.rewards.hps.warmup_iters,
            )
            apply_sequential_dpp_correction(
                trainer, ref_objective, reference_tensors,
                seq_best_images[0], seq_best_rewards,
            )
            # Log the corrected best metrics
            logging.info(f"Best metrics: {seq_best_rewards}")

        sample_dir = f"{save_dir}/sample_{seq_idx:02d}"
        os.makedirs(sample_dir, exist_ok=True)
        save_image_batch_fn(seq_init_images, f"{sample_dir}/init.jpg")
        save_image_batch_fn(seq_best_images, f"{sample_dir}/best.jpg")

        sample_records.append(
            {
                "sample_index": seq_idx,
                "seed": seed_offset,
                "initial_rewards": seq_init_rewards,
                "best_rewards": seq_best_rewards,
                "references_used": seq_idx,
                "diversity_objective": ref_objective if seq_idx > 0 else None,
            }
        )
        all_init_images.extend(seq_init_images)
        all_best_images.extend(seq_best_images)
        init_reward_list.append(seq_init_rewards)
        best_reward_list.append(seq_best_rewards)
        reference_tensors.append(TF.to_tensor(seq_best_images[0]))
        
        # Compute metrics on accumulated image sets (not just averages of per-sample metrics)
        if len(all_init_images) > 1:
            init_tensors_running = torch.stack([TF.to_tensor(img) for img in all_init_images], dim=0)
            best_tensors_running = torch.stack([TF.to_tensor(img) for img in all_best_images], dim=0)
            init_tensors_running = init_tensors_running.to(device=device, dtype=dtype)
            best_tensors_running = best_tensors_running.to(device=device, dtype=dtype)
            running_init = trainer._compute_all_metrics(init_tensors_running, cfg.task.prompt)
            running_best = trainer._compute_all_metrics(best_tensors_running, cfg.task.prompt)
            del init_tensors_running, best_tensors_running
            torch.cuda.empty_cache()
        else:
            running_init = seq_init_rewards
            running_best = seq_best_rewards
        logging.info(
            f"[Sequential] After sample {seq_idx}: "
            f"initial_metrics={running_init} | best_metrics={running_best}"
        )

    reset_diversity_context(trainer)
    save_image_batch_fn(all_best_images, f"{save_dir}/best_image.jpg")
    save_image_batch_fn(all_init_images, f"{save_dir}/init_image.jpg")
    
    # Compute comprehensive metrics on the full set of images
    init_tensors = torch.stack([TF.to_tensor(img) for img in all_init_images], dim=0).to(device=device, dtype=dtype)
    best_tensors = torch.stack([TF.to_tensor(img) for img in all_best_images], dim=0).to(device=device, dtype=dtype)
    
    total_init_rewards = trainer._compute_all_metrics(init_tensors, cfg.task.prompt)
    total_best_rewards = trainer._compute_all_metrics(best_tensors, cfg.task.prompt)
    
    del init_tensors, best_tensors
    torch.cuda.empty_cache()
    
    # Save results to JSON
    results = {
        "per_sample_results": sample_records,
        "final_metrics": {
            "initial": total_init_rewards,
            "best": total_best_rewards,
            "num_images": len(all_init_images),
        },
    }
    
    with open(f"{save_dir}/sequential_results.json", "w") as fp:
        json.dump(results, fp, indent=2)

    # Clean up per-sample subdirectories now that the combined grids are saved.
    for record in sample_records:
        sample_dir = f"{save_dir}/sample_{record['sample_index']:02d}"
        shutil.rmtree(sample_dir, ignore_errors=True)

    return all_init_images, all_best_images, total_init_rewards, total_best_rewards

