"""Utility functions for processing benchmark tasks."""

import json
import logging
import os
from typing import Dict, List, Optional, Set, Tuple, Callable

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from training import get_optimizer
from training.noise_utils import generate_latents
from training.early_stopping import make_sequential_dpp_stop
from training.sequential_utils import (
    apply_sequential_dpp_correction,
    build_sequential_stop_fn,
)


def disable_diversity_context(trainer) -> None:
    """Disable diversity objectives for the trainer (e.g., first sequential sample)."""
    trainer.diversity_objectives = []
    trainer.diversity_reference_images = None
    trainer.diversity_reference_objectives = None


def apply_reference_diversity_context(trainer, reference_tensors, reference_objectives) -> None:
    """Enable diversity with provided reference tensors/objectives."""
    trainer.diversity_objectives = trainer._base_diversity_objectives
    if reference_tensors:
        reference_stack = torch.stack(reference_tensors, dim=0)
        trainer.diversity_reference_images = reference_stack.detach().cpu()
    else:
        trainer.diversity_reference_images = None
    trainer.diversity_reference_objectives = (
        set(reference_objectives) if reference_objectives is not None else None
    )


def reset_diversity_context(trainer) -> None:
    """Restore base diversity objectives without references."""
    trainer.diversity_objectives = trainer._base_diversity_objectives
    trainer.diversity_reference_images = None
    trainer.diversity_reference_objectives = None


def process_prompt_batch(
    prompts_metadata: List[Dict],
    trainer,
    shape: Tuple,
    device: torch.device,
    dtype: torch.dtype,
    cfg,
    outdir: str,
    save_image_fn: Callable,
    multi_apply_fn=None,
    start_idx: int = 0,
    subset_name: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float], List[List[Dict]]]:
    """
    Process a batch of prompts with the trainer.
    
    Args:
        prompts_metadata: List of metadata dicts with 'prompt' key
        trainer: LatentNoiseTrainer instance
        shape: Shape of latents
        device: Device for computation
        dtype: Data type for latents
        cfg: Typed Config (see config.py)
        outdir: Output directory for saving results
        save_image_fn: Function to save images (images, path) -> None
        multi_apply_fn: Optional multi-step refinement function
        start_idx: Starting index for output paths
        subset_name: Optional subset name for logging
        
    Returns:
        total_best_rewards: Accumulated best rewards
        total_init_rewards: Accumulated initial rewards
        all_iteration_histories: Per-iteration metrics for each prompt
    """
    total_best_rewards = {}
    total_init_rewards = {}
    all_iteration_histories = []
    
    # Save output directory path for SLURM merge script
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, ".outdir_path"), "w") as f:
        f.write(os.path.abspath(outdir))
    
    for i, metadata in enumerate(prompts_metadata):
        index = start_idx + i
        
        # Initialize latents and optimizer
        # Use seed + index for each prompt to ensure different but reproducible initialization
        init_latents = generate_latents(
            shape, device, dtype,
            noise_type=cfg.optimization.noise_type,
            seed=cfg.optimization.seed + index,
            noise_exponent=cfg.optimization.noise_exponent,
            flux_schnell_pack=(cfg.model.name == "flux-schnell"),
        )
        latents = torch.nn.Parameter(init_latents, requires_grad=True)
        optimizer = get_optimizer(cfg.optimization.optim, latents, cfg.optimization.lr)
        
        prompt = metadata["prompt"]
        
        # Create output directory
        outpath = f"{outdir}/{index:0>5}"
        os.makedirs(f"{outpath}/samples", exist_ok=True)
        
        # Set up stopping criteria
        dpp_stop_fn = None
        if cfg.stopping.enable_dpp_stopping:
            available_diversity = {obj.name for obj in trainer._base_diversity_objectives}
            if "diversity_dpp" in available_diversity and shape[0] > 1:
                if cfg.stopping.dpp_stopping_absolute is not None:
                    from training.early_stopping import make_dpp_absolute_stop
                    dpp_stop_fn = make_dpp_absolute_stop(threshold=cfg.stopping.dpp_stopping_absolute)
                elif cfg.stopping.dpp_stopping_multiplier is not None:
                    from training.early_stopping import make_dpp_multiplier_stop
                    dpp_stop_fn = make_dpp_multiplier_stop(multiplier=cfg.stopping.dpp_stopping_multiplier)
                else:
                    dpp_stop_fn = make_sequential_dpp_stop(shape[0], threshold=cfg.stopping.dpp_stopping_threshold)

        # HPS stopping/revert parameters
        hps_sample_threshold = None
        hps_absolute_threshold = None
        hps_revert_mode = False
        if cfg.rewards.hps.enable_stopping:
            hps_sample_threshold = cfg.rewards.hps.stopping_threshold
            hps_absolute_threshold = cfg.rewards.hps.absolute_threshold
            hps_revert_mode = cfg.rewards.hps.enable_revert
        
        # Run optimization
        # Note: all_diversity_objectives are already loaded in main.py for comprehensive logging
        (
            init_images,
            best_images,
            init_rewards,
            best_rewards,
            iteration_history,
        ) = trainer.train(
            latents, prompt, optimizer, None, multi_apply_fn,
            return_history=True,
            extra_stop_fn=dpp_stop_fn,
            hps_sample_threshold=hps_sample_threshold,
            hps_absolute_threshold=hps_absolute_threshold,
            hps_revert_mode=hps_revert_mode,
            hps_warmup_iters=cfg.rewards.hps.warmup_iters,
        )
        
        # Store iteration history
        if iteration_history:
            all_iteration_histories.append(iteration_history)
        
        # Log results
        prefix = f"Subset '{subset_name}', " if subset_name else ""
        logging.info(f"{prefix}Prompt {index}: {prompt}")
        logging.info(f"Initial rewards: {init_rewards}")
        logging.info(f"Best rewards: {best_rewards}")
        
        # Save metadata
        with open(f"{outpath}/metadata.jsonl", "w") as fp:
            json.dump(metadata, fp)
        
        # Save per-prompt results JSON
        prompt_results = {
            "prompt_index": index,
            "prompt": prompt,
            "seed": cfg.optimization.seed + index,
            "initial_rewards": init_rewards,
            "best_rewards": best_rewards,
        }
        
        # Add subset info if available
        if subset_name:
            prompt_results["subset"] = subset_name
        
        # Add iteration count and history if available
        if iteration_history:
            prompt_results["num_iterations"] = len(iteration_history)
            # Optionally save full iteration history (can be large)
            if cfg.logging.save_iteration_history:
                prompt_results["iteration_history"] = iteration_history
        
        # Add metadata from original prompt metadata
        prompt_results["metadata"] = metadata
        
        # Save results JSON for this prompt
        with open(f"{outpath}/results.json", "w") as fp:
            json.dump(prompt_results, fp, indent=2)
        
        # Save images using provided function
        save_image_fn(init_images, f"{outpath}/samples/init_{cfg.optimization.seed:05}.jpg")
        save_image_fn(best_images, f"{outpath}/samples/{cfg.optimization.seed:05}.jpg")
        
        # Clean up image objects to free memory
        del init_images, best_images
        torch.cuda.empty_cache()
        
        # Accumulate rewards
        if i == 0:
            total_best_rewards = {k: 0.0 for k in best_rewards.keys()}
            total_init_rewards = {k: 0.0 for k in init_rewards.keys()}
        for k in best_rewards.keys():
            total_best_rewards[k] += best_rewards[k]
            total_init_rewards[k] += init_rewards[k]
        
        # Explicit memory cleanup between prompts
        # Note: init_images and best_images already deleted after saving
        del latents, optimizer
        torch.cuda.empty_cache()
        
        # Clear text feature caches in reward models
        for reward_loss in trainer.reward_losses:
            if hasattr(reward_loss, 'clear_text_cache'):
                reward_loss.clear_text_cache()
        for reward_loss in trainer.all_reward_losses:
            if hasattr(reward_loss, 'clear_text_cache'):
                reward_loss.clear_text_cache()
        
        # Clear diversity objective backend caches
        for diversity_obj in trainer.diversity_objectives:
            if hasattr(diversity_obj, 'dreamsim_backend') and diversity_obj.dreamsim_backend is not None:
                diversity_obj.dreamsim_backend.clear_cache()
            if hasattr(diversity_obj, 'dino_backend') and diversity_obj.dino_backend is not None:
                diversity_obj.dino_backend.clear_cache()
            if hasattr(diversity_obj, 'sscd_backend') and diversity_obj.sscd_backend is not None:
                diversity_obj.sscd_backend.clear_cache()
        for diversity_obj in trainer.all_diversity_objectives:
            if hasattr(diversity_obj, 'dreamsim_backend') and diversity_obj.dreamsim_backend is not None:
                diversity_obj.dreamsim_backend.clear_cache()
            if hasattr(diversity_obj, 'dino_backend') and diversity_obj.dino_backend is not None:
                diversity_obj.dino_backend.clear_cache()
            if hasattr(diversity_obj, 'sscd_backend') and diversity_obj.sscd_backend is not None:
                diversity_obj.sscd_backend.clear_cache()
        
        # Additional cleanup for between-prompt memory
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    return total_best_rewards, total_init_rewards, all_iteration_histories


def process_prompt_sequential(
    prompt: str,
    metadata: Dict,
    trainer,
    device: torch.device,
    dtype: torch.dtype,
    cfg,
    outpath: str,
    save_image_fn: Callable,
    single_sample_shape: Tuple,
    multi_apply_fn=None,
    prompt_index: int = 0,
) -> Tuple[List[Image.Image], List[Image.Image], Dict[str, float], Dict[str, float], List[Dict]]:
    """
    Process a single prompt with sequential diverse sampling.
    
    Args:
        single_sample_shape: Shape for single sample latents (1, channels, h, w) or (1, seq_len, dim)
    
    Returns:
        all_init_images: List of initial images
        all_best_images: List of best images
        total_init_rewards: Metrics computed on full init set
        total_best_rewards: Metrics computed on full best set
        sample_records: Per-sample results
    """
    sequential_count = max(0, cfg.optimization.sequential_diverse_samples)
    if sequential_count <= 0:
        raise ValueError("process_prompt_sequential requires sequential_diverse_samples > 0")

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
        seed_offset = cfg.optimization.seed + prompt_index * 1000 + seq_idx  # Unique seed per prompt+sample
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
            # First sample: no diversity optimization, just generate
            disable_diversity_context(trainer)
            generator_device = trainer.device if hasattr(trainer, "device") else "cuda"
            generator = torch.Generator(device=str(generator_device)).manual_seed(seed_offset)
            with torch.no_grad():
                initial_tensor = trainer.model.apply(
                    latents=seq_latents,
                    prompt=prompt,
                    generator=generator,
                    num_inference_steps=trainer.n_inference_steps,
                    num_images_per_prompt=seq_latents.shape[0],
                )
            if multi_apply_fn is not None:
                init_batch = multi_apply_fn(seq_latents.detach(), prompt)
            else:
                init_batch = initial_tensor
            seq_init_images = trainer._tensor_to_pil(init_batch)
            seq_best_images = seq_init_images
            seq_init_rewards = trainer._compute_all_metrics(init_batch, prompt)
            seq_best_rewards = seq_init_rewards.copy()
            if ref_objective:
                seq_best_rewards[ref_objective] = 0.0
        else:
            # Subsequent samples: optimize with diversity
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
                prompt,
                optimizer,
                outpath,
                multi_apply_fn,
                extra_stop_fn=dpp_stop_fn,
                skip_final_metrics_log=True,
                hps_warmup_iters=cfg.rewards.hps.warmup_iters,
            )
            apply_sequential_dpp_correction(
                trainer, ref_objective, reference_tensors,
                seq_best_images[0], seq_best_rewards,
            )

        # Save per-sample images
        sample_dir = f"{outpath}/sample_{seq_idx:02d}"
        os.makedirs(sample_dir, exist_ok=True)
        save_image_fn(seq_init_images, f"{sample_dir}/init.jpg")
        save_image_fn(seq_best_images, f"{sample_dir}/best.jpg")

        sample_records.append({
            "sample_index": seq_idx,
            "seed": seed_offset,
            "initial_rewards": seq_init_rewards,
            "best_rewards": seq_best_rewards,
            "references_used": seq_idx,
            "diversity_objective": ref_objective if seq_idx > 0 else None,
        })

        all_init_images.extend(seq_init_images)
        all_best_images.extend(seq_best_images)
        init_reward_list.append(seq_init_rewards)
        best_reward_list.append(seq_best_rewards)
        reference_tensors.append(TF.to_tensor(seq_best_images[0]))

    reset_diversity_context(trainer)
    
    # Compute comprehensive metrics on the full set of images
    init_tensors = torch.stack([TF.to_tensor(img) for img in all_init_images], dim=0).to(device=device, dtype=dtype)
    best_tensors = torch.stack([TF.to_tensor(img) for img in all_best_images], dim=0).to(device=device, dtype=dtype)
    
    total_init_rewards = trainer._compute_all_metrics(init_tensors, prompt)
    total_best_rewards = trainer._compute_all_metrics(best_tensors, prompt)
    
    del init_tensors, best_tensors
    torch.cuda.empty_cache()
    
    return all_init_images, all_best_images, total_init_rewards, total_best_rewards, sample_records


def process_prompt_batch_sequential(
    prompts_metadata: List[Dict],
    trainer,
    shape: Tuple,
    device: torch.device,
    dtype: torch.dtype,
    cfg,
    outdir: str,
    save_image_fn: Callable,
    multi_apply_fn=None,
    start_idx: int = 0,
    subset_name: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float], List[List[Dict]]]:
    """
    Process a batch of prompts with sequential diverse sampling.
    
    Each prompt generates multiple images sequentially with diversity constraints.
    
    Args:
        prompts_metadata: List of metadata dicts with 'prompt' key
        trainer: LatentNoiseTrainer instance
        shape: Shape of latents - used to derive single_sample_shape as (1,) + shape[1:]
        device: Device for computation
        dtype: Data type for latents
        cfg: Typed Config (see config.py) (must have sequential_diverse_samples > 0)
        outdir: Output directory for saving results
        save_image_fn: Function to save images (images, path) -> None
        multi_apply_fn: Optional multi-step refinement function
        start_idx: Starting index for output paths
        subset_name: Optional subset name for logging
        
    Returns:
        total_best_rewards: Accumulated best rewards (averaged)
        total_init_rewards: Accumulated initial rewards (averaged)
        all_iteration_histories: Empty list (sequential doesn't use iteration histories the same way)
    """
    # Derive single sample shape from batch shape
    single_sample_shape = (1,) + tuple(shape[1:])
    total_best_rewards = {}
    total_init_rewards = {}
    all_iteration_histories = []
    
    # Save output directory path for SLURM merge script
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, ".outdir_path"), "w") as f:
        f.write(os.path.abspath(outdir))
    
    for i, metadata in enumerate(prompts_metadata):
        index = start_idx + i
        prompt = metadata["prompt"]
        
        # Create output directory
        outpath = f"{outdir}/{index:0>5}"
        os.makedirs(f"{outpath}/samples", exist_ok=True)
        
        # Run sequential optimization for this prompt
        (
            init_images,
            best_images,
            init_rewards,
            best_rewards,
            sample_records,
        ) = process_prompt_sequential(
            prompt=prompt,
            metadata=metadata,
            trainer=trainer,
            device=device,
            dtype=dtype,
            cfg=cfg,
            outpath=outpath,
            save_image_fn=save_image_fn,
            single_sample_shape=single_sample_shape,
            multi_apply_fn=multi_apply_fn,
            prompt_index=index,
        )
        
        # Log results
        prefix = f"Subset '{subset_name}', " if subset_name else ""
        logging.info(f"{prefix}Prompt {index}: {prompt}")
        logging.info(f"Initial rewards (full set): {init_rewards}")
        logging.info(f"Best rewards (full set): {best_rewards}")
        
        # Save metadata
        with open(f"{outpath}/metadata.jsonl", "w") as fp:
            json.dump(metadata, fp)
        
        # Save per-prompt results JSON
        prompt_results = {
            "prompt_index": index,
            "prompt": prompt,
            "seed": cfg.optimization.seed,
            "sequential_samples": cfg.optimization.sequential_diverse_samples,
            "initial_rewards": init_rewards,
            "best_rewards": best_rewards,
            "per_sample_results": sample_records,
        }
        
        if subset_name:
            prompt_results["subset"] = subset_name
        prompt_results["metadata"] = metadata
        
        with open(f"{outpath}/results.json", "w") as fp:
            json.dump(prompt_results, fp, indent=2)
        
        # Save combined images
        save_image_fn(init_images, f"{outpath}/samples/init_{cfg.optimization.seed:05}.jpg")
        save_image_fn(best_images, f"{outpath}/samples/{cfg.optimization.seed:05}.jpg")
        
        # Clean up
        del init_images, best_images
        torch.cuda.empty_cache()
        
        # Accumulate rewards
        if i == 0:
            total_best_rewards = {k: 0.0 for k in best_rewards.keys()}
            total_init_rewards = {k: 0.0 for k in init_rewards.keys()}
        for k in best_rewards.keys():
            total_best_rewards[k] += best_rewards[k]
        for k in init_rewards.keys():
            total_init_rewards[k] += init_rewards[k]
        
        # Memory cleanup
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    return total_best_rewards, total_init_rewards, all_iteration_histories

