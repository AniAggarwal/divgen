import logging
import math
import os

import blobfile as bf
import torch
from PIL import Image
from pytorch_lightning import seed_everything
from typing import List

from config import Config
from models import get_model, get_multi_apply_fn
from rewards import get_reward_losses, get_all_reward_losses_shared
from objectives import get_diversity_objectives, get_additional_logging_objectives
from training import (
    LatentNoiseTrainer,
    get_optimizer,
    process_prompt_batch,
    process_prompt_batch_sequential,
    generate_latents,
    make_dpp_absolute_stop,
    make_dpp_multiplier_stop,
    make_sequential_dpp_stop,
)
from div_utils.sequential import run_single_sequential


def save_image_batch(images, path):
    if not images:
        return
    root, ext = os.path.splitext(path)
    if not ext:
        ext = ".jpg"
    if len(images) == 1:
        images[0].save(root + ext)
    else:
        grid_cols = math.ceil(math.sqrt(len(images)))
        grid_rows = math.ceil(len(images) / grid_cols)
        width, height = images[0].size
        mode = images[0].mode
        grid_image = Image.new(mode, (grid_cols * width, grid_rows * height))
        for idx, image in enumerate(images):
            row = idx // grid_cols
            col = idx % grid_cols
            grid_image.paste(image, (col * width, row * height))
        grid_image.save(root + ext)
        for idx, image in enumerate(images):
            image.save(f"{root}_{idx:02d}{ext}")


def main(cfg):
    seed_everything(cfg.optimization.seed)
    # Clear GPU cache before starting
    torch.cuda.empty_cache()
    bf.makedirs(f"{cfg.paths.save_dir}/logs/{cfg.task.type}")
    # Set up logging and name settings
    logger = logging.getLogger()
    # Build rewards string
    rewards_str = ""
    if cfg.rewards.hps.enable:
        rewards_str += f"_hps{cfg.rewards.hps.weighting}"
    if cfg.rewards.imagereward.enable:
        rewards_str += f"_imr{cfg.rewards.imagereward.weighting}"
    if cfg.rewards.clip.enable:
        rewards_str += f"_clip{cfg.rewards.clip.weighting}"
    if cfg.rewards.clip_b32.enable:
        rewards_str += f"_clipb32_{cfg.rewards.clip_b32.weighting}"
    if cfg.rewards.pickscore.enable:
        rewards_str += f"_pick{cfg.rewards.pickscore.weighting}"

    # Build diversity objectives string
    div_str = ""
    if cfg.diversity.lpips.enable:
        div_str += f"_lpips{cfg.diversity.lpips.weight}"
    if cfg.diversity.dino.enable:
        div_str += f"_dino{cfg.diversity.dino.weight}"
    if cfg.diversity.dreamsim.enable:
        div_str += f"_dream{cfg.diversity.dreamsim.weight}"
    if cfg.diversity.tiny_l2.enable:
        div_str += f"_tinyL2_{cfg.diversity.tiny_l2.weight}"
    if cfg.diversity.color.enable:
        div_str += f"_color{cfg.diversity.color.weight}"
    if cfg.diversity.dpp.enable:
        div_str += f"_dpp{cfg.diversity.dpp.weight}"
    if cfg.diversity.dpp_patch.enable:
        div_str += f"_dpppatch{cfg.diversity.dpp_patch.weight}"
    if cfg.diversity.vendi.enable:
        div_str += f"_vendi{cfg.diversity.vendi.weight}"

    settings = (
        f"{cfg.model.name}{'_' + cfg.task.prompt if cfg.task.type == 't2i-compbench' else ''}"
        f"_{cfg.optimization.seed if cfg.task.type != 'geneval' else ''}"
        f"_lr{cfg.optimization.lr}_gc{cfg.optimization.grad_clip}_iter{cfg.optimization.n_iters}_ns{cfg.optimization.num_samples}"
        f"_reg{cfg.regularization.weight if cfg.regularization.enable else '0'}"
        f"{rewards_str}{div_str}"
    )
    if cfg.optimization.sequential_diverse_samples > 0:
        settings += f"_seq{cfg.optimization.sequential_diverse_samples}"
    if cfg.optimization.noise_type == "pink":
        settings += f"_pink{cfg.optimization.noise_exponent}"
    # buffering=1 → line-buffered file writes so the log file stays current.
    file_stream = open(f"{cfg.paths.save_dir}/logs/{cfg.task.type}/{settings}.txt", "w", buffering=1)
    handler = logging.StreamHandler(file_stream)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel("INFO")
    # reconfigure stdout to line-buffered so log lines flush even when stdout is a pipe (nohup/Slurm/tee).
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logging.info(cfg)
    if cfg.model.device_id is not None:
        logging.info(f"Using CUDA device {cfg.model.device_id}")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.model.device_id)
    device = torch.device("cuda")

    if cfg.model.name in ("flux-schnell", "flux-klein"):
        dtype = torch.bfloat16
    elif cfg.model.dtype == "float32":
        dtype = torch.float32
    elif cfg.model.dtype == "float16":
        dtype = torch.float16
    elif cfg.model.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unknown dtype: {cfg.model.dtype!r} (expected float32, float16, or bfloat16)")
    # Get reward losses (only enabled ones for optimization)
    reward_losses = get_reward_losses(cfg, dtype, device, cfg.paths.cache_dir)

    # Get diversity objectives and thresholds (only enabled ones for optimization)
    diversity_objectives, diversity_thresholds = get_diversity_objectives(cfg, device, cfg.paths.cache_dir)

    logging.info("Loading additional reward losses for comprehensive logging (reusing enabled ones)...")
    all_reward_losses = get_all_reward_losses_shared(
        reward_losses, dtype, device, cfg.paths.cache_dir
    )

    logging.info("Loading additional diversity objectives for comprehensive logging (reusing enabled ones)...")
    additional_objectives, _ = get_additional_logging_objectives(
        diversity_objectives,
        device=device,
        cache_dir=cfg.paths.cache_dir,
        dino_model_name=cfg.diversity.dino.model_name,
        lpips_net=cfg.diversity.lpips.net,
    )
    all_diversity_objectives = diversity_objectives + additional_objectives

    # Get model and noise trainer
    pipe = get_model(
        cfg.model.name, dtype, device, cfg.paths.cache_dir, cfg.model.cpu_offloading
    )
    trainer = LatentNoiseTrainer(
        reward_losses=reward_losses,
        model=pipe,
        n_iters=cfg.optimization.n_iters,
        n_inference_steps=cfg.optimization.n_inference_steps,
        seed=cfg.optimization.seed,
        device=device,
        regularize=cfg.regularization.enable,
        regularization_weight=cfg.regularization.weight,
        grad_clip=cfg.optimization.grad_clip,
        log_metrics=True,
        diversity_objectives=diversity_objectives,
        diversity_thresholds=diversity_thresholds,
        all_reward_losses=all_reward_losses,
        all_diversity_objectives=all_diversity_objectives,
    )

    # Create latents
    if cfg.model.name == "flux-schnell":
        # currently only support 512x512 generation
        # 4D pre-pack shape; generate_latents packs to (B, 1024, 64) when flux_schnell_pack=True
        shape = (cfg.optimization.num_samples, 16, 64, 64)
    elif cfg.model.name == "flux-klein":
        # klein's prepare_latents packs internally, so feed the 4D pre-pack shape
        # at 512x512: (B, transformer.in_channels, H/(vae_scale_factor*2), W/(vae_scale_factor*2))
        shape = (cfg.optimization.num_samples, pipe.transformer.config.in_channels, 32, 32)
    elif cfg.model.name != "pixart":
        height = pipe.unet.config.sample_size * pipe.vae_scale_factor
        width = pipe.unet.config.sample_size * pipe.vae_scale_factor
        shape = (
            cfg.optimization.num_samples,
            pipe.unet.in_channels,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        )
    else:
        height = pipe.transformer.config.sample_size * pipe.vae_scale_factor
        width = pipe.transformer.config.sample_size * pipe.vae_scale_factor
        shape = (
            cfg.optimization.num_samples,
            pipe.transformer.config.in_channels,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        )
    single_sample_shape = (1,) + tuple(shape[1:])
    if cfg.multi_step.enable:
        if cfg.model.name != "flux-schnell":
            raise ValueError(
                f"--multi_step.enable is only meaningful for flux-schnell (which "
                f"optimizes at 1 step but renders init/best at 4 steps); got "
                f"--model.name={cfg.model.name}."
            )
        multi_apply_fn = get_multi_apply_fn(seed=cfg.optimization.seed, pipe=pipe)
    else:
        multi_apply_fn = None

    if cfg.task.type == "single":
        save_dir = f"{cfg.paths.save_dir}/{cfg.task.type}/{settings}/{cfg.task.prompt[:150]}"
        os.makedirs(f"{save_dir}", exist_ok=True)
        sequential_count = max(0, cfg.optimization.sequential_diverse_samples)
        init_images: List[Image.Image] = []
        best_images: List[Image.Image] = []
        if sequential_count > 0:
            (
                init_images,
                best_images,
                total_init_rewards,
                total_best_rewards,
            ) = run_single_sequential(
                cfg=cfg,
                trainer=trainer,
                device=device,
                dtype=dtype,
                single_sample_shape=single_sample_shape,
                multi_apply_fn=multi_apply_fn,
                save_image_batch_fn=save_image_batch,
                save_dir=save_dir,
            )
        else:
            init_latents = generate_latents(
                shape,
                device,
                dtype,
                noise_type=cfg.optimization.noise_type,
                seed=cfg.optimization.seed,
                noise_exponent=cfg.optimization.noise_exponent,
                flux_schnell_pack=(cfg.model.name == "flux-schnell"),
            )
            latents = torch.nn.Parameter(init_latents, requires_grad=True)
            optimizer = get_optimizer(cfg.optimization.optim, latents, cfg.optimization.lr)

            # Set up stopping criteria for batch mode
            dpp_stop_fn = None
            if cfg.stopping.enable_dpp_stopping:
                available_diversity = {obj.name for obj in trainer._base_diversity_objectives}
                if "diversity_dpp" in available_diversity and shape[0] > 1:
                    # Check stopping mode: absolute > multiplier > normalized threshold
                    if cfg.stopping.dpp_stopping_absolute is not None:
                        dpp_stop_fn = make_dpp_absolute_stop(threshold=cfg.stopping.dpp_stopping_absolute)
                    elif cfg.stopping.dpp_stopping_multiplier is not None:
                        dpp_stop_fn = make_dpp_multiplier_stop(multiplier=cfg.stopping.dpp_stopping_multiplier)
                    else:
                        # For batch mode, num_samples is shape[0]
                        dpp_stop_fn = make_sequential_dpp_stop(shape[0], threshold=cfg.stopping.dpp_stopping_threshold)

            # Per-sample HPS threshold (freezes individual samples that reach threshold)
            hps_sample_threshold = None
            hps_absolute_threshold = None
            hps_revert_mode = False
            if cfg.rewards.hps.enable_stopping:
                hps_sample_threshold = cfg.rewards.hps.stopping_threshold
                hps_absolute_threshold = cfg.rewards.hps.absolute_threshold
                hps_revert_mode = cfg.rewards.hps.enable_revert

            (
                init_images,
                best_images,
                total_init_rewards,
                total_best_rewards,
                _,
            ) = trainer.train(
                latents,
                cfg.task.prompt,
                optimizer,
                save_dir,
                multi_apply_fn,
                extra_stop_fn=dpp_stop_fn,
                hps_sample_threshold=hps_sample_threshold,
                hps_absolute_threshold=hps_absolute_threshold,
                hps_revert_mode=hps_revert_mode,
                hps_warmup_iters=cfg.rewards.hps.warmup_iters,
            )
            save_image_batch(best_images, f"{save_dir}/best_image.jpg")
            save_image_batch(init_images, f"{save_dir}/init_image.jpg")
    elif cfg.task.type == "t2i-compbench":
        # Define all T2I-CompBench subsets
        subsets = ["color", "shape", "texture", "spatial", "non-spatial", "complex", "3d_spatial", "numeracy"]

        # Determine which subset to run
        if cfg.task.t2i_subset and cfg.task.t2i_subset != "all":
            if cfg.task.t2i_subset not in subsets:
                raise ValueError(f"Unknown T2I subset {cfg.task.t2i_subset}. Available: {subsets}")
            subsets_to_run = [cfg.task.t2i_subset]
        else:
            subsets_to_run = subsets

        logging.info(f"Running T2I-CompBench on subsets: {subsets_to_run}")

        for subset_idx, subset in enumerate(subsets_to_run):
            prompt_list_file = f"datasets/T2I-CompBench-400/{subset}.txt"

            if not os.path.exists(prompt_list_file):
                logging.warning(f"Subset file not found: {prompt_list_file}, skipping")
                continue

            with open(prompt_list_file) as fp:
                prompts = [line.strip() for line in fp if line.strip()]

            # Convert prompts to metadata format
            metadatas = [{"prompt": prompt, "subset": subset} for prompt in prompts]

            # Apply prompt range filtering if specified
            start_idx = cfg.task.prompt_start_index if cfg.task.prompt_start_index is not None else 0
            end_idx = cfg.task.prompt_end_index if cfg.task.prompt_end_index is not None else len(metadatas)
            metadatas = metadatas[start_idx:end_idx]
            num_prompts = len(metadatas)
            logging.info(f"Processing subset '{subset}': prompts {start_idx} to {end_idx} (total: {num_prompts})")

            outdir = f"{cfg.paths.save_dir}/{cfg.task.type}/{settings}/{subset}"

            # Process all prompts in this subset
            sequential_count = max(0, cfg.optimization.sequential_diverse_samples)
            if sequential_count > 0:
                total_best_rewards, total_init_rewards, all_iteration_histories = process_prompt_batch_sequential(
                    metadatas, trainer, shape, device, dtype, cfg, outdir,
                    save_image_batch, multi_apply_fn, start_idx, subset
                )
            else:
                total_best_rewards, total_init_rewards, all_iteration_histories = process_prompt_batch(
                    metadatas, trainer, shape, device, dtype, cfg, outdir,
                    save_image_batch, multi_apply_fn, start_idx, subset
                )

            # Calculate mean rewards for this subset
            for k in total_best_rewards.keys():
                total_best_rewards[k] /= num_prompts
                total_init_rewards[k] /= num_prompts

            logging.info(f"Subset '{subset}' - Mean initial rewards: {total_init_rewards}")
            logging.info(f"Subset '{subset}' - Mean best rewards: {total_best_rewards}")
    elif cfg.task.type == "geneval" or cfg.task.type == "dpg":
        # Use absolute path relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if cfg.task.type == "geneval":
            prompt_list_file = os.path.join(script_dir, "datasets/GenEval/geneval_generation_prompts.txt")
        else:  # dpg
            prompt_list_file = os.path.join(script_dir, "datasets/DPG/dpg_prompts.txt")
        with open(prompt_list_file) as fp:
            prompts = [line.strip() for line in fp if line.strip()]

        # Convert prompts to metadata format
        metadatas = [{"prompt": prompt} for prompt in prompts]

        # Apply prompt range filtering if specified
        start_idx = cfg.task.prompt_start_index if cfg.task.prompt_start_index is not None else 0
        end_idx = cfg.task.prompt_end_index if cfg.task.prompt_end_index is not None else len(metadatas)
        metadatas = metadatas[start_idx:end_idx]
        num_prompts = len(metadatas)
        logging.info(f"Processing prompts {start_idx} to {end_idx} (total: {num_prompts})")

        outdir = f"{cfg.paths.save_dir}/{cfg.task.type}/{settings}"

        # Process all prompts
        sequential_count = max(0, cfg.optimization.sequential_diverse_samples)
        if sequential_count > 0:
            total_best_rewards, total_init_rewards, all_iteration_histories = process_prompt_batch_sequential(
                metadatas, trainer, shape, device, dtype, cfg, outdir,
                save_image_batch, multi_apply_fn, start_idx
            )
        else:
            total_best_rewards, total_init_rewards, all_iteration_histories = process_prompt_batch(
                metadatas, trainer, shape, device, dtype, cfg, outdir,
                save_image_batch, multi_apply_fn, start_idx
            )

        # Calculate mean rewards
        for k in total_best_rewards.keys():
            total_best_rewards[k] /= num_prompts
            total_init_rewards[k] /= num_prompts
    else:
        raise ValueError(f"Unknown task {cfg.task.type}")
    # log total rewards
    logging.info(f"Mean initial rewards: {total_init_rewards}")
    logging.info(f"Mean best rewards: {total_best_rewards}")


if __name__ == "__main__":
    main(Config.from_args())
