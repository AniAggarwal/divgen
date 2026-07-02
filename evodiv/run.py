"""EvoDiv entry point: evolutionary noise optimisation for collapse recovery.

Usage:
    python -m evodiv.run --config configs/evo_sdxl_geneval.yaml
    python -m evodiv.run --config configs/evo_sdxl_single.yaml --task.prompt "A photo of a cat"

Runs NSGA-II over noise-set genomes for one or more prompts, reusing divgen's
diffusion pipeline and its diversity/quality objectives. Writes, per prompt:
  * init_image.jpg  -- generation-0 (i.i.d. baseline) best-diversity set
  * best_image.jpg  -- final selected set
  * history.json    -- per-generation convergence log
and an aggregate summary at the end.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from typing import List

import torch
from pytorch_lightning import seed_everything

from config import Config  # noqa: F401  (imported for side-effect parity)
from models import get_model, get_multi_apply_fn
from objectives import get_diversity_objectives, get_additional_logging_objectives
from rewards import get_reward_losses

from evodiv.config import EvoConfig
from evodiv.genome import GenomeSpec, init_population
from evodiv.fitness import FitnessEvaluator
from evodiv.nsga2 import evolve
from evodiv.islands import run_islands
from evodiv.io_utils import save_image_set, tensor_set_to_pil


def _latent_spec(cfg, pipe, set_size: int) -> GenomeSpec:
    """Derive the per-image latent (C,H,W) for the model and wrap in a GenomeSpec."""
    name = cfg.model.name
    if name in ("flux-schnell", "flux-klein"):
        raise NotImplementedError(
            "EvoDiv currently targets sdxl-turbo / pixart; flux latent packing is a scale-up TODO."
        )
    elif name != "pixart":
        c = pipe.unet.in_channels
        h = pipe.unet.config.sample_size
        w = pipe.unet.config.sample_size
    else:
        c = pipe.transformer.config.in_channels
        h = pipe.transformer.config.sample_size
        w = pipe.transformer.config.sample_size
    return GenomeSpec(
        channels=c, height=h, width=w, set_size=set_size,
        beta=cfg.evolution.beta,
        sigma_init=cfg.evolution.sigma_init,
        sigma_bounds=(cfg.evolution.sigma_min, cfg.evolution.sigma_max),
        rate_init=cfg.evolution.rate_init,
        rate_bounds=(cfg.evolution.rate_min, cfg.evolution.rate_max),
        tau_sigma=cfg.evolution.tau_sigma, tau_rate=cfg.evolution.tau_rate,
        self_adaptive=cfg.evolution.self_adaptive,
    )


def _load_prompts(cfg) -> List[str]:
    t = cfg.task.type
    if t == "single":
        return [cfg.task.prompt]
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if t == "geneval":
        f = os.path.join(script_dir, "datasets/GenEval/geneval_generation_prompts.txt")
    elif t == "dpg":
        f = os.path.join(script_dir, "datasets/DPG/dpg_prompts.txt")
    else:
        raise ValueError(f"EvoDiv supports task.type in {{single, geneval, dpg}}; got {t!r}")
    with open(f) as fp:
        prompts = [ln.strip() for ln in fp if ln.strip()]
    s = cfg.task.prompt_start_index or 0
    e = cfg.task.prompt_end_index if cfg.task.prompt_end_index is not None else len(prompts)
    return prompts[s:e]


def main(cfg: EvoConfig):
    seed_everything(cfg.evolution.seed)
    torch.cuda.empty_cache()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.info
    device = torch.device("cuda")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[cfg.model.dtype]

    # --- models / objectives / rewards (reuse divgen) --------------------- #
    reward_losses = get_reward_losses(cfg, dtype, device, cfg.paths.cache_dir)
    diversity_objectives, _ = get_diversity_objectives(cfg, device, cfg.paths.cache_dir)
    eval_objectives, _ = get_additional_logging_objectives(
        diversity_objectives, device=device, cache_dir=cfg.paths.cache_dir,
        dino_model_name=cfg.diversity.dino.model_name, lpips_net=cfg.diversity.lpips.net,
    )
    if not reward_losses:
        raise ValueError("Enable at least one quality reward (e.g. --rewards.clip.enable).")
    if not diversity_objectives:
        raise ValueError("Enable at least one diversity objective (e.g. --diversity.dino.enable).")
    log(f"Quality rewards: {[r.name for r in reward_losses]} | "
        f"Diversity objectives: {[o.name for o in diversity_objectives]} | "
        f"Held-out eval: {[o.name for o in eval_objectives]}")

    pipe = get_model(cfg.model.name, dtype, device, cfg.paths.cache_dir, cfg.model.cpu_offloading)
    spec = _latent_spec(cfg, pipe, cfg.evolution.set_size)
    multi_apply_fn = get_multi_apply_fn(seed=cfg.evolution.seed, pipe=pipe) if cfg.multi_step.enable else None

    evaluator = FitnessEvaluator(
        model=pipe, reward_losses=reward_losses, diversity_objectives=diversity_objectives,
        eval_diversity_objectives=eval_objectives, n_inference_steps=cfg.optimization.n_inference_steps,
        seed=cfg.evolution.seed, device=device, model_dtype=dtype,
        render_chunk=cfg.evolution.render_chunk, multi_apply_fn=multi_apply_fn,
    )

    outdir = os.path.join(cfg.paths.save_dir, "evodiv", cfg.model.name, cfg.task.type)
    os.makedirs(outdir, exist_ok=True)

    # --- island mode runs once over prompt-typed subpopulations ----------- #
    if cfg.evolution.islands > 1:
        gen = torch.Generator(device=device).manual_seed(cfg.evolution.seed)
        result = run_islands(cfg, spec, evaluator, None, device, gen, log_fn=log)
        with open(os.path.join(outdir, "islands_history.json"), "w") as fp:
            json.dump({"history": result.history}, fp, indent=2)
        log("Island run complete; see islands_history.json for cross-prompt generalisation.")
        return

    prompts = _load_prompts(cfg)
    log(f"Running EvoDiv on {len(prompts)} prompt(s) -> {outdir}")

    summary = {"init": {}, "best": {}}
    n_done = 0
    for pi, prompt in enumerate(prompts):
        gen = torch.Generator(device=device).manual_seed(cfg.evolution.seed + pi)
        log(f"\n=== prompt {pi + 1}/{len(prompts)}: {prompt!r} ===")

        if cfg.evolution.genome == "gp":
            from evodiv.gp_noise import evolve_gp
            result = evolve_gp(
                spec, evaluator, prompt, cfg.evolution.pop_size,
                cfg.evolution.n_generations, device, gen, log_fn=log,
            )
        else:
            pop = init_population(
                spec, cfg.evolution.pop_size, device,
                noise_type=cfg.optimization.noise_type,
                noise_exponent=cfg.optimization.noise_exponent,
                seed=cfg.evolution.seed + pi * 100003,
            )
            result = evolve(
                pop, evaluator, prompt, cfg.evolution.n_generations, gen,
                p_crossover=cfg.evolution.p_crossover, log_fn=log,
                log_all_every=cfg.evolution.log_all_every,
            )

        hist = result.history
        gen0 = hist[0]
        quality_floor = cfg.evolution.quality_floor_frac * gen0["mean_quality"]
        best_i = result.best_by(cfg.evolution.select_metric if cfg.evolution.select_metric.startswith("diversity")
                                else "_diversity", quality_floor=quality_floor)
        best_raw = result.population.raw[best_i]

        # gen-0 best-diversity set == i.i.d. baseline reference
        init_best = {k[5:]: round(v, 4) for k, v in gen0.items() if k.startswith("best/")}
        best_headline = {k: round(v, 4) for k, v in best_raw.items() if not k.startswith("_")}
        log(f"[prompt {pi}] init(gen0 i.i.d.): {init_best}")
        log(f"[prompt {pi}] best (evolved):   {best_headline}")

        # render + save init and best sets
        p_out = os.path.join(outdir, f"{pi:04d}_{prompt[:60].replace('/', '_')}")
        os.makedirs(p_out, exist_ok=True)
        with torch.no_grad():
            best_imgs = evaluator.render(result.population.latents[best_i], prompt)
        save_image_set(tensor_set_to_pil(best_imgs), os.path.join(p_out, "best_image.jpg"))
        with open(os.path.join(p_out, "history.json"), "w") as fp:
            json.dump({"prompt": prompt, "history": hist,
                       "best_metrics": {k: v for k, v in best_raw.items()},
                       "quality_floor": quality_floor}, fp, indent=2)

        # accumulate summary over the headline metrics present at gen0
        for k, v in gen0.items():
            if k.startswith("best/"):
                metric = k[len("best/"):]
                summary["init"].setdefault(metric, 0.0)
                summary["init"][metric] += v
                summary["best"].setdefault(metric, 0.0)
                summary["best"][metric] += best_raw.get(metric, float("nan"))
        n_done += 1

    for phase in ("init", "best"):
        for k in summary[phase]:
            summary[phase][k] /= max(n_done, 1)
    log(f"\n=== SUMMARY over {n_done} prompt(s) ===")
    log(f"init (i.i.d. gen0): { {k: round(v,4) for k,v in summary['init'].items()} }")
    log(f"best (evolved):     { {k: round(v,4) for k,v in summary['best'].items()} }")
    with open(os.path.join(outdir, "summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)


if __name__ == "__main__":
    main(EvoConfig.from_args())
