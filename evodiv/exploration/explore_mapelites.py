"""Exploration item 2: MAP-Elites vs NSGA-II on 6 GenEval prompts."""
import json, logging, sys, torch
sys.path.insert(0, "/workspace/repos/divgen")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", stream=sys.stdout)
log = logging.info

from config import Config
from models import get_model
from objectives import get_diversity_objectives, get_additional_logging_objectives
from rewards import get_reward_losses
from evodiv.genome import GenomeSpec
from evodiv.fitness import FitnessEvaluator
from evodiv.map_elites import run_map_elites

cfg = Config(); cfg.paths.cache_dir = "/workspace/repos/divgen/cache"
cfg.diversity.dino.enable = True; cfg.diversity.dino.weight = 1.0
cfg.rewards.clip.enable = True; cfg.rewards.clip.weighting = 1.0
device = torch.device("cuda"); dtype = torch.float16
pipe = get_model("sdxl-turbo", dtype, device, cfg.paths.cache_dir)
divs, _ = get_diversity_objectives(cfg, device, cfg.paths.cache_dir)
rews = get_reward_losses(cfg, dtype, device, cfg.paths.cache_dir)
extra, _ = get_additional_logging_objectives(divs, device=device, cache_dir=cfg.paths.cache_dir,
                                             dino_model_name=cfg.diversity.dino.model_name, lpips_net="vgg")
color_obj = next(o for o in extra if o.name == "diversity_color")
tiny_obj = next(o for o in extra if o.name == "diversity_tiny_l2")

ev = FitnessEvaluator(model=pipe, reward_losses=rews, diversity_objectives=divs,
                      n_inference_steps=1, seed=0, device=device, model_dtype=dtype, render_chunk=160)
ev.enable_surrogate(cfg.paths.cache_dir)
pipe.unet = torch.compile(pipe.unet)
spec = GenomeSpec(channels=4, height=64, width=64, set_size=4)

prompts = [l.strip() for l in open("/workspace/repos/divgen/datasets/GenEval/geneval_generation_prompts.txt")][:6]
results = []
for pi, prompt in enumerate(prompts):
    gen = torch.Generator(device=device).manual_seed(pi)
    log(f"=== MAP-Elites prompt {pi}: {prompt!r} ===")
    res = run_map_elites(spec, ev, prompt, color_obj, tiny_obj, device, gen,
                         batch=40, iterations=60, bins=10, log_fn=log)
    results.append(res)
    log(f"[prompt {pi}] coverage={res['coverage']}/100 qd={res['qd_score']:.1f} "
        f"best_dino={res['best']['dino']:.4f} best_clip={res['best']['clip']:.4f} "
        f"top_dino={res['top_dino_in_archive']:.4f}")
with open("/workspace/runs/explore_mapelites_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
avg_cov = sum(r["coverage"] for r in results)/len(results)
avg_best_dino = sum(r["best"]["dino"] for r in results)/len(results)
avg_best_clip = sum(r["best"]["clip"] for r in results)/len(results)
avg_top = sum(r["top_dino_in_archive"] for r in results)/len(results)
log(f"SUMMARY: coverage {avg_cov:.1f}/100 | best-cell dino {avg_best_dino:.4f} clip {avg_best_clip:.4f} | top archive dino {avg_top:.4f}")
log("MAPELITES_DONE")
