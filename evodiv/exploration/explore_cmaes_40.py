"""Exploration item 4: CMA-ES over low-frequency DCT coefficients of the noise.

Parameterization (exploits the paper's Fig. 9 finding): each of the B=4 latents
= fixed white base + IDCT of an 8x8 low-frequency coefficient block per channel,
then chi_d repair. Genome = 4*4*64 = 1024 dims. CMA-ES (single objective:
DINO + 3*CLIP) vs NSGA-II tensor baseline on the same 4 prompts, matched evals.
"""
import json, logging, sys, math, numpy as np, torch
sys.path.insert(0, "/workspace/repos/divgen")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", stream=sys.stdout)
log = logging.info
import cma
from config import Config
from models import get_model
from objectives import get_diversity_objectives
from rewards import get_reward_losses
from evodiv.genome import GenomeSpec, Population, init_population, repair
from evodiv.fitness import FitnessEvaluator

cfg = Config(); cfg.paths.cache_dir = "/workspace/repos/divgen/cache"
cfg.diversity.dino.enable = True; cfg.diversity.dino.weight = 1.0
cfg.rewards.clip.enable = True; cfg.rewards.clip.weighting = 1.0
device = torch.device("cuda"); dtype = torch.float16
pipe = get_model("sdxl-turbo", dtype, device, cfg.paths.cache_dir)
divs,_ = get_diversity_objectives(cfg, device, cfg.paths.cache_dir)
rews = get_reward_losses(cfg, dtype, device, cfg.paths.cache_dir)
ev = FitnessEvaluator(model=pipe, reward_losses=rews, diversity_objectives=divs,
                      n_inference_steps=1, seed=0, device=device, model_dtype=dtype, render_chunk=160)
ev.enable_surrogate(cfg.paths.cache_dir)
pipe.unet = torch.compile(pipe.unet)
spec = GenomeSpec(channels=4, height=64, width=64, set_size=4)
B, C, H, W, K = 4, 4, 64, 64, 8
DIM = B*C*K*K

# IDCT basis for the KxK lowest block on a HxW grid (type-II DCT orthonormal)
n = torch.arange(H).float(); k = torch.arange(K).float()
basis = torch.cos(math.pi*(n[None,:]+0.5)*k[:,None]/H)          # (K,H)
basis[0]*=1/math.sqrt(2); basis*=math.sqrt(2.0/H)
basis = basis.to(device)

def vec_to_latents(x: np.ndarray, base: torch.Tensor) -> torch.Tensor:
    c = torch.tensor(x, dtype=torch.float32, device=device).view(B, C, K, K)
    field = torch.einsum('bckl,kh,lw->bchw', c, basis, basis)
    return repair(base + field)

import os
W_Q = float(os.environ.get("WQ","3.0"))
prompts=[l.strip() for l in open("/workspace/repos/divgen/datasets/GenEval/geneval_generation_prompts.txt")][:40]
res=[]
for pi,prompt in enumerate(prompts):
    g = torch.Generator(device=device).manual_seed(500+pi)
    base = repair(torch.randn(B, C, H, W, device=device, generator=g))
    es = cma.CMAEvolutionStrategy(np.zeros(DIM), 2.0,
        {'popsize': 40, 'seed': 500+pi, 'verbose': -9})
    best=(None,-1e9)
    for it in range(60):
        xs = es.ask()
        lat = torch.stack([vec_to_latents(x, base) for x in xs], 0)  # (40,B,C,H,W)
        pop = Population(lat, torch.zeros(40,device=device), torch.zeros(40,device=device), spec)
        ev.evaluate(pop, prompt, exact=False)
        fits = [r["diversity_dino"] + W_Q*r["CLIP"] for r in pop.raw]
        es.tell(xs, [-f for f in fits])
        bi = int(np.argmax(fits))
        if fits[bi] > best[1]:
            best = ((pop.raw[bi]["diversity_dino"], pop.raw[bi]["CLIP"]), fits[bi])
        if (it+1)%20==0:
            log(f"[{pi}] it{it+1}: best dino={best[0][0]:.4f} clip={best[0][1]:.4f}")
    # exact re-score of the final best
    # exact re-score
    lat=best[2] if len(best)>2 else None
    res.append(best[0])
    log(f"[{pi}] CMA final: dino={best[0][0]:.4f} clip={best[0][1]:.4f}")
d=sum(x[0] for x in res)/len(res); c=sum(x[1] for x in res)/len(res)
log(f"SUMMARY CMA-DCT: dino={d:.4f} clip={c:.4f}  (tensor NSGA-II ref: 0.763 / 0.338)")
json.dump(res, open("/workspace/runs/explore_cmaes_results.json","w"))
log("CMAES_DONE")
