"""Exploration item 3: GP v2 — quality-constrained program search + GP->tensor warm-start hybrid.
Three arms on 4 prompts, matched budget (~60 gen-equivalents of pop 24):
  A. pure tensor NSGA-II, 60 gens
  B. quality-constrained GP, 60 gens (floor = gen0 mean quality)
  C. hybrid: GP 25 gens -> best GP genome seeds a tensor population -> 35 gens
"""
import json, logging, sys, torch
sys.path.insert(0, "/workspace/repos/divgen")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", stream=sys.stdout)
log = logging.info
from config import Config
from models import get_model
from objectives import get_diversity_objectives
from rewards import get_reward_losses
from evodiv.genome import GenomeSpec, Population, init_population, repair, mutate
from evodiv.fitness import FitnessEvaluator
from evodiv.nsga2 import evolve
from evodiv.gp_noise import evolve_gp

cfg = Config(); cfg.paths.cache_dir = "/workspace/repos/divgen/cache"
cfg.diversity.dino.enable = True; cfg.diversity.dino.weight = 1.0
cfg.rewards.clip.enable = True; cfg.rewards.clip.weighting = 1.0
device = torch.device("cuda"); dtype = torch.float16
pipe = get_model("sdxl-turbo", dtype, device, cfg.paths.cache_dir)
divs, _ = get_diversity_objectives(cfg, device, cfg.paths.cache_dir)
rews = get_reward_losses(cfg, dtype, device, cfg.paths.cache_dir)
ev = FitnessEvaluator(model=pipe, reward_losses=rews, diversity_objectives=divs,
                      n_inference_steps=1, seed=0, device=device, model_dtype=dtype, render_chunk=160)
ev.enable_surrogate(cfg.paths.cache_dir)
pipe.unet = torch.compile(pipe.unet)
spec = GenomeSpec(channels=4, height=64, width=64, set_size=4)
POP=24
prompts=[l.strip() for l in open("/workspace/repos/divgen/datasets/GenEval/geneval_generation_prompts.txt")][:4]
out={"A_tensor":[], "B_gp_constrained":[], "C_hybrid":[]}
for pi,prompt in enumerate(prompts):
    gen=torch.Generator(device=device).manual_seed(pi)
    log(f"===== prompt {pi}: {prompt!r} =====")
    # A: pure tensor
    popA=init_population(spec,POP,device,seed=pi*7)
    resA=evolve(popA,ev,prompt,60,gen,log_fn=None)
    bA=max(resA.population.raw,key=lambda r:r["_diversity"])
    out["A_tensor"].append((bA["diversity_dino"],bA["CLIP"]))
    log(f"A tensor60: dino={bA['diversity_dino']:.4f} clip={bA['CLIP']:.4f}")
    # B: quality-constrained GP; floor = gen0 mean quality of A
    floor=resA.history[0]["mean_quality"]
    resB=evolve_gp(spec,ev,prompt,POP,60,device,gen,quality_floor=floor)
    bB=max(resB.population.raw,key=lambda r:r["_diversity"])
    out["B_gp_constrained"].append((bB["diversity_dino"],bB["CLIP"]))
    log(f"B gp-constrained60 (floor={floor:.3f}): dino={bB['diversity_dino']:.4f} clip={bB['CLIP']:.4f}")
    # C: hybrid: GP 25 gens (unconstrained) -> seed tensor pop from best genome
    resC1=evolve_gp(spec,ev,prompt,POP,25,device,gen)
    bi=max(range(resC1.population.size),key=lambda i:resC1.population.raw[i]["_diversity"])
    seed_lat=resC1.population.genomes[bi].render_latents(spec,device)  # (B,C,H,W)
    popC=init_population(spec,POP,device,seed=pi*13+5)
    popC.latents[:POP//2]=repair(seed_lat.unsqueeze(0).expand(POP//2,-1,-1,-1,-1).clone())
    # mutate the seeded half so they are not identical
    lat,ls,lr=mutate(popC.latents[:POP//2],popC.log_sigma[:POP//2],popC.logit_rate[:POP//2],spec,gen)
    popC.latents[:POP//2]=lat; popC.log_sigma[:POP//2]=ls; popC.logit_rate[:POP//2]=lr
    resC=evolve(popC,ev,prompt,35,gen,log_fn=None)
    bC=max(resC.population.raw,key=lambda r:r["_diversity"])
    out["C_hybrid"].append((bC["diversity_dino"],bC["CLIP"]))
    log(f"C hybrid(25GP+35tensor): dino={bC['diversity_dino']:.4f} clip={bC['CLIP']:.4f}")
for k,v in out.items():
    d=sum(x[0] for x in v)/len(v); c=sum(x[1] for x in v)/len(v)
    log(f"SUMMARY {k}: dino={d:.4f} clip={c:.4f}")
json.dump(out,open("/workspace/runs/explore_gp2_results.json","w"),indent=2)
log("GP2_DONE")
