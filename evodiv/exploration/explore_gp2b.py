"""GP v2 corrected: floor-respecting SELECTION (and floored gen0) for arms B/C."""
import json, logging, sys, torch
sys.path.insert(0, "/workspace/repos/divgen")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", stream=sys.stdout)
log = logging.info
from config import Config
from models import get_model
from objectives import get_diversity_objectives
from rewards import get_reward_losses
from evodiv.genome import GenomeSpec, init_population, repair, mutate
from evodiv.fitness import FitnessEvaluator
from evodiv.nsga2 import evolve
from evodiv.gp_noise import evolve_gp

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
POP=24
def pick(pop, floor):
    ok=[i for i,r in enumerate(pop.raw) if r["_quality"]>=floor]
    cands = ok if ok else range(len(pop.raw))
    i=max(cands, key=lambda i: pop.raw[i]["_diversity"])
    return pop.raw[i]
prompts=[l.strip() for l in open("/workspace/repos/divgen/datasets/GenEval/geneval_generation_prompts.txt")][:4]
res={"B":[], "C":[]}
for pi,prompt in enumerate(prompts):
    gen=torch.Generator(device=device).manual_seed(100+pi)
    # floor from an i.i.d. probe population
    probe=init_population(spec,POP,device,seed=pi*7)
    ev.evaluate(probe,prompt,exact=True)
    floor=sum(r["_quality"] for r in probe.raw)/POP
    # B: quality-constrained GP with floored selection
    rB=evolve_gp(spec,ev,prompt,POP,60,device,gen,quality_floor=floor)
    b=pick(rB.population, floor)
    res["B"].append((b["diversity_dino"],b["CLIP"]))
    log(f"[{pi}] B: dino={b['diversity_dino']:.4f} clip={b['CLIP']:.4f} (floor {floor:.3f})")
    # C: hybrid with floored selection
    rC1=evolve_gp(spec,ev,prompt,POP,25,device,gen)
    bi=max(range(rC1.population.size),key=lambda i:rC1.population.raw[i]["_diversity"])
    seed_lat=rC1.population.genomes[bi].render_latents(spec,device)
    popC=init_population(spec,POP,device,seed=pi*13+5)
    half=POP//2
    lat=repair(seed_lat.unsqueeze(0).expand(half,-1,-1,-1,-1).clone())
    lat,ls,lr=mutate(lat,popC.log_sigma[:half],popC.logit_rate[:half],spec,gen)
    popC.latents[:half]=lat; popC.log_sigma[:half]=ls; popC.logit_rate[:half]=lr
    rC=evolve(popC,ev,prompt,35,gen,log_fn=None)
    c=pick(rC.population, floor)
    res["C"].append((c["diversity_dino"],c["CLIP"]))
    log(f"[{pi}] C: dino={c['diversity_dino']:.4f} clip={c['CLIP']:.4f}")
for k,v in res.items():
    log(f"SUMMARY {k}: dino={sum(x[0] for x in v)/len(v):.4f} clip={sum(x[1] for x in v)/len(v):.4f}")
json.dump(res,open("/workspace/runs/explore_gp2b_results.json","w"),indent=2)
log("GP2B_DONE")
