"""Round 2c: NSGA-II over the DCT-subspace genome vs full-space NSGA-II (6 prompts)."""
import json, logging, sys, torch
sys.path.insert(0, "/workspace/repos/divgen")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", stream=sys.stdout)
log = logging.info
from config import Config
from models import get_model
from objectives import get_diversity_objectives
from rewards import get_reward_losses
from evodiv.genome import GenomeSpec, init_population
from evodiv.fitness import FitnessEvaluator
from evodiv.nsga2 import evolve, survival, rank_and_crowding, binary_tournament
from evodiv.dct_genome import DCTPopulation, init_dct_population, dct_variation

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
POP, GENS, K = 32, 45, 8

def pick_at_quality(pop, floor):
    ok=[i for i,r in enumerate(pop.raw) if r["CLIP"]>=floor]
    cands = ok if ok else range(len(pop.raw))
    return pop.raw[max(cands, key=lambda i: pop.raw[i]["_diversity"])]

prompts=[l.strip() for l in open("/workspace/repos/divgen/datasets/GenEval/geneval_generation_prompts.txt")][:6]
res={"dct_nsga":[], "tensor_nsga":[]}
for pi,prompt in enumerate(prompts):
    gen=torch.Generator(device=device).manual_seed(1100+pi)
    # tensor baseline
    popT=init_population(spec,POP,device,seed=pi*41)
    ev.evaluate(popT,prompt,exact=True)
    floor=sum(r["CLIP"] for r in popT.raw)/POP
    rT=evolve(popT,ev,prompt,GENS,gen,log_fn=None)
    bT=pick_at_quality(rT.population, floor)
    res["tensor_nsga"].append((bT["diversity_dino"],bT["CLIP"]))
    log(f"[{pi}] tensor: dino={bT['diversity_dino']:.4f} clip={bT['CLIP']:.4f} (floor {floor:.3f})")
    # DCT-NSGA
    pop=init_dct_population(spec,POP,K,device,seed=pi*41)
    ev.evaluate(pop,prompt,exact=True)
    for g in range(GENS):
        rank,cd,_=rank_and_crowding(pop.F)
        pa=binary_tournament(rank,cd,POP,gen); pb=binary_tournament(rank,cd,POP,gen)
        off=dct_variation(pop,pa,pb,gen)
        ev.evaluate(off,prompt,exact=False)
        pop=survival(DCTPopulation.concat(pop,off),POP)
        if (g+1)%10==0: ev.evaluate(pop,prompt,exact=True)
    ev.evaluate(pop,prompt,exact=True)
    bD=pick_at_quality(pop, floor)
    res["dct_nsga"].append((bD["diversity_dino"],bD["CLIP"]))
    log(f"[{pi}] dct-nsga: dino={bD['diversity_dino']:.4f} clip={bD['CLIP']:.4f}")
for k,v in res.items():
    log(f"SUMMARY {k}: dino={sum(x[0] for x in v)/len(v):.4f} clip={sum(x[1] for x in v)/len(v):.4f}")
json.dump(res,open("/workspace/runs/explore_dctnsga_results.json","w"))
log("DCTNSGA_DONE")
