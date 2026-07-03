"""Exploration item 7: cross-model transfer of bred noise sets, SDXL-Turbo -> PixArt-DMD.
Breed on SDXL (pop 24 x 40 gens), then render the SAME winning noise sets through
PixArt-DMD and compare against PixArt i.i.d. baselines. Both models use (4,64,64).
"""
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
from evodiv.nsga2 import evolve

cfg = Config(); cfg.paths.cache_dir = "/workspace/repos/divgen/cache"
cfg.diversity.dino.enable = True; cfg.diversity.dino.weight = 1.0
cfg.rewards.clip.enable = True; cfg.rewards.clip.weighting = 1.0
device = torch.device("cuda"); dtype = torch.float16
divs,_ = get_diversity_objectives(cfg, device, cfg.paths.cache_dir)
rews = get_reward_losses(cfg, dtype, device, cfg.paths.cache_dir)
spec = GenomeSpec(channels=4, height=64, width=64, set_size=4)
prompts=[l.strip() for l in open("/workspace/repos/divgen/datasets/GenEval/geneval_generation_prompts.txt")][:4]

# --- phase 1: breed on SDXL, keep best latents ---
pipe = get_model("sdxl-turbo", dtype, device, cfg.paths.cache_dir)
ev = FitnessEvaluator(model=pipe, reward_losses=rews, diversity_objectives=divs,
                      n_inference_steps=1, seed=0, device=device, model_dtype=dtype, render_chunk=160)
ev.enable_surrogate(cfg.paths.cache_dir)
pipe.unet = torch.compile(pipe.unet)
bred, iid = [], []
for pi,prompt in enumerate(prompts):
    gen=torch.Generator(device=device).manual_seed(700+pi)
    pop=init_population(spec,24,device,seed=pi*77)
    iid.append(pop.latents[0].clone())          # an i.i.d. reference set
    r=evolve(pop,ev,prompt,40,gen,log_fn=None)
    bi=max(range(r.population.size),key=lambda i:r.population.raw[i]["_diversity"])
    bred.append(r.population.latents[bi].clone())
    b=r.population.raw[bi]
    log(f"[{pi}] bred on SDXL: dino={b['diversity_dino']:.4f} clip={b['CLIP']:.4f}")
del pipe, ev; torch.cuda.empty_cache()

# --- phase 2: render same latents through PixArt-DMD ---
log("loading PixArt-DMD (T5 download may take a while)...")
pipe2 = get_model("pixart", dtype, device, cfg.paths.cache_dir)
ev2 = FitnessEvaluator(model=pipe2, reward_losses=rews, diversity_objectives=divs,
                       n_inference_steps=1, seed=0, device=device, model_dtype=dtype, render_chunk=32)
res=[]
for pi,prompt in enumerate(prompts):
    with torch.no_grad():
        img_b = ev2.render(bred[pi], prompt)
        img_i = ev2.render(iid[pi], prompt)
        db = float(divs[0].compute_pairwise_diversity(img_b).item())
        di = float(divs[0].compute_pairwise_diversity(img_i).item())
        from rewards import clip_img_transform
        prep = clip_img_transform(224)
        cb = 1 - rews[0](prep(img_b), prompt).item()/100
        ci = 1 - rews[0](prep(img_i), prompt).item()/100
    res.append({"prompt":prompt,"pixart_iid_dino":di,"pixart_bred_dino":db,
                "pixart_iid_clip":ci,"pixart_bred_clip":cb})
    log(f"[{pi}] PixArt transfer: iid dino={di:.4f} -> bred dino={db:.4f} | clip {ci:.4f}->{cb:.4f}")
ad_i=sum(r["pixart_iid_dino"] for r in res)/4; ad_b=sum(r["pixart_bred_dino"] for r in res)/4
ac_i=sum(r["pixart_iid_clip"] for r in res)/4; ac_b=sum(r["pixart_bred_clip"] for r in res)/4
log(f"SUMMARY transfer: PixArt dino {ad_i:.4f} -> {ad_b:.4f} | clip {ac_i:.4f} -> {ac_b:.4f}")
json.dump(res,open("/workspace/runs/explore_transfer_results.json","w"),indent=2)
log("TRANSFER_DONE")
