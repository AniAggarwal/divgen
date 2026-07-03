"""Exploration item 8: composition — bred noise warm-starts the paper's gradient optimizer.
Arms per prompt (budget in gradient iters G=10; breeding budget 20 gens pop 16):
  A. gradient-only from i.i.d. (G iters)   [the paper's method, small budget]
  B. bred-only (20 gens)                   [ours, small budget]
  C. bred 20 gens -> gradient G iters      [composition]
"""
import json, logging, sys, torch
sys.path.insert(0, "/workspace/repos/divgen")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", stream=sys.stdout)
log = logging.info
from config import Config
from models import get_model
from objectives import get_diversity_objectives
from rewards import get_reward_losses
from training import LatentNoiseTrainer, get_optimizer
from evodiv.genome import GenomeSpec, init_population
from evodiv.fitness import FitnessEvaluator
from evodiv.nsga2 import evolve

cfg = Config(); cfg.paths.cache_dir = "/workspace/repos/divgen/cache"
cfg.diversity.dino.enable = True; cfg.diversity.dino.weight = 1.0
cfg.rewards.clip.enable = True; cfg.rewards.clip.weighting = 0.01
device = torch.device("cuda"); dtype = torch.float16
pipe = get_model("sdxl-turbo", dtype, device, cfg.paths.cache_dir)
divs,_ = get_diversity_objectives(cfg, device, cfg.paths.cache_dir)
rews = get_reward_losses(cfg, dtype, device, cfg.paths.cache_dir)
ev = FitnessEvaluator(model=pipe, reward_losses=rews, diversity_objectives=divs,
                      n_inference_steps=1, seed=0, device=device, model_dtype=dtype, render_chunk=160)
ev.enable_surrogate(cfg.paths.cache_dir)
spec = GenomeSpec(channels=4, height=64, width=64, set_size=4)
trainer = LatentNoiseTrainer(reward_losses=rews, model=pipe, n_iters=10, n_inference_steps=1,
                             seed=0, device=device, diversity_objectives=divs, log_metrics=False)
def grad_refine(lat0, prompt, iters=10):
    lat = torch.nn.Parameter(lat0.clone().to(dtype), requires_grad=True)
    opt = get_optimizer("sgd", lat, 5.0)
    trainer.n_iters = iters
    _,_,_, best,_ = trainer.train(lat, prompt, opt, skip_final_metrics_log=True)
    return best  # dict of metrics incl diversity_dino & CLIP

def score_set(lat, prompt):
    with torch.no_grad():
        img = ev.render(lat, prompt)
        d = float(divs[0].compute_pairwise_diversity(img).item())
        from rewards import clip_img_transform
        c = 1 - rews[0](clip_img_transform(224)(img), prompt).item()/100
    return d, c

prompts=[l.strip() for l in open("/workspace/repos/divgen/datasets/GenEval/geneval_generation_prompts.txt")][:4]
res={"A_grad":[], "B_bred":[], "C_bred_then_grad":[]}
for pi,prompt in enumerate(prompts):
    gen=torch.Generator(device=device).manual_seed(800+pi)
    pop=init_population(spec,16,device,seed=pi*97)
    iid=pop.latents[0].clone()
    # A: gradient from iid
    mA=grad_refine(iid, prompt, 10)
    res["A_grad"].append((mA.get("diversity_dino",0), mA.get("CLIP",0)))
    log(f"[{pi}] A grad10: dino={mA.get('diversity_dino',0):.4f} clip={mA.get('CLIP',0):.4f}")
    # B: bred only
    r=evolve(pop,ev,prompt,20,gen,log_fn=None)
    bi=max(range(r.population.size),key=lambda i:r.population.raw[i]["_diversity"])
    dB,cB=score_set(r.population.latents[bi],prompt)
    res["B_bred"].append((dB,cB))
    log(f"[{pi}] B bred20: dino={dB:.4f} clip={cB:.4f}")
    # C: bred -> gradient
    mC=grad_refine(r.population.latents[bi], prompt, 10)
    res["C_bred_then_grad"].append((mC.get("diversity_dino",0), mC.get("CLIP",0)))
    log(f"[{pi}] C bred+grad: dino={mC.get('diversity_dino',0):.4f} clip={mC.get('CLIP',0):.4f}")
for k,v in res.items():
    log(f"SUMMARY {k}: dino={sum(x[0] for x in v)/4:.4f} clip={sum(x[1] for x in v)/4:.4f}")
json.dump(res,open("/workspace/runs/explore_hybrid_results.json","w"))
log("HYBRID_DONE")
