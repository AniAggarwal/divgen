"""Exploration item 6: epsilon-lexicase parent selection vs NSGA-II tournament.

Cases for lexicase = the 6 pairwise DINO distances of the set (each pair is a
separate 'test') + the CLIP quality (weighted to appear 3x in the case shuffle).
Survival stays NSGA-II (elitism); only PARENT SELECTION changes.
"""
import json, logging, sys, torch
sys.path.insert(0, "/workspace/repos/divgen")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", stream=sys.stdout)
log = logging.info
import torch.nn.functional as NF
from config import Config
from models import get_model
from objectives import get_diversity_objectives
from rewards import get_reward_losses
from evodiv.genome import GenomeSpec, Population, init_population, crossover, mutate
from evodiv.fitness import FitnessEvaluator
from evodiv.nsga2 import survival, rank_and_crowding, binary_tournament, evolve

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
POP, GENS = 32, 45
B = spec.set_size
iu, ju = torch.triu_indices(B, B, offset=1)

@torch.no_grad()
def eval_with_cases(pop, prompt):
    P = pop.size
    flat = pop.latents.reshape(P*B, spec.channels, spec.height, spec.width)
    images = ev.render(flat, prompt, exact=False).reshape(P, B, 3, 512, 512)
    F, raw = ev._score_population_batched(images, prompt, P, B)
    # per-pair DINO distances as lexicase cases
    feats = divs[0].dino_backend.extract_features(images.reshape(P*B,3,512,512), normalize=False, use_cls=False)
    feats = NF.normalize(feats, p=2, dim=-1).reshape(P, B, *feats.shape[1:])
    sim = torch.einsum("pind,pjnd->pijn", feats, feats)
    dist = (1 - sim).mean(-1)                       # (P,B,B)
    cases = dist[:, iu, ju]                         # (P,6)
    qual = torch.tensor([r["CLIP"] for r in raw], device=device).unsqueeze(1)  # (P,1)
    pop.F, pop.raw = F, raw
    return torch.cat([cases, qual.expand(P,3)], 1)  # (P,9): 6 pair cases + 3x quality

def lexicase_select(cases, n, gen):
    P, C = cases.shape
    eps = cases.std(0) * 0.5
    out = []
    for _ in range(n):
        cand = torch.arange(P, device=device)
        order = torch.randperm(C, generator=gen, device=device)
        for c in order:
            col = cases[cand, c]
            keep = col >= col.max() - eps[c]
            cand = cand[keep]
            if len(cand) == 1: break
        out.append(cand[torch.randint(0, len(cand), (1,), generator=gen, device=device)].item())
    return torch.tensor(out, device=device)

prompts=[l.strip() for l in open("/workspace/repos/divgen/datasets/GenEval/geneval_generation_prompts.txt")][:4]
res={"lexicase":[], "tournament":[]}
for pi,prompt in enumerate(prompts):
    gen=torch.Generator(device=device).manual_seed(900+pi)
    # lexicase arm
    pop=init_population(spec,POP,device,seed=pi*31)
    cases=eval_with_cases(pop,prompt)
    for g in range(GENS):
        pa=lexicase_select(cases,POP,gen); pb=lexicase_select(cases,POP,gen)
        cl,ls,lr=crossover(pop.latents,pop.log_sigma,pop.logit_rate,pa,pb,spec,gen)
        cl,ls,lr=mutate(cl,ls,lr,spec,gen)
        off=Population(cl,ls,lr,spec)
        occ=eval_with_cases(off,prompt)
        comb=Population.concat(pop,off)
        pop=survival(comb,POP)
        cases=eval_with_cases(pop,prompt)   # re-eval survivors' cases (cheap enough)
    b=max(pop.raw,key=lambda r:r["_diversity"])
    res["lexicase"].append((b["diversity_dino"],b["CLIP"]))
    log(f"[{pi}] lexicase: dino={b['diversity_dino']:.4f} clip={b['CLIP']:.4f}")
    # tournament arm (standard evolve, same seeds/budget)
    pop2=init_population(spec,POP,device,seed=pi*31)
    r2=evolve(pop2,ev,prompt,GENS,gen,log_fn=None)
    b2=max(r2.population.raw,key=lambda r:r["_diversity"])
    res["tournament"].append((b2["diversity_dino"],b2["CLIP"]))
    log(f"[{pi}] tournament: dino={b2['diversity_dino']:.4f} clip={b2['CLIP']:.4f}")
for k,v in res.items():
    log(f"SUMMARY {k}: dino={sum(x[0] for x in v)/4:.4f} clip={sum(x[1] for x in v)/4:.4f}")
json.dump(res,open("/workspace/runs/explore_lexicase_results.json","w"))
log("LEXICASE_DONE")
