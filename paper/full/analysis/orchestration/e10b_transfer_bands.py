"""E10b: frequency-band decomposition of cross-model noise transfer.

Protocol mirrors evodiv/exploration/explore_transfer.py (breed on SDXL-Turbo,
replay the winning noise sets zero-shot through PixArt-DMD), then adds the
band question: relative to a fresh i.i.d. reference set, transfer the bred
delta (a) in full, (b) low-frequency bands only (radial DCT index k+l <= CUT,
the "lowest third" of the spectrum), (c) high bands only. If the transfer
effect rides on low frequencies -- the source paper's Fig.-9 story -- (b)
should carry most of (a)'s diversity gain and (c) little of it.

12 prompts, pop 24 x 40 gens (seeds match the archived exploration).
Writes results.json for aggregate.py.
"""
import json, logging, math, os, sys, time
sys.path.insert(0, "/workspace/repos/divgen")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", stream=sys.stdout)
log = logging.info

import torch
from config import Config
from models import get_model
from objectives import get_diversity_objectives
from rewards import get_reward_losses, clip_img_transform
from evodiv.genome import GenomeSpec, init_population, repair
from evodiv.fitness import FitnessEvaluator
from evodiv.nsga2 import evolve

OUT = "/workspace/runs/paperprep/e10_transfer_bands"
N_PROMPTS, POP, GENS = 12, 24, 40
H = W = 64
CUT = 42            # k+l <= 42 of max 126: the lowest third of radial bands

cfg = Config(); cfg.paths.cache_dir = "/workspace/repos/divgen/cache"
cfg.diversity.dino.enable = True
cfg.rewards.clip.enable = True
device = torch.device("cuda"); dtype = torch.float16
divs, _ = get_diversity_objectives(cfg, device, cfg.paths.cache_dir)
rews = get_reward_losses(cfg, dtype, device, cfg.paths.cache_dir)
spec = GenomeSpec(channels=4, height=64, width=64, set_size=4)
prep = clip_img_transform(224)
prompts = [l.strip() for l in open("/workspace/repos/divgen/datasets/GenEval/geneval_generation_prompts.txt") if l.strip()][:N_PROMPTS]

# orthonormal DCT-II basis + band mask
n = torch.arange(H).float(); k = torch.arange(H).float()
BAS = torch.cos(math.pi * (n[None, :] + 0.5) * k[:, None] / H)
BAS[0] *= 1 / math.sqrt(2); BAS *= math.sqrt(2.0 / H)
BAS = BAS.to(device)
kk, ll = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
LOWMASK = ((kk + ll) <= CUT).float().to(device)          # (H, W) in DCT domain


def band_split(delta: torch.Tensor):
    """Return (low, high) band components of a (B, C, H, W) tensor."""
    coeff = torch.einsum("bchw,kh,lw->bckl", delta.float(), BAS, BAS)
    low = torch.einsum("bckl,kh,lw->bchw", coeff * LOWMASK, BAS, BAS)
    high = torch.einsum("bckl,kh,lw->bchw", coeff * (1 - LOWMASK), BAS, BAS)
    return low, high


def score(ev, imgs, prompt):
    d = float(divs[0].compute_pairwise_diversity(imgs).item())
    c = 1 - rews[0](prep(imgs), prompt).item() / 100
    return d, c


state_path = os.path.join(OUT, "state.json")
state = json.load(open(state_path)) if os.path.exists(state_path) else {}

# --- phase 1: breed on SDXL, keep winner + iid reference latents ------------
need = [p for i, p in enumerate(prompts) if str(i) not in state]
if need:
    pipe = get_model("sdxl-turbo", dtype, device, cfg.paths.cache_dir)
    ev = FitnessEvaluator(model=pipe, reward_losses=rews, diversity_objectives=divs,
                          n_inference_steps=1, seed=0, device=device,
                          model_dtype=dtype, render_chunk=160)
    ev.enable_surrogate(cfg.paths.cache_dir)
    pipe.unet = torch.compile(pipe.unet)
    lat_store = {}
    for pi, prompt in enumerate(prompts):
        if str(pi) in state:
            continue
        g = torch.Generator(device=device).manual_seed(700 + pi)
        pop = init_population(spec, POP, device, seed=pi * 77)
        iid_ref = pop.latents[0].clone()
        r = evolve(pop, ev, prompt, GENS, g, log_fn=None)
        bi = max(range(r.population.size), key=lambda i: r.population.raw[i]["_diversity"])
        bred = r.population.latents[bi].clone()
        torch.save({"bred": bred.cpu(), "iid": iid_ref.cpu()},
                   os.path.join(OUT, f"latents_{pi:02d}.pt"))
        _bd = r.population.raw[bi].get('diversity_dino', r.population.raw[bi].get('_diversity'))
        log(f"[{pi}] bred dino~={_bd:.4f}")
    del pipe, ev
    torch.cuda.empty_cache()

# --- phase 2: PixArt replay, full/low/high/iid -------------------------------
pipe2 = get_model("pixart", dtype, device, cfg.paths.cache_dir)
ev2 = FitnessEvaluator(model=pipe2, reward_losses=rews, diversity_objectives=divs,
                       n_inference_steps=1, seed=0, device=device,
                       model_dtype=dtype, render_chunk=32)
for pi, prompt in enumerate(prompts):
    if str(pi) in state:
        continue
    z = torch.load(os.path.join(OUT, f"latents_{pi:02d}.pt"))
    bred, iid = z["bred"].to(device), z["iid"].to(device)
    delta = bred - iid
    low, high = band_split(delta)
    variants = {
        "iid": iid,
        "full": bred,
        "low": repair(iid + low),
        "high": repair(iid + high),
    }
    rec = {"prompt": prompt}
    with torch.no_grad():
        for name, lat in variants.items():
            d, c = score(ev2, ev2.render(lat, prompt), prompt)
            rec[f"{name}_dino"], rec[f"{name}_clip"] = d, c
    state[str(pi)] = rec
    json.dump(state, open(state_path, "w"), indent=1)
    log(f"[{pi}] iid={rec['iid_dino']:.4f} full={rec['full_dino']:.4f} "
        f"low={rec['low_dino']:.4f} high={rec['high_dino']:.4f}")

# --- finalize ----------------------------------------------------------------
rows = list(state.values())
m = lambda key: sum(r[key] for r in rows) / len(rows)
res = {"n": len(rows),
       **{f"{v}_dino": m(f"{v}_dino") for v in ("iid", "full", "low", "high")},
       **{f"{v}_clip": m(f"{v}_clip") for v in ("iid", "full", "low", "high")}}
for v in ("full", "low", "high"):
    res[f"{v}_gain_pct"] = 100.0 * (res[f"{v}_dino"] - res["iid_dino"]) / res["iid_dino"]
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, sort_keys=True)
log("E10B_RESULTS " + json.dumps({k: round(v, 4) for k, v in res.items()}))
log("E10B_DONE")
