"""MAP-Elites quality-diversity archive over noise-set genomes (exploration).

Instead of a single Pareto front (NSGA-II), MAP-Elites keeps an *archive* of
elites indexed by a behaviour descriptor, enforcing diversity in behaviour
space by construction. For collapse recovery this is a natural fit: we want
many *different ways* of being diverse, not one champion.

Descriptor (2D, both cheap, computed from the rendered set):
    d1 = colour diversity of the set   (mean pairwise colour-histogram distance)
    d2 = layout diversity of the set   (tiny 32x32 L2 distance)
Cell fitness = DINO patch diversity + w_q * CLIP quality (scalar within cell).

Loop: sample a batch of elites uniformly from the archive, apply the same
spectral self-adaptive mutation + set crossover as NSGA-II, evaluate the batch
in one render, insert offspring into their cells if fitter.

Reported: archive coverage, QD-score (sum of cell fitness), best genome overall
vs the NSGA-II baseline on the same prompts.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional, Tuple

import torch

from evodiv.genome import GenomeSpec, Population, crossover, init_population, mutate


class Archive:
    def __init__(self, bins: int, lo: Tuple[float, float], hi: Tuple[float, float], spec: GenomeSpec):
        self.bins = bins
        self.lo, self.hi = lo, hi
        self.spec = spec
        self.fit: Dict[Tuple[int, int], float] = {}
        self.lat: Dict[Tuple[int, int], torch.Tensor] = {}
        self.ls: Dict[Tuple[int, int], float] = {}
        self.lr: Dict[Tuple[int, int], float] = {}
        self.meta: Dict[Tuple[int, int], dict] = {}

    def cell(self, d1: float, d2: float) -> Tuple[int, int]:
        import math
        c1 = min(self.bins - 1, max(0, int((d1 - self.lo[0]) / (self.hi[0] - self.lo[0]) * self.bins)))
        c2 = min(self.bins - 1, max(0, int((d2 - self.lo[1]) / (self.hi[1] - self.lo[1]) * self.bins)))
        return (c1, c2)

    def insert(self, key, fit, lat, ls, lr, meta) -> bool:
        if key not in self.fit or fit > self.fit[key]:
            self.fit[key] = fit
            self.lat[key] = lat.detach().clone()
            self.ls[key] = float(ls)
            self.lr[key] = float(lr)
            self.meta[key] = meta
            return True
        return False

    @property
    def coverage(self) -> int:
        return len(self.fit)

    @property
    def qd_score(self) -> float:
        return float(sum(self.fit.values()))

    def best(self):
        key = max(self.fit, key=self.fit.get)
        return key, self.fit[key], self.meta[key]

    def sample_parents(self, n: int, generator: torch.Generator, device) -> Population:
        keys = list(self.fit.keys())
        idx = torch.randint(0, len(keys), (n,), generator=generator, device=device).tolist()
        lat = torch.stack([self.lat[keys[i]] for i in idx], 0).to(device)
        ls = torch.tensor([self.ls[keys[i]] for i in idx], device=device)
        lr = torch.tensor([self.lr[keys[i]] for i in idx], device=device)
        return Population(lat, ls, lr, self.spec)


@torch.no_grad()
def _descriptors(evaluator, images: torch.Tensor, P: int, B: int,
                 color_obj, tiny_obj) -> List[Tuple[float, float]]:
    out = []
    for i in range(P):
        c = float(color_obj.compute_pairwise_diversity(images[i]).item())
        t = float(tiny_obj.compute_pairwise_diversity(images[i]).item())
        out.append((c, t))
    return out


def run_map_elites(
    spec: GenomeSpec,
    evaluator,
    prompt: str,
    color_obj,
    tiny_obj,
    device,
    generator: torch.Generator,
    batch: int = 40,
    iterations: int = 60,
    bins: int = 10,
    w_quality: float = 3.0,
    desc_lo=(0.0, 0.0),
    desc_hi=(0.4, 0.6),
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    """Run MAP-Elites for one prompt; returns summary dict (history + best)."""
    log = log_fn or (lambda s: None)
    arch = Archive(bins, desc_lo, desc_hi, spec)

    def evaluate_batch(pop: Population):
        P, B = pop.size, spec.set_size
        flat = pop.latents.reshape(P * B, spec.channels, spec.height, spec.width)
        images = evaluator.render(flat, prompt, exact=False)
        images = images.reshape(P, B, *images.shape[1:])
        F, raw = evaluator._score_population_batched(images, prompt, P, B)
        descs = _descriptors(evaluator, images, P, B, color_obj, tiny_obj)
        del images, flat
        return raw, descs

    # seed the archive with a random population
    pop = init_population(spec, batch, device, seed=int(torch.randint(0, 10**6, (1,), generator=generator, device=device).item()))
    raw, descs = evaluate_batch(pop)
    for i in range(pop.size):
        fit = raw[i]["diversity_dino"] + w_quality * raw[i]["CLIP"]
        arch.insert(arch.cell(*descs[i]), fit, pop.latents[i], pop.log_sigma[i], pop.logit_rate[i],
                    {"dino": raw[i]["diversity_dino"], "clip": raw[i]["CLIP"], "desc": descs[i]})

    history = []
    for it in range(1, iterations + 1):
        parents_a = arch.sample_parents(batch, generator, device)
        parents_b = arch.sample_parents(batch, generator, device)
        idx = torch.arange(batch, device=device)
        child_lat, ls, lr = crossover(
            torch.cat([parents_a.latents, parents_b.latents], 0),
            torch.cat([parents_a.log_sigma, parents_b.log_sigma], 0),
            torch.cat([parents_a.logit_rate, parents_b.logit_rate], 0),
            idx, idx + batch, spec, generator,
        )
        child_lat, ls, lr = mutate(child_lat, ls, lr, spec, generator)
        children = Population(child_lat, ls, lr, spec)
        raw, descs = evaluate_batch(children)
        inserted = 0
        for i in range(batch):
            fit = raw[i]["diversity_dino"] + w_quality * raw[i]["CLIP"]
            if arch.insert(arch.cell(*descs[i]), fit, children.latents[i], ls[i], lr[i],
                           {"dino": raw[i]["diversity_dino"], "clip": raw[i]["CLIP"], "desc": descs[i]}):
                inserted += 1
        _, best_fit, best_meta = arch.best()
        rec = {"iter": it, "coverage": arch.coverage, "qd_score": round(arch.qd_score, 3),
               "inserted": inserted, "best_fit": round(best_fit, 4),
               "best_dino": round(best_meta["dino"], 4), "best_clip": round(best_meta["clip"], 4)}
        history.append(rec)
        if it % 10 == 0 or it == iterations:
            log(f"[ME it {it:03d}] cover={rec['coverage']}/{bins*bins} qd={rec['qd_score']:.1f} "
                f"ins={inserted} best(dino={rec['best_dino']}, clip={rec['best_clip']})")

    _, best_fit, best_meta = arch.best()
    # top DINO across archive (may differ from best scalar fitness)
    top_dino = max(m["dino"] for m in arch.meta.values())
    return {"prompt": prompt, "history": history, "coverage": arch.coverage,
            "qd_score": arch.qd_score, "best": best_meta, "top_dino_in_archive": top_dino}
