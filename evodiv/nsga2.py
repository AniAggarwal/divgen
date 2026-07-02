"""Multi-objective evolutionary loop (NSGA-II) for EvoDiv.

This is the same algorithm ECAD uses to evolve caching schedules, retargeted at
noise-set genomes. We reuse pymoo's fast non-dominated sorting (the exact
routine behind ECAD's ``NSGA2``) and implement crowding-distance survival and
binary tournament selection over the (quality, diversity) objective pair.

The loop is a textbook (mu + lambda) NSGA-II:
  1. evaluate the population,
  2. rank by non-dominated front + crowding distance,
  3. binary-tournament -> crossover -> mutate to make lambda = P offspring,
  4. evaluate offspring,
  5. survival: sort the combined 2P by (rank, crowding), keep the best P.

Per-generation statistics (best diversity, best quality, hypervolume-ish Pareto
size, mean self-adaptive sigma/rate) are recorded for plotting convergence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from evodiv.genome import Population, crossover, mutate


# --------------------------------------------------------------------------- #
# NSGA-II survival primitives                                                  #
# --------------------------------------------------------------------------- #


def crowding_distance(F: torch.Tensor, fronts: List) -> torch.Tensor:
    """Standard NSGA-II crowding distance, computed per front."""
    n = F.shape[0]
    cd = torch.zeros(n, device=F.device)
    Fnp = F.detach().cpu().numpy()
    for front in fronts:
        if len(front) == 0:
            continue
        front = list(front)
        if len(front) <= 2:
            for i in front:
                cd[i] = float("inf")
            continue
        for m in range(F.shape[1]):
            vals = [(Fnp[i, m], i) for i in front]
            vals.sort(key=lambda t: t[0])
            fmin, fmax = vals[0][0], vals[-1][0]
            cd[vals[0][1]] = float("inf")
            cd[vals[-1][1]] = float("inf")
            span = (fmax - fmin) or 1e-12
            for k in range(1, len(vals) - 1):
                cd[vals[k][1]] += (vals[k + 1][0] - vals[k - 1][0]) / span
    return cd


def rank_and_crowding(F: torch.Tensor):
    """Return (rank per individual, crowding distance per individual, fronts)."""
    fronts = NonDominatedSorting().do(F.detach().cpu().numpy())
    fronts = [list(f) for f in fronts]
    rank = torch.empty(F.shape[0], dtype=torch.long, device=F.device)
    for r, front in enumerate(fronts):
        for i in front:
            rank[i] = r
    cd = crowding_distance(F, fronts)
    return rank, cd, fronts


def survival(pop: Population, n_survive: int) -> Population:
    """Keep the best ``n_survive`` genomes by (front rank, -crowding)."""
    rank, cd, fronts = rank_and_crowding(pop.F)
    keep: List[int] = []
    for front in fronts:
        if len(keep) + len(front) <= n_survive:
            keep.extend(front)
        else:
            remaining = n_survive - len(keep)
            front_sorted = sorted(front, key=lambda i: cd[i].item(), reverse=True)
            keep.extend(front_sorted[:remaining])
            break
    idx = torch.tensor(keep, device=pop.F.device, dtype=torch.long)
    return pop.select(idx)


def binary_tournament(rank: torch.Tensor, cd: torch.Tensor, n: int, generator: torch.Generator):
    """n binary tournaments: lower rank wins; ties broken by larger crowding."""
    device = rank.device
    P = rank.shape[0]
    a = torch.randint(0, P, (n,), device=device, generator=generator)
    b = torch.randint(0, P, (n,), device=device, generator=generator)
    a_better = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (cd[a] > cd[b]))
    return torch.where(a_better, a, b)


# --------------------------------------------------------------------------- #
# Main loop                                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class EvoResult:
    population: Population
    history: List[Dict[str, float]] = field(default_factory=list)

    def best_by(self, key: str = "_diversity", quality_floor: Optional[float] = None):
        """Return the index of the genome maximising ``key`` (optionally subject
        to a quality floor). Ties/empties fall back to the global max of key."""
        raw = self.population.raw
        cands = range(len(raw))
        if quality_floor is not None:
            filt = [i for i in cands if raw[i].get("_quality", -1e9) >= quality_floor]
            if filt:
                cands = filt
        return max(cands, key=lambda i: raw[i].get(key, -1e9))


def evolve(
    pop: Population,
    evaluator,
    prompt: str,
    n_generations: int,
    generator: torch.Generator,
    p_crossover: float = 0.9,
    log_fn: Optional[Callable[[str], None]] = None,
    log_all_every: int = 0,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.002,
    target_diversity: Optional[float] = None,
    target_quality: Optional[float] = None,
    target_quality_frac: Optional[float] = None,
    exact_every: int = 10,
) -> EvoResult:
    """Run NSGA-II for up to ``n_generations`` on a single prompt.

    Early termination (both optional, for large sweeps):
      * patience: stop after ``early_stop_patience`` generations without the
        best diversity improving by ``early_stop_min_delta``;
      * targets: stop once ``best_diversity >= target_diversity`` AND
        ``best_quality >= target_quality`` (if targets given).
    """
    log = log_fn or (lambda s: None)
    spec = pop.spec
    # gen 0 = the reported i.i.d. baseline -> always exact (full VAE)
    evaluator.evaluate(pop, prompt, log_all=True, exact=True)
    history: List[Dict[str, float]] = []

    def record(gen: int):
        raw = pop.raw
        div = [r["_diversity"] for r in raw]
        qual = [r["_quality"] for r in raw]
        rank, _, fronts = rank_and_crowding(pop.F)
        rec = {
            "gen": gen,
            "best_diversity": max(div),
            "mean_diversity": sum(div) / len(div),
            "best_quality": max(qual),
            "mean_quality": sum(qual) / len(qual),
            "pareto_size": int((rank == 0).sum().item()),
            "mean_sigma": float(pop.sigma().mean().item()),
            "mean_rate": float(pop.rate().mean().item()),
            "n_renders": evaluator.n_renders,
        }
        # surface headline raw metrics (e.g. diversity_dino, CLIP) at the best-diversity genome
        best_i = max(range(len(raw)), key=lambda i: raw[i]["_diversity"])
        for k, v in raw[best_i].items():
            if not k.startswith("_"):
                rec[f"best/{k}"] = v
        history.append(rec)
        log(
            f"[gen {gen:03d}] div(best={rec['best_diversity']:.4f} mean={rec['mean_diversity']:.4f}) "
            f"qual(best={rec['best_quality']:.4f} mean={rec['mean_quality']:.4f}) "
            f"pareto={rec['pareto_size']} sigma={rec['mean_sigma']:.3f} rate={rec['mean_rate']:.3f} "
            f"renders={rec['n_renders']}"
        )

    record(0)
    if target_quality is None and target_quality_frac is not None:
        target_quality = target_quality_frac * history[0]["mean_quality"]
    best_div_seen = history[0]["best_diversity"]
    stale = 0
    for gen in range(1, n_generations + 1):
        rank, cd, _ = rank_and_crowding(pop.F)
        n_off = pop.size
        pa = binary_tournament(rank, cd, n_off, generator)
        pb = binary_tournament(rank, cd, n_off, generator)

        child_lat, child_ls, child_lr = crossover(
            pop.latents, pop.log_sigma, pop.logit_rate, pa, pb, spec, generator, p_crossover
        )
        child_lat, child_ls, child_lr = mutate(child_lat, child_ls, child_lr, spec, generator)
        offspring = Population(child_lat, child_ls, child_lr, spec)

        log_all = (log_all_every > 0 and gen % log_all_every == 0) or (gen == n_generations)
        # offspring are scored with the surrogate decoder (if loaded)
        evaluator.evaluate(offspring, prompt, log_all=log_all, exact=False)

        combined = Population.concat(pop, offspring)
        pop = survival(combined, pop.size)
        # periodic re-anchoring: re-score survivors with the exact decoder so
        # surrogate noise (mainly on the quality axis) cannot accumulate
        if exact_every and gen % exact_every == 0:
            evaluator.evaluate(pop, prompt, log_all=False, exact=True)
        record(gen)

        rec = history[-1]
        if rec["best_diversity"] > best_div_seen + early_stop_min_delta:
            best_div_seen = rec["best_diversity"]
            stale = 0
        else:
            stale += 1
        if (target_diversity is not None and rec["best_diversity"] >= target_diversity
                and (target_quality is None or rec["best_quality"] >= target_quality)):
            log(f"[early stop] targets reached at gen {gen} "
                f"(div {rec['best_diversity']:.4f} >= {target_diversity}, "
                f"qual {rec['best_quality']:.4f})")
            break
        if early_stop_patience and stale >= early_stop_patience:
            log(f"[early stop] no diversity improvement for {stale} generations (gen {gen})")
            break

    # final selection must be exact + carry held-out metrics for reporting
    evaluator.evaluate(pop, prompt, log_all=True, exact=True)
    return EvoResult(population=pop, history=history)
