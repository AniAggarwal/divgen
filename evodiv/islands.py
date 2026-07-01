"""Island model with prompt-typed subpopulations (requested idea 1).

Motivation. The ECAD ablations tested prompt distributions (painted
landscapes, 5-word prompts, GPT-generated sets) *separately*. Here we run them
*simultaneously* as isolated islands, each evolving noise-set genomes against
its own prompt type, and migrate the top genomes between islands every
``migration_interval`` generations (ring topology).

Because a genome is a set of noise latents -- not tied to any prompt -- a
migrant carries over verbatim and is simply re-evaluated on the destination
island's prompts. A genome that keeps winning after repeatedly migrating across
prompt types is, by construction, one whose diversity *generalises* across
prompt types rather than overfitting one prompt. At the end we score the
best genomes on the union of all islands' prompts to quantify that.

The per-island engine is the same NSGA-II used by ``evolve`` (non-dominated
sort + crowding survival, binary-tournament selection, spectral self-adaptive
operators); only the migration step is added.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

import torch

from evodiv.genome import Population, crossover, mutate, init_population
from evodiv.nsga2 import EvoResult, binary_tournament, rank_and_crowding, survival


def _load_island_prompts(cfg, log) -> List[List[str]]:
    """Resolve island prompt lists.

    ``evolution.island_prompt_files`` is a comma-separated list of files, one
    per island (each file: one prompt per line). If empty, fall back to slicing
    the task prompt list into ``islands`` contiguous groups.
    """
    files = [f.strip() for f in cfg.evolution.island_prompt_files.split(",") if f.strip()]
    cap = max(1, cfg.evolution.island_eval_prompts)
    if files:
        out = []
        for f in files:
            with open(f) as fp:
                pr = [ln.strip() for ln in fp if ln.strip()]
            out.append(pr[:cap])
            log(f"  island prompts <- {os.path.basename(f)}: {len(pr[:cap])} prompt(s)")
        return out
    raise ValueError(
        "islands>1 requires --evolution.island_prompt_files (comma-separated prompt files)."
    )


def run_islands(
    cfg,
    spec,
    evaluator,
    prompt,                       # ignored; islands use their own prompt files
    device,
    generator: torch.Generator,
    log_fn: Optional[Callable[[str], None]] = None,
) -> EvoResult:
    log = log_fn or (lambda s: None)
    island_prompts = _load_island_prompts(cfg, log)
    K = len(island_prompts)
    P = cfg.evolution.pop_size
    G = cfg.evolution.n_generations
    m_int = max(1, cfg.evolution.migration_interval)
    m_size = max(0, cfg.evolution.migration_size)

    islands: List[Population] = []
    for k in range(K):
        pop = init_population(
            spec, P, device, noise_type=cfg.optimization.noise_type,
            noise_exponent=cfg.optimization.noise_exponent, seed=cfg.evolution.seed + 7919 * k,
        )
        evaluator.evaluate(pop, island_prompts[k], log_all=False)
        islands.append(pop)

    history: List[Dict[str, float]] = []

    def snapshot(gen: int):
        rec = {"gen": gen}
        for k in range(K):
            div = [r["_diversity"] for r in islands[k].raw]
            qual = [r["_quality"] for r in islands[k].raw]
            rec[f"island{k}_best_div"] = max(div)
            rec[f"island{k}_mean_div"] = sum(div) / len(div)
            rec[f"island{k}_best_qual"] = max(qual)
        rec["n_renders"] = evaluator.n_renders
        history.append(rec)
        log(f"[gen {gen:03d}] " + " ".join(
            f"isl{k}(div={rec[f'island{k}_best_div']:.4f})" for k in range(K)))

    snapshot(0)
    for gen in range(1, G + 1):
        for k in range(K):
            pop = islands[k]
            rank, cd, _ = rank_and_crowding(pop.F)
            pa = binary_tournament(rank, cd, P, generator)
            pb = binary_tournament(rank, cd, P, generator)
            cl, cls_, clr = crossover(pop.latents, pop.log_sigma, pop.logit_rate,
                                      pa, pb, spec, generator, cfg.evolution.p_crossover)
            cl, cls_, clr = mutate(cl, cls_, clr, spec, generator)
            off = Population(cl, cls_, clr, spec)
            evaluator.evaluate(off, island_prompts[k], log_all=False)
            islands[k] = survival(Population.concat(pop, off), P)

        # migration (ring): island k sends its top m_size genomes to island k+1
        if m_size and gen % m_int == 0 and K > 1:
            migrants = []
            for k in range(K):
                rank, cd, _ = rank_and_crowding(islands[k].F)
                order = sorted(range(islands[k].size),
                               key=lambda i: (rank[i].item(), -cd[i].item()))
                idx = torch.tensor(order[:m_size], device=device)
                migrants.append(islands[k].select(idx))
            for k in range(K):
                dst = (k + 1) % K
                incoming = migrants[k]
                evaluator.evaluate(incoming, island_prompts[dst], log_all=False)  # re-score on dst
                combined = Population.concat(islands[dst], incoming)
                islands[dst] = survival(combined, P)
            log(f"[gen {gen:03d}] migrated {m_size} genome(s) per island (ring)")

        snapshot(gen)

    # --- generalisation report: score each island's Pareto genomes on ALL prompts ---
    all_prompts = [p for grp in island_prompts for p in grp]
    log(f"[islands] scoring elites on the union of all {len(all_prompts)} prompts for generalisation")
    # pick the best-diversity genome per island, gather into one population
    elites = []
    for k in range(K):
        bi = max(range(islands[k].size), key=lambda i: islands[k].raw[i]["_diversity"])
        elites.append(islands[k].select(torch.tensor([bi], device=device)))
    elite_pop = elites[0]
    for e in elites[1:]:
        elite_pop = Population.concat(elite_pop, e)
    evaluator.evaluate(elite_pop, all_prompts, log_all=True)
    for k in range(K):
        log(f"  island{k} elite -> cross-prompt diversity={elite_pop.raw[k]['_diversity']:.4f} "
            f"quality={elite_pop.raw[k]['_quality']:.4f}")

    # return a merged result whose population is the union of final islands
    merged = islands[0]
    for k in range(1, K):
        merged = Population.concat(merged, islands[k])
    # history[0] must expose best/* headline keys for run.py summary; synthesise from island0
    if merged.raw and not any(k.startswith("best/") for k in history[0]):
        best_i = max(range(merged.size), key=lambda i: merged.raw[i]["_diversity"])
        for k, v in merged.raw[best_i].items():
            if not k.startswith("_"):
                history[0][f"best/{k}"] = v
        history[0]["mean_quality"] = sum(r["_quality"] for r in merged.raw) / merged.size
    return EvoResult(population=merged, history=history)
