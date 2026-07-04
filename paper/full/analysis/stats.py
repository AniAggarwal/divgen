"""Paired significance tests over per-prompt metric vectors.

Reads per_prompt.json (written by aggregate.py) and emits stats_results.json.
Every comparison is paired by prompt key; Wilcoxon signed-rank two-sided,
Holm-corrected within each named family. Comparisons whose inputs are missing
are skipped silently (so this can run at any point of the campaign).

Usage: python paper/full/analysis/stats.py
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))

# (family, name, run_a, run_b, metric) -- positive diff means a > b.
# run keys refer to per_prompt.json entries; metric to best_metrics keys
# (or judge-score keys for paperprep/e5 blocks).
COMPARISONS = [
    # E3 decision rule: CMA-DCT vs breeding at 553 prompts
    ("e3_promotion", "cmadct_vs_bred_dino", "paperprep/e3_cmadct", "geneval_merged", "diversity_dino"),
    ("e3_promotion", "cmadct_vs_bred_clip", "paperprep/e3_cmadct", "geneval_merged", "CLIP"),
    # E1: same-harness gradient vs breeding (paired on the 553 prompts)
    ("e1_gradient", "bred_vs_grad_dino", "geneval_merged", "paperprep/e1_gradient", "diversity_dino"),
    ("e1_gradient", "bred_vs_grad_dreamsim", "geneval_merged", "paperprep/e1_gradient", "diversity_dreamsim"),
    ("e1_gradient", "bred_vs_grad_lpips", "geneval_merged", "paperprep/e1_gradient", "diversity_lpips"),
    ("e1_gradient", "bred_vs_grad_clip", "geneval_merged", "paperprep/e1_gradient", "CLIP"),
    ("e1_gradient", "bred_vs_grad_vendi", "geneval_merged", "paperprep/e1_gradient", "diversity_vendi"),
    ("e1_gradient", "bred_vs_grad_dpp", "geneval_merged", "paperprep/e1_gradient", "diversity_dpp"),
    # E2: breeding vs matched-compute random search
    ("e2_random", "bred_vs_rand_dino", "geneval_merged", "paperprep/e2_random", "diversity_dino"),
    ("e2_random", "bred_vs_rand_dreamsim", "geneval_merged", "paperprep/e2_random", "diversity_dreamsim"),
    ("e2_random", "bred_vs_rand_lpips", "geneval_merged", "paperprep/e2_random", "diversity_lpips"),
    ("e2_random", "bred_vs_rand_clip", "geneval_merged", "paperprep/e2_random", "CLIP"),
    ("e2_random", "bred_vs_rand_vendi", "geneval_merged", "paperprep/e2_random", "diversity_vendi"),
    # E1 (rescored): both methods' saved jpgs scored by identical metric code
    ("e1_rescored", "bred_vs_grad_rescored_dino", "paperprep/e5_judges:bred", "paperprep/e5_judges:grad", "diversity_dino"),
    ("e1_rescored", "bred_vs_grad_rescored_dreamsim", "paperprep/e5_judges:bred", "paperprep/e5_judges:grad", "diversity_dreamsim"),
    ("e1_rescored", "bred_vs_grad_rescored_lpips", "paperprep/e5_judges:bred", "paperprep/e5_judges:grad", "diversity_lpips"),
    ("e1_rescored", "bred_vs_grad_rescored_clip", "paperprep/e5_judges:bred", "paperprep/e5_judges:grad", "CLIP"),
    # E5: judge panel, breeding vs gradient on independent judges
    ("e5_judges", "bred_vs_grad_imagereward", "paperprep/e5_judges:bred", "paperprep/e5_judges:grad", "imagereward"),
    ("e5_judges", "bred_vs_grad_pickscore", "paperprep/e5_judges:bred", "paperprep/e5_judges:grad", "pickscore"),
    ("e5_judges", "bred_vs_grad_hpsv21", "paperprep/e5_judges:bred", "paperprep/e5_judges:grad", "hpsv21"),
    ("e5_judges", "rand_vs_grad_imagereward", "paperprep/e5_judges:rand", "paperprep/e5_judges:grad", "imagereward"),
    ("e5_judges", "bred_vs_rand_imagereward", "paperprep/e5_judges:bred", "paperprep/e5_judges:rand", "imagereward"),
    ("e5_judges", "bred_vs_rand_pickscore", "paperprep/e5_judges:bred", "paperprep/e5_judges:rand", "pickscore"),
    ("e5_judges", "bred_vs_rand_hpsv21", "paperprep/e5_judges:bred", "paperprep/e5_judges:rand", "hpsv21"),
    ("e5_judges", "cmadct_vs_grad_imagereward", "paperprep/e5_judges:cmadct", "paperprep/e5_judges:grad", "imagereward"),
]


def _prompt_key(k: str) -> str:
    """Normalize prompt-dir names so runs pair up.

    Archived dirs look like '0007_a photo of ...' (4-digit) and upstream
    gradient dirs like '00007' (5-digit index only); new runners use the
    archived convention. Pair on the numeric index.
    """
    head = k.split("_", 1)[0]
    if head.isdigit():
        return str(int(head))
    return k


def load_vectors(per_prompt: dict, run: str, metric: str):
    """Return {prompt_key: value}. 'run:sub' selects a sub-block."""
    sub = None
    if ":" in run:
        run, sub = run.split(":", 1)
    block = per_prompt.get(run)
    if block is None:
        return None
    if sub is not None:
        block = block.get(sub)
        if block is None:
            return None
    out = {}
    for k, v in block.items():
        val = v.get(metric)
        if val is not None:
            out[_prompt_key(k)] = val
    return out or None


def holm(pvals):
    """Holm step-down adjusted p-values (returns same order as input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        p = min(1.0, (m - rank) * pvals[i])
        prev = max(prev, p)
        adj[i] = prev
    return adj


def main():
    with open(os.path.join(HERE, "per_prompt.json")) as fp:
        per_prompt = json.load(fp)

    results, families = {}, {}
    for family, name, run_a, run_b, metric in COMPARISONS:
        va = load_vectors(per_prompt, run_a, metric)
        vb = load_vectors(per_prompt, run_b, metric)
        if not va or not vb:
            continue
        keys = sorted(set(va) & set(vb))
        if len(keys) < 8:
            continue
        a = np.array([va[k] for k in keys])
        b = np.array([vb[k] for k in keys])
        d = a - b
        if np.allclose(d, 0):
            w_p = 1.0
        else:
            w_p = float(sps.wilcoxon(a, b, zero_method="wilcox").pvalue)
        results[name] = {
            "family": family, "n": len(keys), "metric": metric,
            "run_a": run_a, "run_b": run_b,
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
            "frac_a_wins": float((d > 0).mean()),
            "wilcoxon_p": w_p,
        }
        families.setdefault(family, []).append(name)

    for family, names in families.items():
        adj = holm([results[n]["wilcoxon_p"] for n in names])
        for n, p in zip(names, adj):
            results[n]["holm_p"] = p
            results[n]["significant_05"] = bool(p < 0.05)

    with open(os.path.join(HERE, "stats_results.json"), "w") as fp:
        json.dump(results, fp, indent=1, sort_keys=True)
    print(f"stats_results.json written: {len(results)} comparisons")
    for n, r in sorted(results.items()):
        print(f"  {n}: n={r['n']} diff={r['mean_diff']:+.4f} "
              f"p={r['wilcoxon_p']:.2e} holm={r['holm_p']:.2e} "
              f"{'SIG' if r['significant_05'] else 'ns'}")


if __name__ == "__main__":
    main()
