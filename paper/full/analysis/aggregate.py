"""Aggregate every archived + new run into paper_numbers.json.

Single source of truth for every number that appears in either paper.
Reads:
  $EVODIV_RUNS               -- the restored 2026-07-02 archive (geneval_merged,
                                geneval_full552, design_ablations, ablation_mut,
                                explore_* results JSONs, explore_setsize)
  $PAPERPREP (default /workspace/runs/paperprep) -- the new E1..E11 runs; each
                                experiment directory provides results.json
                                written by its runner script.

Writes (next to this script):
  paper_numbers.json  -- nested means (everything the tex macros consume)
  per_prompt.json     -- per-prompt metric vectors for paired stats (stats.py)

Usage: python paper/full/analysis/aggregate.py [--check]
  --check: verify the recomputed archived headline against the values committed
           in the 3-pager (fails loudly on drift; run before integrating
           anything new).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("EVODIV_RUNS", "/workspace/runs-restore/runs")
PREP = os.environ.get("PAPERPREP", "/workspace/runs/paperprep")

# metric keys inside history.json:best_metrics / summary.json rows
METRICS = [
    "CLIP", "diversity_dino", "diversity_dreamsim", "diversity_lpips",
    "diversity_vendi", "diversity_vendi_sscd", "diversity_dpp",
    "diversity_dpp_raw", "diversity_dpp_patch", "diversity_dpp_patch_raw",
    "diversity_color", "diversity_tiny_l2",
]

# Rows quoted from other papers (provenance; the only numbers not derived from
# runs -- they are citations, single-sourced here).
REFERENCE = {
    "harrington_iid":    {"diversity_dino": 0.588, "diversity_dreamsim": 0.249,
                          "diversity_lpips": 0.642, "CLIP": 0.335},
    "parmar":            {"diversity_dino": 0.705, "diversity_dreamsim": 0.331,
                          "diversity_lpips": 0.682, "CLIP": 0.333},
    "harrington_grad":   {"diversity_dino": 0.784, "diversity_dreamsim": 0.411,
                          "diversity_lpips": 0.767, "CLIP": 0.349},
    "harrington_grad_s_per_iter_a100": 0.345,   # their Sec. 4.2
    "_provenance": "Harrington et al. CVPR26 Tab.1/Sec 4.2; Parmar arXiv:2508.15773",
}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def collect_history_run(root: str):
    """Mean best_metrics over every <prompt>/history.json under root.

    Returns (means dict, per-prompt dict, n). Prompt key = dir name.
    """
    per = {}
    for hp in sorted(glob.glob(os.path.join(root, "*", "history.json"))):
        with open(hp) as fp:
            h = json.load(fp)
        bm = h.get("best_metrics", {})
        per[os.path.basename(os.path.dirname(hp))] = {
            m: bm.get(m) for m in METRICS if m in bm
        }
    means = {m: _mean([v.get(m) for v in per.values()]) for m in METRICS}
    means = {m: v for m, v in means.items() if v is not None}
    return means, per, len(per)


def read_summary(root: str):
    """Read a nested run's summary.json (init/best means)."""
    sp = os.path.join(root, "summary.json")
    if not os.path.exists(sp):
        return None
    with open(sp) as fp:
        return json.load(fp)


def nested(run: str, experiment: str = "geneval", model: str = "sdxl-turbo") -> str:
    return os.path.join(RUNS, run, "evodiv", model, experiment)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    out = {"reference": REFERENCE}
    per_prompt = {}

    # ---- archived headline: geneval_merged (flat layout, 553 prompts) ---- #
    merged_means, merged_per, n = collect_history_run(os.path.join(RUNS, "geneval_merged"))
    out["geneval_merged"] = {"n": n, "best": merged_means}
    per_prompt["geneval_merged"] = merged_per

    # ---- full552 (pre-extension pass) incl. the i.i.d. init row ---------- #
    s = read_summary(nested("geneval_full552"))
    if s:
        out["geneval_full552"] = {"n": 553, "init": s["init"], "best": s["best"]}
    _, f552_per, _ = collect_history_run(nested("geneval_full552"))
    per_prompt["geneval_full552"] = f552_per

    # ---- design ablations (8 conditions x 6 prompts) --------------------- #
    out["design_ablations"] = {}
    da_root = os.path.join(RUNS, "design_ablations")
    if os.path.isdir(da_root):
        for cond in sorted(os.listdir(da_root)):
            croot = nested(os.path.join("design_ablations", cond))
            s = read_summary(croot)
            if s:
                m, p, nn = collect_history_run(croot)
                out["design_ablations"][cond] = {"n": nn, "init": s["init"], "best": s["best"]}
                per_prompt[f"design_ablations/{cond}"] = p

    # ---- mutation ablation (3 conditions x 4 prompts) -------------------- #
    out["ablation_mut"] = {}
    am_root = os.path.join(RUNS, "ablation_mut")
    if os.path.isdir(am_root):
        for cond in sorted(os.listdir(am_root)):
            croot = nested(os.path.join("ablation_mut", cond))
            s = read_summary(croot)
            if s:
                out["ablation_mut"][cond] = {"init": s["init"], "best": s["best"]}

    # ---- explore_* results JSONs (shapes per the archive) ----------------- #
    explore = {}
    p = os.path.join(RUNS, "explore_cmaes_results.json")
    if os.path.exists(p):
        pairs = json.load(open(p))
        explore["cmaes40"] = {
            "n": len(pairs),
            "diversity_dino": _mean([x[0] for x in pairs]),
            "CLIP": _mean([x[1] for x in pairs]),
        }
        per_prompt["explore_cmaes40"] = {
            str(i): {"diversity_dino": x[0], "CLIP": x[1]} for i, x in enumerate(pairs)
        }
    for name, fn in [("dctnsga", "explore_dctnsga_results.json"),
                     ("gp2", "explore_gp2_results.json"),
                     ("gp2b", "explore_gp2b_results.json"),
                     ("hybrid", "explore_hybrid_results.json"),
                     ("lexicase", "explore_lexicase_results.json")]:
        p = os.path.join(RUNS, fn)
        if not os.path.exists(p):
            continue
        arms = json.load(open(p))
        explore[name] = {
            arm: {"n": len(v),
                  "diversity_dino": _mean([x[0] for x in v]),
                  "CLIP": _mean([x[1] for x in v])}
            for arm, v in arms.items()
        }
    p = os.path.join(RUNS, "explore_transfer_results.json")
    if os.path.exists(p):
        rows = json.load(open(p))
        iid = _mean([r["pixart_iid_dino"] for r in rows])
        bred = _mean([r["pixart_bred_dino"] for r in rows])
        explore["transfer"] = {
            "n": len(rows),
            "pixart_iid_dino": iid, "pixart_bred_dino": bred,
            "pixart_iid_clip": _mean([r["pixart_iid_clip"] for r in rows]),
            "pixart_bred_clip": _mean([r["pixart_bred_clip"] for r in rows]),
            "dino_gain_pct": 100.0 * (bred - iid) / iid,
        }
    # gp_full: program-space search (single-task nested run)
    s = read_summary(nested("gp_full", "single"))
    if s:
        explore["gp_full"] = {"best": s["best"], "init": s["init"]}
    # islands: elite cross-prompt generalisation, parsed from the run log
    ilog = os.path.join(RUNS, "islands_full.log")
    if os.path.exists(ilog):
        import re
        vals = [float(m.group(1)) for m in
                re.finditer(r"cross-prompt diversity=([0-9.]+)", open(ilog).read())]
        if vals:
            explore["islands"] = {"cross_prompt_dino": vals,
                                  "min": min(vals), "max": max(vals)}
    # set-size study: B4/B8/B16 nested runs
    ss_root = os.path.join(RUNS, "explore_setsize")
    if os.path.isdir(ss_root):
        explore["setsize"] = {}
        for b in sorted(os.listdir(ss_root)):
            s = read_summary(os.path.join(ss_root, b, "evodiv", "sdxl-turbo", "geneval"))
            if s:
                explore["setsize"][b] = {"init": s["init"], "best": s["best"]}
    out["explore"] = explore

    # ---- new paperprep experiments (E1..E11) ------------------------------ #
    # Each runner writes <exp>/results.json in a self-describing format;
    # trust it and mount it under its experiment key.
    out["paperprep"] = {}
    if os.path.isdir(PREP):
        for exp in sorted(os.listdir(PREP)):
            rp = os.path.join(PREP, exp, "results.json")
            if os.path.exists(rp):
                with open(rp) as fp:
                    out["paperprep"][exp] = json.load(fp)
            pp = os.path.join(PREP, exp, "per_prompt.json")
            if os.path.exists(pp):
                with open(pp) as fp:
                    per_prompt[f"paperprep/{exp}"] = json.load(fp)

    # E4's memory table exists as soon as the probes ran (before breeding ends)
    mt = os.path.join(PREP, "e4_flux", "memory_table.json")
    if os.path.exists(mt) and "memory_table" not in out["paperprep"].get("e4_flux", {}):
        out["paperprep"].setdefault("e4_flux", {})["memory_table"] = json.load(open(mt))

    # E3 covariance dumps -> low-band variance fraction (E10a)
    cov_files = glob.glob(os.path.join(PREP, "e3_cmadct", "*", "cma_cov.npz"))
    if len(cov_files) >= 10:
        import numpy as np
        K = 8
        kk, ll = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
        radial = (kk + ll).reshape(-1)
        low = radial <= 4          # lowest third of the 0..14 radial bands
        fracs = []
        for f in cov_files:
            cv = np.load(f)["coord_var"].reshape(-1, K * K).mean(0)
            fracs.append(float(cv[low].sum() / cv.sum()))
        out.setdefault("paperprep", {}).setdefault("e3_cmadct", {})[
            "cov_lowband_frac"] = float(np.mean(fracs))
        out["paperprep"]["e3_cmadct"]["cov_n"] = len(fracs)

    # E8 vendi plateau: generation where the mean best-vendi curve reaches
    # 99% of its final value
    e8 = glob.glob(os.path.join(PREP, "e8_vendi", "evodiv", "sdxl-turbo",
                                "*", "[0-9]*", "history.json"))
    if len(e8) >= 10:
        import numpy as np
        G = 0
        curves = []
        for hp in e8:
            h = json.load(open(hp))["history"]
            c = [e.get("best_diversity") for e in h]
            curves.append(c)
            G = max(G, len(c))
        m = np.full((len(curves), G), np.nan)
        for i, c in enumerate(curves):
            m[i, :len(c)] = c
            m[i, len(c):] = c[-1]
        mean = np.nanmean(m, 0)
        target = mean[0] + 0.99 * (mean[-1] - mean[0])
        plateau = int(np.argmax(mean >= target))
        out["paperprep"].setdefault("e8_vendi", {}).update(
            {"plateau_gen": plateau, "final_best_vendi_curve": float(mean[-1]),
             "init_vendi_curve": float(mean[0])})

    # ---- write ------------------------------------------------------------ #
    with open(os.path.join(HERE, "paper_numbers.json"), "w") as fp:
        json.dump(out, fp, indent=1, sort_keys=True)
    with open(os.path.join(HERE, "per_prompt.json"), "w") as fp:
        json.dump(per_prompt, fp)
    print(f"paper_numbers.json written; geneval_merged n={n}")
    print("headline:", {k: round(v, 4) for k, v in merged_means.items()
                        if k in ("diversity_dino", "diversity_dreamsim",
                                 "diversity_lpips", "CLIP", "diversity_vendi",
                                 "diversity_dpp")})

    # ---- sanity gate vs the committed 3-pager ----------------------------- #
    if args.check:
        gate = {"diversity_dino": 0.790, "diversity_dreamsim": 0.437,
                "diversity_lpips": 0.745, "CLIP": 0.393,
                "diversity_vendi": 2.852, "diversity_dpp": 0.878}
        bad = []
        for k, want in gate.items():
            got = merged_means[k]
            if abs(got - want) > 5e-4 + 1e-3 * abs(want):
                bad.append(f"{k}: recomputed {got:.4f} != committed {want}")
        init = out.get("geneval_full552", {}).get("init", {})
        gate_init = {"diversity_dino": 0.658, "diversity_dreamsim": 0.297,
                     "diversity_lpips": 0.670, "CLIP": 0.375}
        for k, want in gate_init.items():
            got = init.get(k)
            if got is None or abs(got - want) > 2e-3:
                bad.append(f"init {k}: recomputed {got} != committed {want}")
        if n != 553:
            bad.append(f"geneval_merged has {n} prompts, expected 553")
        if bad:
            print("SANITY GATE FAILED:\n  " + "\n  ".join(bad))
            sys.exit(1)
        print("sanity gate PASSED: archive reproduces the committed 3-pager numbers")


if __name__ == "__main__":
    main()
