"""Aggregate + plot EvoDiv runs.

Reads the per-prompt ``history.json`` files written by ``evodiv.run`` under a
run directory, produces:
  * a markdown table of init (i.i.d. gen-0) vs evolved diversity/quality,
    averaged across prompts, alongside the paper's reported numbers; and
  * convergence plots (diversity + quality + self-adaptive sigma/rate vs gen).

Usage:
    python -m evodiv.analyze --run_dir outputs/evodiv/sdxl-turbo/geneval [--out report]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

# Paper's reported SDXL-Turbo / GenEval numbers (Tab. 1, white noise) for context.
# These use the paper's own eval harness averaged over 552 prompts; our single-GPU
# runs use the same objective code but far fewer prompts, so treat as a reference band.
PAPER_TAB1_WHITE = {
    "i.i.d.": {"DINO": 0.588, "DreamSim": 0.249, "LPIPS": 0.642, "CLIP": 0.335},
    "Parmar": {"DINO": 0.705, "DreamSim": 0.331, "LPIPS": 0.682, "CLIP": 0.333},
    "Ours(grad)": {"DINO": 0.784, "DreamSim": 0.411, "LPIPS": 0.767, "CLIP": 0.349},
}

# map our metric keys -> display names used above
KEY2NAME = {
    "diversity_dino": "DINO", "diversity_dreamsim": "DreamSim",
    "diversity_lpips": "LPIPS", "CLIP": "CLIP", "diversity_vendi": "Vendi",
    "diversity_dpp": "DPP", "HPS": "HPS",
}


def load_histories(run_dir: str) -> List[dict]:
    files = sorted(glob.glob(os.path.join(run_dir, "**", "history.json"), recursive=True))
    return [json.load(open(f)) for f in files]


def aggregate(histories: List[dict]) -> Dict[str, Dict[str, float]]:
    init_acc, best_acc = defaultdict(float), defaultdict(float)
    n = 0
    for h in histories:
        gen0 = h["history"][0]
        best = h["best_metrics"]
        got = False
        for k, v in gen0.items():
            if k.startswith("best/"):
                metric = k[len("best/"):]
                init_acc[metric] += v
                best_acc[metric] += best.get(metric, float("nan"))
                got = True
        if got:
            n += 1
    n = max(n, 1)
    return {"init": {k: v / n for k, v in init_acc.items()},
            "best": {k: v / n for k, v in best_acc.items()}, "n": n}


def markdown_table(agg) -> str:
    metrics = [("diversity_dino", "DINO"), ("diversity_dreamsim", "DreamSim"),
               ("diversity_lpips", "LPIPS"), ("diversity_vendi", "Vendi"),
               ("diversity_dpp", "DPP"), ("CLIP", "CLIP"), ("HPS", "HPS")]
    present = [(k, name) for k, name in metrics if k in agg["init"]]
    lines = [f"EvoDiv results averaged over {agg['n']} prompt(s).", "",
             "| Method | " + " | ".join(name for _, name in present) + " |",
             "|" + "---|" * (len(present) + 1)]
    # paper reference rows (only for metrics we can name)
    for label, row in PAPER_TAB1_WHITE.items():
        cells = [f"{row.get(name, ''):.3f}" if name in row else "-" for _, name in present]
        lines.append(f"| paper {label} | " + " | ".join(cells) + " |")
    lines.append("| **ours init (i.i.d.)** | " +
                 " | ".join(f"{agg['init'][k]:.3f}" for k, _ in present) + " |")
    lines.append("| **ours evolved** | " +
                 " | ".join(f"{agg['best'][k]:.3f}" for k, _ in present) + " |")
    return "\n".join(lines)


# Styling follows the ECAD (editing-dits) notebooks: seaborn whitegrid,
# 2.5-width lines with black-edged markers, dashed black reference lines,
# "#1f77b4" blue for ours, constrained layout.
ECAD_COLORS = {
    "ours": "#1f77b4",      # blue   (matches "Ours" in the ECAD frontier plots)
    "mean": "#7f7f7f",      # gray
    "quality": "#2ca02c",   # green
    "sigma": "#d62728",     # red
    "rate": "#ff7f0e",      # orange
}


def plot_convergence(histories: List[dict], out_png: str,
                     ref_lines: Optional[Dict[str, float]] = None):
    """Convergence figure in the ECAD notebook style.

    ref_lines: optional {label: y} dashed black reference lines for the
    diversity panel (e.g. the paper's gradient result and i.i.d. baseline).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.75)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    # early-stopped prompts hold their final value (they stopped because they
    # converged), so curves are averaged over ALL prompts at every generation
    max_len = max(len(h["history"]) for h in histories)
    gens = list(range(max_len))
    mark_every = max(1, max_len // 10)
    marker_kw = dict(marker="o", markersize=6, markeredgecolor="black",
                     markeredgewidth=1.0, markevery=mark_every)

    def avg_curve(key):
        out = []
        for g in range(max_len):
            vals = [h["history"][min(g, len(h["history"]) - 1)][key] for h in histories]
            out.append(sum(vals) / len(vals))
        return out

    ax = axes[0]
    ax.plot(gens, avg_curve("best_diversity"), label="best", lw=2.5, alpha=0.9,
            color=ECAD_COLORS["ours"], zorder=3, **marker_kw)
    ax.plot(gens, avg_curve("mean_diversity"), label="population mean", lw=2.0,
            ls="--", alpha=0.9, color=ECAD_COLORS["mean"], zorder=2)
    for label, y in (ref_lines or {}).items():
        ax.axhline(y=y, color="black", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.annotate(label, xy=(0.02, y), xycoords=("axes fraction", "data"),
                    fontsize=10, va="bottom")
    ax.set_title("Diversity (bred objective)")
    ax.set_xlabel("generation"); ax.set_ylabel("DINOv2 diversity")
    ax.legend(frameon=True)

    ax = axes[1]
    ax.plot(gens, avg_curve("best_quality"), label="best", lw=2.5, alpha=0.9,
            color=ECAD_COLORS["quality"], zorder=3, **marker_kw)
    ax.plot(gens, avg_curve("mean_quality"), label="population mean", lw=2.0,
            ls="--", alpha=0.9, color=ECAD_COLORS["mean"], zorder=2)
    ax.set_title("Quality (constraint)")
    ax.set_xlabel("generation"); ax.set_ylabel("CLIPScore")
    ax.legend(frameon=True)

    ax = axes[2]
    if "mean_sigma" in histories[0]["history"][0]:
        ax.plot(gens, avg_curve("mean_sigma"), label=r"step $\sigma$", lw=2.5,
                alpha=0.9, color=ECAD_COLORS["sigma"], zorder=3, **marker_kw)
        ax.plot(gens, avg_curve("mean_rate"), label=r"rate $\rho$", lw=2.5,
                alpha=0.9, color=ECAD_COLORS["rate"], zorder=2,
                marker="s", markersize=6, markeredgecolor="black",
                markeredgewidth=1.0, markevery=mark_every)
        ax.set_title("Self-tuned strategy parameters")
        ax.set_xlabel("generation"); ax.set_ylabel("value")
    else:
        ax.plot(gens, avg_curve("pareto_size"), label="Pareto size", lw=2.5,
                alpha=0.9, color=ECAD_COLORS["rate"], **marker_kw)
        ax.set_title("Pareto front size"); ax.set_xlabel("generation")
    ax.legend(frameon=True)

    sns.despine(fig)
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", default="evodiv_report")
    args = ap.parse_args()
    hists = load_histories(args.run_dir)
    if not hists:
        raise SystemExit(f"no history.json under {args.run_dir}")
    agg = aggregate(hists)
    table = markdown_table(agg)
    print(table)
    with open(args.out + ".md", "w") as f:
        f.write(table + "\n")
    refs = {"Harrington et al. (gradient)": PAPER_TAB1_WHITE["Ours(grad)"]["DINO"],
            "i.i.d. (paper)": PAPER_TAB1_WHITE["i.i.d."]["DINO"]}
    plot_convergence(hists, args.out + ".png", ref_lines=refs)


if __name__ == "__main__":
    main()
