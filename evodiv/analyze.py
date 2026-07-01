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
from typing import Dict, List

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


def plot_convergence(histories: List[dict], out_png: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    # average per-generation curves across prompts (truncate to shortest)
    min_len = min(len(h["history"]) for h in histories)
    gens = list(range(min_len))

    def avg_curve(key):
        return [sum(h["history"][g][key] for h in histories) / len(histories) for g in range(min_len)]

    axes[0].plot(gens, avg_curve("best_diversity"), label="best", lw=2)
    axes[0].plot(gens, avg_curve("mean_diversity"), label="mean", ls="--")
    axes[0].set_title("Diversity (optimized objective)"); axes[0].set_xlabel("generation")
    axes[0].set_ylabel("diversity"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(gens, avg_curve("best_quality"), label="best", lw=2, color="tab:green")
    axes[1].plot(gens, avg_curve("mean_quality"), label="mean", ls="--", color="tab:green")
    axes[1].set_title("Quality (constraint)"); axes[1].set_xlabel("generation")
    axes[1].set_ylabel("quality reward"); axes[1].legend(); axes[1].grid(alpha=0.3)

    if "mean_sigma" in histories[0]["history"][0]:
        axes[2].plot(gens, avg_curve("mean_sigma"), label="mean sigma", color="tab:red")
        axes[2].plot(gens, avg_curve("mean_rate"), label="mean rate", color="tab:purple")
        axes[2].set_title("Self-adaptive strategy params"); axes[2].set_xlabel("generation")
        axes[2].legend(); axes[2].grid(alpha=0.3)
    else:
        axes[2].plot(gens, avg_curve("pareto_size"), label="Pareto size", color="tab:orange")
        axes[2].set_title("Pareto front size"); axes[2].set_xlabel("generation")
        axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
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
    plot_convergence(hists, args.out + ".png")


if __name__ == "__main__":
    main()
