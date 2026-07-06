"""Regenerate every data-driven figure in both papers from the runs.

Each figure has its own builder; builders skip cleanly when their inputs do
not exist yet, so this can run at any point in the campaign. Outputs land in
paper/figures/ (shared by the 3-pager and the full paper).

Style: Okabe-Ito colorblind-safe palette, one hue per METHOD held fixed across
every figure; thin marks, recessive grids, direct labels where they fit.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("EVODIV_RUNS", "/workspace/runs-restore/runs")
PREP = os.environ.get("PAPERPREP", "/workspace/runs/paperprep")
FIGS = os.path.normpath(os.path.join(HERE, "..", "..", "figures"))

# one color per method, fixed order everywhere (Okabe-Ito)
C = {"bred": "#0072B2", "grad": "#D55E00", "rand": "#009E73",
     "cmadct": "#CC79A7", "iid": "#8C8C8C", "accent": "#E69F00"}
LABEL = {"bred": "bred (ours)", "grad": "gradient", "rand": "random search",
         "cmadct": "CMA-DCT", "iid": "i.i.d."}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "lines.linewidth": 1.6, "figure.dpi": 200, "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def _histories(root):
    out = []
    for hp in sorted(glob.glob(os.path.join(root, "*", "history.json"))):
        with open(hp) as fp:
            out.append(json.load(fp)["history"])
    return out


def fig_convergence():
    """3-panel convergence averaged over the full 553-prompt run."""
    hists = _histories(os.path.join(RUNS, "geneval_full552", "evodiv",
                                    "sdxl-turbo", "geneval"))
    if not hists:
        return print("skip convergence (no full552 histories)")
    G = max(len(h) for h in hists)

    def series(key):
        m = np.full((len(hists), G), np.nan)
        for i, h in enumerate(hists):
            for g, e in enumerate(h):
                m[i, g] = e.get(key, np.nan)
            m[i, len(h):] = m[i, len(h) - 1]      # freeze after early stop
        return np.nanmean(m, 0)

    x = np.arange(G)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.9))
    ax = axes[0]
    ax.plot(x, series("best_diversity"), color=C["bred"])
    ax.plot(x, series("mean_diversity"), color=C["bred"], alpha=0.45)
    ax.text(x[-1], series("best_diversity")[-1], " best", va="center", fontsize=7, color=C["bred"])
    ax.text(x[-1], series("mean_diversity")[-1], " mean", va="center", fontsize=7,
            color=C["bred"], alpha=0.6)
    ax.set_xlabel("generation"); ax.set_ylabel("DINO diversity")
    ax = axes[1]
    ax.plot(x, series("mean_quality"), color=C["accent"])
    ax.set_xlabel("generation"); ax.set_ylabel("CLIP quality (pop.\\ mean)")
    ax = axes[2]
    ax.plot(x, series("mean_sigma"), color=C["cmadct"])
    ax.set_ylabel(r"mutation step $\sigma$", color=C["cmadct"])
    ax2 = ax.twinx()
    ax2.plot(x, series("mean_rate"), color=C["rand"])
    ax2.set_ylabel(r"mutation rate $\rho$", color=C["rand"])
    ax2.grid(False); ax2.spines["top"].set_visible(False)
    ax.set_xlabel("generation")
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIGS, "convergence.png"))
    plt.close(fig)
    print("wrote convergence.png")


def fig_judges():
    """Judge panel (E5): per-judge dot comparison across methods."""
    p = os.path.join(PREP, "e5_judges", "results.json")
    if not os.path.exists(p):
        return print("skip judges (no e5 results)")
    R = json.load(open(p))
    judges = [("imagereward", "ImageReward"), ("pickscore", "PickScore"),
              ("hpsv21", "HPSv2.1")]
    methods = [m for m in ("bred", "grad", "rand", "cmadct") if m in R]
    if len(methods) < 2:
        return print("skip judges (need >=2 methods)")
    fig, axes = plt.subplots(1, len(judges), figsize=(7.0, 1.7))
    for ax, (jk, jn) in zip(axes, judges):
        ys = np.arange(len(methods))[::-1]
        for y, m in zip(ys, methods):
            v = R[m].get(jk)
            if v is None:
                continue
            ax.scatter([v], [y], s=28, color=C[m], zorder=3)
            ax.text(v, y + 0.32, f"{v:.3f}", ha="center", fontsize=6.5, color=C[m])
        ax.set_yticks(ys)
        ax.set_yticklabels([LABEL[m] for m in methods] if ax is axes[0] else [""] * len(methods))
        ax.set_title(jn)
        ax.grid(axis="x", alpha=0.25); ax.grid(axis="y", visible=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIGS, "judges.png"))
    plt.close(fig)
    print("wrote judges.png")


def fig_spectrum():
    """E10a: where CMA-DCT search concentrates -- coordinate variance by DCT
    radial band + covariance eigenvalue decay, averaged over prompts."""
    files = sorted(glob.glob(os.path.join(PREP, "e3_cmadct", "*", "cma_cov.npz")))
    if len(files) < 10:
        return print(f"skip spectrum ({len(files)} cov dumps)")
    K = 8
    kk, ll = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
    radial = (kk + ll).reshape(-1)                       # 0..14 band index per (k,l)
    bands = np.arange(radial.max() + 1)
    band_vals, eig_curves = [], []
    for f in files:
        z = np.load(f)
        cv = z["coord_var"].reshape(-1, K * K).mean(0)   # avg over B*C blocks
        band_vals.append([cv[radial == b].mean() for b in bands])
        ev = z["eigvals"]
        eig_curves.append(ev / ev.sum())
    bv = np.mean(band_vals, 0)
    ec = np.mean([e[:64] for e in eig_curves], 0)

    fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.8))
    axes[0].bar(bands, bv, color=C["cmadct"], width=0.82)
    axes[0].set_xlabel("DCT band $k{+}l$ (low $\\to$ high frequency)")
    axes[0].set_ylabel("search variance")
    axes[1].semilogy(np.arange(1, len(ec) + 1), ec, color=C["cmadct"])
    axes[1].set_xlabel("eigenvalue rank")
    axes[1].set_ylabel("normalized eigenvalue")
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIGS, "cma_spectrum.png"))
    plt.close(fig)
    print(f"wrote cma_spectrum.png (n={len(files)})")


def fig_vendi_budget():
    """E8: Vendi under a long selection budget."""
    hp = glob.glob(os.path.join(PREP, "e8_vendi", "**", "history.json"), recursive=True)
    hists = [json.load(open(p))["history"] for p in hp]
    if not hists:
        return print("skip vendi budget (no e8)")
    G = max(len(h) for h in hists)
    m = np.full((len(hists), G), np.nan)
    for i, h in enumerate(hists):
        for g, e in enumerate(h):
            m[i, g] = e.get("best/diversity_vendi", e.get("best_diversity", np.nan))
        m[i, len(h):] = m[i, len(h) - 1]
    fig, ax = plt.subplots(figsize=(3.3, 1.8))
    ax.plot(np.arange(G), np.nanmean(m, 0), color=C["bred"])
    ax.set_xlabel("generation"); ax.set_ylabel("Vendi (bred objective)")
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIGS, "vendi_budget.png"))
    plt.close(fig)
    print("wrote vendi_budget.png")


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    fig_convergence()
    fig_judges()
    fig_spectrum()
    fig_vendi_budget()
