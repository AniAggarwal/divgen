"""Aggregate the night-queue experiments into night/NIGHT_SUMMARY.json and a
human-readable NIGHT_QUEUE.md for the user to review on waking.

These are exploratory extensions -- reported separately from the paper's
headline numbers. The user decides which to fold in after verification.
"""
import glob, json, os, statistics

NB = "/workspace/runs/paperprep/night"
KEYS = ["diversity_dino", "diversity_dreamsim", "diversity_lpips", "CLIP",
        "diversity_vendi", "diversity_dpp", "diversity_dpp_raw"]

rows = {}
for rp in sorted(glob.glob(os.path.join(NB, "*", "results.json"))):
    name = os.path.basename(os.path.dirname(rp))
    with open(rp) as fp:
        r = json.load(fp)
    best = r.get("best", {})
    rows[name] = {"n": r.get("n"),
                  **{k: round(best[k], 4) for k in KEYS if k in best},
                  "init": {k: round(r["init"][k], 4) for k in KEYS
                           if k in r.get("init", {})}}

with open(os.path.join(NB, "NIGHT_SUMMARY.json"), "w") as fp:
    json.dump(rows, fp, indent=1, sort_keys=True)

# markdown table grouped by experiment family
GROUPS = [
    ("NA_beta", "Spectral-bias (beta) curve, 64 prompts", "beta {0,0.5,1(=main),2}"),
    ("NB_B", "Set-size / diversity ceiling, 64 prompts", "B images per set"),
    ("NC_", "Which diversity objective breeds best, 64 prompts", "bred objective"),
    ("ND_", "Pink vs white initialization, 128 prompts", ""),
    ("NE_cmaK", "CMA-DCT dimensionality sweep, 40 prompts", "K = DCT block edge; DIM=4*4*K^2"),
    ("NF_", "GP program-synthesis genome, 32 prompts", ""),
    ("NG_", "Crossover / repair ablations at scale, 64 prompts", ""),
]
lines = ["# Night-queue experiments (exploratory — for verification)",
         "",
         "Extra genetic-method runs launched after the planned E1–E11 to keep the",
         "paid GPU busy. Forward-only, SDXL-Turbo, surrogate+compile, resume-safe,",
         "never bred on ImageReward. **Not folded into the paper's headline numbers**",
         "— review and tell me which to promote (each has saved image sets + history).",
         "", "Data: `runs/paperprep/night/<name>/`, mirrored to Tigris",
         "`paperprep-night-<name>.tar.gz`.", ""]
for prefix, title, note in GROUPS:
    grp = {k: v for k, v in rows.items() if k.startswith(prefix)}
    if not grp:
        continue
    lines.append(f"## {title}")
    if note:
        lines.append(f"*{note}*")
    lines.append("")
    lines.append("| run | n | DINO | DreamSim | LPIPS | CLIP | Vendi | DPP |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for name in sorted(grp):
        v = grp[name]
        g = lambda k: f"{v[k]:.3f}" if k in v else "—"
        lines.append(f"| {name} | {v.get('n','?')} | {g('diversity_dino')} | "
                     f"{g('diversity_dreamsim')} | {g('diversity_lpips')} | "
                     f"{g('CLIP')} | {g('diversity_vendi')} | {g('diversity_dpp')} |")
    lines.append("")
with open(os.path.join(NB, "NIGHT_QUEUE.md"), "w") as fp:
    fp.write("\n".join(lines) + "\n")
print(f"night_aggregate: {len(rows)} experiments -> NIGHT_QUEUE.md")
