"""Turn an experiment's per-prompt history.json files into the root-level
results.json + per_prompt.json that paper/full/analysis/aggregate.py mounts.

Usage: python finalize_exp.py <exp_dir> [--expect N]
Works for any run whose prompt dirs contain history.json with best_metrics
(e2_random, e3_cmadct, and the evodiv-native nested layouts via --glob).
"""
import argparse, glob, json, os, statistics, sys

ap = argparse.ArgumentParser()
ap.add_argument("exp_dir")
ap.add_argument("--expect", type=int, default=0)
ap.add_argument("--glob", default="*/history.json",
                help="pattern under exp_dir for per-prompt histories")
args = ap.parse_args()

per, extras = {}, {"elapsed_s": [], "renders_used": []}
for hp in sorted(glob.glob(os.path.join(args.exp_dir, args.glob))):
    with open(hp) as fp:
        h = json.load(fp)
    key = os.path.relpath(os.path.dirname(hp), args.exp_dir)
    if "best_metrics" in h:
        per[key] = h["best_metrics"]
    for f in extras:
        if f in h:
            extras[f].append(h[f])

if args.expect and len(per) != args.expect:
    print(f"INCOMPLETE: {len(per)}/{args.expect} prompts have results", file=sys.stderr)
    sys.exit(1)

keys = sorted({k for v in per.values() for k in v if not k.startswith("_")})
means = {k: statistics.mean([v[k] for v in per.values() if k in v]) for k in keys}
results = {"n": len(per), "best": means}
# nested evodiv runs also carry an init (gen-0 i.i.d.) row in summary.json
for sp in glob.glob(os.path.join(args.exp_dir, "evodiv", "*", "*", "summary.json")):
    with open(sp) as fp:
        s = json.load(fp)
    if s.get("init"):
        results["init"] = s["init"]
    break
for f, vals in extras.items():
    if vals:
        results[f"total_{f}"] = sum(vals)
        results[f"mean_{f}"] = statistics.mean(vals)

with open(os.path.join(args.exp_dir, "results.json"), "w") as fp:
    json.dump(results, fp, indent=1, sort_keys=True)
with open(os.path.join(args.exp_dir, "per_prompt.json"), "w") as fp:
    json.dump(per, fp)
print(f"{args.exp_dir}: n={len(per)}")
print({k: round(v, 4) for k, v in means.items()
       if k in ("diversity_dino", "diversity_dreamsim", "diversity_lpips",
                "CLIP", "diversity_vendi", "diversity_dpp")})
