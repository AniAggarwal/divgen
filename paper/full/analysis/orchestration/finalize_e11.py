"""Combine E11's three seed subruns into one results.json with across-seed
standard deviations of the per-seed prompt means."""
import json, os, statistics, sys

E11 = "/workspace/runs/paperprep/e11_seeds"
seeds = []
for s in (0, 1, 2):
    rp = os.path.join(E11, f"seed{s}", "results.json")
    if not os.path.exists(rp):
        print(f"missing seed{s}", file=sys.stderr)
        sys.exit(1)
    seeds.append(json.load(open(rp)))

keys = sorted(set.intersection(*[set(s["best"]) for s in seeds]))
res = {"n_seeds": len(seeds), "n_prompts": seeds[0]["n"],
       "per_seed_best": {f"seed{i}": s["best"] for i, s in enumerate(seeds)},
       "mean": {k: statistics.mean(s["best"][k] for s in seeds) for k in keys},
       "std": {k: statistics.stdev(s["best"][k] for s in seeds) for k in keys}}
res["std_dino"] = res["std"]["diversity_dino"]
res["std_clip"] = res["std"]["CLIP"]
json.dump(res, open(os.path.join(E11, "results.json"), "w"), indent=1, sort_keys=True)
print("E11 combined:", {k: round(res["std"][k], 4) for k in ("diversity_dino", "CLIP")})
