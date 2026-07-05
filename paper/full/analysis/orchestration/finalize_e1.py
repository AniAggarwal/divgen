"""Finalize E1: upstream main.py layout -> results.json + per_prompt.json.

- best/init reward means over prompts (trainer metric conventions)
- timing: s_per_iter median overall AND over the solo window (prompts 0-9 ran
  with the GPU otherwise idle; later prompts shared the GPU, so the SOLO
  number is the one the paper quotes)
- peak memory: max + median over prompts
- rescored: if E5 has scored the 'grad' method (same jpgs, evodiv metric
  code), its per-prompt means are mounted here for Table-1 comparability.
"""
import glob, json, os, statistics, sys

E1 = "/workspace/runs/paperprep/e1_gradient"
SOLO_PROMPTS = set(range(10))     # documented solo-timing window

per, rows = {}, []
for rp in sorted(glob.glob(os.path.join(E1, "geneval", "*", "[0-9]*", "results.json"))):
    with open(rp) as fp:
        r = json.load(fp)
    rows.append(r)
    per[str(r["prompt_index"])] = r["best_rewards"]

expect = int(os.environ.get("E1_EXPECT", "553"))
if len(rows) != expect:
    print(f"INCOMPLETE: {len(rows)}/{expect}", file=sys.stderr)
    sys.exit(1)

keys = sorted(rows[0]["best_rewards"])
res = {
    "n": len(rows),
    "best": {k: statistics.mean(r["best_rewards"][k] for r in rows) for k in keys},
    "init": {k: statistics.mean(r["initial_rewards"][k] for r in rows) for k in keys},
    "s_per_iter_median": statistics.median(r["s_per_iter"] for r in rows),
    "s_per_iter_solo_median": statistics.median(
        r["s_per_iter"] for r in rows if r["prompt_index"] in SOLO_PROMPTS),
    "peak_mem_alloc_gb_max": max(r["peak_mem_alloc_gb"] for r in rows),
    "peak_mem_alloc_gb_median": statistics.median(r["peak_mem_alloc_gb"] for r in rows),
    "total_elapsed_h": sum(r["elapsed_s"] for r in rows) / 3600,
}

# mount E5's rescored means when available (evodiv metric code over saved jpgs)
e5p = "/workspace/runs/paperprep/e5_judges/per_prompt.json"
if os.path.exists(e5p):
    grad = json.load(open(e5p)).get("grad", {})
    if grad:
        mk = sorted({k for v in grad.values() for k in v})
        res["rescored"] = {k: statistics.mean(v[k] for v in grad.values() if k in v)
                           for k in mk}
        res["rescored_n"] = len(grad)

json.dump(res, open(os.path.join(E1, "results.json"), "w"), indent=1, sort_keys=True)
json.dump(per, open(os.path.join(E1, "per_prompt.json"), "w"))
print(f"E1 finalized: n={res['n']}, s/iter solo={res['s_per_iter_solo_median']:.3f}, "
      f"peak={res['peak_mem_alloc_gb_max']:.1f}GB")
