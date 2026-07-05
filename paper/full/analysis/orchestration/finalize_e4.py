"""Finalize E4: flux breeding histories -> results.json (+ memory table).

Reported numbers are the 4-step-rendered metrics (best_metrics_multistep /
init_metrics_multistep), matching the repository's flux protocol; the 1-step
search metrics are kept under search_*.
"""
import glob, json, os, statistics, sys

E4 = "/workspace/runs/paperprep/e4_flux"
hists = sorted(glob.glob(os.path.join(E4, "breeding", "evodiv", "flux-schnell",
                                      "*", "[0-9]*", "history.json")))
expect = int(os.environ.get("E4_EXPECT", "60"))
rows = [json.load(open(h)) for h in hists]
rows = [r for r in rows if r.get("best_metrics_multistep")]
if len(rows) < expect:
    print(f"INCOMPLETE: {len(rows)}/{expect}", file=sys.stderr)
    sys.exit(1)

def means(key):
    ks = sorted({k for r in rows for k in (r.get(key) or {})})
    return {k: statistics.mean(r[key][k] for r in rows if k in (r.get(key) or {}))
            for k in ks}

res = {"n": len(rows),
       "best": means("best_metrics_multistep"),
       "init": means("init_metrics_multistep"),
       "search_best": means("best_metrics")}
mt = os.path.join(E4, "memory_table.json")
if os.path.exists(mt):
    res["memory_table"] = json.load(open(mt))
json.dump(res, open(os.path.join(E4, "results.json"), "w"), indent=1, sort_keys=True)

per = {}
for h, r in zip(hists, rows):
    key = os.path.basename(os.path.dirname(h))
    per[str(int(key.split("_", 1)[0]))] = r["best_metrics_multistep"]
json.dump(per, open(os.path.join(E4, "per_prompt.json"), "w"))
print(f"E4 finalized: n={res['n']}, "
      f"init dino {res['init'].get('diversity_dino'):.4f} -> best {res['best'].get('diversity_dino'):.4f}")
