#!/usr/bin/env python3
"""Merge two EvoDiv run dirs, keeping the better (higher select-metric) result
per prompt. Usage: merge_best.py <base_dir> <ext_dir> <merged_dir> [metric]"""
import glob, json, os, shutil, sys

base, ext, merged = sys.argv[1], sys.argv[2], sys.argv[3]
metric = sys.argv[4] if len(sys.argv) > 4 else "diversity_dino"
os.makedirs(merged, exist_ok=True)
n_base = n_ext = 0
for f in sorted(glob.glob(os.path.join(base, "*", "history.json"))):
    pdir = os.path.basename(os.path.dirname(f))
    d_base = json.load(open(f))
    src = os.path.dirname(f)
    ef = os.path.join(ext, pdir, "history.json")
    if os.path.exists(ef):
        d_ext = json.load(open(ef))
        if d_ext["best_metrics"].get(metric, -1) > d_base["best_metrics"].get(metric, -1):
            src = os.path.dirname(ef); n_ext += 1
        else:
            n_base += 1
    else:
        n_base += 1
    dst = os.path.join(merged, pdir)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
print(f"merged: {n_base} from base, {n_ext} from extension -> {merged}")
