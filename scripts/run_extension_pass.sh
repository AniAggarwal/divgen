#!/bin/bash
# Extension pass: re-breed prompts whose selected genome's DINO diversity fell
# below a threshold, with a larger budget (90 gens, patience 20) and a fresh
# base seed. Results go to a separate save_dir; merge_best.py then builds a
# merged view taking the better history per prompt (adaptive per-prompt budget).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="${HF_HOME:-$PWD/cache/hf_home}"
BASE="${1:?usage: run_extension_pass.sh <base_run_dir> <ext_save_dir> [threshold]}"
EXT="${2:?}"
THRESH="${3:-0.784}"
# find under-threshold prompt indices from the base run
IDXS=$(python3 - "$BASE" "$THRESH" <<'PY'
import glob,json,os,sys
base,thr=sys.argv[1],float(sys.argv[2])
out=[]
for f in sorted(glob.glob(os.path.join(base,"*","history.json"))):
    d=json.load(open(f))
    div=d["best_metrics"].get("diversity_dino",0)
    if div<thr:
        out.append(os.path.basename(os.path.dirname(f)).split("_")[0])
print(" ".join(out))
PY
)
N=$(echo $IDXS | wc -w)
echo "[ext] $N prompts below $THRESH"
for i in $IDXS; do
  s=$((10#$i)); e=$((s+1))
  python -m evodiv.run --config configs/evo_sdxl_geneval_full.yaml \
    --task.prompt_start_index $s --task.prompt_end_index $e \
    --evolution.n_generations 90 --evolution.early_stop_patience 20 \
    --evolution.seed 1000 \
    --paths.save_dir "$EXT" >> "$EXT.log" 2>&1
done
echo "[ext] done"
