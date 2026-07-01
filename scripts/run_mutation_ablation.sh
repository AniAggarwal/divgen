#!/bin/bash
# ECAD Table 9 story: fixed low / mid / high mutation rate vs self-adaptive.
# Runs 4 conditions on the same prompt subset and writes them under runs/ablation_mut/.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="${HF_HOME:-$PWD/cache/hf_home}"
CFG=configs/evo_sdxl_geneval.yaml
COMMON="--paths.save_dir runs/ablation_mut --task.prompt_end_index 6 --evolution.n_generations 60 --evolution.pop_size 40"
for cond in "fixed01:--no-evolution.self_adaptive --evolution.rate_init 0.01" \
            "fixed10:--no-evolution.self_adaptive --evolution.rate_init 0.10" \
            "fixed15:--no-evolution.self_adaptive --evolution.rate_init 0.15" \
            "selfadapt:--evolution.self_adaptive"; do
  name="${cond%%:*}"; flags="${cond#*:}"
  echo "[ablation] $name : $flags"
  python -m evodiv.run --config $CFG $COMMON --paths.save_dir "runs/ablation_mut/$name" $flags \
    > "runs/ablation_mut_${name}.log" 2>&1
done
echo "done; analyze each with: python -m evodiv.analyze --run_dir runs/ablation_mut/<name>"
