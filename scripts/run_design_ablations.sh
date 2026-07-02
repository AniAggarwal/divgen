#!/bin/bash
# Design-ablation suite for the paper: one variant per condition, all on the
# same prompts (GenEval 0-5), same seeds, same budget (45 gens x pop 32).
# Groups: mutation rate | spectral bias beta | crossover | chi_d repair | init.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="${HF_HOME:-$PWD/cache/hf_home}"
OUT="${1:-runs/design_ablations}"
CFG=configs/evo_sdxl_geneval_full.yaml
COMMON="--task.prompt_end_index 6 --evolution.n_generations 45 --evolution.pop_size 32 \
 --evolution.early_stop_patience 0 --evolution.target_diversity 99 --evolution.log_all_every 0"
run() {
  name=$1; shift
  echo "[ablation] $name : $*"
  python -m evodiv.run --config $CFG $COMMON --paths.save_dir "$OUT/$name" "$@" \
    > "$OUT/${name}.log" 2>&1
}
mkdir -p "$OUT"
run default
run mut01     --no-evolution.self_adaptive --evolution.rate_init 0.01
run mut15     --no-evolution.self_adaptive --evolution.rate_init 0.15
run beta0     --evolution.beta 0.0
run beta2     --evolution.beta 2.0
run noxover   --evolution.p_crossover 0.0
run norepair  --no-evolution.repair_ops
run pink      --optimization.noise_type pink --optimization.noise_exponent 0.2
echo "[ablation] all conditions done"
for d in default mut01 mut15 beta0 beta2 noxover norepair pink; do
  echo "== $d =="
  python -m evodiv.analyze --run_dir "$OUT/$d/evodiv/sdxl-turbo/geneval" --out "$OUT/report_$d" 2>/dev/null | grep -E "ours (init|evolved)"
done
