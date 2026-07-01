#!/bin/bash
# Island model: prompt-typed subpopulations (landscape / 5-word / GPT-generated)
# evolving simultaneously with noise migration; reports cross-prompt generalization.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="${HF_HOME:-$PWD/cache/hf_home}"
python -m evodiv.run --config configs/evo_sdxl_islands.yaml "$@" 2>&1 | tee runs/islands.log
