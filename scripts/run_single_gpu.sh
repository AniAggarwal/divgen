#!/bin/bash
# EvoDiv single-GPU launcher (mirrors ECAD's nohup pattern).
# Usage: ./scripts/run_single_gpu.sh <config.yaml> [extra --flag val ...]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="${HF_HOME:-$PWD/cache/hf_home}"
CONFIG="${1:?usage: run_single_gpu.sh <config.yaml> [overrides...]}"; shift || true
STAMP=$(basename "$CONFIG" .yaml)
mkdir -p runs
echo "[EvoDiv] config=$CONFIG  -> runs/${STAMP}.log"
nohup python -m evodiv.run --config "$CONFIG" "$@" > "runs/${STAMP}.log" 2>&1 &
echo "[EvoDiv] PID $! (tail -f runs/${STAMP}.log)"
