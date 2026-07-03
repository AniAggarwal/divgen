#!/bin/bash
source /venv/main/bin/activate
cd /workspace/repos/divgen
export PYTHONPATH=$PWD HF_HOME=/workspace/hf_home PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
for WQ in 1.0 3.0 6.0; do
  echo "=== WQ=$WQ ==="
  WQ=$WQ python /workspace/runs/explore_cmaes_sweep.py 2>&1 | grep -E "SUMMARY|CMA final"
done
echo "CMA_SWEEP_DONE"
