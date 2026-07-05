#!/bin/bash
# NIGHT QUEUE — extra genetic-method experiments to keep the GPU busy after the
# planned E1-E11 finish (user asked to not waste paid GPU time while asleep).
#
# These are EXPLORATORY / for-verification: they extend the paper's ablations
# and method comparisons at scale, using only the existing forward-only search
# harness. They are written under runs/paperprep/night/<name>, aggregated
# separately (night_aggregate.py -> NIGHT_QUEUE.md), and are NOT folded into
# the paper's headline numbers until the user verifies them. All use SDXL-Turbo,
# surrogate+compile for speed, are resume-safe (skip_existing), and never breed
# on ImageReward.
#
# Each experiment is smoke-gated (1 prompt x 2 gens) so a broken config is
# skipped and logged, not fatal to the queue.
source /workspace/evodiv.env
cd /workspace/repos/divgen
P=/workspace/runs/paperprep
NB=$P/night
mkdir -p "$NB"
GLOG=$NB/night.log
nlog(){ echo "[$(date -u +%FT%TZ)] $*" >> "$GLOG"; }

# night_run <name> <expect_prompts> <run.py args...>
night_run(){
  local NAME=$1 EXPECT=$2; shift 2
  local OUT=$NB/$NAME
  [ -f "$OUT/DONE" ] && { nlog "$NAME already done"; return 0; }
  mkdir -p "$OUT"
  # smoke: 1 prompt x 2 gens to a throwaway dir
  local SM=$OUT/.smoke
  rm -rf "$SM"
  if ! python -m evodiv.run "$@" --paths.save_dir "$SM" \
        --task.prompt_end_index 1 --evolution.n_generations 2 --evolution.pop_size 6 \
        > "$OUT/smoke.log" 2>&1 || ! grep -q 'SUMMARY over' "$OUT/smoke.log"; then
    nlog "$NAME SMOKE FAILED -> skipping"; tail -3 "$OUT/smoke.log" >> "$GLOG"; rm -rf "$SM"; return 1
  fi
  rm -rf "$SM"
  nlog "$NAME start (expect $EXPECT prompts)"
  for attempt in 1 2 3; do
    python -m evodiv.run "$@" --paths.save_dir "$OUT" >> "$OUT/run.log" 2>&1
    if grep -q 'SUMMARY over' "$OUT/run.log"; then
      python $P/finalize_exp.py "$OUT" --glob "evodiv/*/*/[0-9]*/history.json" \
        --expect "$EXPECT" >> "$OUT/run.log" 2>&1 && touch "$OUT/DONE"
      nlog "$NAME COMPLETE"; bash $P/upload_exp.sh "night/$NAME" 2>/dev/null; return 0
    fi
    nlog "$NAME attempt $attempt incomplete; resuming"; sleep 15
  done
  nlog "$NAME FAILED after retries"; return 1
}

Cc="--evolution.use_surrogate --evolution.compile_model --evolution.exact_every 10"
BASE="--config configs/evo_sdxl_geneval.yaml"

# --- NA: spectral-bias (beta) curve at scale (64 prompts) -------------------
for B in 0.0 0.5 2.0; do
  night_run "NA_beta${B}" 64 $BASE --task.prompt_end_index 64 \
    --evolution.beta $B --evolution.n_generations 60 --evolution.early_stop_patience 12 $Cc
done

# --- NB: set-size Vendi ceiling (B=8, 16) at 64 prompts ---------------------
for SZ in 8 16; do
  night_run "NB_B${SZ}" 64 $BASE --task.prompt_end_index 64 \
    --evolution.set_size $SZ --evolution.n_generations 60 --evolution.early_stop_patience 12 \
    --evolution.render_chunk 128 $Cc
done

# --- NC: which diversity currency breeds best (DPP, Vendi) at 64 ------------
night_run "NC_dppobj" 64 $BASE --task.prompt_end_index 64 \
  --no-diversity.dino.enable --diversity.dpp.enable --diversity.dpp.weight 1.0 \
  --evolution.select_metric diversity_dpp --evolution.n_generations 60 \
  --evolution.early_stop_patience 12 $Cc
night_run "NC_vendiobj" 64 $BASE --task.prompt_end_index 64 \
  --no-diversity.dino.enable --diversity.vendi.enable --diversity.vendi.weight 1.0 \
  --evolution.select_metric diversity_vendi --evolution.n_generations 60 \
  --evolution.early_stop_patience 12 $Cc

# --- ND: pink vs white initialization at 128 prompts ------------------------
night_run "ND_pink" 128 $BASE --task.prompt_end_index 128 \
  --optimization.noise_type pink --optimization.noise_exponent 0.2 \
  --evolution.n_generations 60 --evolution.early_stop_patience 12 $Cc

# --- NE: CMA-DCT dimensionality sweep (K=4,6,12; K=8 is E3) at 40 -----------
for K in 4 6 12; do
  OUT=$NB/NE_cmaK${K}
  [ -f "$OUT/DONE" ] && { nlog "NE_cmaK${K} done"; } || {
    mkdir -p "$OUT"; nlog "NE_cmaK${K} start"
    E3_OUT="$OUT" E3_K=$K E3_END=40 python $P/e3_cmadct/e3_cmadct.py >> "$OUT/run.log" 2>&1
    if grep -q 'E3_ALL_PROMPTS_DONE' "$OUT/run.log"; then
      python $P/finalize_exp.py "$OUT" --expect 40 >> "$OUT/run.log" 2>&1 && touch "$OUT/DONE"
      nlog "NE_cmaK${K} COMPLETE"; bash $P/upload_exp.sh "night/NE_cmaK${K}" 2>/dev/null
    else nlog "NE_cmaK${K} INCOMPLETE"; fi
  }
done

# --- NF: GP program-synthesis genome at 32 prompts (far end of front) -------
night_run "NF_gp" 32 $BASE --task.prompt_end_index 32 --evolution.genome gp \
  --evolution.n_generations 40 --evolution.pop_size 32

# --- NG: crossover-rate + repair ablations at scale (64) --------------------
night_run "NG_noxover" 64 $BASE --task.prompt_end_index 64 \
  --evolution.p_crossover 0.0 --evolution.n_generations 60 --evolution.early_stop_patience 12 $Cc
night_run "NG_norepair" 64 $BASE --task.prompt_end_index 64 \
  --no-evolution.repair_ops --evolution.n_generations 60 --evolution.early_stop_patience 12 $Cc

python $P/night_aggregate.py >> "$GLOG" 2>&1
nlog "NIGHT QUEUE COMPLETE"
