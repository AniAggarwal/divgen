#!/bin/bash
# Master driver: STRICTLY SERIAL — exactly one GPU job at a time (per user
# directive: no concurrent jobs that contend; measured E1+E3 slowed E1 ~67%).
# Even the light E5 judging sub-runs run in the foreground between compute
# stages, not overlapping them.
#
# Order (strict priority): E1 -> E5-grad -> E5-rand -> E3 -> E5-cmadct ->
# tail (E6-E9,E11) -> E10b -> STOP. The night queue of EXTRA experiments does
# NOT auto-run; it is gated behind $P/NIGHT_OK (create it to allow the extras).
# Every stage is DONE-flag guarded and resume-safe.
source /workspace/evodiv.env
cd /workspace/repos/divgen
P=/workspace/runs/paperprep
LOG=$P/driver.log
log(){ echo "[$(date -u +%FT%TZ)] DRIVER: $*" >> "$LOG"; }
waitdone(){ while [ ! -f "$1" ]; do sleep 60; done; }

# E5 judging sub-run, FOREGROUND (serial), then upload
e5run(){ local m=$1; log "E5-$m start (serial)"
  E5_METHODS=$m python $P/e5_judges/e5_judges.py >> $P/e5_judges/run_$m.log 2>&1
  bash $P/upload_exp.sh e5_judges >> "$LOG" 2>&1; log "E5-$m done"; }

log "driver (serial) up; pid $$"

# ---- E1 (launched externally via run_e1.sh) -------------------------------
log "waiting for E1/DONE"
waitdone $P/e1_gradient/DONE
python $P/finalize_e1.py >> "$LOG" 2>&1
bash $P/upload_exp.sh e1_gradient >> "$LOG" 2>&1
e5run grad                                     # judge E1's images (serial)
python $P/finalize_e1.py >> "$LOG" 2>&1        # re-finalize with rescored row
bash $P/upload_exp.sh e1_gradient >> "$LOG" 2>&1

# ---- E2 already complete; judge its images (serial) -----------------------
[ -f $P/e2_random/DONE ] && bash $P/upload_exp.sh e2_random >> "$LOG" 2>&1
e5run rand

# ---- E3 (CMA-DCT, 553) solo -----------------------------------------------
if pgrep -f '[e]3_cmadct.py' >/dev/null; then
  log "E3 already running -- waiting"; waitdone $P/e3_cmadct/DONE
elif [ ! -f $P/e3_cmadct/DONE ]; then
  log "E3 start (solo)"; bash $P/e3_cmadct/run_e3.sh >> "$LOG" 2>&1
fi
[ -f $P/e3_cmadct/DONE ] && bash $P/upload_exp.sh e3_cmadct >> "$LOG" 2>&1
e5run cmadct

# ---- tail experiments E6-E9, E11 (serial) ---------------------------------
if [ ! -f $P/tail.DONE ]; then
  log "tail (E6-E9,E11) start"
  bash $P/run_tail_experiments.sh >> "$LOG" 2>&1 && touch $P/tail.DONE
fi
for e in e6_dpp_hps e7_dpg e8_vendi e9_lpips e11_seeds; do
  [ -f $P/$e/DONE ] && bash $P/upload_exp.sh $e >> "$LOG" 2>&1
done

# ---- E10b transfer bands --------------------------------------------------
if [ ! -f $P/e10_transfer_bands/DONE ]; then
  log "E10b start"
  python $P/e10_transfer_bands/e10b_transfer_bands.py >> "$LOG" 2>&1 \
    && touch $P/e10_transfer_bands/DONE
fi
[ -f $P/e10_transfer_bands/DONE ] && bash $P/upload_exp.sh e10_transfer_bands >> "$LOG" 2>&1

log "PLANNED PIPELINE E1-E11 COMPLETE"
touch $P/PIPELINE_DONE

# ---- night queue: GATED. Only runs if the user opts in via NIGHT_OK. -------
if [ -f $P/NIGHT_OK ]; then
  log "NIGHT_OK present -> entering night queue (extra experiments)"
  bash $P/night_queue.sh >> "$LOG" 2>&1
  touch $P/NIGHT_DONE
  log "ALL WORK COMPLETE (planned pipeline + night queue)"
else
  log "night queue gated (no NIGHT_OK); planned pipeline done, GPU idle. STOP."
  touch $P/NIGHT_DONE   # lets the safety daemon exit
fi
