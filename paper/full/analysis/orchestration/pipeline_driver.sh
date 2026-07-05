#!/bin/bash
# Master driver: runs the whole campaign as a SERIAL compute pipeline so exactly
# one diffusion/gradient job uses the B200 at a time (measured: 3-way contention
# cost ~70% throughput on E1). The light, IO-bound E5 judging sub-runs are the
# only jobs allowed to overlap a compute job (they fill its GPU bubbles).
#
# Order (strict priority): E1 (running externally) -> E2(256) -> E3 -> tail
# (E6-E9,E11) -> E10b -> night queue. Every stage is DONE-flag guarded and
# resume-safe, so the driver can be re-launched and picks up where it left off.
# GPU never idles until the night queue drains.
source /workspace/evodiv.env
cd /workspace/repos/divgen
P=/workspace/runs/paperprep
LOG=$P/driver.log
log(){ echo "[$(date -u +%FT%TZ)] DRIVER: $*" >> "$LOG"; }
waitdone(){ while [ ! -f "$1" ]; do sleep 60; done; }

# light E5 judging sub-run (overlaps the next compute stage), then upload
e5bg(){ local m=$1; ( E5_METHODS=$m python $P/e5_judges/e5_judges.py \
        >> $P/e5_judges/run_$m.log 2>&1; bash $P/upload_exp.sh e5_judges \
        >> "$LOG" 2>&1 ) & }

log "driver up; pid $$"

# ---- E1 (launched externally via run_e1.sh) -------------------------------
log "waiting for E1/DONE"
waitdone $P/e1_gradient/DONE
python $P/finalize_e1.py >> "$LOG" 2>&1
bash $P/upload_exp.sh e1_gradient >> "$LOG" 2>&1
log "E1 finalized+uploaded; kicking E5-grad (overlaps E2), re-finalize E1 after"
( E5_METHODS=grad python $P/e5_judges/e5_judges.py >> $P/e5_judges/run_grad.log 2>&1
  python $P/finalize_e1.py >> "$LOG" 2>&1          # now with rescored row
  bash $P/upload_exp.sh e1_gradient >> "$LOG" 2>&1
  bash $P/upload_exp.sh e5_judges >> "$LOG" 2>&1 ) &

# ---- E2 (matched-compute random control, capped at 256) -------------------
# Guard: if an E2 is already running (e.g. a concurrency test paired with E1),
# wait for it rather than launching a duplicate.
if pgrep -f 'e2_random/run_e2.sh' >/dev/null || pgrep -f 'e2_random.py' >/dev/null; then
  log "E2 already running (concurrent) -- waiting for its DONE instead of relaunching"
  waitdone $P/e2_random/DONE
elif [ ! -f $P/e2_random/DONE ]; then
  log "E2 start (cap 256)"
  EXPECT=256 E2_END=256 bash $P/e2_random/run_e2.sh >> "$LOG" 2>&1
fi
[ -f $P/e2_random/DONE ] && bash $P/upload_exp.sh e2_random >> "$LOG" 2>&1
log "E2 done; kicking E5-rand (overlaps E3)"
e5bg rand

# ---- E3 (CMA-DCT, 553) ----------------------------------------------------
# Guard: E3 may have been launched early to pair with the still-running E1
# (fills E1's 23%-util idle). If so, wait for it rather than duplicating.
if pgrep -f 'e3_cmadct/run_e3.sh' >/dev/null || pgrep -f '[e]3_cmadct.py' >/dev/null; then
  log "E3 already running (early-launched, paired with E1) -- waiting for DONE"
  waitdone $P/e3_cmadct/DONE
elif [ ! -f $P/e3_cmadct/DONE ]; then
  log "E3 start"
  bash $P/e3_cmadct/run_e3.sh >> "$LOG" 2>&1
fi
[ -f $P/e3_cmadct/DONE ] && bash $P/upload_exp.sh e3_cmadct >> "$LOG" 2>&1
log "E3 done; kicking E5-cmadct (overlaps tail)"
e5bg cmadct

# ---- tail experiments E6-E9, E11 ------------------------------------------
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

# ---- night queue (extra experiments; user asked to keep GPU busy) ---------
log "entering night queue"
bash $P/night_queue.sh >> "$LOG" 2>&1
touch $P/NIGHT_DONE
log "ALL WORK COMPLETE (planned pipeline + night queue)"
