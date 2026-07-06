#!/bin/bash
# Post-E1..E9/E11 finisher: run E10b (transfer bands, planned-pipeline priority)
# solo, then the opted-in night queue, then mark done. Replaces the driver's
# remaining work after E10b's bugfix. Strictly serial (one GPU job at a time).
source /workspace/evodiv.env
cd /workspace/repos/divgen
P=/workspace/runs/paperprep
LOG=$P/driver.log
log(){ echo "[$(date -u +%FT%TZ)] FINISH: $*" >> "$LOG"; }

log "finisher up; pid $$"

# ---- E10b transfer bands (retry loop; resumable via its state.json) --------
if [ ! -f $P/e10_transfer_bands/DONE ]; then
  for attempt in 1 2 3; do
    log "E10b attempt $attempt"
    python $P/e10_transfer_bands/e10b_transfer_bands.py >> $P/e10_transfer_bands/e10b.log 2>&1
    if grep -q 'E10B_DONE' $P/e10_transfer_bands/e10b.log; then
      touch $P/e10_transfer_bands/DONE
      bash $P/upload_exp.sh e10_transfer_bands >> "$LOG" 2>&1
      log "E10b COMPLETE"; break
    fi
    log "E10b attempt $attempt incomplete; retrying"; sleep 15
  done
fi

# ---- night queue (opted in via NIGHT_OK) ----------------------------------
if [ -f $P/NIGHT_OK ]; then
  log "entering night queue"
  bash $P/night_queue.sh >> "$LOG" 2>&1
fi
touch $P/NIGHT_DONE
log "ALL WORK COMPLETE"
