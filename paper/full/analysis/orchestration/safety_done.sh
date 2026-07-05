#!/bin/bash
# Safety net: ensure each compute stage's DONE flag gets set even if its
# launcher wrapper dies (e.g. the CC session ends and the setsid'd python
# finishes orphaned). Detached (PPID 1). Idempotent and non-interfering.
#
# Counts COMPLETED outputs (results.json for the gradient run, history.json for
# the breeding-style runs) -- NOT prompt dirs, which are created at prompt start
# and would overcount the in-flight prompt. Only touches DONE if finalize
# actually succeeds. Sources the venv so `python` resolves under setsid.
source /venv/main/bin/activate
source /workspace/.envrc 2>/dev/null
export PYTHONPATH=/workspace/repos/divgen HF_HOME=/workspace/hf
P=/workspace/runs/paperprep
LOG=$P/safety_done.log
echo "[$(date -u +%FT%TZ)] safety_done v2 up pid $$" >> "$LOG"
check(){  # dir  expected  countcmd  finalizecmd
  local d=$1 exp=$2 cnt
  [ -f "$P/$d/DONE" ] && return
  cnt=$(eval "$3")
  [ "${cnt:-0}" -ge "$exp" ] || return
  sleep 180                          # grace for the wrapper to finalize itself
  [ -f "$P/$d/DONE" ] && return
  cnt=$(eval "$3"); [ "${cnt:-0}" -ge "$exp" ] || return
  echo "[$(date -u +%FT%TZ)] $d reached $cnt/$exp, no DONE -> finalize+touch" >> "$LOG"
  if eval "$4" >> "$LOG" 2>&1; then
    touch "$P/$d/DONE"
    echo "[$(date -u +%FT%TZ)] $d finalized + DONE set" >> "$LOG"
  else
    echo "[$(date -u +%FT%TZ)] $d finalize FAILED -- leaving DONE unset for retry" >> "$LOG"
  fi
}
while true; do
  check e1_gradient 553 \
    "find $P/e1_gradient/geneval -name results.json 2>/dev/null | wc -l" \
    "python $P/finalize_e1.py"
  check e2_random 256 \
    "find $P/e2_random -maxdepth 2 -name history.json 2>/dev/null | wc -l" \
    "python $P/finalize_exp.py $P/e2_random --expect 256"
  check e3_cmadct 553 \
    "find $P/e3_cmadct -maxdepth 2 -name history.json 2>/dev/null | wc -l" \
    "python $P/finalize_exp.py $P/e3_cmadct --expect 553"
  [ -f "$P/NIGHT_DONE" ] && { echo "[$(date -u +%FT%TZ)] all done, safety exit" >> "$LOG"; exit 0; }
  sleep 300
done
