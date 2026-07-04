#!/bin/bash
# Utilization watchdog for long GPU runs (not a liveness check).
# Usage: watchdog.sh <logfile> <pidfile> [stall_minutes]
# A run is healthy only if its log tail is fresh OR the GPU is busy.
# If BOTH the log is stale for >stall_minutes AND GPU util stays 0%,
# kill the process group so the launcher's retry loop restarts it (resume mode
# skips completed prompts). Appends one status line per minute to
# <logfile>.health.
LOG="$1"; PIDFILE="$2"; STALL_MIN="${3:-15}"
HEALTH="${LOG}.health"
zero_streak=0
while true; do
  sleep 60
  [ -f "$PIDFILE" ] || { echo "$(date -u +%FT%T) pidfile gone, watchdog exiting" >> "$HEALTH"; exit 0; }
  PID=$(cat "$PIDFILE")
  kill -0 "$PID" 2>/dev/null || { echo "$(date -u +%FT%T) pid $PID dead, watchdog exiting" >> "$HEALTH"; exit 0; }
  UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  AGE=$(( $(date +%s) - $(stat -c %Y "$LOG" 2>/dev/null || echo 0) ))
  echo "$(date -u +%FT%T) util=${UTIL}% log_age=${AGE}s pid=$PID" >> "$HEALTH"
  if [ "$UTIL" -eq 0 ] 2>/dev/null; then zero_streak=$((zero_streak+1)); else zero_streak=0; fi
  if [ "$AGE" -gt $((STALL_MIN*60)) ] && [ "$zero_streak" -ge "$STALL_MIN" ]; then
    echo "$(date -u +%FT%T) STALL: log stale ${AGE}s and util 0% for ${zero_streak}m -> killing $PID" >> "$HEALTH"
    kill -9 -- -"$PID" 2>/dev/null || kill -9 "$PID" 2>/dev/null
    zero_streak=0
  fi
done
