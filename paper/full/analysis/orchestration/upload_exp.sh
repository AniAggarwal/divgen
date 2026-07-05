#!/bin/bash
# Tar one finished experiment and upload to today's Tigris prefix; refresh
# MANIFEST.md. Usage: upload_exp.sh <exp-dir-name>
set -euo pipefail
EXP=$1
P=/workspace/runs/paperprep
DATE=2026-07-04   # campaign prefix pinned to its start date
PREFIX=s3://anirud-dev/evodiv-divgen-fork/$DATE
SAFE=${EXP//\//-}                       # night/NA -> night-NA for the artifact name
TAR=/tmp/claude-0/paperprep-$SAFE.tar.gz
tar czf "$TAR" -C "$P" "$EXP"
/workspace/tigris.sh cp "$TAR" "$PREFIX/paperprep-$SAFE.tar.gz" | tail -1
rm -f "$TAR"
# refresh manifest: list of uploaded artifacts + statuses
{
  echo "# EvoDiv GPU results campaign — $DATE"
  echo
  echo "Canonical repo: github.com/AniAggarwal/divgen branch evolutionary-noise-opt."
  echo "Brief: paper/full/GPU_PLAN.md (reconstructed; see provenance note)."
  echo
  echo "## Uploaded artifacts"
  /workspace/tigris.sh ls "$PREFIX/" | awk '{print "- " $4 "  (" $3 " bytes)"}'
  echo
  echo "## Experiment status ($(date -u +%FT%TZ))"
  for d in "$P"/e*/; do
    name=$(basename "$d")
    if [ -f "$d/DONE" ]; then s="DONE"; elif [ -f "$d/run.log" ] || [ -d "$d/breeding" ]; then s="running/partial"; else s="queued"; fi
    echo "- $name: $s"
  done
  echo
  echo "Prior archive: s3://anirud-dev/evodiv-divgen-fork/2026-07-02/"
} > /tmp/claude-0/MANIFEST.md
/workspace/tigris.sh cp /tmp/claude-0/MANIFEST.md "$PREFIX/MANIFEST.md" | tail -1
echo "UPLOADED $EXP"
