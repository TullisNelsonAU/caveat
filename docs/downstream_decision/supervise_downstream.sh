#!/usr/bin/env bash
# Overnight babysitter for the RQ7 full downstream run. The engine run is memory-heavy (~10GB/proc)
# and background jobs got reaped twice earlier tonight, so I don't trust a single launch to survive 9h
# unattended. This loops run_downstream.sh until it exits clean. The run itself resumes per-binary from
# boundaries_meta.csv, so a restart never redoes finished holdout binaries — worst case it re-does the
# ~33min fit (fit isn't resumable) and picks up where it left off. Strictly serial: exactly one
# run_downstream.sh at a time, never concurrent — concurrency OOMs the box, which is what we're avoiding.
set -uo pipefail
cd "$(dirname "$0")"
n=0
until ./run_downstream.sh; do
  n=$((n+1))
  echo "════ [$(date '+%F %T')] run_downstream exited nonzero — restart #$n (resumes from boundaries_meta.csv) ════"
  if [ "$n" -ge 40 ]; then
    echo "════ [$(date '+%F %T')] gave up after $n restarts — needs a human ════"
    exit 1
  fi
  sleep 10
done
echo "════ [$(date '+%F %T')] run_downstream completed clean (exit 0) after $n restart(s) ════"
