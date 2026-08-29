#!/usr/bin/env bash
# Label-free fire-rate probe over the wild Debian corpus.
#
# Engine of record: probdisasm `feat/chainfwd-prior` @ c62ead9. Serial, one binary in memory at a
# time, resumable — re-running after a kill picks up where it stopped.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BINS="${BINS:-$HOME/lab/projects/probablistic/corpus/wild-debian/bins}"

cd "$ROOT"
cargo build --release --bin firerate 2>&1 | tail -2

./target/release/firerate \
  --bins "$BINS" \
  --out "$HERE/firerate.csv" \
  --summary "$HERE/firerate_summary.json" \
  --max-code-bytes "${MAX_CODE:-3000000}"

# Record which engine actually produced the CSV. Do this here, in the runner, rather than trusting a
# prose "engine of record" line to stay true: probdisasm is a *path* dependency, so Cargo.lock pins
# only version 0.2.2 and a rebuild against a different checkout leaves no trace in the results.
python3 "$ROOT/docs/tools/engine_manifest.py" stamp "$HERE"
