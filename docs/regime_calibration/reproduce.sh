#!/usr/bin/env bash
# Reproduce the regime-adaptive calibration probe end to end.
# Engine of record: probdisasm feat/chainfwd-prior @ c62ead9 (#18). Build from HEAD.
set -euo pipefail
cd "$(dirname "$0")/../.."                       # upd-suite worktree root
OUT=docs/regime_calibration
C=~/lab/projects/probablistic/corpus/regime-opt
# 0. confirm the engine is #18
( cd ../probdisasm && test "$(git rev-parse HEAD)" = c62ead9b8d14b25164e68974b8822ac104eb1be5 || { echo "probdisasm not @ #18"; exit 1; } )
# 1. corpus (idempotent)
[ -d "$C/O2/bins" ] || bash "$OUT/build_regime_corpus.sh"
# 2. engine + harness from HEAD
cargo build --release --bin optregime
# 3. run the probe (resumable; delete the CSV to recompute from scratch)
./target/release/optregime \
  --level O0 "$C/O0/bins" "$C/O0/gt" --level O1 "$C/O1/bins" "$C/O1/gt" \
  --level O2 "$C/O2/bins" "$C/O2/gt" --level O3 "$C/O3/bins" "$C/O3/gt" \
  --default O2 --n-fit 12 --seed 1 \
  --out "$OUT/optregime.csv" --summary "$OUT/optregime_summary.json"
# 4. tables + confound audit
python3 "$OUT/analyze_optregime.py" "$OUT/optregime.csv"
