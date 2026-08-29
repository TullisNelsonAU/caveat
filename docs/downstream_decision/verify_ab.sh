#!/usr/bin/env bash
# Drift guard for the shared bank/classifier/split module.
#
# `experimental/consistency/src/lib.rs` holds the machinery lifted out of `bin/switching.rs` so that
# `bin/downstream.rs` grades the *same* held-out binaries under the *same* bank. That leaves two
# copies of the logic in the tree, which is only safe if something checks they still agree. This is
# that check: run both binaries over a deliberately tiny corpus with identical flags and seed, then
# diff every column they have in common (the benign-engine signature, all four GT-free picks, the
# always-benign and oracle ECE, base rate, candidate count).
#
# If this fails, the two implementations have diverged — fix the divergence, don't relax the check.
#
# Cheap by construction: 4+3+3 fit binaries and 5 held-out, d3_max only (size-capped). Minutes, not
# hours, and small enough to run alongside another job without breaking the one-binary memory rule.
set -euo pipefail
cd "$(dirname "$0")/../.."                         # upd-suite root
CORP=~/lab/projects/probablistic/corpus
PK=docs/consistency_credibility/packed
ROOT=$(pwd)
WORK=${AB_WORK:-/tmp/downstream_ab}
BASE=${AB_BASELINE:?set AB_BASELINE to a switching binary built before the refactor}
mkdir -p "$WORK"

# A handful of packed specimens is plenty; take the first four NRV carvings deterministically.
PACKED_ARGS=()
for f in $(ls "$PK"/*_upx_nrv | head -4); do PACKED_ARGS+=(--packed-spec nrv "$f" "$f.upxgt"); done

COMMON=(
  --clean-bins "$CORP/x86_64-binaries/elf/coreutils" --clean-gt "$CORP/gt"
  --desync-level d3_max "$CORP/desync-dense/d3_max/stripped_small" "$CORP/desync-dense/d3_max/gt_small"
  "${PACKED_ARGS[@]}"
  --n-clean-fit 4 --n-clean-holdout 2
  --n-desync-fit 3 --n-desync-holdout 2
  --n-packed-fit 3 --n-packed-holdout 1
  --n-tig-holdout 0
  --entropy-strength 1.0 --chainfwd-strength 0.5 --seed 1
)

rm -f "$WORK"/switching_ab.csv "$WORK"/downstream_ab.csv "$WORK"/downstream_meta_ab.csv

echo "── baseline: switching ──"
"$BASE" "${COMMON[@]}" \
  --out "$WORK/switching_ab.csv" --summary "$WORK/switching_ab.json" >"$WORK/switching_ab.log" 2>&1

echo "── new: downstream ──"
./target/release/downstream "${COMMON[@]}" --tau 0.9 \
  --func-gt-root docs/downstream_decision/func_gt \
  --out "$WORK/downstream_ab.csv" --meta "$WORK/downstream_meta_ab.csv" \
  --summary "$WORK/downstream_ab.json" >"$WORK/downstream_ab.log" 2>&1

echo "── diffing shared columns ──"
python3 docs/downstream_decision/compare_ab.py "$WORK/switching_ab.csv" "$WORK/downstream_meta_ab.csv"
