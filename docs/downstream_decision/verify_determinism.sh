#!/usr/bin/env bash
# Is the engine bit-deterministic across processes?
#
# `verify_ab.sh` flagged a mismatch in `mmae_pick` on clean binaries while every float column it is
# derived from agreed to 1e-9. On clean binaries the benign and packed engines produce the *same*
# S_glob (the entropy prior has nothing to bite on), so the argmin is decided by whatever is left in
# the low bits — a coin flip. That could be my refactor, or it could be run-to-run nondeterminism in
# the engine itself.
#
# This tells the two apart without involving the new code at all: run the SAME baseline binary twice
# with identical arguments and diff. If the picks disagree here, the nondeterminism is in the engine
# and predates this work. If they agree, the drift is mine and I have a bug to find.
set -euo pipefail
cd "$(dirname "$0")/../.."
CORP=~/lab/projects/probablistic/corpus
PK=docs/consistency_credibility/packed
WORK=${AB_WORK:-/tmp/downstream_ab}
BASE=${AB_BASELINE:?set AB_BASELINE to the baseline switching binary}
mkdir -p "$WORK"

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

for i in 1 2; do
  rm -f "$WORK/determinism_$i.csv"
  echo "── baseline switching, run $i ──"
  "$BASE" "${COMMON[@]}" \
    --out "$WORK/determinism_$i.csv" --summary "$WORK/determinism_$i.json" \
    >"$WORK/determinism_$i.log" 2>&1
done

echo "── diffing baseline against itself ──"
if diff -u "$WORK/determinism_1.csv" "$WORK/determinism_2.csv"; then
  echo "IDENTICAL — the engine is bit-deterministic across processes, so the verify_ab mismatch is a"
  echo "real difference introduced by the shared-module refactor. Find it."
else
  echo
  echo "DIFFERS — the same binary disagrees with itself on identical input. The engine is not"
  echo "bit-deterministic across processes; the verify_ab mismatch is that nondeterminism surfacing"
  echo "through a knife-edge argmin, not a refactor bug. Compare the columns above against the"
  echo "verify_ab mismatch set before drawing that conclusion."
fi
