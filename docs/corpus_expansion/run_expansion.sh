#!/usr/bin/env bash
# Corpus expansion (Job 5): grow the evaluated clean + desync corpus WITHOUT rebuilding anything and
# WITHOUT disturbing the published fit. The switching harness splits each regime's seeded pool into
# fit (first N) then holdout (next M); keeping the fit counts identical (15 clean / 25 desync / 9 UPX,
# seed 1) means the bank + thresholds are bit-identical to every published run, and only the holdout
# grows. We just raise --n-clean-holdout / --n-desync-holdout to pull in binaries already built and on
# disk but never evaluated (clean 848 avail, 35 used; desync pool 234, 50 used). No `seed random`
# rebuild — that would overwrite the published corpus and break the committed RQ7/ablation results.
#
# Resumable per binary (read_existing_csv on --out), so a partial overnight run is still usable; the
# fit (~33 min) is redone on any restart but no holdout binary is recomputed. Deterministic (seed 1).
set -euo pipefail
cd "$(dirname "$0")/../.."                          # upd-suite-regime root
BIN=./target/release/switching
CORP=~/lab/projects/probablistic/corpus
OUT=docs/corpus_expansion
PK=docs/consistency_credibility/packed              # published UPX fit corpus (bank only)
ROOT=$(pwd)

# Published packed FIT set — bank only, so the expanded signatures use the identical calibration bank.
FIT_ARGS=()
for f in "$PK"/*_upx_nrv;  do [ -e "$f" ] && FIT_ARGS+=(--packed-spec nrv  "$f" "$f.upxgt"); done
for f in "$PK"/*_upx_lzma; do [ -e "$f" ] && FIT_ARGS+=(--packed-spec lzma "$f" "$f.upxgt"); done
FIT_ARGS+=(--packed-spec ls "$ROOT/corpus_packed/ls_packed" "$ROOT/corpus_packed/ls_packed.upxgt")

# Desync pool = the SAME dirs the published runs draw from (pilot + the dense _small subsets), so the
# first-25 fit prefix is unchanged; we only take more of the holdout tail from that same seeded pool.
"$BIN" \
  --clean-bins "$CORP/x86_64-binaries/elf/coreutils" --clean-gt "$CORP/gt" \
  --desync-level pilot    "$CORP/desync-pilot/stripped"               "$CORP/desync-gt" \
  --desync-level d1_med   "$CORP/desync-dense/d1_med/stripped_small"   "$CORP/desync-dense/d1_med/gt_small" \
  --desync-level d2_heavy "$CORP/desync-dense/d2_heavy/stripped_small" "$CORP/desync-dense/d2_heavy/gt_small" \
  --desync-level d3_max   "$CORP/desync-dense/d3_max/stripped_small"   "$CORP/desync-dense/d3_max/gt_small" \
  "${FIT_ARGS[@]}" \
  --n-clean-fit 15 --n-clean-holdout 300 \
  --n-desync-fit 25 --n-desync-holdout 200 \
  --n-packed-fit 9 --n-packed-holdout 0 \
  --n-tig-holdout 0 \
  --entropy-strength 1.0 --chainfwd-strength 0.5 --seed 1 \
  --out "$OUT/expanded.csv" --summary "$OUT/expanded.json"
echo "wrote $OUT/expanded.csv"
