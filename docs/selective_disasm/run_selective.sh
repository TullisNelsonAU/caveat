#!/usr/bin/env bash
# Selective-disassembly precision demo on the PACKED corpus — the regime where the calibration error
# actually lives (packed ECE 0.366; desync only 0.076). Reproduces the published consistency-switching
# fit (15 clean / 25 desync / 9 UPX, seed 1, identical bank + thresholds), then evaluates the in-band
# packer families through it and dumps, per held-out binary, the calibrated posterior of every
# candidate inside the packer's provable-data window under each arm. That window is provable data, so
# any address an arm asserts as code there is a fabricated head — the raw material for the offline
# requested-vs-achieved precision sweep (analyze_selective.py).
#
# Only IN-BAND packers are scored here: upxnrv / upxlzma (UPX compressed, format-exact b_info window)
# and kite (kiteshield RC4, entropy-validated encrypted-tail window). kiten/ezuri are OUT-of-band
# (analyzed region is genuine loader/Go .text, no in-band data oracle) — a "code head" there is not a
# fabrication, so they are excluded from a precision-against-data-window demo by construction.
set -euo pipefail
cd "$(dirname "$0")/../.."                          # upd-suite-regime root
BIN=./target/release/switching
CORP=~/lab/projects/probablistic/corpus
OUT=docs/selective_disasm
COR=docs/packer_breadth/corpus/out                  # reuse the breadth corpus (never re-carved)
PK=docs/consistency_credibility/packed              # the published UPX fit corpus
ROOT=$(pwd)

# Published packed FIT set (identical bank): every carved NRV/LZMA UPX binary + ls_packed.
FIT_ARGS=()
for f in "$PK"/*_upx_nrv;  do [ -e "$f" ] && FIT_ARGS+=(--packed-spec nrv  "$f" "$f.upxgt"); done
for f in "$PK"/*_upx_lzma; do [ -e "$f" ] && FIT_ARGS+=(--packed-spec lzma "$f" "$f.upxgt"); done
FIT_ARGS+=(--packed-spec ls "$ROOT/corpus_packed/ls_packed" "$ROOT/corpus_packed/ls_packed.upxgt")

PACK_ARGS=()
add() { # add --packed-holdout LABEL ELF GT for every corpus binary with the given extension
  local label="$1" ext="$2"
  for f in "$COR"/*."$ext"; do
    [ -e "$f" ] || continue
    PACK_ARGS+=(--packed-holdout "$label" "$f" "$f.upxgt")
  done
}
add upxnrv  upxnrv
add upxlzma upxlzma
add kite    kite

CSV=$OUT/selective.csv
SUM=$OUT/selective.json
DUMP=$OUT/selective_posteriors.csv

"$BIN" \
  --clean-bins "$CORP/x86_64-binaries/elf/coreutils" --clean-gt "$CORP/gt" \
  --desync-level pilot    "$CORP/desync-pilot/stripped"               "$CORP/desync-gt" \
  --desync-level d1_med   "$CORP/desync-dense/d1_med/stripped_small"   "$CORP/desync-dense/d1_med/gt_small" \
  --desync-level d2_heavy "$CORP/desync-dense/d2_heavy/stripped_small" "$CORP/desync-dense/d2_heavy/gt_small" \
  --desync-level d3_max   "$CORP/desync-dense/d3_max/stripped_small"   "$CORP/desync-dense/d3_max/gt_small" \
  "${FIT_ARGS[@]}" \
  "${PACK_ARGS[@]}" \
  --n-clean-fit 15 --n-clean-holdout 0 \
  --n-desync-fit 25 --n-desync-holdout 0 \
  --n-packed-fit 9 --n-packed-holdout 0 \
  --n-tig-holdout 0 \
  --entropy-strength 1.0 --chainfwd-strength 0.5 --seed 1 \
  --out "$CSV" --summary "$SUM" --selective-dump "$DUMP"
echo "wrote $CSV and $DUMP"
