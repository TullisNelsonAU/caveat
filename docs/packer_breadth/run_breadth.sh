#!/usr/bin/env bash
# Packer-breadth eval: reproduce the PUBLISHED consistency-switching fit (clean + desync + UPX,
# seed 1, identical bank + benign-default rule thresholds) and evaluate STRUCTURALLY-DISTINCT,
# held-out non-UPX packers through it via --packed-holdout (never in the fit set). This tests
# whether the UPX-fit packed regime + GT-free signature rule GENERALIZE across packer families.
#
# The original holdouts are shrunk to a sanity floor (the fit — and thus the bank/thresholds — is
# unchanged, since fit uses the first N of the seeded order and holdout the next M). Compute is
# focused on the new packers. Tigress is dropped (separate blind-spot probe).
#
# Usage: bash run_breadth.sh <arm>   where arm ∈ {main, ezuri}
#   main  — upxnrv / upxlzma (in-band compressed, format-exact b_info oracle),
#           kite (in-band RC4, entropy-validated tail oracle), kiten (out-of-band, routing-only)
#   ezuri — overlay crypter, 766 KB genuine Go .text (heavy engine run; separate for memory headroom)
set -euo pipefail
cd "$(dirname "$0")/../.."                          # upd-suite root
BIN=./target/release/switching
CORP=~/lab/projects/probablistic/corpus
OUT=docs/packer_breadth
COR=$OUT/corpus/out
ARM="${1:-main}"
PK=docs/consistency_credibility/packed                # the published UPX fit corpus
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

case "$ARM" in
  main)
    add upxnrv  upxnrv
    add upxlzma upxlzma
    add kite    kite
    add kiten   kiten
    CSV=$OUT/breadth_main.csv; SUM=$OUT/breadth_main.json ;;
  ezuri)
    add ezuri   ezuri
    CSV=$OUT/breadth_ezuri.csv; SUM=$OUT/breadth_ezuri.json ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac

"$BIN" \
  --clean-bins "$CORP/x86_64-binaries/elf/coreutils" --clean-gt "$CORP/gt" \
  --desync-level pilot    "$CORP/desync-pilot/stripped"               "$CORP/desync-gt" \
  --desync-level d1_med   "$CORP/desync-dense/d1_med/stripped_small"   "$CORP/desync-dense/d1_med/gt_small" \
  --desync-level d2_heavy "$CORP/desync-dense/d2_heavy/stripped_small" "$CORP/desync-dense/d2_heavy/gt_small" \
  --desync-level d3_max   "$CORP/desync-dense/d3_max/stripped_small"   "$CORP/desync-dense/d3_max/gt_small" \
  "${FIT_ARGS[@]}" \
  "${PACK_ARGS[@]}" \
  --n-clean-fit 15 --n-clean-holdout 6 \
  --n-desync-fit 25 --n-desync-holdout 6 \
  --n-packed-fit 9 --n-packed-holdout 0 \
  --n-tig-holdout 0 \
  --entropy-strength 1.0 --chainfwd-strength 0.5 --seed 1 \
  --out "$CSV" --summary "$SUM"
echo "wrote $CSV"
