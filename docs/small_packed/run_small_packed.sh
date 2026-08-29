#!/usr/bin/env bash
# Small-packed eval (paper Limitation 3). Reproduce the PUBLISHED consistency-switching fit
# verbatim — the same clean/desync/UPX fit corpora, the same seed 1, the same
# --n-*-fit/--n-*-holdout counts, the same engine strengths as docs/packer_breadth/run_breadth.sh —
# and push SMALL packed binaries through that frozen bank via --packed-holdout.
#
# Because the fit args are byte-identical to the breadth run, the bank, the null thresholds and the
# operating point are identical, so these rows are directly comparable to Table IV.
#
# Corpus: the 9 small C programs the Tigress / CFG arms use (docs/derisk/programs), cross-compiled
# with the same flags as those arms, packed by docs/small_packed/corpus/genpack.sh with upx -9 (NRV),
# upx --lzma -9 and kiteshield (default, inner encryption).
set -euo pipefail
cd "$(dirname "$0")/../.."                          # upd-suite-regime root
BIN=./target/release/switching
CORP=~/lab/projects/probablistic/corpus
OUT=docs/small_packed
COR=$OUT/corpus/out
PK=docs/consistency_credibility/packed                # the published UPX fit corpus
ROOT=$(pwd)

# Published packed FIT set (identical bank): every carved NRV/LZMA UPX binary + ls_packed.
FIT_ARGS=()
for f in "$PK"/*_upx_nrv;  do [ -e "$f" ] && FIT_ARGS+=(--packed-spec nrv  "$f" "$f.upxgt"); done
for f in "$PK"/*_upx_lzma; do [ -e "$f" ] && FIT_ARGS+=(--packed-spec lzma "$f" "$f.upxgt"); done
FIT_ARGS+=(--packed-spec ls "$ROOT/corpus_packed/ls_packed" "$ROOT/corpus_packed/ls_packed.upxgt")

PACK_ARGS=()
add() { # add --packed-holdout LABEL ELF GT for every small packed binary with the given extension
  local label="$1" ext="$2"
  for f in "$COR"/*."$ext"; do
    [ -e "$f" ] || continue
    PACK_ARGS+=(--packed-holdout "$label" "$f" "$f.upxgt")
  done
}
add small_upxnrv  upxnrv
add small_upxlzma upxlzma
add small_kite    kite

CSV=$OUT/switching_small_packed.csv; SUM=$OUT/switching_small_packed.json

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
