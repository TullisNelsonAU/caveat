#!/usr/bin/env bash
# Paper-2 payoff: consistency-driven calibration-map switching.
#
# One invocation fits the regime-config bank (benign / packed / obfuscated maps, each under its own
# engine setting), trains the GT-free signature classifier, and runs the three-arm held-out
# evaluation (always-benign vs oracle vs consistency-selected). Serial, one binary in memory at a
# time (the three engine runs per held-out binary are sequential), resumable CSV.
#
# Corpus, four regimes:
#   benign      — clean coreutils (fit map + benign centroid)          GT: symbols of the original
#   obfuscated  — desync-cc density ladder, size-capped .text<180KB    GT: per-level desync_gt.py
#   packed      — UPX coreutils, NRV + LZMA backends + ls_packed        GT: UPX b_info window
#   tigress     — HELD-OUT ONLY, the semantic-obfuscation blind spot    GT: gen-gt symtab
#
# Tigress is regenerable via ../consistency_credibility/build_tigress_graded.sh (writes /tmp/tig_graded).
# If /tmp/tig_graded is gone, rebuild it first, or drop the --tigress-level flags.
set -euo pipefail
cd "$(dirname "$0")/../.."                         # upd-suite root
BIN=./target/release/switching
CORP=~/lab/projects/probablistic/corpus            # clean + desync corpora
PK=docs/consistency_credibility/packed             # multi-packer corpus (make_upxgt.py)
ROOT=$(pwd)
TIG=/tmp/tig_graded                                # graded Tigress (build_tigress_graded.sh)
OUT=docs/consistency_switching

# Packed specimens: every carved NRV/LZMA binary + the original ls_packed.
PACKED_ARGS=()
for f in "$PK"/*_upx_nrv;  do [ -e "$f" ] && PACKED_ARGS+=(--packed-spec nrv  "$f" "$f.upxgt"); done
for f in "$PK"/*_upx_lzma; do [ -e "$f" ] && PACKED_ARGS+=(--packed-spec lzma "$f" "$f.upxgt"); done
PACKED_ARGS+=(--packed-spec ls "$ROOT/corpus_packed/ls_packed" "$ROOT/corpus_packed/ls_packed.upxgt")

# Tigress levels only if the corpus is on disk.
TIG_ARGS=()
for lvl in tigL tigM tigH; do
  [ -d "$TIG/$lvl/bins" ] && TIG_ARGS+=(--tigress-level "$lvl" "$TIG/$lvl/bins" "$TIG/$lvl/gt")
done

"$BIN" \
  --clean-bins "$CORP/x86_64-binaries/elf/coreutils" --clean-gt "$CORP/gt" \
  --desync-level pilot    "$CORP/desync-pilot/stripped"               "$CORP/desync-gt" \
  --desync-level d1_med   "$CORP/desync-dense/d1_med/stripped_small"   "$CORP/desync-dense/d1_med/gt_small" \
  --desync-level d2_heavy "$CORP/desync-dense/d2_heavy/stripped_small" "$CORP/desync-dense/d2_heavy/gt_small" \
  --desync-level d3_max   "$CORP/desync-dense/d3_max/stripped_small"   "$CORP/desync-dense/d3_max/gt_small" \
  "${PACKED_ARGS[@]}" \
  "${TIG_ARGS[@]}" \
  --n-clean-fit 15 --n-clean-holdout 20 \
  --n-desync-fit 25 --n-desync-holdout 25 \
  --n-packed-fit 9 --n-packed-holdout 8 \
  --n-tig-holdout 27 \
  --entropy-strength 1.0 --chainfwd-strength 0.5 --seed 1 \
  --out "$OUT/switching.csv" --summary "$OUT/switching_summary.json"
