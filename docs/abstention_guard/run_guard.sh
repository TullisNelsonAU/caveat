#!/usr/bin/env bash
# Abstention-guard probe: keep the calibration switch from harming semantic obfuscation without
# touching the structural wins. Same corpus, same machinery, same seed as run_switching.sh — the only
# change is the guarded rule variant (region-entropy gate on the packed route) plus a held-out
# legitimate-VM false-positive gate. One invocation fits the bank + classifier (now also the guard
# entropy threshold) and runs the held-out eval, streaming both the old `rule` arm (which harms
# Tigress) and the new `guard` arm side by side so the fix is directly comparable.
#
# Corpus, same as the switching probe, plus:
#   benign-holdout vmlegit — p05_vm baseline + virtualized, vmbig (the CFG-probe legit VMs).
#                            True regime benign: they decode cleanly and are already calibrated under
#                            the benign map, so the correct GT-free action is to abstain (stay benign).
#                            gen-gt instruction GT. Held-out only.
#
# Tigress is regenerable via ../consistency_credibility/build_tigress_graded.sh (writes /tmp/tig_graded).
set -euo pipefail
cd "$(dirname "$0")/../.."                         # upd-suite(-regime) root
BIN=./target/release/switching
CORP=~/lab/projects/probablistic/corpus            # clean + desync corpora
PK=docs/consistency_credibility/packed             # multi-packer corpus (make_upxgt.py)
ROOT=$(pwd)
TIG=/tmp/tig_graded                                # graded Tigress (build_tigress_graded.sh)
VM=~/lab/projects/probablistic/corpus/vm-legit     # legit-VM FP gate (build_vm_legit.sh)
OUT=docs/abstention_guard

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

# Legit-VM FP gate (held-out, true regime benign).
VM_ARGS=()
[ -d "$VM/bins" ] && VM_ARGS+=(--benign-holdout vmlegit "$VM/bins" "$VM/gt")

"$BIN" \
  --clean-bins "$CORP/x86_64-binaries/elf/coreutils" --clean-gt "$CORP/gt" \
  --desync-level pilot    "$CORP/desync-pilot/stripped"               "$CORP/desync-gt" \
  --desync-level d1_med   "$CORP/desync-dense/d1_med/stripped_small"   "$CORP/desync-dense/d1_med/gt_small" \
  --desync-level d2_heavy "$CORP/desync-dense/d2_heavy/stripped_small" "$CORP/desync-dense/d2_heavy/gt_small" \
  --desync-level d3_max   "$CORP/desync-dense/d3_max/stripped_small"   "$CORP/desync-dense/d3_max/gt_small" \
  "${PACKED_ARGS[@]}" \
  "${TIG_ARGS[@]}" \
  "${VM_ARGS[@]}" \
  --n-clean-fit 15 --n-clean-holdout 20 \
  --n-desync-fit 25 --n-desync-holdout 25 \
  --n-packed-fit 9 --n-packed-holdout 8 \
  --n-tig-holdout 27 \
  --entropy-strength 1.0 --chainfwd-strength 0.5 --seed 1 \
  --out "$OUT/guard.csv" --summary "$OUT/guard_summary.json"
