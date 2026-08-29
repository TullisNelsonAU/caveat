#!/usr/bin/env bash
# The downstream payoff: what stale calibration costs a real analysis task, and what the switch buys
# back. The task is function-boundary recovery — the first thing every stripped-binary tool has to do.
#
# Same corpus, same seed, same split, same bank as the switching / abstention-guard runs — this is
# deliberately a re-reading of the *already-published* calibration numbers in task currency, not
# a new experiment on a new split. Change the counts here and the result stops being comparable to
# `docs/consistency_switching/` and `docs/abstention_guard/`, so don't.
#
# The task under test: recover the function head set. A head is recovered when a direct call reaches
# it and both ends of that call clear τ under the arm's calibrated posterior. Graded as boundary
# precision / recall / F1 against the unstripped original's `.symtab` FUNC entries — symbol-table
# rows, never a disassembly. Thresholds 0.5 / 0.7 / 0.9; the headline is τ=0.9.
#
# Run `gen_boundary_gt.py` first; it materializes that GT under `func_gt/<sublabel>/<name>.func.gt`
# and hard-gates on the stripped/unstripped `.text` matching. Packed gets no GT file on purpose —
# UPX's b_info chain proves data and never code, so packed F1 is reported as undefined.
#
# Serial, one binary in memory at a time (the three engine runs per held-out binary are sequential),
# resumable — re-running picks up from the meta CSV. Local only, no push.
set -euo pipefail
cd "$(dirname "$0")/../.."                         # upd-suite root
BIN=./target/release/downstream
CORP=~/lab/projects/probablistic/corpus            # clean + desync corpora
PK=docs/consistency_credibility/packed             # multi-packer corpus (make_upxgt.py)
ROOT=$(pwd)
TIG=/tmp/tig_graded                                # graded Tigress (build_tigress_graded.sh)
VM=$CORP/vm-legit                                  # legit-VM FP gate (build_vm_legit.sh)
OUT=docs/downstream_decision

# Packed specimens: every carved NRV/LZMA binary + the original ls_packed.
PACKED_ARGS=()
for f in "$PK"/*_upx_nrv;  do [ -e "$f" ] && PACKED_ARGS+=(--packed-spec nrv  "$f" "$f.upxgt"); done
for f in "$PK"/*_upx_lzma; do [ -e "$f" ] && PACKED_ARGS+=(--packed-spec lzma "$f" "$f.upxgt"); done
PACKED_ARGS+=(--packed-spec ls "$ROOT/corpus_packed/ls_packed" "$ROOT/corpus_packed/ls_packed.upxgt")

# Tigress levels only if the corpus is on disk. Held-out only — the semantic-obfuscation blind spot,
# carried here so the report can say plainly where the switch does *not* rescue the decision.
TIG_ARGS=()
for lvl in tigL tigM tigH; do
  [ -d "$TIG/$lvl/bins" ] && TIG_ARGS+=(--tigress-level "$lvl" "$TIG/$lvl/bins" "$TIG/$lvl/gt")
done

# Legit-VM false-positive gate (held-out, true regime benign): the switch must not wreck a decision
# on a binary that was never obfuscated in the first place.
VM_ARGS=()
[ -d "$VM/bins" ] && VM_ARGS+=(--benign-holdout vmlegit "$VM/bins" "$VM/gt")

"$BIN" \
  --clean-bins "$CORP/x86_64-binaries/elf/coreutils" --clean-gt "$CORP/gt" \
  --func-gt-root "$OUT/func_gt" \
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
  --tau 0.5 --tau 0.7 --tau 0.9 \
  --entropy-strength 1.0 --chainfwd-strength 0.5 --seed 1 \
  --out "$OUT/boundaries.csv" --meta "$OUT/boundaries_meta.csv" --summary "$OUT/boundaries_summary.json"
