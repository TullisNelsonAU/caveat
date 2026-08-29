#!/usr/bin/env bash
# Phase-0 RE-PROBE on obfuscated binaries — the sweet spot the desync probe
# could not test. STAGED, not yet run: needs Tigress binaries, which are Ubuntu-
# box work. Run the instant the box is up. Same harness as reproduce.sh.
#
# Reframed question (see SYMEX_RESULTS.md): junk-pruning is dead (angr already
# avoids cleanly-disconnected decoys). The live question is INDIRECT-JUMP TARGET
# SELECTION under obfuscation — virtualization fan-out + opaque-predicate reached-
# but-fake paths — where symex actually explodes. The probe now emits, per binary:
#   indirect_src_blocks       computed-jump (Ijk_Boring, non-const) dispatch sites
#   indirect_targets_taken    distinct in-.text targets angr followed out of them
#   indirect_targets_GT_junk  of those, how many are GT-junk (fake dispatch)
#   indirect_junk_share       GT_junk / taken  <-- THE GATE NUMBER
# plus the desync metrics (exec %GT-real, low-conf false-negative split).
#
# GO for the Phase-1 build iff, on Tigress, indirect_junk_share is materially
# > 0 AND those junk targets are the ones our posterior marks low-confidence
# (indirect_targets_lowconf tracks the overlap) — i.e. a "only follow high-
# confidence resolved targets" filter would gate real junk without dropping the
# GT-real targets. If indirect_junk_share ~ 0 (angr resolves cleanly even under
# obfuscation), that's another honest NO-GO — report it and rethink again.
set -euo pipefail

# --- point these at the Ubuntu box checkout ---------------------------------
PROB="${PROB:-$HOME/lab/projects/probablistic}"
DISS="$PROB/target/release/dissertation"
TIG="${TIG:-$PROB/corpus/tigress}"     # obfuscated ELFs + per-binary .gt (real starts)
VENV="${VENV:-/tmp/angr-venv}"
OUT="${OUT:-$(dirname "$0")/tigress_results.json}"
WORK="${WORK:-/tmp/symex_tigress}"; mkdir -p "$WORK"
# transforms to sweep — where indirect dispatch / reached-but-fake actually live
TRANSFORMS="${TRANSFORMS:-Virtualize Flatten EncodeArithmetic}"  # + opaque-predicate variants
# ---------------------------------------------------------------------------

PY="$VENV/bin/python"
PAIRS=()
# Expected layout: $TIG/<transform>/<name>  with GT at $TIG/<transform>/gt/<name>.gt
for tf in $TRANSFORMS; do
  for bin in "$TIG/$tf"/*; do
    [ -f "$bin" ] || continue
    nm="$(basename "$bin")"
    gt="$TIG/$tf/gt/$nm.gt"
    [ -f "$gt" ] || { echo "WARN no GT for $tf/$nm, skipping" >&2; continue; }
    post="$WORK/post_${tf}_${nm}.json"
    [ -f "$post" ] || "$DISS" disassemble "$bin" --mode soft --output "$post"
    PAIRS+=("$bin::$post::$gt")
  done
done

# heavier budget than desync: obfuscated dispatch needs more steps to fan out
"$PY" "$(dirname "$0")/probe.py" \
  --steps 800 --active-cap 60 --time-cap 180 \
  --out "$OUT" --pairs "${PAIRS[@]}"

echo "results -> $OUT"
echo "GATE: inspect indirect_junk_share + indirect_targets_lowconf per binary."
