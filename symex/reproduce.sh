#!/usr/bin/env bash
# Phase-0 symex probe — reproduce. Read-only wrt probdisasm; consumes the Soft
# posterior sidecar (dissertation disassemble --mode soft) + the desync GT.
#
# Needs: angr in a venv, the prebuilt `dissertation` binary from the probdisasm
# repo (target/release/dissertation), and the desync-dense corpus.
set -euo pipefail

# --- point these at your checkout -------------------------------------------
PROB="${PROB:-$HOME/lab/projects/probablistic}"      # probdisasm repo
DISS="$PROB/target/release/dissertation"             # prebuilt emitter
CORP="$PROB/corpus/desync-dense"                     # d1_med / d2_heavy / d3_max
VENV="${VENV:-/tmp/angr-venv}"                       # python venv with angr
OUT="${OUT:-$(dirname "$0")/probe_results.json}"
WORK="${WORK:-/tmp/symex_probe}"; mkdir -p "$WORK"
# ---------------------------------------------------------------------------

# one-time: python3 -m venv "$VENV" && "$VENV/bin/pip" install angr
PY="$VENV/bin/python"

# sample: (level, name) across the three junk-density levels
SAMP=(
  "d1_med:echo"   "d1_med:cat"   "d2_heavy:base32"
  "d3_max:basename" "d3_max:true"
)

PAIRS=()
for s in "${SAMP[@]}"; do
  lvl=${s%%:*}; short=${s##*:}
  nm="desync_coreutils_64_O0_${short}"
  bin="$CORP/$lvl/unstripped/$nm"
  post="$WORK/post_${lvl}_${short}.json"
  gt="$CORP/$lvl/gt/$nm.gt"
  [ -f "$post" ] || "$DISS" disassemble "$bin" --mode soft --output "$post"
  PAIRS+=("$bin::$post::$gt")
done

"$PY" "$(dirname "$0")/probe.py" \
  --steps 400 --active-cap 40 --time-cap 90 \
  --out "$OUT" --pairs "${PAIRS[@]}"

echo "results -> $OUT"
