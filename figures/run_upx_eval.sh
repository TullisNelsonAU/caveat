#!/usr/bin/env bash
# One-shot: build, run the labeled UPX over-commitment measurement, redraw the figure.
# Perfect ground truth — negatives come from UPX's own b_info chain, not entropy or a disassembler.
set -euo pipefail

ROOT=~/lab/projects/upd-suite
PACKED="$ROOT/corpus_packed/ls_packed"
FIG="$ROOT/figures"

cd "$ROOT"
cargo build --release -p upxeval

# stdout = CSV (region,strength,n,fp_rate,mean_p,brier,max_p); stderr = human-readable layout + table
./target/release/upxeval "$PACKED" --strengths 0,30 > "$FIG/upx_labeled.csv"

echo
echo "== labeled result =="
cat "$FIG/upx_labeled.csv"

python3 "$FIG/plot_upx_labeled.py"
echo "done — see $FIG/upx_labeled.{png,pdf,csv}"
