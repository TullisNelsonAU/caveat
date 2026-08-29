#!/usr/bin/env bash
# Turn the raw OLLVM build output (_work/out/<arm>/<prog>) into the optregime corpus layout:
#   corpus/ollvm-obf/<arm>/bins/<prog>   +   corpus/ollvm-obf/<arm>/gt/<prog>.gt
# GT is gen-gt's insn_max (emitted instruction stream, construction-based — faithful for
# clang/OLLVM x86_64 since .text carries no inline data; this is why coreutils GT survives
# the transforms). Host-side (gen-gt runs on the Mac, cross-disassembles x86_64 via capstone).
set -euo pipefail
GENGT="$ROOT"/target/release/gen-gt
WORK=~/lab/projects/probablistic/corpus/ollvm-obf/_work
OUT=~/lab/projects/probablistic/corpus/ollvm-obf
ARMS="benign sub bcf fla"
for arm in $ARMS; do
  src="$WORK/out/$arm"
  [ -d "$src" ] || { echo "MISSING arm dir $src"; exit 1; }
  mkdir -p "$OUT/$arm/bins" "$OUT/$arm/gt"
  n=0
  for b in "$src"/*; do
    p=$(basename "$b")
    cp "$b" "$OUT/$arm/bins/$p"
    tmp=$(mktemp -d); "$GENGT" "$b" "$tmp" >/dev/null 2>&1
    cp "$tmp/insn_max.txt" "$OUT/$arm/gt/$p.gt"; rm -rf "$tmp"
    n=$((n+1))
  done
  echo "  $arm: $n binaries + gt"
done
echo "corpus at $OUT (4 arms)"
