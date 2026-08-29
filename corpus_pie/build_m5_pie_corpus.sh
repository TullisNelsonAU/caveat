#!/usr/bin/env bash
# Build the M5 PIE code-in-data corpus.
#
# Compiles the purpose-built PIE sources (corpus_pie/src/*.{c,cpp}) at -O2 -fPIC -pie -g
# with the x86_64-unknown-linux-gnu cross toolchain, so they carry the exact indirect
# idioms M4 targets: `pie_rel` 4-byte-relative switch/computed-goto jump tables (invisible
# to M3's 8-byte scan) and C++ `.data.rel.ro` vtables (whose PIE slots are R_X86_64_RELATIVE
# relocs). Then, on each seed:
#   * gen-gt (DWARF+symtab, crates/groundtruth) → instruction GT  = insn_max (the emitted
#     stream; construction-based, never a disassembler's opinion).
#   * cross objdump -t ` F .text` → function GT (FUNC symbols, the M2/M3/M4 method — PIE
#     changes addresses, not the GT method).
#   * gauntlet native-code-in-data → the code-in-data specimen (real .text ++ tiled code
#     decoy), with its own .gt (clipped to the code half), .regions, .manifest.json.
#
# GT is construction-based; tools are never GT. Deterministic and re-runnable.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/corpus_pie/src"
SEEDS="$REPO/corpus_pie/seeds"
OUT="$REPO/corpus_pie/cid"
XTC="/usr/local/opt/x86_64-unknown-linux-gnu/bin"
CC="$XTC/x86_64-unknown-linux-gnu-gcc"
CXX="$XTC/x86_64-unknown-linux-gnu-g++"
OBJDUMP="$XTC/x86_64-unknown-linux-gnu-objdump"
GENGT="$REPO/target/release/gen-gt"
GAUNTLET="$REPO/target/release/gauntlet"
CFLAGS="-O2 -fPIC -pie -g -fno-stack-protector"

mkdir -p "$SEEDS" "$OUT"
rm -f "$OUT"/*.elf "$OUT"/*.gt "$OUT"/*.regions "$OUT"/*.manifest.json "$OUT"/index.json 2>/dev/null || true

build_one() {
  local name="$1" src="$2" compiler="$3"
  echo "== $name =="
  "$compiler" $CFLAGS -o "$SEEDS/$name.elf" "$src"
  # instruction GT: gen-gt insn_max (emitted stream) for the seed .text
  local gdir="$SEEDS/$name.gtdir"; mkdir -p "$gdir"
  "$GENGT" "$SEEDS/$name.elf" "$gdir" >/dev/null
  cp "$gdir/insn_max.txt" "$SEEDS/$name.gt"
  # code-in-data specimen (region-bytes: ~a function's worth of decoy)
  "$GAUNTLET" generate --gen native-code-in-data \
    --seed-elf "$SEEDS/$name.elf" --seed-gt "$SEEDS/$name.gt" \
    --region-bytes 4096 --out "$OUT" >/dev/null
  # function GT (FUNC symbols of the seed) as <stem>.func.gt, the M2/M3/M4 method.
  # gauntlet keeps the seed's filename (incl .elf) in the artifact stem.
  local stem="$OUT/${name}.elf__native-code-in-data"
  "$OBJDUMP" -t "$SEEDS/$name.elf" \
    | awk '/ F \.text\t/ {print "0x"$1}' | sort -u > "$stem.func.gt"
  printf "   seed .text=%s  insn_gt=%s  func_gt=%s\n" \
    "$(python3 -c "import struct;d=open('$SEEDS/$name.elf','rb').read();import subprocess" 2>/dev/null; echo ok)" \
    "$(wc -l < "$SEEDS/$name.gt" | tr -d ' ')" \
    "$(wc -l < "$stem.func.gt" | tr -d ' ')"
}

build_one switch_dense    "$SRC/switch_dense.c"    "$CC"
build_one switch_tailcall "$SRC/switch_tailcall.c" "$CC"
build_one computed_goto   "$SRC/computed_goto.c"   "$CC"
build_one vtable_shapes   "$SRC/vtable_shapes.cpp" "$CXX"
build_one vtable_codec    "$SRC/vtable_codec.cpp"  "$CXX"

echo "== validate =="
"$GAUNTLET" validate "$OUT"
echo "corpus at $OUT"
