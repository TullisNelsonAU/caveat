#!/usr/bin/env bash
# Build the confidence-gated-rewriter corpus: runnable x86_64 PIE binaries with construction-based
# instruction GT and calibrated stack marginals. Every binary is a deterministic function of argv
# content → stdout, so the reference I/O (the behavioural oracle) is stable across runs and machines.
#
#   clean_calc   — straight-line -O0 code, no in-.text data: the CLEAN sanity case.
#   datatab      — reads a lookup table embedded in .text: the code-in-data HARD case (a linear-sweep
#                  rewriter instruments the table bytes and corrupts the constants the program reads).
#   junk_desync  — carries overlapping-instruction anti-disassembly traps: the DESYNC case.
#
# GT = gen-gt insn_max (the emitted stream; never a disassembler's opinion). Marginals = the coupled,
# recalibrated stack posterior P̂ per instruction (udstack --milestone b --layers 3 --dump-instr), fit
# transductively on each binary's own GT — the calibrated belief the rewriter gates on.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/corpus_rw/src"
BIN="$REPO/corpus_rw/bin"
XTC="/usr/local/opt/x86_64-unknown-linux-gnu/bin"
CC="$XTC/x86_64-unknown-linux-gnu-gcc"
GENGT="$REPO/target/release/gen-gt"
UDSTACK="$REPO/target/release/udstack"
mkdir -p "$BIN"

build_one() {
  local name="$1" opt="$2"
  echo "== $name ($opt) =="
  "$CC" $opt -fPIC -pie -g -fno-stack-protector -o "$BIN/$name.elf" "$SRC/$name.c"
  local gd="$BIN/$name.gtdir"; mkdir -p "$gd"
  "$GENGT" "$BIN/$name.elf" "$gd" >/dev/null
  cp "$gd/insn_max.txt" "$BIN/$name.gt"
  # calibrated coupled marginals P̂ (the gate signal). Transductive fit on the binary's own GT.
  "$UDSTACK" "$BIN/$name.elf" "$BIN/$name.gt" --milestone b --layers 3 --dump-instr 2>/dev/null \
    | grep '^instr_bel' > "$BIN/$name.marg"
  printf "   gt=%s marg=%s\n" "$(wc -l < "$BIN/$name.gt" | tr -d ' ')" "$(wc -l < "$BIN/$name.marg" | tr -d ' ')"
}

build_one clean_calc  "-O0"
build_one datatab     "-O1"
build_one junk_desync "-O1"
echo "corpus at $BIN"
