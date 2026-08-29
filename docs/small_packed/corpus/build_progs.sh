#!/usr/bin/env bash
# Build the nine small C programs the Tigress and CFG arms use, with the SAME compiler and the SAME
# flags those arms use (docs/consistency_credibility/build_tigress_graded.sh,
# docs/cfg_obf_probe/build_corpus.sh) so the only difference between this corpus and theirs is the
# packer. Output: corpus/src/, the unpacked inputs for genpack.sh.
set -euo pipefail
# Run from the suite root with the same relative source path the original build used: with -g the
# compilation directory and the source path land in DWARF, so anything else changes the bytes.
cd "$(dirname "$0")/../../.."
PROG=docs/derisk/programs
OUT=docs/small_packed/corpus/src
GCC=x86_64-unknown-linux-gnu-gcc
SR=$($GCC -print-sysroot)
CFLAGS="--sysroot=$SR -O2 -g -no-pie"

rm -rf "$OUT"; mkdir -p "$OUT"
for p in p01_statemachine p02_insertsort p05_vm p06_parser p07_crc p08_binsearch p09_matmul \
         p10_collatz p12_modpow; do
  $GCC $CFLAGS "$PROG/$p.c" -o "$OUT/$p"
  echo "  $OUT/$p  $(stat -c%s "$OUT/$p")B"
done
