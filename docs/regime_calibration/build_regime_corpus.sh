#!/usr/bin/env bash
# Build the CORE opt-level corpus for the regime-calibration probe: the SAME coreutils programs
# compiled at gcc O0/O1/O2/O3, so optimization level is the sole variable. We reuse the existing
# unstripped+debug gcc coreutils builds on disk (real, sizeable .text — stable ECE/Moran) and label
# them with gen-gt (construction-based instruction stream, never a disassembler's opinion). These are
# ET_EXEC not PIE; PIE is irrelevant to a calibration measurement (disassembly is identical) and no
# PIE multi-opt source set exists on disk — see RESULTS.md §corpus.
set -euo pipefail
SRC=~/lab/projects/probablistic/corpus/x86_64-binaries/elf/coreutils
GENGT="$ROOT"/target/release/gen-gt   # insn_max = emitted stream GT
OUT=~/lab/projects/probablistic/corpus/regime-opt
# 24 programs spanning ~15-50 KB .text (size spread makes the confound audit meaningful).
PROGS="tty cksum sleep expand paste pwd id comm base64 wc head md5sum tr cut printf seq cat dd mkdir date chown pr tail stat"
for L in O0 O1 O2 O3; do mkdir -p "$OUT/$L/bins" "$OUT/$L/gt"; done
for p in $PROGS; do
  for L in O0 O1 O2 O3; do
    b="$SRC/gcc_coreutils_64_${L}_${p}"; [ -e "$b" ] || { echo "MISSING $b"; exit 1; }
    cp "$b" "$OUT/$L/bins/$p"
    tmp=$(mktemp -d); "$GENGT" "$b" "$tmp" >/dev/null 2>&1
    cp "$tmp/insn_max.txt" "$OUT/$L/gt/$p.gt"; rm -rf "$tmp"
  done
  echo "  $p done"
done
echo "corpus at $OUT (24 programs x 4 levels)"
