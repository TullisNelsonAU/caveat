#!/usr/bin/env bash
# Legit-VM false-positive gate corpus for the abstention guard. Three unambiguously-real bytecode
# interpreters whose dispatch loops trip the switch's spatial threshold but which decode cleanly and
# are normal-entropy code — the exact case the guard must abstain on:
#   p05_vm_baseline — the derisk VM program, clean-compiled (a real interpreter, no obfuscation)
#   p05_vm_virt     — the same VM, Tigress-virtualized (interpreter-in-interpreter)
#   vmbig           — the hand-written 40-opcode bytecode VM from the CFG probe
# GT from gen-gt (emitted-stream, construction-based). p05_vm_{baseline,virt} are reused from the
# rewrite_headtohead corpus (already carry gen-gt insn.gt); vmbig is built + gen-gt'd here.
set -euo pipefail
GCC=x86_64-unknown-linux-gnu-gcc
SR=$($GCC -print-sysroot)
GENGT="$ROOT"/target/release/gen-gt
VMSRC=~/lab/projects/upd-suite/docs/cfg_obf_probe/programs/vmbig.c
REW=~/lab/projects/probablistic/eval/rewrite_headtohead/corpus
AG=~/lab/projects/probablistic/corpus/vm-legit

rm -rf "$AG"; mkdir -p "$AG/bins" "$AG/gt"

echo "== vmbig (-O2 -g -no-pie, cross-gcc) + gen-gt =="
$GCC --sysroot="$SR" -O2 -g -no-pie "$VMSRC" -o "$AG/bins/vmbig"
$GENGT "$AG/bins/vmbig" /tmp/vmbig_gt >/dev/null
cp /tmp/vmbig_gt/insn_max.txt "$AG/gt/vmbig.gt"

echo "== p05_vm baseline + virtualize (reuse rewrite_headtohead gen-gt GT) =="
cp "$REW/p05_vm__baseline/obf.elf"   "$AG/bins/p05_vm_baseline"
cp "$REW/p05_vm__baseline/insn.gt"   "$AG/gt/p05_vm_baseline.gt"
cp "$REW/p05_vm__Virtualize/obf.elf" "$AG/bins/p05_vm_virt"
cp "$REW/p05_vm__Virtualize/insn.gt" "$AG/gt/p05_vm_virt.gt"

echo "== done =="; ls "$AG/bins"
