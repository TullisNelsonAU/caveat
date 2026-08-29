#!/usr/bin/env bash
# CFG-obfuscation probe corpus. Four groups, all x86_64 ELF built with the SAME flags as the Tigress
# graded corpus (-O2 -g -no-pie, cross-gcc) so the only difference between clean and obfuscated is the
# transform — not the compiler or ABI.
#
#   obf_flatten  — Tigress --Flatten    (control-flow flattening)
#   obf_virt     — Tigress --Virtualize (bytecode virtualization)
#   clean_ctrl   — the SAME small programs, clean-compiled (controlled negative: isolates the
#                  obfuscation effect from program identity — clean vs obfuscated of the same source)
#   benign_switch— switch-heavy real coreutils (the real-world FP surface: legit dispatch code)
#   legit_interp — the interpreter FP GATE: clean bytecode VM / parser / state machine + a 40-op VM
#
# Tigress groups are reused from /tmp/tig_graded (build_tigress_graded.sh). Clean groups are built
# here. GT is not needed (the probe is GT-free topology), so we don't run gen-gt.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)                                           # docs/cfg_obf_probe
SUITE=$(cd ../.. && pwd)
PROG=$SUITE/docs/derisk/programs
CORE=~/lab/projects/probablistic/corpus/x86_64-binaries/elf/coreutils
TIG=/tmp/tig_graded
GCC=x86_64-unknown-linux-gnu-gcc
SR=$($GCC -print-sysroot)
CFLAGS="--sysroot=$SR -O2 -g -no-pie"

C=$ROOT/corpus
rm -rf "$C"; mkdir -p "$C"/{obf_flatten,obf_virt,clean_ctrl,benign_switch,legit_interp}

echo "── legit interpreters (the FP gate) ──"
# Interpreter family = the legit switch-heavy FP gate.
for p in p01_statemachine p05_vm p06_parser; do
  $GCC $CFLAGS "$PROG/$p.c" -o "$C/legit_interp/$p" 2>/dev/null && echo "  legit_interp/$p"
done
# The hand-written 40-opcode bytecode VM (a bigger, unambiguously-real interpreter).
$GCC $CFLAGS "$ROOT/programs/vmbig.c" -o "$C/legit_interp/vmbig" 2>/dev/null && echo "  legit_interp/vmbig"

echo "── clean controls (same programs Tigress obfuscates, clean-compiled) ──"
# The controlled negative: clean builds of the exact programs in obf_flatten/obf_virt, so the only
# difference from the obfuscated groups is the transform.
for p in p02_insertsort p03_ackermann p07_crc p08_binsearch p09_matmul p10_collatz p11_sieve p12_modpow; do
  $GCC $CFLAGS "$PROG/$p.c" -o "$C/clean_ctrl/$p" 2>/dev/null && echo "  clean_ctrl/$p"
done

echo "── switch-heavy real coreutils (real-world FP surface) ──"
# Real programs with genuine large switch statements — the real-world FP surface (legit dispatch
# code, not interpreters). Reuse the existing clean coreutils build.
for u in factor expr printf seq od stty dd date pr tail; do
  f=$(ls "$CORE"/*_"$u" 2>/dev/null | head -1)
  [ -n "$f" ] && cp "$f" "$C/benign_switch/coreutils_$u" && echo "  benign_switch/coreutils_$u"
done

echo "── Tigress obfuscated (reused from $TIG) ──"
if [ -d "$TIG/tigH/bins" ]; then
  for f in "$TIG/tigH/bins"/*; do cp "$f" "$C/obf_flatten/$(basename "$f")"; done
  echo "  obf_flatten: $(ls "$C/obf_flatten" | wc -l | tr -d ' ') bins"
else echo "  !! $TIG/tigH missing — run build_tigress_graded.sh"; fi
if [ -d "$TIG/tigL/bins" ]; then
  for f in "$TIG/tigL/bins"/*; do cp "$f" "$C/obf_virt/$(basename "$f")"; done
  echo "  obf_virt: $(ls "$C/obf_virt" | wc -l | tr -d ' ') bins"
else echo "  !! $TIG/tigL missing"; fi

echo "── corpus summary ──"
for g in obf_flatten obf_virt clean_ctrl benign_switch legit_interp; do
  echo "  $g: $(ls "$C/$g" 2>/dev/null | wc -l | tr -d ' ')"
done
