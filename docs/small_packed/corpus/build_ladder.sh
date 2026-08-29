#!/usr/bin/env bash
# Freestanding size ladder: the crossover instrument for the size-aware gate.
#
# The glibc arm (genpack.sh) cannot reach the candidate counts Limitation 3 is actually about —
# glibc's crt/stub code floors a "hello-world-sized" dynamic ELF at n ~ 3800 once packed. So this
# ladder links k of the SAME small programs (docs/derisk/programs, k = 1..12, distinct programs, no
# repeats — repeated code would just compress away and break the ladder) with -nostdlib and a
# 2-instruction _start. Unpacked text runs from a few hundred bytes to a few KB, so the packed
# candidate count sweeps the n <= 2000 regime the paper flags and up into the glibc arm's range.
#
# Same compiler and same flags as the Tigress/CFG arms apart from -nostdlib -static.
set -euo pipefail
cd "$(dirname "$0")"
PROG=../../derisk/programs
GCC=x86_64-unknown-linux-gnu-gcc
SR=$($GCC -print-sysroot)
CFLAGS="--sysroot=$SR -O2 -g -no-pie -nostdlib -static -ffreestanding -fno-stack-protector"
# The minimal-layout link drops -g and the padding a normal layout carries. Objects are shared with
# the -g ladder above; only the link differs.
MINCFLAGS="--sysroot=$SR -O2 -no-pie -nostdlib -static -ffreestanding -fno-stack-protector"
MINLDFLAGS="-Wl,-z,noseparate-code -Wl,--build-id=none"

# Fixed unit order: the 9 programs of the main arm first, then the 3 remaining derisk programs.
UNITS=(p01_statemachine p02_insertsort p05_vm p06_parser p07_crc p08_binsearch p09_matmul \
       p10_collatz p12_modpow p03_ackermann p04_fnptr p11_sieve)
KS=(1 2 3 4 6 8 10 12)

OUT=ladder_src
rm -rf "$OUT" ladder ladder_min ladder_minu; mkdir -p "$OUT" ladder ladder_min ladder_minu

for k in "${KS[@]}"; do
  drv=$OUT/driver_k$k.c
  {
    echo "/* freestanding driver, k=$k units */"
    for ((i=0;i<k;i++)); do echo "int main_${UNITS[$i]}(void);"; done
    echo "static int run(void){ int s=0;"
    for ((i=0;i<k;i++)); do echo "  s += main_${UNITS[$i]}();"; done
    echo "  return s & 0xff; }"
    cat <<'C'
void _start(void) {
    int r = run();
    __asm__ volatile("syscall" :: "a"(60L), "D"((long)r) : "memory");
    __builtin_unreachable();
}
C
  } > "$drv"

  objs=("$drv")
  for ((i=0;i<k;i++)); do
    p=${UNITS[$i]}
    o=$OUT/${p}_k$k.o
    $GCC $CFLAGS -Dmain=main_$p -c "$PROG/$p.c" -o "$o"
    objs+=("$o")
  done
  $GCC $CFLAGS -e _start "${objs[@]}" -o "ladder/k$(printf %02d $k)"
  # Minimal-layout variants of the SAME objects: no separate code segment, no build-id. `ladder_minu`
  # is unstripped, `ladder_min` is stripped. These are the smallest images this substrate produces,
  # and they are what the packer-floor probe (probe_floor.sh) runs over.
  $GCC $MINCFLAGS $MINLDFLAGS -e _start "${objs[@]}" -o "ladder_minu/u$(printf %02d $k)"
  $GCC $MINCFLAGS $MINLDFLAGS -s -e _start "${objs[@]}" -o "ladder_min/m$(printf %02d $k)"

  sz=$(x86_64-unknown-linux-gnu-size "ladder/k$(printf %02d $k)" 2>/dev/null | awk 'NR==2{print $1}')
  printf "k=%-2s text=%-6sB  ladder=%-7sB  minu=%-7sB  min=%sB\n" "$k" "$sz" \
    "$(stat -c%s "ladder/k$(printf %02d $k)")" \
    "$(stat -c%s "ladder_minu/u$(printf %02d $k)")" \
    "$(stat -c%s "ladder_min/m$(printf %02d $k)")"
done
