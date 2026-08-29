#!/usr/bin/env bash
set -e
W=/tmp/tig_graded; rm -rf $W; mkdir -p $W
SR=$(x86_64-unknown-linux-gnu-gcc -print-sysroot)
PROG=~/lab/projects/upd-suite/docs/derisk/programs
TIG=/Applications/tigress/4.0.11/tigress
export TIGRESS_HOME=/Applications/tigress/4.0.11; export PATH=$TIGRESS_HOME:$PATH
GENGT="$ROOT"/target/release/gen-gt

progs="p01_statemachine p02_insertsort p05_vm p06_parser p07_crc p08_binsearch p09_matmul p10_collatz p12_modpow"
fns_for() { case "$1" in
  p01_statemachine) echo "trans,main";; p02_insertsort) echo "isort,main";;
  p05_vm) echo "run,main";; p06_parser) echo "is_digit,eval,main";;
  p07_crc) echo "crc_step,main";; p08_binsearch) echo "bsearch_i,main";;
  p09_matmul) echo "matmul,main";; p10_collatz) echo "collatz_len,main";;
  p12_modpow) echo "gcd,modpow,main";; esac; }

# graded by family strength (de-risk drift ordering): Virtualize < EncodeArithmetic < Flatten
levels="tigL:Virtualize tigM:EncodeArithmetic tigH:Flatten"
for spec in $levels; do lvl=${spec%%:*}; mkdir -p $W/$lvl/bins $W/$lvl/gt; done

for p in $progs; do
  fns=$(fns_for $p)
  for spec in $levels; do
    lvl=${spec%%:*}; xf=${spec##*:}
    obf=$W/$lvl/$p.c
    $TIG --Environment=x86_64:Linux:Gcc:0 --Seed=20260707 --Transform=$xf --Functions=$fns --out=$obf $PROG/$p.c >/dev/null 2>&1 || { echo "tig fail $p/$lvl"; continue; }
    x86_64-unknown-linux-gnu-gcc --sysroot=$SR -O2 -g -no-pie $obf -o $W/$lvl/bins/$p 2>/dev/null || { echo "gcc fail $p/$lvl"; continue; }
    if $GENGT $W/$lvl/bins/$p $W/$lvl/gtd_$p >/dev/null 2>&1; then cp $W/$lvl/gtd_$p/insn_max.txt $W/$lvl/gt/$p.gt; else echo "gt fail $p/$lvl"; fi
  done
done
for spec in $levels; do lvl=${spec%%:*}; echo "$lvl: $(ls $W/$lvl/bins 2>/dev/null | wc -l | tr -d ' ') bins / $(ls $W/$lvl/gt 2>/dev/null | wc -l | tr -d ' ') gt"; done
echo "gt sample:"; head -2 $W/tigM/gt/p05_vm.gt 2>/dev/null
