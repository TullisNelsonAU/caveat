#!/usr/bin/env bash
# Reproduce the OLLVM staleness probe end to end.
# Engine of record: probdisasm feat/chainfwd-prior @ c62ead9 (#18). Local only.
#
# OLLVM toolchain: prebuilt heroims/obfuscator clang 13.0.1 in the Debian image
# `icyguider/ollvm` (glibc, so GNU coreutils builds cleanly; classic -mllvm -sub/-bcf/-fla).
set -euo pipefail
cd "$(dirname "$0")/../.."                       # upd-suite worktree root
OUT=docs/ollvm_staleness
WORK=~/lab/projects/probablistic/corpus/ollvm-obf/_work
IMG=icyguider/ollvm:latest

# 0. confirm the engine is #18
( cd ../probdisasm && test "$(git rev-parse HEAD)" = c62ead9b8d14b25164e68974b8822ac104eb1be5 || { echo "probdisasm not @ #18"; exit 1; } )

# 1. corpus: build the 4 OLLVM arms in the container, then label with gen-gt (idempotent)
if [ ! -d ~/lab/projects/probablistic/corpus/ollvm-obf/fla/bins ]; then
  [ -d "$WORK/coreutils-9.5" ] || { echo "unpack coreutils-9.5 into $WORK first"; exit 1; }
  for arm in benign sub bcf fla; do
    docker run --rm --platform linux/amd64 -v "$WORK":/w --entrypoint bash "$IMG" \
      -lc "export PATH=\$PATH:/usr/local/bin; bash /w/build_arm.sh $arm"
  done
  bash "$OUT/finalize_corpus.sh"
fi

# 2. harness from HEAD (regime-agnostic optregime, reused verbatim; benign = default)
cargo build --release --bin optregime
C=~/lab/projects/probablistic/corpus/ollvm-obf
./target/release/optregime \
  --level benign "$C/benign/bins" "$C/benign/gt" \
  --level sub    "$C/sub/bins"    "$C/sub/gt" \
  --level bcf    "$C/bcf/bins"    "$C/bcf/gt" \
  --level fla    "$C/fla/bins"    "$C/fla/gt" \
  --default benign --n-fit 12 --seed 1 \
  --out "$OUT/ollvm_staleness.csv" --summary "$OUT/ollvm_staleness_summary.json"

# 3. per-transform staleness tables + go/no-go
python3 "$OUT/analyze_ollvm.py" "$OUT/ollvm_staleness.csv" benign
