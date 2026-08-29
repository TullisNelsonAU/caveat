#!/usr/bin/env bash
# Build the stratum-B sample: binaries too big for the census pass.
#
# Why a sample at all. Engine cost grows ~n^1.5 in .text bytes (measured: 267 KiB → 139 s, 812 KiB →
# >600 s), so a census of the whole corpus is not affordable — the Go median .text is 2.3 MiB, which
# is roughly an hour per binary. The census pass therefore caps at 256 KiB, and that cap reaches
# *zero* Go binaries: the smallest Go .text in the corpus is 347 KiB. Since quantifying Go is one of
# the study's stated goals, the large tail has to be sampled rather than skipped.
#
# Selection is deterministic and size-blind: candidates in the band are ordered by their content
# sha256 — already a uniform random key — and the first k per language are taken. Re-running picks
# the same set. Nothing about the fire rate influences selection.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORP="${CORP:-$HOME/lab/projects/probablistic/corpus/wild-debian}"
DEST="${DEST:-$CORP/sample_large}"

LO=${LO:-262144}     # above the census cap
HI=${HI:-786432}     # 768 KiB — keeps per-binary cost near ~9 min at the top of the band

rm -rf "$DEST"; mkdir -p "$DEST"

pick() {  # $1 = lang, $2 = how many
  awk -F, -v l="$1" -v lo="$LO" -v hi="$HI" \
    'NR>1 && $10==l && $11>lo && $11<=hi {print $6, $5}' "$CORP/provenance.csv" \
    | sort | head -n "$2" | awk '{print $2}'
}

total=0
for spec in "go:5" "rust:4" "cxx:3" "c:3"; do
  lang="${spec%%:*}"; k="${spec##*:}"
  n=0
  while read -r name; do
    [ -z "$name" ] && continue
    ln -sf "$CORP/bins/$name" "$DEST/$name"
    n=$((n+1)); total=$((total+1))
  done < <(pick "$lang" "$k")
  echo "[*] $lang: $n selected"
done

echo "[*] stratum-B sample: $total binaries in $DEST"
