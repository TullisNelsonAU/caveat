#!/usr/bin/env bash
# Host-side driver: fetch third-party ELF x86-64 executables straight out of Debian bookworm.
#
# The corpus lands in probablistic/corpus/, which that repo's .gitignore excludes wholesale — the
# binaries are never committed, only provenance.csv and the results are.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${DEST:-$HOME/lab/projects/probablistic/corpus/wild-debian}"

mkdir -p "$DEST"
cp "$HERE/pkglist.txt" "$DEST/pkglist.txt"
cp "$HERE/build_corpus_inner.sh" "$DEST/build_corpus_inner.sh"

echo "[*] corpus destination: $DEST"
docker run --rm \
  -v "$DEST:/out" \
  -w /work \
  debian:bookworm-slim \
  bash /out/build_corpus_inner.sh 2>&1 | tee "$DEST/build.log"

echo "[*] done: $(find "$DEST/bins" -type f | wc -l | tr -d ' ') binaries"
