#!/usr/bin/env bash
# Prepare the graded-drift + multi-packer corpora for run_credibility.sh. Idempotent.
#
#  1. Per-level desync GT: desync-cc changes the byte layout at each density, so each level needs its
#     own instruction GT. scripts/desync_gt.py recovers it from the *unstripped* desync binary's
#     junk-symbol markers (never by disassembling the stripped input under test).
#  2. Size-capped subsets (stripped_small / gt_small, .text < ~180 KB): the Soft engine's factor graph
#     is O(.text); a single 700 KB d3_max binary spikes to many GB. Capping bounds peak RSS to
#     ~2 GB so the serial one-binary-at-a-time run is memory-safe. Junk fraction is size-independent,
#     so the gradient (3.4% -> 17.9%) is preserved.
#  3. Multi-packer corpus: UPX-pack coreutils with two backends (NRV2, LZMA); carve format-exact GT
#     from each one's b_info chain (make_upxgt.py).
set -euo pipefail
PROB=~/lab/projects/probablistic
DESYNC_GT_PY="$PROB/scripts/desync_gt.py"
CU="$PROB/corpus/x86_64-binaries/elf/coreutils"
HERE="$(cd "$(dirname "$0")" && pwd)"
SIZE_CAP=180k

# readelf shim (macOS ships the cross-toolchain readelf under a prefixed name)
SHIM=$(mktemp -d); ln -sf "$(command -v x86_64-unknown-linux-gnu-readelf)" "$SHIM/readelf"
export PATH="$SHIM:$PATH"

echo "== 1+2. per-level desync GT + size-capped subsets =="
for lvl in d1_med d2_heavy d3_max; do
  U="$PROB/corpus/desync-dense/$lvl/unstripped"
  S="$PROB/corpus/desync-dense/$lvl/stripped"
  GT="$PROB/corpus/desync-dense/$lvl/gt";           mkdir -p "$GT"
  SS="$PROB/corpus/desync-dense/$lvl/stripped_small"; rm -rf "$SS"; mkdir -p "$SS"
  GS="$PROB/corpus/desync-dense/$lvl/gt_small";       rm -rf "$GS"; mkdir -p "$GS"
  n=0
  for f in "$U"/desync_coreutils_64_O0_*; do
    [ -f "$f" ] || continue
    b=$(basename "$f")
    [ -s "$GT/$b.gt" ] || python3 "$DESYNC_GT_PY" "$f" "$GT/$b.gt" >/dev/null 2>&1 || continue
    # size cap on the stripped input the engine actually analyzes
    if [ -n "$(find "$S" -name "$b" -size -$SIZE_CAP 2>/dev/null)" ]; then
      ln -sf "$S/$b" "$SS/$b"; ln -sf "$GT/$b.gt" "$GS/$b.gt"; n=$((n+1))
    fi
  done
  echo "  $lvl: GT built; $n binaries <$SIZE_CAP linked into stripped_small"
done

echo "== 3. multi-packer corpus (UPX NRV + LZMA) =="
PK="$HERE/packed"; mkdir -p "$PK"
for bin in sort od du ls cp dir base64 sha256sum; do
  SRC="$CU/gcc_coreutils_64_O0_$bin"; [ -f "$SRC" ] || continue
  for m in nrv lzma; do
    OUT="$PK/${bin}_upx_${m}"; [ -s "$OUT.upxgt" ] && continue
    cp "$SRC" "$OUT"
    if [ "$m" = lzma ]; then FL="--lzma -9 -q -f"; else FL="-9 -q -f"; fi
    if upx $FL "$OUT" >/dev/null 2>&1; then python3 "$HERE/make_upxgt.py" "$OUT" "$OUT.upxgt" >/dev/null
    else rm -f "$OUT"; echo "  skip ${bin}_${m} (upx declined)"; fi
  done
done
echo "  packed specimens: $(ls "$PK"/*.upxgt 2>/dev/null | wc -l | tr -d ' ')"
rm -rf "$SHIM"
echo "done — now: bash $HERE/run_credibility.sh"
