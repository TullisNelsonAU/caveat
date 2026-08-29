#!/usr/bin/env bash
# Re-derive provenance for an already-harvested corpus, in place.
#
# Split out from the harvester on purpose: the download is slow and the classification heuristics are
# the part that needs iterating. This reads the package/version/source columns from the existing
# provenance.csv (that mapping is only knowable at download time) and recomputes everything that is a
# pure function of the bytes on disk — build-id, strippedness, language, .text size, DWARF producer.
#
# Runs inside a Debian container so readelf/strings are present and consistent.
set -uo pipefail

BINS=/out/bins
OLD=/out/provenance.csv
NEW=/out/provenance.new.csv

apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq --no-install-recommends binutils file >/dev/null 2>&1
command -v strings >/dev/null || { echo "FATAL: strings missing"; exit 1; }
command -v readelf >/dev/null || { echo "FATAL: readelf missing"; exit 1; }

echo "pkg,version,source,arch,path,sha256,size,build_id,stripped,lang,text_bytes,dwarf_producer,opt_hint" >"$NEW"

# path -> "pkg,version,source" from the harvest run
declare -A PKG VER SRC
while IFS=, read -r pkg version source arch path rest; do
  [ "$pkg" = "pkg" ] && continue
  PKG[$path]="$pkg"; VER[$path]="$version"; SRC[$path]="$source"
done <"$OLD"

n=0
for f in "$BINS"/*; do
  [ -f "$f" ] || continue
  base=$(basename "$f")
  n=$((n+1))

  sha=$(sha256sum "$f" | cut -d' ' -f1)
  size=$(stat -c %s "$f")

  bid=$(readelf -n "$f" 2>/dev/null | awk '/Build ID/ {print $3; exit}')
  [ -z "$bid" ] && bid=NA

  sections=$(readelf -SW "$f" 2>/dev/null)

  stripped=yes
  echo "$sections" | grep -q ' \.symtab ' && stripped=no

  # readelf -SW pads the index column ("[ 9]" vs "[15]"), so a fixed field number is wrong for
  # single-digit sections. Anchor on `.text` itself and take Size = 4 fields further along
  # (Name, Type, Address, Off, Size).
  texthex=$(echo "$sections" | awk '{for(i=1;i<=NF;i++) if($i==".text"){print $(i+4); exit}}')
  textb=0
  [ -n "$texthex" ] && textb=$(printf '%d' "0x$texthex" 2>/dev/null || echo 0)

  # One strings pass, reused by every language test — strings over a 75 MB Go binary is not cheap.
  str=$(strings -a "$f" 2>/dev/null | head -400000)

  # Order matters. Rust must be tested before C++: rustc itself links libstdc++ (for LLVM) and uses
  # _ZN-style mangling, so a C++-first order would misfile every Rust binary. Go must be first
  # because static Go binaries match almost nothing else.
  lang=c
  if echo "$sections" | grep -qE ' \.(gopclntab|go\.buildinfo)( |$)'; then
    lang=go
  elif echo "$str" | grep -qm1 -E 'Go build ID:|runtime\.goexit|go1\.[0-9]+\.[0-9]+'; then
    lang=go
  elif echo "$str" | grep -qm1 -E 'RUST_BACKTRACE|library/std/src|cargo/registry|/rustc/[0-9a-f]{8}|rustc-[0-9]+\.[0-9]+'; then
    lang=rust
  elif readelf -dW "$f" 2>/dev/null | grep -qE 'NEEDED.*lib(stdc\+\+|c\+\+)'; then
    lang=cxx
  elif echo "$str" | grep -qm1 -E '_ZNSt|_ZNK?[0-9]+[A-Za-z]'; then
    lang=cxx
  elif echo "$str" | grep -qm1 -E 'GHC [0-9]|camlProgram|caml_program|FPC [0-9]|GNAT[0-9 ]'; then
    lang=other
  fi

  prod=NA; opt=NA
  if echo "$sections" | grep -q ' \.debug_info '; then
    prod=$(readelf --debug-dump=info "$f" 2>/dev/null | grep -m1 -o 'DW_AT_producer.*' | \
           sed 's/DW_AT_producer *: *//; s/[",]/ /g' | tr -s ' ' | cut -c1-120)
    [ -z "$prod" ] && prod=NA
    opt=$(echo "$prod" | grep -o -- '-O[0-3sgz]' | head -1)
    [ -z "$opt" ] && opt=NA
  fi

  printf '%s,%s,%s,amd64,%s,%s,%s,%s,%s,%s,%s,"%s",%s\n' \
    "${PKG[$base]:-unknown}" "${VER[$base]:-NA}" "${SRC[$base]:-unknown}" "$base" "$sha" "$size" \
    "$bid" "$stripped" "$lang" "$textb" "$prod" "$opt" >>"$NEW"

  [ $((n % 100)) -eq 0 ] && echo "  ...$n"
done

mv "$NEW" "$OLD"
echo "[*] re-derived provenance for $n binaries"
