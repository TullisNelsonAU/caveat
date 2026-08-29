#!/usr/bin/env bash
# Runs *inside* a Debian bookworm container. Downloads amd64 binary packages, unpacks them, and
# harvests every ELF x86-64 executable they ship, recording provenance as it goes.
#
# We deliberately do not build anything. The whole point of this corpus is that we did not compile
# it: these are the exact bytes Debian ships to users, produced by upstream toolchains we never
# touched. `apt-get download` + `dpkg-deb -x` never executes package code, so the container is a
# fetch-and-unpack sandbox, not a build box.
#
# Host arch is arm64, so we run the container natively and add amd64 as a *foreign* dpkg arch. apt
# then downloads amd64 .debs while readelf/file stay native-speed. binutils reads foreign-arch ELF
# fine, so provenance extraction is unaffected.
set -uo pipefail

OUT=/out/bins
META=/out/provenance.csv
WORK=/work
mkdir -p "$OUT" "$WORK"

echo "[*] configuring apt for foreign amd64"
dpkg --add-architecture amd64
cat >/etc/apt/sources.list.d/amd64.list <<'EOF'
deb [arch=amd64] http://deb.debian.org/debian bookworm main contrib non-free non-free-firmware
deb [arch=amd64] http://deb.debian.org/debian bookworm-updates main contrib non-free non-free-firmware
EOF
# The stock sources.list is arch-unqualified; pin it to the native arch so apt does not try to fetch
# amd64 twice under different names.
if [ -f /etc/apt/sources.list ]; then
  sed -i 's|^deb |deb [arch=arm64] |' /etc/apt/sources.list 2>/dev/null || true
fi
if [ -f /etc/apt/sources.list.d/debian.sources ]; then
  sed -i '/^Architectures:/d' /etc/apt/sources.list.d/debian.sources
  sed -i 's|^Components:|Architectures: arm64\nComponents:|' /etc/apt/sources.list.d/debian.sources
fi

apt-get update -qq 2>&1 | tail -3
apt-get install -y -qq --no-install-recommends binutils file python3 xz-utils 2>&1 | tail -3

TOTAL=0
declare -A SEEN_SHA

# Incremental: a second pass over a different package list appends to the same corpus. Pre-seed the
# dedup set from the existing provenance so a binary shipped by two packages is still counted once
# across runs, and so re-running a list is a no-op rather than a way to double-count.
if [ -s "$META" ]; then
  while IFS=, read -r _pkg _ver _src _arch _path sha _rest; do
    [ "$_pkg" = "pkg" ] && continue
    [ -n "$sha" ] && SEEN_SHA[$sha]=1
  done <"$META"
  echo "[*] resuming: ${#SEEN_SHA[@]} binaries already in corpus"
else
  echo "pkg,version,source,arch,path,sha256,size,build_id,stripped,lang,text_bytes,dwarf_producer,opt_hint" >"$META"
fi

# ── provenance for one harvested file ─────────────────────────────────────────
probe() {
  local pkg="$1" ver="$2" src="$3" f="$4"

  local sha size
  sha=$(sha256sum "$f" | cut -d' ' -f1)
  [ -n "${SEEN_SHA[$sha]:-}" ] && return 1   # same bytes shipped by two packages; count once
  SEEN_SHA[$sha]=1
  size=$(stat -c %s "$f")

  # build-id
  local bid
  bid=$(readelf -n "$f" 2>/dev/null | awk '/Build ID/ {print $3; exit}')
  [ -z "$bid" ] && bid=NA

  local sections
  sections=$(readelf -SW "$f" 2>/dev/null)

  # stripped = no .symtab
  local stripped=yes
  echo "$sections" | grep -q ' \.symtab ' && stripped=no

  # .text size. readelf -SW prints it as bare hex; convert with printf rather than awk's strtonum,
  # which is a gawk extension and silently absent under the container's mawk.
  # readelf -SW pads the index column ("[ 9]" vs "[15]"), so a fixed field number is wrong for
  # single-digit sections. Anchor on `.text` itself and take Size = 4 fields further along.
  local texthex textb
  texthex=$(echo "$sections" | awk '{for(i=1;i<=NF;i++) if($i==".text"){print $(i+4); exit}}')
  textb=0
  [ -n "$texthex" ] && textb=$(printf '%d' "0x$texthex" 2>/dev/null || echo 0)

  # ── language / toolchain ────────────────────────────────────────────────────
  # Order matters. Rust must be tested before C++: rustc itself links libstdc++ (for LLVM) and uses
  # _ZN-style mangling, so a C++-first order would misfile every Rust binary. Go must be first
  # because static Go binaries match almost nothing else.
  #
  # Note on the Rust markers: Debian ships these stripped, so `rust_begin_unwind` is gone and there
  # is no `/rustc/<40-hex>` build path. What survives is panic-machinery strings and the embedded
  # crate paths — verified against ripgrep/exa/lsd before being trusted.
  local str
  str=$(strings -a "$f" 2>/dev/null | head -400000)
  local lang=c
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

  # ── DWARF producer + opt level, when debug info survived (rare in shipped debs) ──
  local prod=NA opt=NA
  if echo "$sections" | grep -q ' \.debug_info '; then
    prod=$(readelf --debug-dump=info "$f" 2>/dev/null | grep -m1 -o 'DW_AT_producer.*' | \
           sed 's/DW_AT_producer *: *//; s/[",]/ /g' | tr -s ' ' | cut -c1-120)
    [ -z "$prod" ] && prod=NA
    opt=$(echo "$prod" | grep -o -- '-O[0-3sgz]' | head -1)
    [ -z "$opt" ] && opt=NA
  fi

  local dest="$OUT/${pkg}__$(basename "$f")"
  # two packages can ship the same basename; disambiguate with a sha prefix
  [ -e "$dest" ] && dest="$OUT/${pkg}__${sha:0:8}__$(basename "$f")"
  cp "$f" "$dest" 2>/dev/null || return 1

  printf '%s,%s,%s,amd64,%s,%s,%s,%s,%s,%s,%s,"%s",%s\n' \
    "$pkg" "$ver" "$src" "$(basename "$dest")" "$sha" "$size" "$bid" \
    "$stripped" "$lang" "$textb" "$prod" "$opt" >>"$META"
  return 0
}

# ── main loop: one package at a time, unpacked into a scratch dir and thrown away ──
while read -r pkg; do
  [ -z "$pkg" ] && continue
  case "$pkg" in \#*) continue ;; esac

  rm -rf "$WORK/x"; mkdir -p "$WORK/x"
  ( cd "$WORK/x" && apt-get download "${pkg}:amd64" -qq >/dev/null 2>&1 )
  deb=$(ls "$WORK/x"/*.deb 2>/dev/null | head -1)
  if [ -z "$deb" ]; then echo "[skip] $pkg (no amd64 deb)"; continue; fi

  ver=$(dpkg-deb -f "$deb" Version 2>/dev/null | tr -d ' ,')
  src=$(dpkg-deb -f "$deb" Source 2>/dev/null | awk '{print $1}')
  [ -z "$src" ] && src="$pkg"

  rm -rf "$WORK/r"; mkdir -p "$WORK/r"
  dpkg-deb -x "$deb" "$WORK/r" 2>/dev/null || { echo "[skip] $pkg (unpack failed)"; continue; }

  n=0
  # Only harvest from executable dirs. This is what makes "executable" well defined: modern Debian
  # ships PIE, so ET_DYN no longer distinguishes a program from a shared library, and `file` calls
  # both "shared object". Where the distro *installs* the file is the reliable signal.
  while IFS= read -r f; do
    [ -L "$f" ] && continue
    head -c 4 "$f" 2>/dev/null | grep -q ELF || continue
    readelf -hW "$f" 2>/dev/null | grep -q 'X86-64' || continue
    if probe "$pkg" "$ver" "$src" "$f"; then n=$((n+1)); TOTAL=$((TOTAL+1)); fi
  done < <(find "$WORK/r" \( -path '*/bin/*' -o -path '*/sbin/*' -o -path '*/games/*' \) -type f -perm -u+x 2>/dev/null)

  echo "[ok] $pkg $ver -> $n binaries (total $TOTAL)"
  rm -rf "$WORK/x" "$WORK/r"
done < /out/pkglist.txt

echo "[*] harvested $TOTAL unique ELF x86-64 executables"
