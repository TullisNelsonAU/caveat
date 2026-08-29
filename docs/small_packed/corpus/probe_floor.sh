#!/usr/bin/env bash
# Packer-floor probe: run every packer over both minimal ladders and record the exact outcome
# (or the exact exception) per input. Writes packer_floor.csv. Runs inside `packerbox` with this
# directory mounted at /w.
set -u
cd /w
CSV=packer_floor.csv
echo "input,bytes,stripped,upxnrv,upxlzma,kite" > $CSV

status_upx() { # <file> <lzma?>
  local f="$1" lz="$2" tmp
  tmp=$(mktemp); cp "$f" "$tmp"; chmod +x "$tmp"   # UPX refuses a non-executable file
  local log
  if [ "$lz" = 1 ]; then log=$(upx --lzma -9 -f "$tmp" 2>&1); else log=$(upx -9 -f "$tmp" 2>&1); fi
  if grep -q "^upx: .*Exception" <<<"$log"; then
    sed -n 's/^upx: [^:]*: \(.*Exception.*\)$/\1/p' <<<"$log" | head -1 | tr ',' ';'
  elif grep -qi "error" <<<"$log"; then
    echo "error"
  else
    echo "ok:$(stat -c%s "$tmp")B"
  fi
  rm -f "$tmp"
}

status_kite() {
  local f="$1" out log
  out=$(mktemp); log=$(kiteshield "$f" "$out" 2>&1)
  if [ -s "$out" ]; then echo "ok:$(stat -c%s "$out")B"
  else grep -v Copyright <<<"$log" | grep -v '^$' | tail -1 | tr ',' ';'; fi
  rm -f "$out"
}

for d in ladder_min ladder_minu; do
  [ -d "$d" ] || continue
  strip_flag=$([ "$d" = ladder_min ] && echo yes || echo no)
  for b in $(cd $d && ls); do
    f=$d/$b
    printf "%s,%s,%s,%s,%s,%s\n" "$b" "$(stat -c%s $f)" "$strip_flag" \
      "$(status_upx $f 0)" "$(status_upx $f 1)" "$(status_kite $f)" >> $CSV
  done
done
cat $CSV
