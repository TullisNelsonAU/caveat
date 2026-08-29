#!/usr/bin/env bash
# Pack the unstripped minimal-layout ladder (ladder_minu, 4.2 KB - 25.6 KB ELFs) with the same three
# configurations as the main arm. These are the smallest inputs the packers still accept, so this is
# the arm that pins the lowest reachable candidate count. Rows where a packer refuses are simply
# absent — probe_floor.sh records the refusals and the exact exception.
#
# Runs inside `packerbox` with this directory mounted at /w.
set -u
cd /w
mkdir -p minu_out
for b in $(cd ladder_minu && ls); do
  printf "%-4s in=%6sB " "$b" "$(stat -c%s ladder_minu/$b)"
  cp ladder_minu/$b minu_out/${b}.upxnrv
  if upx -9 -q minu_out/${b}.upxnrv >/dev/null 2>&1 \
     && python3 make_upxgt.py minu_out/${b}.upxnrv minu_out/${b}.upxnrv.upxgt >/dev/null; then
    printf "nrv=%sB " "$(stat -c%s minu_out/${b}.upxnrv)"
  else printf "nrv=refused "; rm -f minu_out/${b}.upxnrv minu_out/${b}.upxnrv.upxgt; fi

  cp ladder_minu/$b minu_out/${b}.upxlzma
  if upx --lzma -9 -q -f minu_out/${b}.upxlzma >/dev/null 2>&1 \
     && python3 make_upxgt.py minu_out/${b}.upxlzma minu_out/${b}.upxlzma.upxgt >/dev/null; then
    printf "lzma=%sB " "$(stat -c%s minu_out/${b}.upxlzma)"
  else printf "lzma=refused "; rm -f minu_out/${b}.upxlzma minu_out/${b}.upxlzma.upxgt; fi

  if kiteshield ladder_minu/$b minu_out/${b}.kite >/dev/null 2>&1; then
    printf "kite=%sB\n" "$(stat -c%s minu_out/${b}.kite)"
  else printf "kite=refused\n"; rm -f minu_out/${b}.kite; fi
done
echo "=== minu images ==="; ls minu_out | grep -v upxgt | wc -l
