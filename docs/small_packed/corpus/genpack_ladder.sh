#!/usr/bin/env bash
# Pack the freestanding size ladder with the same three configurations as the main arm.
# Runs inside `packerbox` with the corpus directory mounted at /w.
set -u
cd /w
mkdir -p ladder_out
for b in $(cd ladder && ls); do
  echo "── $b ──"
  cp ladder/$b ladder_out/${b}.upxnrv
  upx -9 -q ladder_out/${b}.upxnrv >/dev/null 2>&1 \
    && python3 make_upxgt.py ladder_out/${b}.upxnrv ladder_out/${b}.upxnrv.upxgt >/dev/null \
    && echo "  upxnrv ok" || { echo "  upxnrv FAIL"; rm -f ladder_out/${b}.upxnrv ladder_out/${b}.upxnrv.upxgt; }

  cp ladder/$b ladder_out/${b}.upxlzma
  upx --lzma -9 -q -f ladder_out/${b}.upxlzma >/dev/null 2>&1 \
    && python3 make_upxgt.py ladder_out/${b}.upxlzma ladder_out/${b}.upxlzma.upxgt >/dev/null \
    && echo "  upxlzma ok" || { echo "  upxlzma FAIL"; rm -f ladder_out/${b}.upxlzma ladder_out/${b}.upxlzma.upxgt; }

  kiteshield ladder/$b ladder_out/${b}.kite >/dev/null 2>&1 && echo "  kite ok" || { echo "  kite FAIL"; rm -f ladder_out/${b}.kite; }
done
echo "=== ladder images ==="; ls -l ladder_out | grep -v upxgt
