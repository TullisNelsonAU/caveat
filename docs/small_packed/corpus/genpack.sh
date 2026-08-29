#!/usr/bin/env bash
# Pack the small Tigress/CFG-arm programs with the SAME packers and configurations as the
# packer-breadth corpus (docs/packer_breadth/corpus/genall.sh): upx -9 (NRV), upx --lzma -9,
# and kiteshield default (in-band per-function RC4).
#
# Runs inside the `packerbox` image (upx 4.2.4, kiteshield) with this directory mounted at /w.
# Ground truth is the packer's own provable-data window, identically to the existing packed corpus:
#   * UPX  — format-exact b_info chain via make_upxgt.py (copied verbatim from packer_breadth).
#   * kite — entropy-validated exec-segment tail; the loader is a constant-size prefix, measured
#            per program from the corresponding `kiteshield -n` build rather than hardcoded.
set -u
cd /w
mkdir -p out
PROGS=$(cd src && ls)

# Emit the kiteshield loader prefix (bytes) for a program: the exec PT_LOAD filesz of its `-n`
# build, which contains the loader stub and no inner-encrypted function bodies.
seg_filesz() { python3 - "$1" <<'PY'
import struct,sys
b=open(sys.argv[1],'rb').read()
phoff=struct.unpack_from('<Q',b,0x20)[0];phes=struct.unpack_from('<H',b,0x36)[0];phn=struct.unpack_from('<H',b,0x38)[0]
for i in range(phn):
    o=phoff+i*phes
    t,fl=struct.unpack_from('<I',b,o)[0],struct.unpack_from('<I',b,o+4)[0]
    if t==1 and fl&1:
        print(struct.unpack_from('<Q',b,o+32)[0]); break
PY
}

carve_seg_tail() { python3 - "$1" "$2" "$3" <<'PY'
import struct,sys,math
from collections import Counter
path,out,loader=sys.argv[1],sys.argv[2],int(sys.argv[3])
b=open(path,'rb').read()
phoff=struct.unpack_from('<Q',b,0x20)[0];phes=struct.unpack_from('<H',b,0x36)[0];phn=struct.unpack_from('<H',b,0x38)[0]
reg=None
for i in range(phn):
    o=phoff+i*phes;t,fl=struct.unpack_from('<I',b,o)[0],struct.unpack_from('<I',b,o+4)[0]
    off,va,fs=struct.unpack_from('<Q',b,o+8)[0],struct.unpack_from('<Q',b,o+16)[0],struct.unpack_from('<Q',b,o+32)[0]
    if t==1 and fl&1: reg=(off,va,fs);break
off,va,fs=reg
lo_f=off+loader; hi_f=off+fs; lo_v=va+loader; hi_v=va+fs
w=b[lo_f:hi_f]; c=Counter(w);n=len(w);H=-sum((v/n)*math.log2(v/n) for v in c.values()) if n else 0
open(out,'w').write(
  f"# Entropy-validated NEGATIVE window for {path.split('/')[-1]} (kiteshield in-place RC4).\n"
  f"# loader prefix={loader}B (= -n build segment); tail={n}B entropy={H:.3f} b/byte = encrypted data.\n"
  f"exec_segment  -         0x{va:x}  0x{hi_v:x}  0x{off:x}  0x{hi_f:x}\n"
  f"compressed    NEGATIVE  0x{lo_v:x}  0x{hi_v:x}  0x{lo_f:x}  0x{hi_f:x}  encrypted data ({n} B, H={H:.2f})\n")
print(f"{path.split('/')[-1]}: kite window v0x{lo_v:x}..0x{hi_v:x} {n}B H={H:.3f}")
PY
}

for b in $PROGS; do
  echo "── $b ──"
  cp src/$b out/${b}.upxnrv
  if upx -9 -q out/${b}.upxnrv >/dev/null 2>&1; then
    python3 make_upxgt.py out/${b}.upxnrv out/${b}.upxnrv.upxgt >/dev/null \
      && echo "  upxnrv ok" || { echo "  upxnrv GT FAIL"; rm -f out/${b}.upxnrv; }
  else
    echo "  upxnrv PACK FAIL"; rm -f out/${b}.upxnrv
  fi

  cp src/$b out/${b}.upxlzma
  if upx --lzma -9 -q -f out/${b}.upxlzma >/dev/null 2>&1; then
    python3 make_upxgt.py out/${b}.upxlzma out/${b}.upxlzma.upxgt >/dev/null \
      && echo "  upxlzma ok" || { echo "  upxlzma GT FAIL"; rm -f out/${b}.upxlzma; }
  else
    echo "  upxlzma PACK FAIL"; rm -f out/${b}.upxlzma
  fi

  if kiteshield src/$b out/${b}.kite >/dev/null 2>&1 \
     && kiteshield -n src/$b out/${b}.kiten >/dev/null 2>&1; then
    L=$(seg_filesz out/${b}.kiten)
    echo "  kite loader prefix = ${L}B"
    carve_seg_tail out/${b}.kite out/${b}.kite.upxgt "$L" && echo "  kite ok"
  else
    echo "  kite PACK FAIL"; rm -f out/${b}.kite out/${b}.kiten
  fi
done

echo "=== packed images ==="; ls -l out | grep -v upxgt
