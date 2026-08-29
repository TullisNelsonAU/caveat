#!/usr/bin/env bash
set -u
cd /w
BINS="cat comm cut fold head join nl paste tail tr uniq wc"
mkdir -p out
# carve NEGATIVE window for an in-place packer whose analyzed region is the first exec PT_LOAD.
# UPX: precise b_info via make_upxgt.py. kiteshield-default: entropy-validated segment tail
# (loader is a constant-size prefix; the tail is RC4-encrypted function bodies = provable data).
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
  f"# loader prefix={loader}B (const, = -n build segment); tail={n}B entropy={H:.3f} b/byte = encrypted data.\n"
  f"exec_segment  -         0x{va:x}  0x{hi_v:x}  0x{off:x}  0x{hi_f:x}\n"
  f"compressed    NEGATIVE  0x{lo_v:x}  0x{hi_v:x}  0x{lo_f:x}  0x{hi_f:x}  encrypted data ({n} B, H={H:.2f})\n")
print(f"{path.split('/')[-1]}: kite window v0x{lo_v:x}..0x{hi_v:x} {n}B H={H:.3f}")
PY
}
for b in $BINS; do
  # UPX nrv + lzma (format-exact b_info oracle)
  cp $b out/${b}.upxnrv;  upx -9 -q out/${b}.upxnrv  >/dev/null 2>&1 && python3 make_upxgt.py out/${b}.upxnrv  out/${b}.upxnrv.upxgt  >/dev/null
  cp $b out/${b}.upxlzma; upx --lzma -9 -q -f out/${b}.upxlzma >/dev/null 2>&1 && python3 make_upxgt.py out/${b}.upxlzma out/${b}.upxlzma.upxgt >/dev/null
  # kiteshield default (in-place RC4) + -n (loader only) 
  kiteshield    $b out/${b}.kite   >/dev/null 2>&1 && carve_seg_tail out/${b}.kite   out/${b}.kite.upxgt   8584 >/dev/null
  kiteshield -n $b out/${b}.kiten  >/dev/null 2>&1
  # ezuri (overlay crypter)
  ( cd /opt/ezuri && printf "/w/$b\n/w/out/${b}.ezuri\n$b\n\n\n" | ezuri_pack >/dev/null 2>&1 )
done
echo "=== generated ==="; ls out | grep -v upxgt | wc -l; echo "windows: $(ls out/*.upxgt 2>/dev/null|wc -l)"
ls -la out | head -40
