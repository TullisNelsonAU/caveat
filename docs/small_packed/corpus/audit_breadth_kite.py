#!/usr/bin/env python3
"""Read-only audit of the PUBLISHED kiteshield ground-truth windows in the breadth corpus.

Writes breadth_kite_window_audit.csv. Touches nothing in docs/packer_breadth — it only measures how
much of each published NEGATIVE window (exec segment minus the constant 8,584-byte loader prefix) is
actually high-entropy packer data, using the same 512-byte / 7.0 b/byte validation as
kite_gt_validate.py.

Usage: audit_breadth_kite.py <breadth_out_dir> <out.csv>
"""
import glob
import math
import os
import struct
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kite_gt_validate import BLOCK, LOADER, THR, entropy, exec_seg  # noqa: E402


def main():
    src, out = sys.argv[1], sys.argv[2]
    rows = ["name,seg_bytes,published_window_bytes,published_window_ent,"
            "validated_window_bytes,validated_window_ent,plaintext_prefix_bytes,plaintext_frac"]
    for path in sorted(glob.glob(os.path.join(src, "*.kite"))):
        b = open(path, "rb").read()
        off, va, fs = exec_seg(b)
        end = off + fs
        pub_lo = off + LOADER
        lo = end
        while lo - BLOCK >= pub_lo and entropy(b[lo - BLOCK:lo]) >= THR:
            lo -= BLOCK
        pub, val = end - pub_lo, end - lo
        pre = pub - val
        rows.append(f"{os.path.basename(path)},{fs},{pub},{entropy(b[pub_lo:end]):.4f},"
                    f"{val},{entropy(b[lo:end]):.4f},{pre},{pre / pub:.4f}")
        print(rows[-1])
    open(out, "w").write("\n".join(rows) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
