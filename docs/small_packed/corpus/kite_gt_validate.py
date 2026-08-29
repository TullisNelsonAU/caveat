#!/usr/bin/env python3
"""Entropy-validate the kiteshield in-band ground-truth window, and emit the window only if it
survives validation.

The packer-breadth corpus carves kiteshield GT as "exec segment minus a constant 8584-byte loader
prefix = RC4 payload". Measured on these binaries that assumption does not hold: the default
(inner-encryption) build appends TWO things to the 8584-byte loader — a plaintext inner-decryption
runtime, and then the per-function key/trap table, which is the only genuinely random part. The
plaintext runtime is real x86 (it contains `0f 1f 84 00 ...` / `66 2e 0f 1f 84 00` alignment NOPs,
byte sequences RC4 output does not produce), so labelling it NEGATIVE would assert that real code
is provable data.

So: scan backwards from the end of the first executable PT_LOAD in 512-byte blocks while block
entropy stays >= THR. The surviving suffix is the provable-data window. If it is shorter than
MIN_WINDOW, this binary has NO in-band provable-data window and gets a ROUTING-ONLY placeholder
(same convention the breadth corpus uses for kiten/ezuri): signature and routing columns are still
computed, ECE is N/A.

Usage: kite_gt_validate.py <out_dir> <validation_csv>
"""
import glob
import math
import os
import struct
import sys
from collections import Counter

THR = 7.0          # bits/byte a 512-byte block must clear to count as packed data
BLOCK = 512
MIN_WINDOW = 1024  # a window smaller than this is not a usable oracle
LOADER = 8584      # the constant kiteshield loader prefix (= the `-n` build's exec segment)


def entropy(w):
    if not w:
        return 0.0
    c = Counter(w)
    n = len(w)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def exec_seg(b):
    phoff = struct.unpack_from("<Q", b, 0x20)[0]
    phes = struct.unpack_from("<H", b, 0x36)[0]
    phn = struct.unpack_from("<H", b, 0x38)[0]
    for i in range(phn):
        o = phoff + i * phes
        t, fl = struct.unpack_from("<I", b, o)[0], struct.unpack_from("<I", b, o + 4)[0]
        if t == 1 and fl & 1:
            return (
                struct.unpack_from("<Q", b, o + 8)[0],
                struct.unpack_from("<Q", b, o + 16)[0],
                struct.unpack_from("<Q", b, o + 32)[0],
            )
    raise SystemExit("no executable PT_LOAD")


def main():
    out_dir, csv_path = sys.argv[1], sys.argv[2]
    rows = ["name,seg_bytes,loader_prefix,legacy_window_bytes,legacy_window_ent,"
            "validated_window_bytes,validated_window_ent,gt_kind"]
    for path in sorted(glob.glob(os.path.join(out_dir, "*.kite"))):
        name = os.path.basename(path)
        b = open(path, "rb").read()
        off, va, fs = exec_seg(b)
        end = off + fs

        # What the breadth-corpus rule would have carved.
        legacy_lo = off + LOADER
        legacy = b[legacy_lo:end]

        # Entropy-validated suffix.
        lo = end
        while lo - BLOCK >= legacy_lo and entropy(b[lo - BLOCK:lo]) >= THR:
            lo -= BLOCK
        win = b[lo:end]

        gt = path + ".upxgt"
        if len(win) >= MIN_WINDOW:
            kind = "provable_data"
            lo_v, hi_v = va + (lo - off), va + (end - off)
            with open(gt, "w") as f:
                f.write(
                    f"# Entropy-validated NEGATIVE window for {name} (kiteshield in-band).\n"
                    f"# Suffix of the first exec PT_LOAD whose {BLOCK}B blocks all clear "
                    f"{THR} b/byte: {len(win)} B, H={entropy(win):.3f}.\n"
                    f"exec_segment  -         0x{va:x}  0x{va + fs:x}  0x{off:x}  0x{end:x}\n"
                    f"compressed    NEGATIVE  0x{lo_v:x}  0x{hi_v:x}  0x{lo:x}  0x{end:x}  "
                    f"encrypted data ({len(win)} B, H={entropy(win):.2f})\n"
                )
        else:
            kind = "routing_only"
            with open(gt, "w") as f:
                f.write(
                    f"# ROUTING-ONLY placeholder window (whole first exec PT_LOAD) for {name}.\n"
                    f"# NOT a provable-data oracle: no {BLOCK}B block in the post-loader tail clears "
                    f"{THR} b/byte\n"
                    f"# (best tail suffix = {len(win)} B), i.e. kiteshield's inner-encryption key/trap "
                    f"table is empty or\n"
                    f"# negligible at this program size and the outer-encrypted payload is out-of-band "
                    f"(second, non-exec\n"
                    f"# PT_LOAD). Present only so the pipeline computes the GT-free signature/routing "
                    f"columns. ECE is N/A.\n"
                    f"exec_segment  -         0x{va:x}  0x{va + fs:x}  0x{off:x}  0x{end:x}\n"
                    f"compressed    NEGATIVE  0x{va:x}  0x{va + fs:x}  0x{off:x}  0x{end:x}  "
                    f"PLACEHOLDER (code, not data)\n"
                )
        rows.append(
            f"{name},{fs},{LOADER},{len(legacy)},{entropy(legacy):.4f},"
            f"{len(win)},{entropy(win):.4f},{kind}"
        )
        print(f"{name:30s} seg={fs:6d} legacy={len(legacy):6d}B H={entropy(legacy):.3f} "
              f"-> validated={len(win):6d}B H={entropy(win):.3f} [{kind}]")
    open(csv_path, "w").write("\n".join(rows) + "\n")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
