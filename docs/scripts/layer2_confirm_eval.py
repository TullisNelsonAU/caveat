#!/usr/bin/env python3
"""Layer-2 function-confirmation eval on the code-in-data corpus.

Answers LAYER2_CONFIRM_SPEC §4:
  (1) does --confirm suppress the decoy (leak floor toward ~0)?
  (2) confirmed_real_recall  — the hard-gate recall ceiling (from the CONFIRM stderr line).
  (3) precision at matched recall vs cover — does confirm beat cover in the band the ceiling permits?
  (4) honesty line identical (cover vs confirm).
  (6) indirect-only fraction = 1 - confirmed_real_recall.
"""
import subprocess, glob, os, re

BENCH = os.path.expanduser("~/lab/projects/upd-suite/target/release/bench")
BIASES = ",".join(f"{b/4:.2f}" for b in range(-80, 81))          # dense bias sweep
TARGETS = [0.30, 0.40, 0.42, 0.50, 0.60, 0.70, 0.80, 0.90]

def decoy_from(stem):
    for l in open(stem + ".regions"):
        if "junk_decoy" in l:
            return l.split()[0]                                   # already 0x-prefixed
    raise SystemExit("no junk_decoy region in " + stem)

def run(elf, gt, d, extra):
    cmd = [BENCH, elf, gt, "--decoy-from", d, "--biases", BIASES] + extra
    o = subprocess.run(cmd, capture_output=True, text=True)
    rows, calib, crr = [], None, None
    for ln in o.stderr.splitlines():
        m = re.search(r"confirmed_real_recall=([0-9.]+)", ln)
        if m:
            crr = float(m.group(1))
    for ln in o.stdout.splitlines():
        p = ln.split(",")
        if p[0] == "calibration":
            calib = ln
        elif len(p) == 7:
            try:
                rows.append((float(p[3]), float(p[4]), int(p[6])))  # recall, precision, leak
            except ValueError:
                pass
    return rows, calib, crr

def at_recall(rows, t):
    c = [r for r in rows if r[0] >= t - 1e-9]
    return max(c, key=lambda r: r[1]) if c else None

def leak_floor(rows):
    """min decoy-leak reached while still keeping recall >= 0.30 (the confirmed band)."""
    c = [r for r in rows if r[0] >= 0.30]
    return min((r[2] for r in c), default=None)

modes = {
    "cover":      ["--cover"],
    "confirm g4": ["--confirm", "--gamma", "4"],
    "confirm g8": ["--confirm", "--gamma", "8"],
    "confirm g16":["--confirm", "--gamma", "16"],
    "confirm g64":["--confirm", "--gamma", "64"],
}

specimens = sorted(glob.glob("/tmp/cid/*__native-code-in-data.elf"))
agg = {m: {t: [] for t in TARGETS} for m in modes}
floors = {m: [] for m in modes}
crrs, honesty_ok = [], True

for elf in specimens:
    stem = elf[:-4]; gt = stem + ".gt"; d = decoy_from(stem)
    name = os.path.basename(stem).split("__")[0]
    cover_cal = None
    for m, ex in modes.items():
        rows, cal, crr = run(elf, gt, d, ex)
        if m == "cover":
            cover_cal = cal
        else:
            if cal != cover_cal:
                honesty_ok = False
            if crr is not None:
                crrs.append((name, crr))
        for t in TARGETS:
            r = at_recall(rows, t)
            if r:
                agg[m][t].append(r)
        f = leak_floor(rows)
        if f is not None:
            floors[m].append(f)

print("=== §4.2 / §4.6  confirmed_real_recall ceiling (indirect-only = 1 - ceiling) ===")
for name, crr in crrs:
    print(f"  {name:>34}: ceiling={crr:.4f}  indirect-only={1-crr:.4f}")
if crrs:
    mean = sum(c for _, c in crrs) / len(crrs)
    print(f"  {'MEAN':>34}: ceiling={mean:.4f}  indirect-only={1-mean:.4f}")

print("\n=== §4.1 / §4.3  mean precision (and decoy-leak) at matched recall ===")
print(f"{'recall':>7} | " + " | ".join(f"{m:>22}" for m in modes))
for t in TARGETS:
    cells = []
    for m in modes:
        v = agg[m][t]
        if not v:
            cells.append("unreachable")
            continue
        p = sum(x[1] for x in v) / len(v)
        lk = sum(x[2] for x in v) / len(v)
        cells.append(f"P={p:.3f} lk={lk:6.0f} (n={len(v)})")
    print(f"{t:>7.2f} | " + " | ".join(f"{c:>22}" for c in cells))

print("\n=== §4.1  decoy-leak FLOOR (min leak with recall>=0.30) — suppression test ===")
for m in modes:
    v = floors[m]
    if v:
        print(f"  {m:>12}: mean floor = {sum(v)/len(v):7.0f}  (per-specimen {[int(x) for x in v]})")

print(f"\n=== §4.4  honesty line identical cover vs confirm: {'YES' if honesty_ok else 'NO'} ===")
