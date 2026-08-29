#!/usr/bin/env python3
"""Selective-disassembly precision demo, computed offline from selective_posteriors.csv.

The analyst asks the selective disassembler for a precision guarantee p* (0.90 / 0.95 / 0.99). For
each arm the calibration map maps a raw engine posterior to a calibrated P(code). We honour the
guarantee the only way a calibrated probability lets us: assert a candidate is a code head iff its
calibrated posterior >= p* (every selected head then carries a >= p* self-reported chance of being
code, so the map CLAIMS the selection is >= p* precise). We then measure the precision the arm
ACTUALLY delivered against construction-based ground truth.

Ground truth is the packer's provable-data payload window (UPX b_info / kiteshield entropy-validated
tail) — never a disassembly. Every candidate in that window is provable DATA (label 0). So every head
an arm asserts inside it is a fabricated code head (a false positive), and:

    achieved precision = TP / (TP + FP) = 0 / asserted = 0.000   whenever the arm asserts >=1 head,
                       = undefined (no assertions, no false discovery)   when it asserts 0.

A well-calibrated packed map assigns ~0 P(code) to compressed/encrypted data, so at p*=0.99 it
asserts nothing and keeps its promise vacuously (0 fabricated heads). The stale (always-benign) map
is overconfident on high-entropy bytes, so it keeps asserting code into the payload — promising 0.99
and delivering 0.000. That gap, in fabricated-head counts, is the demo.

Only in-band packers appear in the dump (upxnrv/upxlzma/kite): their analyzed window IS the data
oracle. kiten/ezuri are out-of-band (genuine loader/Go code) and carry no in-band data oracle, so a
"code head" there is not a fabrication and they are excluded by construction.
"""
import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DUMP = HERE / "selective_posteriors.csv"
if not DUMP.exists():
    DUMP = HERE / "selective_posteriors.csv.gz"
TARGETS = [0.90, 0.95, 0.99]
ARMS = ["stale", "oracle", "switch_rule", "switch_guard"]
FAMILIES = ["upxnrv", "upxlzma", "kite"]


def load(path):
    """rows[(arm)] -> list of (name, sublabel, posterior); also per-binary window sizes."""
    rows = defaultdict(list)
    win_n = defaultdict(int)          # (arm, name) -> candidates in window (== n_window, same all arms)
    seen_bin = set()
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for r in csv.DictReader(f):
            arm = r["arm"]
            name = r["name"]
            p = float(r["posterior"])
            rows[arm].append((name, r["sublabel"], p))
            win_n[(arm, name)] += 1
            seen_bin.add((r["sublabel"], name))
    return rows, win_n, seen_bin


def fam_of(sublabel):
    return sublabel


def main():
    if not DUMP.exists():
        sys.exit(f"missing {DUMP} — run run_selective.sh first")
    rows, win_n, seen_bin = load(DUMP)

    n_bins = len({name for (_, name) in seen_bin})
    fam_bins = defaultdict(set)
    for sub, name in seen_bin:
        fam_bins[sub].add(name)

    print(f"# Selective-disassembly precision demo — {n_bins} in-band packed binaries")
    print("  families:", {k: len(v) for k, v in sorted(fam_bins.items())})
    print()

    # ── Requested-vs-achieved precision + coverage + fabricated heads, pooled over all binaries ──
    print("## Pooled requested-vs-achieved precision (all in-band binaries)\n")
    hdr = f"{'arm':13} {'target':>6} {'claimed':>8} {'achieved':>9} {'coverage':>9} {'fab/bin':>8} {'fab total':>10} {'n_bin>0':>8}"
    print(hdr)
    print("-" * len(hdr))
    table = {}
    for arm in ARMS:
        data = rows.get(arm, [])
        # total window candidates for this arm, pooled and per binary
        per_bin_win = defaultdict(int)
        for name, _, _ in data:
            per_bin_win[name] += 1
        total_win = sum(per_bin_win.values())
        for t in TARGETS:
            sel = [(name, p) for (name, _, p) in data if p >= t]
            asserted = len(sel)
            claimed = (sum(p for _, p in sel) / asserted) if asserted else float("nan")
            achieved = 0.0 if asserted else float("nan")   # window is all-negative
            coverage = asserted / total_win if total_win else float("nan")
            per_bin_fab = defaultdict(int)
            for name, _ in sel:
                per_bin_fab[name] += 1
            fab_total = asserted
            fab_per_bin = fab_total / max(1, len(per_bin_win))
            n_bin_pos = sum(1 for v in per_bin_fab.values() if v > 0)
            claimed_s = f"{claimed:.3f}" if asserted else "  —"
            achieved_s = f"{achieved:.3f}" if asserted else "  — (0 assert)"
            print(f"{arm:13} {t:>6.2f} {claimed_s:>8} {achieved_s:>9} {coverage:>9.4f} "
                  f"{fab_per_bin:>8.2f} {fab_total:>10} {n_bin_pos:>8}")
            table[(arm, t)] = dict(claimed=claimed, achieved=achieved, coverage=coverage,
                                   fab_total=fab_total, fab_per_bin=fab_per_bin,
                                   n_bin_pos=n_bin_pos, n_bin=len(per_bin_win))
        print()

    # ── Per-family fabricated-head counts at each target ──
    print("## Fabricated code heads asserted inside the provable-data window (per family)\n")
    for t in TARGETS:
        print(f"### target precision {t:.2f}")
        print(f"{'family':9} {'n':>3} " + " ".join(f"{a:>13}" for a in ARMS))
        for fam in FAMILIES:
            names = fam_bins.get(fam, set())
            if not names:
                continue
            cells = []
            for arm in ARMS:
                fab = sum(1 for (name, sub, p) in rows.get(arm, [])
                          if sub == fam and p >= t)
                cells.append(f"{fab:>13}")
            print(f"{fam:9} {len(names):>3} " + " ".join(cells))
        print()

    # ── Per-binary fabricated-head detail at the strictest target (0.99) ──
    print("## Per-binary fabricated heads at target 0.99 (stale vs honest arms)\n")
    t = 0.99
    per = defaultdict(lambda: {a: 0 for a in ARMS})
    meta = {}
    for arm in ARMS:
        for (name, sub, p) in rows.get(arm, []):
            meta[name] = sub
            if p >= t:
                per[name][arm] += 1
    print(f"{'binary':32} {'fam':9} " + " ".join(f"{a:>13}" for a in ARMS))
    for name in sorted(per, key=lambda n: (meta[n], n)):
        c = per[name]
        print(f"{name:32} {meta[name]:9} " + " ".join(f"{c[a]:>13}" for a in ARMS))
    return table


if __name__ == "__main__":
    main()
