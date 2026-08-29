#!/usr/bin/env python3
"""Semantic-obfuscation arm: per-transform S_spat, size-standardized z, and the
paired Wilcoxon contrasts against the no-dispatcher control.

Standardization follows the size-aware spatial null:

    z = (S_spat - mu) * sqrt(n) / c        mu = 0.069231, c = 4.034322

both estimated on the 20 clean-fit binaries (see ../spatial_null_repair/).

The design is paired: the same nine programs appear under all three transforms,
so each dispatcher transform is contrasted against EncodeArithmetic (which
installs no dispatch structure) with a two-sided Wilcoxon signed-rank test over
the nine matched pairs.

PROVENANCE. The paper's semantic-obfuscation table is built from the boundary
corpus, `../downstream_decision/boundaries_meta.csv`, which this script reads.
The sublabels map to transforms by construction: tigL = Virtualize,
tigM = EncodeArithmetic (the no-dispatcher control), tigH = Flatten.

A later re-run, `tigress_rerun.tsv`, also sits in this directory. It is a
different scoring pass over a rebuilt corpus and gives slightly different values
(Virtualize z 0.98, EncodeArithmetic z 0.29, 22/27 routes rather than 23/27).
Pass --rerun to read it instead. The paper reports the boundary-corpus run.
"""
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

MU = 0.069231
C = 4.034322
CONTROL = "EncodeArithmetic"
HERE = Path(__file__).parent
SRC = HERE.parent / "downstream_decision" / "boundaries_meta.csv"
ALT = HERE / "tigress_rerun.tsv"   # the later re-run; see PROVENANCE below


def zscore(s_spat, n):
    return (s_spat - MU) * math.sqrt(n) / C


NAMES = {"tigL": "Virtualize", "tigM": "EncodeArithmetic", "tigH": "Flatten"}


def load(use_rerun=False):
    z, s, ece = {}, {}, {}
    if use_rerun:
        with open(ALT) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                key = (r["transform_name"], r["name"])
                s[key] = float(r["s_spat"])
                z[key] = zscore(s[key], float(r["n"]))
        return z, s, ece
    with open(SRC) as f:
        for r in csv.DictReader(f):
            if not r["sublabel"].startswith("tig"):
                continue
            key = (NAMES[r["sublabel"]], r["name"])
            s[key] = float(r["s_spat_benign_eng"])
            z[key] = zscore(s[key], float(r["n"]))
            ece[key] = (float(r["ece_always_benign"]), float(r["ece_oracle"]),
                        int(r["n"]), r["rule_pick"])
    return z, s, ece


def wilcoxon_signed_rank(a, b):
    """Two-sided exact Wilcoxon signed-rank. n is small, so enumerate all sign
    flips rather than depend on scipy."""
    d = [x - y for x, y in zip(a, b) if x != y]
    n = len(d)
    if n == 0:
        return None, None
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, dv in zip(ranks, d) if dv > 0)
    total = sum(ranks)
    stat = min(w_plus, total - w_plus)
    count = 0
    for mask in range(1 << n):
        wp = sum(ranks[i] for i in range(n) if mask >> i & 1)
        if min(wp, total - wp) <= stat:
            count += 1
    return stat, min(1.0, count / (1 << n))


def main():
    use_rerun = "--rerun" in sys.argv
    z, s, ece = load(use_rerun)
    print("source:", (ALT if use_rerun else SRC).name)
    transforms = sorted({t for t, _ in z})
    progs = sorted({p for _, p in z})

    print(f"Semantic-obfuscation arm  --  {len(progs)} programs x "
          f"{len(transforms)} transforms = {len(z)} binaries")
    print(f"size-aware null: mu={MU}, c={C}\n")
    hdr = f"{'transform':<18}{'n_bin':>6}{'benign ECE':>12}{'oracle ECE':>12}{'S_spat':>9}{'z':>7}{'->packed':>10}{'cands':>14}"
    print(hdr if ece else f"{'transform':<18}{'n_bin':>6}{'S_spat':>9}{'z':>7}")
    for t in transforms:
        zs = [z[(t, p)] for p in progs]
        ss = [s[(t, p)] for p in progs]
        if not ece:
            print(f"{t:<18}{len(zs):>6}{statistics.mean(ss):>9.3f}{statistics.mean(zs):>7.2f}")
            continue
        e = [ece[(t, p)] for p in progs]
        ns = [x[2] for x in e]
        print(f"{t:<18}{len(zs):>6}{statistics.mean(x[0] for x in e):>12.4f}"
              f"{statistics.mean(x[1] for x in e):>12.4f}{statistics.mean(ss):>9.3f}"
              f"{statistics.mean(zs):>7.2f}"
              f"{sum(1 for x in e if x[3] == 'packed'):>7}/{len(e)}"
              f"{min(ns):>8}-{max(ns):<5}")
    if ece:
        alln = [ece[(t, p)][2] for t in transforms for p in progs]
        print(f"\ncandidate range over the arm: {min(alln)}-{max(alln)}   "
              f"routes to packed: {sum(1 for t in transforms for p in progs if ece[(t,p)][3]=='packed')}/{len(alln)}")

    print(f"\npaired Wilcoxon signed-rank vs {CONTROL}, two-sided, "
          f"{len(progs)} matched pairs")
    print(f"{'contrast':<34}{'space':>7}{'W':>7}{'p':>9}")
    for t in transforms:
        if t == CONTROL:
            continue
        for label, table in (("z", z), ("raw", s)):
            a = [table[(t, p)] for p in progs]
            b = [table[(CONTROL, p)] for p in progs]
            w, p = wilcoxon_signed_rank(a, b)
            print(f"{t + ' vs ' + CONTROL:<34}{label:>7}{w:>7.1f}{p:>9.4f}")


if __name__ == "__main__":
    main()
