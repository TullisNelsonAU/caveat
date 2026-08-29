#!/usr/bin/env python3
"""Ablations for the consistency-switching detector + GT-free routing rule.

Everything here is computed OFFLINE from the committed per-binary signature table
docs/consistency_credibility/credibility.csv (142 binaries: 20 clean_fit / 25 clean_holdout /
80 desync / 17 packed), with per-binary benign-engine S_glob (s_glob_surprise), S_spat
(s_spat_moran), and region_entropy. No inference is re-run — the ablation only re-thresholds and
re-routes signatures that were already produced by the engine of record.

Faithful reproduction of the shipped rule (switching.rs::SignatureClassifier::train / classify_rule):
    glob_hi = percentile(clean_fit S_glob, 95) * 2.5      # 2.5x margin over the clean p95 tail
    spat_hi = percentile(clean_fit S_spat, 95)
    route:   obf   if S_glob > glob_hi                    # desync peeled off first (huge surprise)
             packed elif S_spat > spat_hi                 # packed carried by the spatial statistic
             benign else

True regime for scoring: clean_holdout -> benign (held-out FP gate), desync -> obfuscated,
packed -> packed. clean_fit is used only to set thresholds, never scored.

Three ablations:
  (a) which axis: S_glob-only vs S_spat-only vs both, on DETECTION (benign vs not) and 3-way ROUTING.
  (b) sensitivity to the clean-null percentile (90 / 95 / 99).
  (c) sensitivity of routing accuracy to the (glob_hi, spat_hi) thresholds, local grid incl. the
      paper's 2.57 / 0.108 operating point.
"""
import csv
from pathlib import Path

HERE = Path(__file__).parent
CSV = HERE.parent / "consistency_credibility" / "credibility.csv"

# true regime by role
TRUE = {"clean_holdout": "benign", "desync": "obfuscated", "packed": "packed"}


def load():
    rows = list(csv.DictReader(open(CSV)))
    fit = [(float(r["s_glob_surprise"]), float(r["s_spat_moran"]))
           for r in rows if r["role"] == "clean_fit"]
    test = [(TRUE[r["role"]], float(r["s_glob_surprise"]), float(r["s_spat_moran"]))
            for r in rows if r["role"] in TRUE]
    return fit, test


def percentile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = q * (len(xs) - 1)
    lo = int(i); hi = min(lo + 1, len(xs) - 1); f = i - lo
    return xs[lo] * (1 - f) + xs[hi] * f


def thresholds(fit, q=0.95, margin=2.5):
    g = percentile([f[0] for f in fit], q) * margin
    s = percentile([f[1] for f in fit], q)
    return g, s


def route(sg, ss, glob_hi, spat_hi, use_glob=True, use_spat=True):
    """Ordered rule; an axis switched off is treated as never firing."""
    if use_glob and sg > glob_hi:
        return "obfuscated"
    if use_spat and ss > spat_hi:
        return "packed"
    return "benign"


def score(test, glob_hi, spat_hi, use_glob=True, use_spat=True):
    regimes = ["benign", "obfuscated", "packed"]
    n = {r: 0 for r in regimes}
    route_ok = {r: 0 for r in regimes}
    det_ok = {r: 0 for r in regimes}     # detection: benign wants "no alarm", others want "alarm"
    fp = 0
    for true, sg, ss in test:
        n[true] += 1
        pick = route(sg, ss, glob_hi, spat_hi, use_glob, use_spat)
        if pick == true:
            route_ok[true] += 1
        alarm = pick != "benign"
        if true == "benign":
            if not alarm:
                det_ok[true] += 1
            else:
                fp += 1
        else:
            if alarm:
                det_ok[true] += 1
    n_test = sum(n.values())
    route_acc = sum(route_ok.values()) / n_test
    n_pos = n["obfuscated"] + n["packed"]
    tpr = (det_ok["obfuscated"] + det_ok["packed"]) / n_pos
    fpr = fp / n["benign"]
    return dict(n=n, route_ok=route_ok, det_ok=det_ok, route_acc=route_acc,
                tpr=tpr, fpr=fpr, fp=fp, n_test=n_test)


def main():
    fit, test = load()
    glob_hi, spat_hi = thresholds(fit, 0.95)
    print(f"# Ablations — {len(test)} held-out binaries, thresholds from {len(fit)} clean_fit\n")
    print(f"operating point: glob_hi = p95(clean S_glob)x2.5 = {glob_hi:.4f}   "
          f"spat_hi = p95(clean S_spat) = {spat_hi:.4f}\n")
    counts = {}
    for _, _, _ in test:
        pass
    from collections import Counter
    c = Counter(t for t, _, _ in test)
    print("test set:", dict(c), "\n")

    # ── (a) which axis ──
    print("## (a) S_glob-only vs S_spat-only vs both\n")
    hdr = (f"{'axes':12} {'route acc':>9} {'benign':>10} {'obf(desync)':>12} {'packed':>9} "
           f"{'det TPR':>8} {'det FPR':>8}")
    print(hdr); print("-" * len(hdr))
    for label, ug, us in [("S_glob only", True, False), ("S_spat only", False, True), ("both", True, True)]:
        r = score(test, glob_hi, spat_hi, ug, us)
        b = f"{r['route_ok']['benign']}/{r['n']['benign']}"
        o = f"{r['route_ok']['obfuscated']}/{r['n']['obfuscated']}"
        p = f"{r['route_ok']['packed']}/{r['n']['packed']}"
        print(f"{label:12} {r['route_acc']:>9.3f} {b:>10} {o:>12} {p:>9} {r['tpr']:>8.3f} {r['fpr']:>8.3f}")
    print()
    print("  route acc = 3-way (benign/obfuscated/packed) correct / total.")
    print("  det TPR = alarm raised on obfuscated+packed; det FPR = false alarm on held-out benign.\n")

    # ── (b) clean-null percentile sweep ──
    print("## (b) clean-null percentile sweep (glob_hi = pXX x2.5, spat_hi = pXX)\n")
    hdr = f"{'pctile':6} {'glob_hi':>8} {'spat_hi':>8} {'route acc':>9} {'det TPR':>8} {'det FPR':>8}"
    print(hdr); print("-" * len(hdr))
    for q in (0.90, 0.95, 0.99):
        gh, sh = thresholds(fit, q)
        r = score(test, gh, sh)
        print(f"p{int(q*100):<5} {gh:>8.4f} {sh:>8.4f} {r['route_acc']:>9.3f} {r['tpr']:>8.3f} {r['fpr']:>8.3f}")
    print()

    # ── (c) threshold sensitivity grid ──
    print("## (c) routing-accuracy sensitivity to (glob_hi, spat_hi) — local grid\n")
    glob_grid = [1.5, 2.0, 2.53, 2.57, 3.0, 5.0, 10.0]
    spat_grid = [0.09, 0.1053, 0.108, 0.12, 0.15]
    print("rows = glob_hi, cols = spat_hi; cell = 3-way routing accuracy")
    corner = "glob\\spat"
    print(f"{corner:>10} " + " ".join(f"{s:>7.4f}" for s in spat_grid))
    for gh in glob_grid:
        cells = []
        for sh in spat_grid:
            r = score(test, gh, sh)
            cells.append(f"{r['route_acc']:>7.3f}")
        star = "  <- op" if abs(gh - 2.53) < 0.06 else ""
        print(f"{gh:>10.2f} " + " ".join(cells) + star)
    print()
    print("  paper operating point ~ (2.53-2.57, 0.105-0.108). Neighbouring cells show the plateau.")


if __name__ == "__main__":
    main()
