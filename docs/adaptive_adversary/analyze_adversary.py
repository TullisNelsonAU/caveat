#!/usr/bin/env python3
"""Turn adaptive_adversary.csv into the tables the results doc quotes, and check them against the
values Table V of the NDSS draft prints.

Three things happen here and nothing else:

  1. Per-construction tables, one per substrate/donor pair, straight out of the CSV.
  2. The Table V reconstruction — the paper's six rows, each aggregated over the scope the paper
     actually used (see SCOPE_NOTE below), formatted at the paper's own precision.
  3. A verification table: the reconstructed string against the printed string, per cell. A cell
     that does not reproduce is marked FAIL and the run is non-zero exit. The printed values are
     transcribed constants here; they are never recomputed from the CSV, so this is a real check.

Nothing is fit, tuned, or thresholded on this data. The detection null (S_glob 1.01, S_spat 0.105)
is the published constant, baked into the driver.
"""
import csv
import os
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "adaptive_adversary.csv")

DET_GLOB_HI, DET_SPAT_HI = 1.01, 0.105

PRIMARY = "gcc_coreutils_64_O1_sum"  # base of the primary pair; the other pair is the swap

SCOPE_NOTE = """\
`scope` records which substrate/donor pairs a printed cell was aggregated over: `primary` is
base=`sum`/donor=`printf`, `both` pools the primary and the swap. The paper's Table V does not use
one scope throughout — the packing row pools both pairs, the interleave and desync rows are the
primary pair alone, except the desync `S_spat` cell, which pools. Every printed value reproduces
under the scope recorded here; the scope column is what makes that auditable.

The `fires` column above lists *every* prong above the detection null, which is not what Table V's
`fires` column prints: that one names the prong of record, the one the surrounding argument turns
on. Both interleave and desync also clear the other prong. No number differs; the column means
something narrower than it reads."""


def short(name):
    return name.replace("gcc_coreutils_64_O1_", "")


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("s_glob", "s_spat", "ece_cal", "ece_raw", "true_recov",
                  "region_mean_pi_cal", "region_s_glob", "region_s_spat"):
            r[k] = float(r[k])
        for k in ("n_cand", "gt_positives", "adv_bytes", "adv_positives", "region_n",
                  "region_fp_conf", "fire_glob", "fire_spat"):
            r[k] = int(r[k])
    return rows


def pairs(rows):
    """OrderedDict pair-id -> rows, primary first."""
    out = OrderedDict()
    for r in rows:
        out.setdefault((r["base"], r["donor"]), []).append(r)
    return OrderedDict(sorted(out.items(), key=lambda kv: kv[0][0] != PRIMARY))


def select(rows, family, scope):
    sel = [r for r in rows if r["family"] == family]
    if scope == "primary":
        sel = [r for r in sel if r["base"] == PRIMARY]
    return sel


# ── the printed Table V, transcribed. `kind` says how the cell was formed from the selected rows. ──
# kind: point (single row), range (min--max), max (<=max), range_desc (max--min, the way the paper
# prints the desync recovery as a fall).
TABLE_V = [
    ("clean base (null)", "clean", [
        ("s_glob",     "point",      "%.2f", "primary", "0.65"),
        ("s_spat",     "point",      "%.2f", "primary", "0.06"),
        ("ece_cal",    "point",      "%.3f", "primary", "0.002"),
        ("true_recov", "point",      "%.2f", "primary", "0.98"),
    ]),
    ("self-consistent decoy", "decoy", [
        ("s_glob",     "point",      "%.1f", "primary", "1.4"),
        ("s_spat",     "point",      "%.2f", "primary", "0.08"),
        ("ece_cal",    "point",      "%.2f", "primary", "0.16"),
        ("true_recov", "point",      "%.2f", "primary", "0.98"),
    ]),
    ("decoy, relabeled", "decoy_relabeled", [
        ("s_glob",     "point",      "%.1f", "primary", "1.4"),
        ("s_spat",     "point",      "%.2f", "primary", "0.08"),
        ("ece_cal",    "point",      "%.3f", "primary", "0.003"),
        ("true_recov", "point",      "%.2f", "primary", "0.98"),
    ]),
    ("packing (rand/DEFLATE/NOP)", "packing", [
        ("s_glob",     "max",        "%.1f", "both",    "<=1.8"),
        ("s_spat",     "range",      "%.2f", "both",    "0.14--0.33"),
        ("ece_cal",    "range",      "%.2f", "both",    "0.14--0.52"),
        ("true_recov", "point",      "%.2f", "both",    "0.98"),
    ]),
    ("interleave (k=512->8)", "interleave", [
        ("s_glob",     "range",      "%.1f", "primary", "1.1--2.5"),
        ("s_spat",     "range",      "%.2f", "primary", "0.14--0.17"),
        ("ece_cal",    "range",      "%.2f", "primary", "0.19--0.20"),
        ("true_recov", "point",      "%.2f", "primary", "0.98"),
    ]),
    ("desync (rho=.02->.40)", "desync", [
        ("s_glob",     "range",      "%.1f", "primary", "6.1--15.5"),
        ("s_spat",     "range",      "%.2f", "both",    "0.16--0.31"),
        ("ece_cal",    "range",      "%.2f", "primary", "0.04--0.20"),
        ("true_recov", "range_desc", "%.2f", "primary", "0.90--0.59"),
    ]),
]


def cell(sel, col, kind, fmt):
    vals = [r[col] for r in sel]
    lo, hi = min(vals), max(vals)
    if kind == "point":
        # A "point" cell over several rows is only honest if they agree at the printed precision.
        strs = {fmt % v for v in vals}
        return sorted(strs)[0] if len(strs) == 1 else "%s--%s" % (fmt % lo, fmt % hi)
    if kind == "max":
        return "<=" + fmt % hi
    if kind == "range":
        return (fmt % lo) if (fmt % lo) == (fmt % hi) else "%s--%s" % (fmt % lo, fmt % hi)
    if kind == "range_desc":
        return "%s--%s" % (fmt % hi, fmt % lo)
    raise ValueError(kind)


def fires_of(sel):
    g = any(r["fire_glob"] for r in sel)
    s = any(r["fire_spat"] for r in sel)
    return {(True, True): "S_glob+S_spat", (True, False): "S_glob",
            (False, True): "S_spat", (False, False): "none"}[(g, s)]


def per_construction(rows):
    out = []
    for (base, donor), rs in pairs(rows).items():
        tag = "primary" if base == PRIMARY else "swap"
        out.append("### %s pair — base=`%s`, donor=`%s`\n" % (tag, short(base), short(donor)))
        out.append("| construction | family | n | S_glob | S_spat | ECE (true) | recov. (true) | "
                   "GT+ | adv bytes | adv GT+ | fires |")
        out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in rs:
            out.append("| `%s` | %s | %d | %.3f | %.3f | %.4f | %.3f | %d | %d | %d | %s |" % (
                r["construction"], r["family"], r["n_cand"], r["s_glob"], r["s_spat"],
                r["ece_cal"], r["true_recov"], r["gt_positives"], r["adv_bytes"],
                r["adv_positives"], r["fires"]))
        out.append("")
    return out


def table_v(rows):
    out = ["| construction | S_glob | S_spat | ECE | recov. | fires (all prongs) | scope |",
           "|---|---:|---:|---:|---:|---|---|"]
    for label, family, cols in TABLE_V:
        scopes = {c[3] for c in cols}
        vals = []
        for col, kind, fmt, scope, _printed in cols:
            sel = select(rows, family, scope)
            vals.append(cell(sel, col, kind, fmt))
        fires = fires_of(select(rows, family, "primary"))
        out.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            label, vals[0], vals[1], vals[2], vals[3], fires,
            "mixed" if len(scopes) > 1 else scopes.pop()))
    out.append("")
    out.append(SCOPE_NOTE)
    return out


def verify(rows):
    out = ["| construction | cell | scope | printed in Table V | from CSV | |",
           "|---|---|---|---|---|---|"]
    ok = True
    for label, family, cols in TABLE_V:
        for col, kind, fmt, scope, printed in cols:
            got = cell(select(rows, family, scope), col, kind, fmt)
            good = got == printed
            ok &= good
            out.append("| %s | %s | %s | %s | %s | %s |" % (
                label, col, scope, printed, got, "ok" if good else "**FAIL**"))
    out.append("")
    out.append("**%s** — %d of %d printed cells reproduce from the committed CSV." % (
        "All cells reproduce" if ok else "MISMATCH", sum(
            1 for l in out if l.endswith("| ok |")), sum(len(c) for _, _, c in TABLE_V)))
    return out, ok


def uniform_primary(rows):
    """What Table V would print if every row used the primary pair alone. Recorded because the
    paper's scope is not uniform; this is the comparison that shows exactly which cells move."""
    out = ["| construction | cell | as printed | primary-only | moves |", "|---|---|---|---|---|"]
    for label, family, cols in TABLE_V:
        for col, kind, fmt, scope, printed in cols:
            got = cell(select(rows, family, "primary"), col, kind, fmt)
            out.append("| %s | %s | %s | %s | %s |" % (
                label, col, printed, got, "" if got == printed else "**yes**"))
    return out


def candidate_counts(rows):
    """n per constructed binary. Prompt 1 re-thresholds this table against a size-aware spatial
    null T(n), so n is carried per row rather than summarized."""
    out = ["| construction | " + " | ".join(
        ("n (%s)" % ("primary" if b == PRIMARY else "swap")) for b, _ in pairs(rows)) + " |"]
    out.append("|---" * (1 + len(pairs(rows))) + "|")
    order = [r["construction"] for r in list(pairs(rows).values())[0]]
    by = {(r["construction"], r["base"]): r["n_cand"] for r in rows}
    for c in order:
        cells = [str(by.get((c, b), "")) for b, _ in pairs(rows)]
        out.append("| `%s` | %s |" % (c, " | ".join(cells)))
    return out


def main():
    rows = load(CSV_PATH)
    ver, ok = verify(rows)
    blocks = [
        ("## Per-construction results", per_construction(rows)),
        ("## Table V as printed, rebuilt from the CSV", table_v(rows)),
        ("## Verification against the printed Table V", ver),
        ("## The same table on the primary pair alone", uniform_primary(rows)),
        ("## Candidate counts", candidate_counts(rows)),
    ]
    for head, body in blocks:
        print(head)
        print()
        print("\n".join(body))
        print()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
