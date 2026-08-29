#!/usr/bin/env python3
"""Diff the columns `switching` and `downstream` have in common, row-for-row.

The two binaries share a bank, a classifier and a corpus split via `consistency::*`, but the shared
module is a lift out of `bin/switching.rs` rather than a refactor of it — `switching.rs` is left
untouched because its CSV is what the paper quotes. So the guarantee that the two agree has to be
numeric. This asserts it: same held-out set, same signature, same picks, same ECE.

Floats are compared to 1e-9, which is tight enough that any real logic change trips it and loose
enough to survive the CSV's 6-decimal round-trip. Picks and labels must match exactly.

Exit 0 = the implementations agree. Exit 1 = they have drifted; fix the code, not this script.
"""
import csv
import sys

FLOAT_COLS = [
    "base_rate",
    "ece_always_benign",
    "ece_oracle",
    "region_ent",
    "s_glob_benign_eng",
    "s_spat_benign_eng",
    "s_glob_packed_eng",
    "s_glob_obf_eng",
    "nis_benign_eng",
    "nis_packed_eng",
    "nis_obf_eng",
]
EXACT_COLS = [
    "regime",
    "sublabel",
    "n",
    "code_bytes",
    "mmae_pick",
    "mmae_nis_pick",
    "clf_pick",
    "rule_pick",
    "guard_pick",
]
TOL = 1e-9


def load(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {(r["name"], r["sublabel"]): r for r in rows}


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    a, b = load(a_path), load(b_path)

    problems = []
    if set(a) != set(b):
        only_a = sorted(set(a) - set(b))
        only_b = sorted(set(b) - set(a))
        problems.append(f"held-out sets differ: only in switching={only_a} only in downstream={only_b}")

    shared = sorted(set(a) & set(b))
    if not shared:
        problems.append("no overlapping rows at all — did both runs produce output?")

    for key in shared:
        ra, rb = a[key], b[key]
        for col in EXACT_COLS:
            if col not in ra or col not in rb:
                continue
            if ra[col] != rb[col]:
                problems.append(f"{key} {col}: switching={ra[col]!r} downstream={rb[col]!r}")
        for col in FLOAT_COLS:
            if col not in ra or col not in rb:
                continue
            va, vb = float(ra[col]), float(rb[col])
            if abs(va - vb) > TOL:
                problems.append(f"{key} {col}: switching={va!r} downstream={vb!r} (delta {va - vb:.3e})")

    n_cols = len(EXACT_COLS) + len(FLOAT_COLS)
    if problems:
        print(f"DRIFT — {len(problems)} mismatch(es) across {len(shared)} shared binaries:")
        for p in problems[:40]:
            print(f"  {p}")
        if len(problems) > 40:
            print(f"  … and {len(problems) - 40} more")
        return 1

    print(f"OK — {len(shared)} held-out binaries agree on all {n_cols} shared columns "
          f"(floats to {TOL:g}); the shared module matches bin/switching.rs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
