#!/usr/bin/env python3
"""Assemble the committed master CSV for the small-packed probe.

Inputs (all committed):
  sig_unpacked.tsv        signature pass, 9 clean glibc builds of the Tigress/CFG-arm programs
  sig_packed.tsv          signature pass, the same 9 programs packed (upxnrv/upxlzma/kite/kiten)
  sig_ladder_clean.tsv    signature pass, freestanding size ladder, unpacked
  sig_ladder_packed.tsv   signature pass, freestanding size ladder, packed
  sig_minu.tsv            signature pass, minimal (noseparate-code) freestanding ladder, packed
  switching_small_packed.csv       the frozen-bank switching run: ECE + routing for the main arm
  corpus/kite_gt_validation*.csv   whether each kite image has a provable-data window
  ../packer_breadth/breadth_main.csv   Table IV reference rows (large packed binaries)

Output: small_packed_master.csv — one row per binary, deterministic (inputs sorted by name).
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FLAT = 0.105178
MU, C, Z95 = 0.069231, 4.034322, 1.645


def t_floored(n):
    return max(FLAT, MU + Z95 * C / (n ** 0.5))


COLS = [
    "name", "arm", "program", "packer", "build", "packed",
    "n", "code_bytes", "region_ent", "s_glob", "s_spat",
    "t_n", "fire_flat", "fire_tn",
    "rule_pick", "guard_pick", "rule_pick_tn", "guard_pick_tn",
    "gt_kind", "ece_always_benign", "ece_oracle", "ece_rule", "ece_guard",
]

BUILD = {
    "glibc_clean": "glibc_O2_g_nopie",
    "glibc_packed": "glibc_O2_g_nopie",
    "ladder_clean": "freestanding_O2_g_nopie",
    "ladder_packed": "freestanding_O2_g_nopie",
    "minu_packed": "freestanding_O2_g_nopie_minlayout",
    "breadth_reference": "debian_coreutils",
}


def read_tsv(path):
    with open(os.path.join(HERE, path)) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def split_name(name):
    """`p07_crc.upxlzma` -> ('p07_crc', 'upxlzma'); `k04` -> ('k04', 'none')."""
    if "." in name:
        prog, packer = name.rsplit(".", 1)
        return prog, packer
    return name, "none"


def gt_kinds():
    kinds = {}
    for fn in ("corpus/kite_gt_validation.csv", "corpus/kite_gt_validation_ladder.csv",
               "corpus/kite_gt_validation_minu.csv"):
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                kinds[r["name"]] = r["gt_kind"]
    return kinds


def switching_rows():
    """name -> ECE columns, from the frozen-bank run. Keys are `<file>__<label>`."""
    p = os.path.join(HERE, "switching_small_packed.csv")
    out = {}
    if not os.path.exists(p):
        return out
    with open(p) as f:
        for r in csv.DictReader(f):
            base = r["name"].split("__")[0]
            out[base] = r
    return out


def main():
    kinds = gt_kinds()
    sw = switching_rows()
    rows = []

    for path, arm in [
        ("sig_unpacked.tsv", "glibc_clean"),
        ("sig_packed.tsv", "glibc_packed"),
        ("sig_ladder_clean.tsv", "ladder_clean"),
        ("sig_ladder_packed.tsv", "ladder_packed"),
        ("sig_minu.tsv", "minu_packed"),
    ]:
        for r in read_tsv(path):
            name = r["name"]
            prog, packer = split_name(name)
            n = int(r["n"])
            e = sw.get(name, {})
            # Cross-check: `switching` and `small_signature` are independent code paths that both
            # count benign-engine candidates. If they ever disagree, one of them is measuring a
            # different region and every joined row is suspect.
            if e and int(e["n"]) != n:
                raise SystemExit(f"n mismatch on {name}: signature={n} switching={e['n']}")
            # UPX ground truth is the b_info window; kite is entropy-validated (see corpus/).
            if packer.startswith("upx"):
                kind = "provable_data"
            elif packer == "kite":
                kind = kinds.get(name, "")
            elif packer == "kiten":
                kind = "routing_only"
            else:
                kind = ""
            rows.append({
                "name": name,
                "arm": arm,
                "program": prog,
                "packer": packer,
                "build": BUILD[arm],
                "packed": packer != "none",
                "n": n,
                "code_bytes": r["code_bytes"],
                "region_ent": f'{float(r["region_ent"]):.6f}',
                "s_glob": f'{float(r["s_glob"]):.6f}',
                "s_spat": f'{float(r["s_spat"]):.6f}',
                "t_n": f"{t_floored(n):.6f}",
                "fire_flat": r["fire_flat"],
                "fire_tn": r["fire_floored"],
                "rule_pick": r["rule_pick"],
                "guard_pick": r["guard_pick"],
                "rule_pick_tn": r["rule_pick_floored"],
                "guard_pick_tn": r["guard_pick_floored"],
                "gt_kind": kind,
                "ece_always_benign": e.get("ece_always_benign", ""),
                "ece_oracle": e.get("ece_oracle", ""),
                "ece_rule": e.get("ece_rule", ""),
                "ece_guard": e.get("ece_guard", ""),
            })

    # Table IV reference rows: the large packed binaries, same statistic, same frozen bank.
    bp = os.path.abspath(os.path.join(HERE, "..", "packer_breadth", "breadth_main.csv"))
    with open(bp) as f:
        for r in csv.DictReader(f):
            if r["regime"] != "packed" or r["sublabel"] not in ("upxnrv", "upxlzma", "kite", "kiten"):
                continue
            n = int(r["n"])
            ss = float(r["s_spat_benign_eng"])
            t = t_floored(n)
            rows.append({
                "name": r["name"],
                "arm": "breadth_reference",
                "program": r["name"].split(".")[0],
                "packer": r["sublabel"],
                "build": BUILD["breadth_reference"],
                "packed": True,
                "n": n,
                "code_bytes": r["code_bytes"],
                "region_ent": f'{float(r["region_ent"]):.6f}',
                "s_glob": f'{float(r["s_glob_benign_eng"]):.6f}',
                "s_spat": f"{ss:.6f}",
                "t_n": f"{t:.6f}",
                "fire_flat": str(ss > FLAT).lower(),
                "fire_tn": str(ss > t).lower(),
                "rule_pick": r["rule_pick"],
                "guard_pick": r["guard_pick"],
                # The published run reports no floored-gate pick; recompute is out of scope here.
                "rule_pick_tn": "",
                "guard_pick_tn": "",
                # `provable_data_mixed`: the published kite window is carved with a constant loader
                # prefix and, as corpus/audit_breadth_kite.py measures, still contains ~7 KB of
                # plaintext loader code. Its ECE is reported but is not a clean data oracle.
                "gt_kind": ("routing_only" if r["sublabel"] == "kiten"
                            else "provable_data_mixed" if r["sublabel"] == "kite"
                            else "provable_data"),
                "ece_always_benign": r["ece_always_benign"],
                "ece_oracle": r["ece_oracle"],
                "ece_rule": r["ece_rule"],
                "ece_guard": r["ece_guard"],
            })

    rows.sort(key=lambda r: (r["arm"], r["packer"], r["name"]))
    out = os.path.join(HERE, "small_packed_master.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
