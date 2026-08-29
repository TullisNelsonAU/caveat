#!/usr/bin/env python3
"""Staircase figures from the CSVs (no engine calls). Emits PNG + PDF + the plotted CSV.
  Fig 1  the recoverability staircase: mean calibrated U_entropy vs evidence rung, one series per
         obfuscation/structure (the headline; SPEC sec 2a).
  Fig 2  decode-side decoy pruning: decoy-leak floor, cover (E1) vs confirm (E2), per structure --
         the anchored-vs-anchorless contrast the entropy staircase under-states.
"""
import csv, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "staircase")
RUNGS = ["R0", "R2", "R3", "R4", "R5"]
XLAB = ["E0\nraw pi", "E2\nconfirm", "E3\nresolve", "E4\ntrace", "E5\noracle"]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(name):
    p = os.path.join(D, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def fig_staircase(rows):
    ins = [r for r in rows if r["obj_class"] == "instruction-start" and r["status"] == "ok"]
    groups = {}
    for r in ins:
        groups.setdefault((r["obf"], r["struct"]), {}).setdefault(r["rung"], []).append(fnum(r["U_entropy"]))
    plt.figure(figsize=(8, 5))
    csv_out = [["obf", "struct"] + RUNGS]
    for g in sorted(groups):
        ys = []
        for rk in RUNGS:
            v = [x for x in groups[g].get(rk, []) if x is not None]
            ys.append(st.mean(v) if v else None)
        xs = [i for i, y in enumerate(ys) if y is not None]
        yv = [ys[i] for i in xs]
        plt.plot(xs, yv, marker="o", label="%s / %s" % g)
        csv_out.append([g[0], g[1]] + ["%.4f" % y if y is not None else "" for y in ys])
    plt.xticks(range(len(RUNGS)), XLAB)
    plt.ylabel("U_k  (mean binary entropy of calibrated posterior, bits)")
    plt.xlabel("evidence rung")
    plt.title("Recoverability staircase (instruction-start)")
    plt.legend(fontsize=7, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(D, "staircase.png"), dpi=140)
    plt.savefig(os.path.join(D, "staircase.pdf"))
    with open(os.path.join(D, "staircase_plot.csv"), "w", newline="") as f:
        csv.writer(f).writerows(csv_out)
    plt.close()


def fig_decode(rows):
    if not rows:
        return
    structs = sorted({r["struct"] for r in rows})
    cover = {r["struct"]: fnum(r["leak_floor"]) for r in rows if r["mode"] == "E1_cover"}
    conf = {r["struct"]: fnum(r["leak_floor"]) for r in rows if r["mode"] == "E2_confirm"}
    x = range(len(structs))
    plt.figure(figsize=(8, 4.5))
    plt.bar([i - 0.2 for i in x], [cover.get(s) or 0 for s in structs], 0.4, label="E1 cover")
    plt.bar([i + 0.2 for i in x], [conf.get(s) or 0 for s in structs], 0.4, label="E2 confirm")
    plt.xticks(list(x), structs, rotation=30, ha="right", fontsize=8)
    plt.ylabel("decoy-leak floor (junk starts selected, recall>=0.30)")
    plt.title("Decoy pruning by reachability confirmation")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(D, "decoy_leak.png"), dpi=140)
    plt.savefig(os.path.join(D, "decoy_leak.pdf"))
    plt.close()


if __name__ == "__main__":
    fig_staircase(load("staircase_raw.csv"))
    fig_decode(load("decode_leak.csv"))
    print("figures -> docs/staircase/{staircase,decoy_leak}.{png,pdf}")
