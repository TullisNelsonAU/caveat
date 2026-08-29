#!/usr/bin/env python3
"""K3_DECOY_HEAVY figure from k3_decoy_heavy.csv (no engine calls). Two panels:
   (left)  the discrimination axis: func AUROC K2 vs K3 per structure (paired bars);
   (right) the split: module belief F_c mean on real vs disconnected-decoy vs reached-decoy heads —
           the mechanism. Disconnected pins to ~0 (crushed); reached stays high (not prunable).
Emits PNG + PDF. This is the K3_DECOY_HEAVY_SPEC deliverable figure."""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "staircase")


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main():
    p = os.path.join(D, "k3_decoy_heavy.csv")
    if not os.path.exists(p):
        print("no k3_decoy_heavy.csv"); return
    rows = list(csv.DictReader(open(p)))
    structs = [r["struct"] for r in rows]
    x = np.arange(len(structs))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    # left: func AUROC K2 vs K3
    k2 = [f(r["func_auroc_k2"]) for r in rows]
    k3 = [f(r["func_auroc_k3"]) for r in rows]
    axL.bar(x - 0.18, k2, 0.36, label="K=2", color="tab:gray")
    axL.bar(x + 0.18, k3, 0.36, label="K=3 (module layer)", color="tab:green")
    axL.set_ylim(0.5, 1.0)
    axL.axhline(0.5, color="k", lw=0.7, ls=":")
    axL.set_xticks(x); axL.set_xticklabels(structs, rotation=20, ha="right")
    axL.set_ylabel("function AUROC (real vs candidate)")
    axL.set_title("Discrimination axis — K=2 vs K=3")
    axL.legend(fontsize=9); axL.grid(alpha=0.3, axis="y")

    # right: F_c mean on real / disconnected / reached
    real = [f(r["fc_mean_real"]) for r in rows]
    disc = [f(r["fc_mean_disc"]) for r in rows]
    reach = [f(r["fc_mean_reach"]) for r in rows]
    axR.bar(x - 0.26, real, 0.26, label="real heads", color="tab:blue")
    axR.bar(x + 0.00, disc, 0.26, label="disconnected decoys", color="tab:orange")
    axR.bar(x + 0.26, reach, 0.26, label="reached decoys (self-anchor/interleaved)", color="tab:red")
    axR.set_ylim(0, 1.0)
    axR.set_xticks(x); axR.set_xticklabels(structs, rotation=20, ha="right")
    axR.set_ylabel("mean module belief F_c")
    axR.set_title("The split — F_c pins disconnected to 0, not reached")
    axR.legend(fontsize=8); axR.grid(alpha=0.3, axis="y")

    fig.suptitle("L4 module layer on decoy-heavy — the win is co-extensive with disconnected structure")
    fig.tight_layout()
    fig.savefig(os.path.join(D, "k3_decoy_heavy.png"), dpi=140)
    fig.savefig(os.path.join(D, "k3_decoy_heavy.pdf"))
    plt.close()
    print("figure -> docs/staircase/k3_decoy_heavy.{png,pdf}")


if __name__ == "__main__":
    main()
