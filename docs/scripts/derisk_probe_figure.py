#!/usr/bin/env python3
"""DERISK_PROBE figure from derisk_probe.csv (no engine calls). Two panels, fair GT (insn_max):
   (left)  calibration — raw ECE vs recalibrated-ceiling ECE per transform family (mean over programs);
   (right) discrimination — AUROC per transform (box of per-program spread) with the 0.85 GO line and
           the 0.5 chance line. The desync failure would sit at AUROC~0.5 / exploding ECE; nothing does.
Emits PNG + PDF."""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "derisk"))
ORDER = ["baseline", "Virtualize", "Flatten", "AddOpaque", "EncodeArithmetic", "EncodeLiterals"]


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main():
    p = os.path.join(D, "derisk_probe.csv")
    if not os.path.exists(p):
        print("no derisk_probe.csv"); return
    rows = [r for r in csv.DictReader(open(p)) if r["ok"] == "1" and r.get("gt_kind", "max") == "max"]
    by = {t: [r for r in rows if r["transform"] == t] for t in ORDER}
    labels = [t.replace("Encode", "Enc") for t in ORDER]
    x = np.arange(len(ORDER))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    raw = [np.mean([f(r["ece_raw"]) for r in by[t]]) if by[t] else 0 for t in ORDER]
    rec = [np.mean([f(r["ece_recal"]) for r in by[t]]) if by[t] else 0 for t in ORDER]
    axL.bar(x - 0.19, raw, 0.38, label="raw ECE", color="tab:orange")
    axL.bar(x + 0.19, rec, 0.38, label="recal-ceiling ECE", color="tab:green")
    axL.axhline(0.05, color="k", lw=0.8, ls="--", label="GO threshold 0.05")
    axL.set_xticks(x); axL.set_xticklabels(labels, rotation=20, ha="right")
    axL.set_ylabel("Expected Calibration Error")
    axL.set_title("Calibration survives — raw stays low, fully recalibratable")
    axL.legend(fontsize=8); axL.grid(alpha=0.3, axis="y")

    data = [[f(r["auroc_raw"]) for r in by[t]] for t in ORDER]
    axR.boxplot(data, positions=x, widths=0.5, showmeans=True)
    axR.axhline(0.85, color="tab:green", lw=0.9, ls="--", label="GO AUROC 0.85")
    axR.axhline(0.5, color="tab:red", lw=0.9, ls=":", label="chance / desync collapse")
    axR.set_ylim(0.45, 1.0)
    axR.set_xticks(x); axR.set_xticklabels(labels, rotation=20, ha="right")
    axR.set_ylabel("AUROC (real vs candidate)")
    axR.set_title("Discrimination holds — no collapse toward chance")
    axR.legend(fontsize=8, loc="lower right"); axR.grid(alpha=0.3, axis="y")

    fig.suptitle("De-risk probe: Soft posterior stays calibrated under real Tigress obfuscation (GO)")
    fig.tight_layout()
    fig.savefig(os.path.join(D, "derisk_probe.png"), dpi=140)
    fig.savefig(os.path.join(D, "derisk_probe.pdf"))
    plt.close()
    print("figure -> docs/derisk/derisk_probe.{png,pdf}")


if __name__ == "__main__":
    main()
