#!/usr/bin/env python3
"""Labeled UPX over-commitment figure: FP rate over provably-compressed bytes.

Reads upx_labeled.csv (produced by `upxeval` — region,strength,n,fp_rate,mean_p,brier,max_p) and
draws the honesty result as a *measurement* against perfect ground truth: the false-positive rate
over bytes UPX's own format proves are compressed payload, with the entropy prior off vs on. No
entropy binning, no disassembler labels. Bars grouped by region (exact compressed extent vs the
conservative interior block), each showing prior OFF and ON.
"""
from pathlib import Path
import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
rows = list(csv.DictReader((HERE / "upx_labeled.csv").open()))

regions = []
for r in rows:
    if r["region"] not in regions:
        regions.append(r["region"])
strengths = sorted({float(r["strength"]) for r in rows})


def cell(region, s, col):
    for r in rows:
        if r["region"] == region and float(r["strength"]) == s:
            return float(r[col])
    return float("nan")


label_for = {0.0: "entropy prior OFF", 30.0: "entropy prior ON (strength 30)"}
colors = {min(strengths): "#c0392b", max(strengths): "#1f8a4c"}

fig, ax = plt.subplots(figsize=(7.0, 4.3))
x = np.arange(len(regions))
w = 0.36
for k, s in enumerate(strengths):
    vals = [100.0 * cell(reg, s, "fp_rate") for reg in regions]
    bars = ax.bar(x + (k - (len(strengths) - 1) / 2) * w, vals, w,
                  color=colors.get(s, "#888888"),
                  label=label_for.get(s, f"strength {s:g}"))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.6)

n_by_region = {reg: int(cell(reg, strengths[0], "n")) for reg in regions}
xticklabels = [
    f"{reg}\n(n={n_by_region[reg]:,} bytes,\nall provably data)" for reg in regions
]
ax.set_xticks(x)
ax.set_xticklabels(xticklabels, fontsize=9)
ax.set_ylabel("false-positive rate  (P >= 0.9 over known-data)  %")
ax.set_ylim(0, max(2.0, 1.30 * max(100.0 * cell(reg, strengths[0], "fp_rate") for reg in regions)))
ax.grid(True, axis="y", alpha=0.25, lw=0.6)
ax.set_title(
    "Soft over-commitment on a real UPX packer body, measured against perfect GT\n"
    "negatives = UPX compressed blocks (from the b_info chain) — no entropy, no disassembly",
    fontsize=10.3,
)
ax.legend(loc="upper center", fontsize=9, framealpha=0.92)

fig.tight_layout()
for ext in ("png", "pdf"):
    out = HERE / f"upx_labeled.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
