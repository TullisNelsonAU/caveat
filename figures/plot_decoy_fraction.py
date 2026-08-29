#!/usr/bin/env python3
"""Plot the Layer-1 code-in-code degradation curve.

Reads decoy_fraction_curve.csv (sweep --kind code over rising decoy sizes on O2 ls,
entropy strength 30) and renders the headline figure: as the planted real-code
("decoy") fraction grows, ECE climbs and AUROC sags, while confident-code rate over
real code stays flat. The entropy + DASSA levers are inert on this input (shown in the
sweep ablations), so the curve is the boundary of what Layer-1 priors can do.
"""
from pathlib import Path
import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
rows = list(csv.DictReader((HERE / "decoy_fraction_curve.csv").open()))

frac = [100.0 * float(r["decoy_frac"]) for r in rows]
ece = [float(r["ece"]) for r in rows]
auroc = [float(r["auroc"]) for r in rows]
code_conf = [float(r["code_conf"]) for r in rows]
data_conf = [float(r["data_conf"]) for r in rows]

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.linewidth": 0.8})

fig, ax_ece = plt.subplots(figsize=(7.0, 4.3))
ax_auroc = ax_ece.twinx()

# Honest axis (ECE) — left, red, the thing that degrades.
c_ece, c_auc = "#c0392b", "#2471a3"
(l_ece,) = ax_ece.plot(frac, ece, "o-", color=c_ece, lw=2.2, ms=6, label="ECE (calibration error — lower better)")
ax_ece.set_xlabel("Code-in-code decoy fraction of .text  (%)")
ax_ece.set_ylabel("ECE", color=c_ece)
ax_ece.tick_params(axis="y", labelcolor=c_ece)
ax_ece.set_ylim(0, 0.15)

# Accurate axis (AUROC) — right, blue.
(l_auc,) = ax_auroc.plot(frac, auroc, "s--", color=c_auc, lw=2.2, ms=6, label="AUROC (discrimination — higher better)")
ax_auroc.set_ylabel("AUROC", color=c_auc)
ax_auroc.tick_params(axis="y", labelcolor=c_auc)
ax_auroc.set_ylim(0.90, 1.0)

ax_ece.grid(True, alpha=0.25, lw=0.6)
ax_ece.set_title(
    "Layer-1 priors cap out on code-in-code\n"
    "entropy prior + DASSA inert here; only reachability (CFG / Layer 2) separates decoy from real code",
    fontsize=10.5,
)

# Real-code recall is untouched across the whole sweep — state it rather than plot it off-scale.
cc_lo, cc_hi = min(code_conf), max(code_conf)
ax_ece.annotate(
    f"real-code confident rate flat at {cc_lo:.3f}–{cc_hi:.3f}\n(recall preserved; damage is all on the decoy)",
    xy=(0.035, 0.04), xycoords="axes fraction", fontsize=8.6, color="#555555",
    ha="left", va="bottom",
    bbox=dict(boxstyle="round,pad=0.35", fc="#f4f4f4", ec="#cccccc", lw=0.6),
)

lines = [l_ece, l_auc]
ax_ece.legend(lines, [l.get_label() for l in lines], loc="center left", fontsize=8.8, framealpha=0.92)

fig.tight_layout()
for ext in ("png", "pdf"):
    out = HERE / f"decoy_fraction_curve.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
