#!/usr/bin/env python3
"""The real-world UPX figure: entropy prior backs off on a packed body, not the stub.

Data is the `honesty` curve on a UPX-packed coreutils `ls` (no ground truth, no section
headers — pure segment-fallback path), at entropy strength 0 vs 30. The packed image is
bimodal: a small low-entropy unpacker stub and a large high-entropy compressed body. The
prior collapses confident-code over the body (30.5% -> 1.3%) while leaving the stub flat,
and the 6-bit floor is visible as the bin where the two curves split.
"""
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent

# bin low-edge, candidate count, confident-code% at strength 0, at strength 30
DATA = [
    (1.0, 70, 5.7, 5.7), (1.5, 148, 30.4, 30.4), (2.0, 21, 19.0, 19.0),
    (2.5, 38, 26.3, 26.3), (3.0, 20, 40.0, 40.0), (3.5, 48, 27.1, 27.1),
    (4.0, 63, 14.3, 14.3), (4.5, 219, 29.2, 29.2), (5.0, 406, 26.4, 27.1),
    (5.5, 459, 32.7, 32.2), (6.0, 2363, 32.8, 11.6), (6.5, 14172, 31.7, 2.1),
    (7.0, 27540, 30.5, 1.3),
]
center = [lo + 0.25 for lo, _, _, _ in DATA]
count = [c for _, c, _, _ in DATA]
s0 = [a for _, _, a, _ in DATA]
s30 = [b for _, _, _, b in DATA]

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.linewidth": 0.8})
fig, ax = plt.subplots(figsize=(7.4, 4.4))

# Background: where the candidates actually live (log count), to show the bimodality.
ax_n = ax.twinx()
ax_n.bar(center, count, width=0.46, color="#dfe6ec", edgecolor="#c4ced6", lw=0.4, zorder=0)
ax_n.set_yscale("log")
ax_n.set_ylabel("candidates per bin  (log)", color="#9aa7b2")
ax_n.tick_params(axis="y", labelcolor="#9aa7b2")
ax_n.set_ylim(10, 1e5)

# Foreground: confident-code rate, prior off vs on.
ax.set_zorder(ax_n.get_zorder() + 1)
ax.patch.set_visible(False)
c0, c30 = "#c0392b", "#1f8a4c"
ax.plot(center, s0, "o-", color=c0, lw=2.3, ms=6, label="entropy prior OFF (strength 0)", zorder=5)
ax.plot(center, s30, "s-", color=c30, lw=2.3, ms=6, label="entropy prior ON (strength 30)", zorder=5)

# The 6-bit floor — where the gate starts firing.
ax.axvline(6.0, color="#555555", ls="--", lw=1.1, zorder=2)
ax.text(6.06, 41, "6-bit floor", color="#555555", fontsize=8.6, rotation=90, va="top")

# Call out the two regimes.
ax.annotate("compressed body\n30.5% -> 1.3%", xy=(7.25, 1.3), xytext=(6.45, 18),
            fontsize=8.8, color=c30, ha="left",
            arrowprops=dict(arrowstyle="->", color=c30, lw=1.1))
ax.text(1.15, 3.0, "curves identical below the floor (stub ~25%, untouched);\n"
                   "they split only where the prior gates on, at 6 bits",
        fontsize=8.4, color="#555555", ha="left", va="bottom")

ax.set_xlabel("local byte-entropy  (bits)")
ax.set_ylabel("confident-code rate  (P >= 0.9)  %", color="#333333")
ax.set_ylim(0, 48)
ax.set_xlim(1.0, 7.75)
ax.grid(True, axis="y", alpha=0.25, lw=0.6)
ax.set_title(
    "Entropy prior backs off on a real UPX packer body, not the stub\n"
    "UPX-packed coreutils ls — no ground truth, no section headers (segment fallback)",
    fontsize=10.5,
)
ax.legend(loc="upper center", fontsize=8.8, framealpha=0.92)

fig.tight_layout()
for ext in ("png", "pdf"):
    out = HERE / f"upx_packed_curve.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
