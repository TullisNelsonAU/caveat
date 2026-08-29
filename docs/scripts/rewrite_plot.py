#!/usr/bin/env python3
"""Coverage-vs-correctness curve for the confidence-gated rewriter (datatab, the code-in-data case).

Reads rewrite_curve.csv (produced by rewrite_eval.py) and plots, against τ: instrumentation coverage,
the number of detours that land in the in-.text data table (the corruption source), and whether the
rewritten binary still passes its reference I/O. The message: calibration lets you raise τ to an
operating point where the decoy sites vanish and the binary works — trading coverage for correctness at
a meaningful threshold, which the commit-everywhere baseline cannot do.
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [r for r in csv.DictReader(open(os.path.join(HERE, "rewrite_curve.csv"))) if r["binary"] == "datatab"]
ours = [r for r in rows if r["arm"].startswith("ours")]
taus = [float(r["arm"].split("=")[1]) for r in ours]
cov = [float(r["coverage"]) for r in ours]
decoy = [int(r["decoy_sites"]) for r in ours]
works = [int(r["works"]) for r in ours]
base = next(r for r in rows if r["arm"] == "baseline")

fig, ax1 = plt.subplots(figsize=(6.2, 3.8))
ax1.set_xlabel(r"confidence threshold $\tau$")
ax1.set_ylabel("coverage (fraction of leaders instrumented)", color="tab:blue")
ax1.plot(taus, cov, "o-", color="tab:blue", label="ours coverage")
ax1.axhline(float(base["coverage"]), ls=":", color="tab:blue", alpha=.6,
            label=f"baseline coverage ({base['coverage']})")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_ylim(0, 0.42)

ax2 = ax1.twinx()
ax2.set_ylabel("detours inside the in-.text data table", color="tab:red")
ax2.plot(taus, decoy, "s--", color="tab:red", label="ours decoy sites")
ax2.axhline(int(base["decoy_sites"]), ls=":", color="tab:red", alpha=.6,
            label=f"baseline decoy sites ({base['decoy_sites']})")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.set_ylim(0, 28)

# shade the τ region where the rewrite still WORKS
for t, w in zip(taus, works):
    if w:
        ax1.axvspan(t - 0.012, t + 0.012, color="tab:green", alpha=0.12)
first_ok = next((t for t, w in zip(taus, works) if w), None)
if first_ok is not None:
    ax1.annotate("works ✓ (0 decoy sites)", xy=(first_ok, 0.03), xytext=(first_ok - 0.42, 0.18),
                 arrowprops=dict(arrowstyle="->", color="tab:green"), color="tab:green")
ax1.set_title("datatab: calibration trades coverage for a working rewrite\n"
              "(baseline commits everywhere → corrupts the table → broken)")
l1, la1 = ax1.get_legend_handles_labels()
l2, la2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, la1 + la2, fontsize=7, loc="upper center")
fig.tight_layout()
out = os.path.join(HERE, "..", "..", "figures", "rewrite_coverage_correctness.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=140)
print("wrote", os.path.abspath(out))
