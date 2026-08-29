#!/usr/bin/env python3
"""FU2 effort-curve figure from eig_faithful.csv (no engine calls): recovered true mass at P̂≥0.9 vs
#queries, EIG (F_h·ΔH) vs certified (max conditional entropy), one panel per arm. Emits PNG + PDF.
The mass is plotted as Δ from step 0 so specimens of different baselines overlay on one axis."""
import csv, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "staircase")


def load():
    p = os.path.join(D, "eig_faithful.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def main():
    rows = load()
    if not rows:
        print("no eig_faithful.csv"); return
    arms = sorted({r["arm"] for r in rows})
    fig, axes = plt.subplots(1, len(arms), figsize=(6.2 * len(arms), 5), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        # per (specimen, rule) -> {step: tp9}; plot Δmass vs step, mean over specimens (bold) + faint each
        cur = {}
        for r in rows:
            if r["arm"] != arm:
                continue
            cur.setdefault((r["specimen"], r["rule"]), {})[int(r["step"])] = int(r["tp9"])
        for rule, color in (("eig", "tab:blue"), ("certent", "tab:orange")):
            series = []
            for (spec, ru), d in cur.items():
                if ru != rule or not d:
                    continue
                steps = sorted(d)
                base = d[steps[0]]
                xs = [0] + steps
                ys = [0] + [d[s] - base for s in steps]  # Δ from the first recorded step's baseline
                ax.plot(xs, ys, color=color, alpha=0.25, lw=1)
                series.append({s: d[s] - base for s in steps})
            # mean curve over specimens at each step
            allsteps = sorted({s for sr in series for s in sr})
            mean_y = [st.mean([sr[s] for sr in series if s in sr]) for s in allsteps] if series else []
            if mean_y:
                ax.plot([0] + allsteps, [0] + mean_y, color=color, lw=2.6,
                        marker="o", label="%s (mean)" % ("F_h·ΔH (EIG)" if rule == "eig" else "certified h(F)"))
        ax.set_xlabel("# queries"); ax.set_ylabel("Δ recovered true mass (instr @ P̂≥0.9)")
        ax.set_title("Effort curve — %s arm" % arm); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle("Faithful EIG vs certified greedy — recovered mass vs queries (same q, internal re-ranking)")
    fig.tight_layout()
    fig.savefig(os.path.join(D, "eig_faithful.png"), dpi=140)
    fig.savefig(os.path.join(D, "eig_faithful.pdf"))
    plt.close()
    print("figure -> docs/staircase/eig_faithful.{png,pdf}")


if __name__ == "__main__":
    main()
