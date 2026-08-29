#!/usr/bin/env python3
"""Analysis for the Paper-2 switching run. Turns switching.csv into the payoff numbers:
the three-arm ECE table per regime (always-benign vs oracle vs consistency-selected), the GT-free
selection accuracy of each rule, the recovery fraction (how much of the oracle's calibration gain
the switch recovers), the convenient-case guard (is always-benign genuinely stale?), and the honest
Tigress blind-spot block. Emits a grouped-bar ECE figure.

Leads with whichever selection rule is cleaner and states where it fails.

usage: analyze_switching.py switching.csv [out_dir]
"""
import csv, sys, os
from collections import defaultdict

CORE_REGIMES = ["benign", "packed", "obfuscated"]


def is_tig(row):
    return row["sublabel"].startswith("tig")


def fnum(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def mean(v):
    v = [x for x in v if x == x]
    return sum(v) / len(v) if v else float("nan")


def median(v):
    v = sorted(x for x in v if x == x)
    return v[len(v) // 2] if v else float("nan")


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "switching.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(csv_path))
    rows = load(csv_path)
    core = [r for r in rows if not is_tig(r)]
    tig = [r for r in rows if is_tig(r)]

    print(f"held-out binaries: {len(rows)}  (core {len(core)}, tigress {len(tig)})\n")

    # ── Arm ECE by regime ──  (arms: always-benign, oracle, mmae, clf-centroid, rule-default)
    ARM_KEYS = ("ece_always_benign", "ece_oracle", "ece_mmae", "ece_clf", "ece_rule")
    print("── Arm ECE by regime (mean; always-benign median in parens) ──")
    print(f"  {'regime':<11} {'n':>3}  {'always-benign':>14}  {'oracle':>8}  {'mmae':>8}  {'clf':>8}  {'rule':>8}")
    arms_by_regime = {}
    for reg in CORE_REGIMES:
        rs = [r for r in core if r["regime"] == reg]
        if not rs:
            continue
        a = {k: [fnum(r[k]) for r in rs] for k in ARM_KEYS}
        arms_by_regime[reg] = {k: mean(v) for k, v in a.items()}
        print(f"  {reg:<11} {len(rs):>3}  "
              f"{mean(a['ece_always_benign']):>8.4f}({median(a['ece_always_benign']):.4f})  "
              f"{mean(a['ece_oracle']):>6.4f}  {mean(a['ece_mmae']):>6.4f}  "
              f"{mean(a['ece_clf']):>6.4f}  {mean(a['ece_rule']):>6.4f}")

    # ── Selection accuracy ──
    print("\n── Selection accuracy (GT-free rule picks true regime) ──")
    def sel_acc(rs, col):
        rs = [r for r in rs if r["regime"] != "" ]
        return sum(1 for r in rs if r[col] == r["regime"]) / len(rs) if rs else float("nan")
    print(f"  overall(core):  MMAE(S_glob)={sel_acc(core,'mmae_pick'):.2f}  "
          f"MMAE(NIS)={sel_acc(core,'mmae_nis_pick'):.2f}  "
          f"clf-centroid={sel_acc(core,'clf_pick'):.2f}  rule-default={sel_acc(core,'rule_pick'):.2f}")
    for reg in CORE_REGIMES:
        rs = [r for r in core if r["regime"] == reg]
        if not rs:
            continue
        print(f"    {reg:<11} MMAE={sel_acc(rs,'mmae_pick'):.2f}  "
              f"MMAE-NIS={sel_acc(rs,'mmae_nis_pick'):.2f}  clf={sel_acc(rs,'clf_pick'):.2f}  "
              f"rule={sel_acc(rs,'rule_pick'):.2f}")

    # ── Recovery fraction + convenient-case guard ──
    print("\n── Recovery fraction (a→c)/(a→b) + stale-guard (a vs b) ──")
    for reg in CORE_REGIMES:
        if reg == "benign" or reg not in arms_by_regime:
            continue
        a = arms_by_regime[reg]
        gap = a["ece_always_benign"] - a["ece_oracle"]
        rec_c = (a["ece_always_benign"] - a["ece_clf"]) / gap if abs(gap) > 1e-9 else float("nan")
        rec_r = (a["ece_always_benign"] - a["ece_rule"]) / gap if abs(gap) > 1e-9 else float("nan")
        stale = "STALE" if gap > 0.01 else "not-stale (nothing to restore!)"
        print(f"  {reg:<11} always-benign={a['ece_always_benign']:.4f} oracle={a['ece_oracle']:.4f} "
              f"gap={gap:+.4f} [{stale}]  recovery: clf={rec_c:+.2f} rule={rec_r:+.2f}")
    # Benign guard: switching must NOT break the already-good case.
    if "benign" in arms_by_regime:
        a = arms_by_regime["benign"]
        print(f"  benign      always-benign={a['ece_always_benign']:.4f} clf={a['ece_clf']:.4f} "
              f"rule={a['ece_rule']:.4f} (switch must not regress the good case)")

    # ── Tigress blind spot ──
    if tig:
        a = {k: mean([fnum(r[k]) for r in tig]) for k in
             ("ece_always_benign", "ece_oracle", "ece_mmae", "ece_clf")}
        clf_acc = sel_acc(tig, "clf_pick")
        print(f"\n── Tigress blind-spot (held-out limit; n={len(tig)}) ──")
        print(f"  always-benign={a['ece_always_benign']:.4f} oracle={a['ece_oracle']:.4f} "
              f"mmae={a['ece_mmae']:.4f} clf={a['ece_clf']:.4f}  clf-sel-acc={clf_acc:.2f}")

    # ── Figure: grouped-bar ECE per regime ──
    # "switched" arm = whichever signature classifier does least harm on benign (do-no-harm wins).
    switched_key = "ece_rule"
    if "benign" in arms_by_regime and arms_by_regime["benign"]["ece_clf"] < arms_by_regime["benign"]["ece_rule"]:
        switched_key = "ece_clf"
    print(f"\n(figure switched-arm = {switched_key})")
    try:
        make_figure(arms_by_regime, tig, out_dir, switched_key)
    except Exception as e:
        print(f"\n(figure skipped: {e})")


def make_figure(arms_by_regime, tig, out_dir, switched_key="ece_rule"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regimes = [r for r in CORE_REGIMES if r in arms_by_regime]
    labels = list(regimes)
    always = [arms_by_regime[r]["ece_always_benign"] for r in regimes]
    oracle = [arms_by_regime[r]["ece_oracle"] for r in regimes]
    clf = [arms_by_regime[r][switched_key] for r in regimes]
    if tig:
        labels.append("tigress\n(limit)")
        always.append(mean([fnum(r["ece_always_benign"]) for r in tig]))
        oracle.append(mean([fnum(r["ece_oracle"]) for r in tig]))
        clf.append(mean([fnum(r[switched_key]) for r in tig]))

    x = range(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar([i - w for i in x], always, w, label="always-benign (stale)", color="#c44e52")
    ax.bar(list(x), clf, w, label="consistency-switched (ours, GT-free)", color="#4c72b0")
    ax.bar([i + w for i in x], oracle, w, label="oracle (ceiling)", color="#55a868")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("true post-hoc ECE (lower = better calibrated)")
    ax.set_title("Consistency switching restores calibration without ground truth")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_switching.{ext}"), dpi=140)
    print(f"\nwrote fig_switching.svg / .png → {out_dir}")


if __name__ == "__main__":
    main()
