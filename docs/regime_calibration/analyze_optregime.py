#!/usr/bin/env python3
"""Regenerate the regime-calibration probe tables + confound audit from optregime.csv.

Phase A  — does the O2-fit calibration map drift when applied to held-out O0/O1/O3?
Phase B  — does the (S_glob, S_spat) signature select the right per-regime map, GT-free?
Confound — (a) does a size/entropy-only baseline select as well?  (b) is the signature
           additive over size+entropy (partial association controlling for both)?

Everything is read off the per-binary CSV the Rust harness emits; this script does no engine work.
It prints a markdown-ish report to stdout — the RESULTS.md tables are lifted from here.
"""
import csv
import sys
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

CSV = sys.argv[1] if len(sys.argv) > 1 else "optregime.csv"


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            def fnum(k):
                v = r[k]
                return None if v == "NA" else float(v)
            rows.append(dict(
                stem=r["stem"], level=r["level"], split=r["split"],
                n=int(r["n"]), code_bytes=int(r["code_bytes"]),
                entropy=float(r["entropy"]), base_rate=float(r["base_rate"]),
                s_glob=float(r["s_glob"]), s_spat=float(r["s_spat"]),
                ece_raw=fnum("ece_raw"), ece_default=fnum("ece_default"),
                ece_oracle=fnum("ece_oracle"), ece_sig=fnum("ece_sig"),
                ece_se=fnum("ece_se"), sig_pick=r["sig_pick"], se_pick=r["se_pick"],
            ))
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def median(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else float("nan")


def main():
    rows = load(CSV)
    levels = []
    for r in rows:
        if r["level"] not in levels:
            levels.append(r["level"])
    levels_sorted = sorted(levels)  # O0,O1,O2,O3
    default = "O2"
    hold = [r for r in rows if r["split"] == "holdout"]
    allr = rows

    print(f"# Regime-calibration probe — analysis of `{CSV}`\n")
    print(f"- levels: {levels_sorted} | default = **{default}**")
    print(f"- binaries: {len(rows)} total, {len(hold)} held-out "
          f"({len(set(r['stem'] for r in rows if r['split']=='fit'))} fit programs, "
          f"{len(set(r['stem'] for r in hold))} holdout programs)\n")

    # ── Phase A: drift of the O2 map across regimes ──
    print("## Phase A — calibration drift (O2 map applied unchanged)\n")
    print("Held-out instruction-level ECE. `raw` = uncalibrated; `O2-map` = the default map applied "
          "to every regime (the Phase-A arm); `own-map` = each regime's own map (oracle, context).\n")
    print("| regime | n | raw ECE | **O2-map ECE** | own-map ECE | drift vs O2-holdout | ratio |")
    print("|---|---|---|---|---|---|---|")
    base = mean([r["ece_default"] for r in hold if r["level"] == default])
    phaseA = {}
    for lv in levels_sorted:
        rs = [r for r in hold if r["level"] == lv]
        raw = mean([r["ece_raw"] for r in rs])
        dfl = mean([r["ece_default"] for r in rs])
        own = mean([r["ece_oracle"] for r in rs])
        drift = dfl - base
        ratio = dfl / base if base > 0 else float("nan")
        phaseA[lv] = dict(raw=raw, dfl=dfl, own=own, drift=drift, ratio=ratio, n=len(rs))
        star = " (default baseline)" if lv == default else ""
        print(f"| {lv}{star} | {len(rs)} | {raw:.4f} | {dfl:.4f} | {own:.4f} | {drift:+.4f} | {ratio:.2f}× |")
    off = [lv for lv in levels_sorted if lv != default]
    max_ratio = max(phaseA[lv]["ratio"] for lv in off)
    max_drift = max(phaseA[lv]["drift"] for lv in off)
    print(f"\n*Off-default worst-case: drift {max_drift:+.4f} ECE, {max_ratio:.2f}× the O2 held-out baseline "
          f"({base:.4f}).*\n")

    # ── Phase B: three-arm + selection ──
    print("## Phase B — signature-selected map (GT-free) vs always-default vs oracle\n")
    print("| regime | n | always-default | oracle | **switched (sig)** | size/ent | sel-acc sig | sel-acc size/ent | recovery(sig) |")
    print("|---|---|---|---|---|---|---|---|---|")
    phaseB = {}
    for lv in levels_sorted:
        rs = [r for r in hold if r["level"] == lv]
        a = mean([r["ece_default"] for r in rs])
        o = mean([r["ece_oracle"] for r in rs])
        s = mean([r["ece_sig"] for r in rs])
        e = mean([r["ece_se"] for r in rs])
        acc_s = np.mean([r["sig_pick"] == lv for r in rs])
        acc_e = np.mean([r["se_pick"] == lv for r in rs])
        rec = (a - s) / (a - o) if abs(a - o) > 1e-9 else float("nan")
        phaseB[lv] = dict(a=a, o=o, s=s, e=e, acc_s=acc_s, acc_e=acc_e, rec=rec, n=len(rs))
        print(f"| {lv} | {len(rs)} | {a:.4f} | {o:.4f} | {s:.4f} | {e:.4f} | {acc_s:.2f} | {acc_e:.2f} | {rec:+.2f} |")
    sel_sig = np.mean([r["sig_pick"] == r["level"] for r in hold])
    sel_se = np.mean([r["se_pick"] == r["level"] for r in hold])
    chance = 1.0 / len(levels_sorted)
    print(f"\n*Overall GT-free selection accuracy: signature **{sel_sig:.2f}**, size/entropy {sel_se:.2f} "
          f"(chance = {chance:.2f}).*\n")

    # confusion matrix for the signature selector (holdout)
    print("Signature selector confusion (rows = true regime, cols = picked):\n")
    idx = {lv: i for i, lv in enumerate(levels_sorted)}
    cm = np.zeros((len(levels_sorted), len(levels_sorted)), int)
    for r in hold:
        cm[idx[r["level"]], idx[r["sig_pick"]]] += 1
    print("| true\\pick | " + " | ".join(levels_sorted) + " |")
    print("|" + "---|" * (len(levels_sorted) + 1))
    for lv in levels_sorted:
        print(f"| {lv} | " + " | ".join(str(cm[idx[lv], idx[c]]) for c in levels_sorted) + " |")
    print()

    # ── Confound audit ──
    print("## Confound audit\n")
    # (a) already have sel_se vs sel_sig above; restate + ECE arm
    mean_sig = mean([r["ece_sig"] for r in hold])
    mean_se = mean([r["ece_se"] for r in hold])
    mean_dfl = mean([r["ece_default"] for r in hold])
    mean_ora = mean([r["ece_oracle"] for r in hold])
    print("**(a) Does a size/entropy-only baseline select as well as the signature?**\n")
    print(f"- selection accuracy: signature {sel_sig:.2f} vs size/entropy {sel_se:.2f}")
    print(f"- mean held-out ECE: always-default {mean_dfl:.4f} | oracle {mean_ora:.4f} | "
          f"signature {mean_sig:.4f} | size/entropy {mean_se:.4f}\n")

    # (b) partial association: does the signature carry regime info beyond size+entropy?
    print("**(b) Is the signature *additive* over size and entropy?**\n")
    # ordinal opt level 0..3
    ordv = np.array([idx[r["level"]] for r in allr], float)
    lnsize = np.log(np.array([r["code_bytes"] for r in allr], float))
    ent = np.array([r["entropy"] for r in allr], float)
    sglob = np.array([r["s_glob"] for r in allr], float)
    sspat = np.array([r["s_spat"] for r in allr], float)

    def partial_corr(y, x, controls):
        # correlation of residuals of y~controls and x~controls (OLS with intercept)
        C = np.column_stack([np.ones_like(controls[0])] + list(controls))
        ry = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
        rx = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
        r = np.corrcoef(ry, rx)[0, 1]
        n = len(y)
        k = C.shape[1]  # #params in controls incl intercept
        dfr = n - k - 1
        t = r * np.sqrt(dfr / max(1e-12, 1 - r * r))
        p = 2 * stats.t.sf(abs(t), dfr)
        return r, t, p, dfr

    for name, x in [("s_glob", sglob), ("s_spat", sspat)]:
        r0 = np.corrcoef(ordv, x)[0, 1]
        rp, t, p, dfr = partial_corr(ordv, x, [lnsize, ent])
        print(f"- `{name}` vs opt-ordinal: raw r = {r0:+.3f}; **partial r (control ln_size, entropy) "
              f"= {rp:+.3f}** (t={t:.2f}, df={dfr}, p≈{p:.1e})")
    # also how well size+entropy alone explain the signature (is signature ~ size proxy?)
    for name, x in [("s_glob", sglob), ("s_spat", sspat)]:
        C = np.column_stack([np.ones_like(lnsize), lnsize, ent])
        beta = np.linalg.lstsq(C, x, rcond=None)[0]
        pred = C @ beta
        ss_res = np.sum((x - pred) ** 2)
        ss_tot = np.sum((x - x.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        print(f"- `{name}` explained by size+entropy alone: R² = {r2:.3f}")
    print()

    # added-value classification: LOGO(program) CV accuracy, S+E vs S+E+signature
    groups = np.array([r["stem"] for r in allr])
    ylab = np.array([idx[r["level"]] for r in allr])
    Xse = np.column_stack([lnsize, ent])
    Xsig = np.column_stack([lnsize, ent, sglob, sspat])
    logo = LeaveOneGroupOut()

    def cv_acc(X):
        preds = np.zeros_like(ylab)
        for tr, te in logo.split(X, ylab, groups):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
            clf.fit(sc.transform(X[tr]), ylab[tr])
            preds[te] = clf.predict(sc.transform(X[te]))
        return np.mean(preds == ylab)

    acc_se_cv = cv_acc(Xse)
    acc_sig_cv = cv_acc(Xsig)
    print("Leave-one-program-out CV multinomial-logistic regime accuracy (all binaries):\n")
    print(f"- size+entropy only:        {acc_se_cv:.2f}")
    print(f"- size+entropy + signature: {acc_sig_cv:.2f}")
    print(f"- **added value of signature: {acc_sig_cv - acc_se_cv:+.2f}** (chance {chance:.2f})\n")

    # ── Verdict inputs summarized ──
    print("## Verdict inputs\n")
    gateA = max_ratio >= 1.5 or max_drift >= 0.01
    gateB_recover = (mean_sig <= mean_ora * 1.5) and (mean_sig < mean_dfl * 0.9)
    gateB_sel = sel_sig >= 0.6
    conf_sig_beats_se = (sel_sig - sel_se) >= 0.1
    conf_additive = (acc_sig_cv - acc_se_cv) >= 0.05
    print(f"- Gate A (drift real): worst ratio {max_ratio:.2f}×, worst drift {max_drift:+.4f} → "
          f"{'PASS' if gateA else 'FAIL'}")
    print(f"- Gate B (restores toward oracle): sig {mean_sig:.4f} vs oracle {mean_ora:.4f} vs default "
          f"{mean_dfl:.4f} → {'PASS' if gateB_recover else 'FAIL'}")
    print(f"- Gate B (selection ≥0.60): {sel_sig:.2f} → {'PASS' if gateB_sel else 'FAIL'}")
    print(f"- Confound (sig beats size/ent by ≥0.10): {sel_sig:.2f} vs {sel_se:.2f} → "
          f"{'PASS' if conf_sig_beats_se else 'FAIL'}")
    print(f"- Confound (signature additive, +CV ≥0.05): {acc_sig_cv - acc_se_cv:+.2f} → "
          f"{'PASS' if conf_additive else 'FAIL'}")
    go = gateA and gateB_recover and gateB_sel and conf_sig_beats_se and conf_additive
    print(f"\n### Combined: {'GO' if go else 'NO-GO'}\n")


if __name__ == "__main__":
    main()
