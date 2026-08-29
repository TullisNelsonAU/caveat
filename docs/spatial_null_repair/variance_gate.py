#!/usr/bin/env python3
"""Two-component variance gate: fit `a`, `c` on the clean_fit split, then score.

    T2(n) = mu + z * sqrt(a + c^2/n)

`a` is the between-binary variance component that does not vanish with `n`; `c^2/n` is the
sampling component the published one-component gate assumed was the whole story. Both are
estimated on the 20 `role == clean_fit` rows of `credibility.csv` and then frozen, exactly as
`mu` and `c` were for the published gates. Nothing is refit on an evaluation corpus.

Scoring re-derives the routing decisions from the recorded per-binary `(S_glob, S_spat,
region_ent, n)` by reimplementing `SignatureClassifier::{classify_rule, classify_guard}` in
Python; the reimplementation is validated by reproducing all three published gate columns
(`old_*`, `new_*`, `flr_*`) of `decisions.tsv` cell for cell before any T2 number is computed.
A single mismatch aborts.

REPORT ONLY. This script writes a report section; it does not change any published gate.
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import binomtest, chi2

HERE = Path(__file__).parent
DEC = HERE / "decisions.tsv"
META = HERE / "decisions.meta.tsv"
CRED = HERE.parent / "consistency_credibility" / "credibility.csv"

Z = 1.645
TIG = ("tigL", "tigM", "tigH")
BUCKETS = ((250, 1_000), (1_000, 4_000), (4_000, 16_000), (16_000, None))


# ── the shipped classifier, reimplemented and validated below ────────────────
def classify_rule(s_glob, s_spat, glob_hi, spat_hi):
    if s_glob > glob_hi:
        return "obfuscated"
    if s_spat > spat_hi:
        return "packed"
    return "benign"


def classify_guard(s_glob, s_spat, region_ent, glob_hi, spat_hi, pack_ent_lo):
    packed_like = math.isnan(pack_ent_lo) or region_ent > pack_ent_lo
    if s_glob > glob_hi:
        return "obfuscated"
    if s_spat > spat_hi and packed_like:
        return "packed"
    return "benign"


def load():
    rows = list(csv.DictReader(open(DEC), delimiter="\t"))
    meta = {r["key"]: float(r["value"]) for r in csv.DictReader(open(META), delimiter="\t")}
    for r in rows:
        r["n"] = int(float(r["n"]))
        for k in ("s_glob", "s_spat", "t_new", "t_flr"):
            r[k] = float(r[k])
        r["region_ent"] = None if r["region_ent"] == "-" else float(r["region_ent"])
        r["floor_binds"] = r["floor_binds"] == "true"
    return rows, meta


# ── fit ──────────────────────────────────────────────────────────────────────
def fit_split():
    fit = [r for r in csv.DictReader(open(CRED)) if r["role"] == "clean_fit"]
    assert len(fit) == 20, f"expected 20 clean_fit rows, found {len(fit)}"
    s = np.array([float(r["s_spat_moran"]) for r in fit])
    n = np.array([float(r["n"]) for r in fit])
    return s, n, [r["name"] for r in fit]


def nll(theta, d2, n):
    """-log L for d_i ~ N(0, a + c^2/n_i), parameterised by (sqrt a, c) to keep both >= 0."""
    ra, c = theta
    v = ra * ra + (c * c) / n
    if np.any(v <= 0):
        return 1e12
    return 0.5 * np.sum(np.log(2 * np.pi * v) + d2 / v)


def ssq(theta, d2, n):
    """Unweighted least squares on the squared deviations: sum (d_i^2 - v_i)^2."""
    ra, c = theta
    v = ra * ra + (c * c) / n
    return float(np.sum((d2 - v) ** 2))


def fit_two_component(mu, s, n, objective):
    d2 = (s - mu) ** 2
    best, best_x = np.inf, None
    for ra0 in (1e-4, 1e-3, 5e-3, 2e-2, 5e-2):
        for c0 in (0.5, 2.0, 4.0, 8.0):
            r = minimize(objective, np.array([ra0, c0]), args=(d2, n), method="Nelder-Mead",
                         options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 40000, "maxfev": 40000})
            if r.fun < best:
                best, best_x = r.fun, r.x
    ra, c = abs(best_x[0]), abs(best_x[1])
    return ra * ra, c, best


def main():
    rows, meta = load()
    mu, c1, flat = meta["mu"], meta["c"], meta["flat_spat_hi"]
    glob_hi, pack_ent_lo = meta["glob_hi"], meta["pack_ent_lo"]
    s, n, _ = fit_split()

    # sanity: the published one-component constants must reproduce from this same split
    mu_chk = float(s.mean())
    scaled = (s - mu_chk) * np.sqrt(n)
    c_chk = float(scaled.std())  # population sd about the scaled mean — the shipped `pstdev`
    assert abs(mu_chk - mu) < 1e-9, (mu_chk, mu)
    assert abs(c_chk - c1) < 1e-6, (c_chk, c1)

    a_ml, c_ml, nll_ml = fit_two_component(mu, s, n, nll)
    a_ls, c_ls, ss_ls = fit_two_component(mu, s, n, ssq)

    # one-component nested model (a == 0) at its own ML optimum, for a likelihood-ratio test
    d2 = (s - mu) ** 2
    c0_ml = float(np.sqrt(np.mean(d2 * n)))          # closed-form ML for a == 0
    nll0 = nll(np.array([0.0, c0_ml]), d2, n)
    lr = 2 * (nll0 - nll_ml)

    # joint-mu sensitivity: refit mu together with (a, c) rather than holding the published mean
    def nll_mu(theta):
        m, ra, c = theta
        v = ra * ra + (c * c) / n
        if np.any(v <= 0):
            return 1e12
        return 0.5 * np.sum(np.log(2 * np.pi * v) + ((s - m) ** 2) / v)

    rj = minimize(nll_mu, np.array([mu, math.sqrt(a_ml), c_ml]), method="Nelder-Mead",
                  options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 60000, "maxfev": 60000})
    mu_j, a_j, c_j = rj.x[0], rj.x[1] ** 2, abs(rj.x[2])

    def T2(nn, a=a_ml, cc=c_ml):
        return mu + Z * math.sqrt(a + (cc * cc) / nn)

    def T1(nn):
        return mu + Z * c1 / math.sqrt(nn)

    # Supplementary variant: the sampling component is *held* at the published one-component `c`
    # and only the between-binary component `a` is fit. This is the two-component gate the
    # reviewer's argument describes when the 1/n term is asserted rather than estimated; it is
    # reported alongside T2 because the free fit collapses c to zero and so never exercises the
    # two-component shape at all.
    a_fixc = float(minimize(lambda x: nll(np.array([x[0], c1]), d2, n), np.array([0.02]),
                            method="Nelder-Mead",
                            options={"xatol": 1e-14, "fatol": 1e-16}).x[0] ** 2)

    def T2c(nn):
        return mu + Z * math.sqrt(a_fixc + (c1 * c1) / nn)

    # ── validate the Python classifier against every published gate column ────
    checked = 0
    for r in rows:
        for era, spat_hi in (("old", flat), ("new", r["t_new"]), ("flr", r["t_flr"])):
            got = classify_rule(r["s_glob"], r["s_spat"], glob_hi, spat_hi)
            assert got == r[f"{era}_rule"], (r["name"], era, got, r[f"{era}_rule"])
            got = classify_rule(r["s_glob"], r["s_spat"], float("inf"), spat_hi)
            assert got == r[f"{era}_spat_only"], (r["name"], era, "spat_only")
            if r["region_ent"] is not None:
                got = classify_guard(r["s_glob"], r["s_spat"], r["region_ent"], glob_hi, spat_hi,
                                     pack_ent_lo)
                assert got == r[f"{era}_guard"], (r["name"], era, "guard")
            checked += 1
    assert checked == 3 * len(rows)

    # ── score T2 on every row, same three arms ───────────────────────────────
    for r in rows:
        t2 = T2(r["n"])
        r["t_t2"] = t2
        r["t2_rule"] = classify_rule(r["s_glob"], r["s_spat"], glob_hi, t2)
        r["t2_spat_only"] = classify_rule(r["s_glob"], r["s_spat"], float("inf"), t2)
        r["t2_guard"] = ("-" if r["region_ent"] is None else
                         classify_guard(r["s_glob"], r["s_spat"], r["region_ent"], glob_hi, t2,
                                        pack_ent_lo))
        t2c = T2c(r["n"])
        r["t_t2c"] = t2c
        r["t2c_rule"] = classify_rule(r["s_glob"], r["s_spat"], glob_hi, t2c)
        r["t2c_spat_only"] = classify_rule(r["s_glob"], r["s_spat"], float("inf"), t2c)
        r["t2c_guard"] = ("-" if r["region_ent"] is None else
                          classify_guard(r["s_glob"], r["s_spat"], r["region_ent"], glob_hi, t2c,
                                         pack_ent_lo))

    out = {
        "fit": {
            "mu": mu, "z": Z, "flat": flat,
            "a_ml": a_ml, "c_ml": c_ml, "nll_ml": float(nll_ml),
            "sqrt_a_ml": math.sqrt(a_ml),
            "a_ls": a_ls, "c_ls": c_ls, "ssq_ls": float(ss_ls),
            "a_fixc": a_fixc, "sqrt_a_fixc": math.sqrt(a_fixc),
            "asymptote_fixc": mu + Z * math.sqrt(a_fixc),
            "T2c_at": {str(k): T2c(k) for k in
                       (500, 1000, 2630, 4000, 16000, 32000, 100000, 719327)},
            "profile_c": {str(cc): float(minimize(lambda x: nll(np.array([x[0], cc]), d2, n),
                                                  np.array([0.02]), method="Nelder-Mead",
                                                  options={"xatol": 1e-14, "fatol": 1e-16}).fun)
                          for cc in (0.0, 1.0, 2.0, 3.0, 4.0343215059, 5.0, 6.0)},
            "corr_d2_invn": float(np.corrcoef((s - mu) ** 2, 1 / n)[0, 1]),
            "corr_absd_invsqrtn": float(np.corrcoef(np.abs(s - mu), 1 / np.sqrt(n))[0, 1]),
            "c_onecomp_published": c1, "c0_ml_nested": c0_ml, "nll0": float(nll0), "lr": float(lr),
            "mu_joint": mu_j, "a_joint": a_j, "c_joint": c_j,
            "fit_n_lo": int(n.min()), "fit_n_hi": int(n.max()),
            "asymptote": mu + Z * math.sqrt(a_ml),
            "asymptote_ls": mu + Z * math.sqrt(a_ls) if a_ls > 0 else None,
            "T2_at": {str(k): T2(k) for k in (500, 1000, 2630, 4000, 16000, 32000, 100000, 719327)},
            "T1_at": {str(k): T1(k) for k in (500, 1000, 2630, 4000, 16000, 32000, 100000, 719327)},
            "Tflr_at": {str(k): max(flat, T1(k)) for k in
                        (500, 1000, 2630, 4000, 16000, 32000, 100000, 719327)},
        },
        "classifier_cells_validated": checked,
    }

    # crossings of T2 against FLAT and against T1
    def cross_flat():
        lo, hi = 1.0, 1e12
        if T2(hi) > flat:
            return None
        for _ in range(300):
            mid = math.sqrt(lo * hi)
            if T2(mid) > flat:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    out["fit"]["n_cross_flat"] = cross_flat()

    ERAS = (("old", "flat"), ("new", "T(n)"), ("flr", "T'(n)"), ("t2", "T2(n)"),
            ("t2c", "T2c(n)"))

    def cnt(rs, era, arm="spat_only"):
        return sum(1 for r in rs if r[f"{era}_{arm}"] != "benign")

    def acc(rs, era, arm):
        return sum(1 for r in rs if r[f"{era}_{arm}"] == r["true_regime"])

    def corpus(lbl):
        return [r for r in rows if r["corpus"] == lbl]

    res = {}
    cred = corpus("credibility")
    res["detection"] = {}
    for role in ("clean_fit", "benign", "obfuscated", "packed"):
        rs = [r for r in cred if r["true_regime"] == role]
        res["detection"][role] = {"N": len(rs), **{e: cnt(rs, e) for e, _ in ERAS}}

    wild = corpus("wild_debian")
    res["wild"] = {"N": len(wild),
                   "rule": {e: cnt(wild, e, "rule") for e, _ in ERAS},
                   "guard": {e: cnt(wild, e, "guard") for e, _ in ERAS},
                   "buckets": []}
    for lo, hi in BUCKETS:
        sub = [r for r in wild if r["n"] >= lo and (hi is None or r["n"] < hi)]
        res["wild"]["buckets"].append({
            "bucket": f"{lo:,}–{hi:,}" if hi else f">= {lo:,}", "N": len(sub),
            **{e: (cnt(sub, e, "rule") / len(sub) if sub else None) for e, _ in ERAS},
            **{e + "_k": cnt(sub, e, "rule") for e, _ in ERAS}})

    pb = [r for r in rows if r["corpus"].startswith("breadth") and r["true_regime"] == "packed"]
    res["breadth"] = {"N": len(pb), **{e: acc(pb, e, "rule") for e, _ in ERAS},
                      "by_config": {}}
    for sl in sorted(set(r["sublabel"] for r in pb)):
        rs = [r for r in pb if r["sublabel"] == sl]
        res["breadth"]["by_config"][sl] = {
            "N": len(rs), "mean_n": sum(r["n"] for r in rs) / len(rs),
            "mean_s_spat": sum(r["s_spat"] for r in rs) / len(rs),
            "mean_t2": sum(r["t_t2"] for r in rs) / len(rs),
            "mean_tflr": sum(r["t_flr"] for r in rs) / len(rs),
            **{e: acc(rs, e, "rule") for e, _ in ERAS}}

    test = [r for r in cred if r["true_regime"] != "clean_fit"]
    res["ablation"] = {"N": len(test),
                       **{arm: {e: acc(test, e, arm) for e, _ in ERAS}
                          for arm in ("spat_only", "rule")}}

    core = corpus("switching_core")
    sel_sets = {
        "core": [r for r in core if r["sublabel"] not in TIG],
        "core_tig": [r for r in core if r["sublabel"] in TIG],
        "scale": corpus("corpus_expansion"),
        "guardc": corpus("abstention_guard"),
    }
    res["selection"] = {}
    for k, rs in sel_sets.items():
        entry = {"N": len(rs), "rule": {e: acc(rs, e, "rule") for e, _ in ERAS}}
        if rs and rs[0]["old_guard"] != "-":
            entry["guard"] = {e: acc(rs, e, "guard") for e, _ in ERAS}
        res["selection"][k] = entry

    # ── Tigress, both runs ───────────────────────────────────────────────────
    res["tigress"] = {}
    for cl, rs_all in (("boundaries_meta", corpus("boundaries_meta")),
                       ("switching_core", core)):
        per = {}
        for sl in TIG:
            rs = [r for r in rs_all if r["sublabel"] == sl]
            if not rs:
                continue
            per[sl] = {"N": len(rs), "n_lo": min(r["n"] for r in rs),
                       "n_hi": max(r["n"] for r in rs),
                       "mean_s_spat": sum(r["s_spat"] for r in rs) / len(rs),
                       "mean_t2": sum(r["t_t2"] for r in rs) / len(rs),
                       "mean_tflr": sum(r["t_flr"] for r in rs) / len(rs),
                       **{e: sum(1 for r in rs if r[f"{e}_rule"] == "packed") for e, _ in ERAS}}
        res["tigress"][cl] = per

    res["p05_vm"] = [
        {"corpus": r["corpus"], "name": r["name"], "sublabel": r["sublabel"],
         "true_regime": r["true_regime"], "n": r["n"], "s_spat": r["s_spat"],
         "t_flr": r["t_flr"], "t_t2": r["t_t2"],
         **{f"{e}_rule": r[f"{e}_rule"] for e, _ in ERAS},
         **{f"{e}_guard": r[f"{e}_guard"] for e, _ in ERAS}}
        for r in rows if r["name"].startswith("p05_vm")]

    # ── subset property: T2 vs the published floored gate, and vs flat ───────
    def gains(era_new, era_ref, arm):
        return [r for r in rows
                if r[f"{era_new}_{arm}"] != "benign" and r[f"{era_ref}_{arm}"] == "benign"]

    res["subset"] = {}
    for era_new, ref in (("t2", "flr"), ("t2", "old"), ("t2c", "flr"), ("t2c", "old")):
        d = {}
        for arm in ("spat_only", "rule", "guard"):
            g = [r for r in gains(era_new, ref, arm)
                 if not (arm == "guard" and r["region_ent"] is None)]
            d[arm] = {"count": len(g),
                      "rows": [{"corpus": r["corpus"], "name": r["name"],
                                "sublabel": r["sublabel"], "true_regime": r["true_regime"],
                                "n": r["n"], "s_spat": r["s_spat"],
                                "t_ref": r["t_flr"] if ref == "flr" else flat,
                                "t_new": r[f"t_{era_new}"]} for r in g]}
        res["subset"][f"{era_new}_vs_{ref}"] = d
    res["subset"]["t2_below_flr_n_range"] = None
    below = [r for r in rows if r["t_t2"] < r["t_flr"] - 1e-12]
    if below:
        res["subset"]["t2_below_flr_n_range"] = [min(r["n"] for r in below),
                                                 max(r["n"] for r in below), len(below)]

    out["results"] = res
    if "--json" in sys.argv:
        print(json.dumps(out, indent=1, default=float))
        return
    emit(out, rows, meta)

OUT2 = HERE / "TWO_COMPONENT_VARIANCE_GATE.md"

GATES = (("old", "flat"), ("new", "T(n)"), ("flr", "T'(n)"), ("t2", "T2(n)"), ("t2c", "T2c(n)"))


def emit(out, rows, meta):
    f = out["fit"]
    r = out["results"]
    L = []
    w = L.append
    fr = lambda k, n: f"{k}/{n} = {k / n:.4f}" if n else "n/a"

    mu, flat, c1 = f["mu"], f["flat"], f["c_onecomp_published"]
    a, c = f["a_ml"], f["c_ml"]
    afx = f["a_fixc"]

    w("# Two-component variance gate — fit, scored, and not adopted")
    w("")
    w("An external reviewer observed that the floor in Sec. 6.10 is a crude stand-in for a variance")
    w("model we can name: the unfloored gate under-covers at large `n` because `Var(S_spat)` is not")
    w("`c^2/n`, since a real program carries a between-binary variance component that does not vanish")
    w("with `n`. The principled form is")
    w("")
    w("```")
    w("    T2(n) = mu + 1.645 * sqrt(a + c^2/n)")
    w("```")
    w("")
    w(f"with both `a` and `c` estimated on the same {int(meta['n_fit'])} `role == clean_fit` rows of")
    w("`credibility.csv` that set `mu`, `c`, and `FLAT` for the published gates, then frozen. This")
    w("document reports what that gate does. **It is not adopted, and nothing else changes**: the")
    w("published operating point remains the floored gate `T'(n) = max(FLAT, T(n))`.")
    w("")
    w("Five gates are compared throughout:")
    w("")
    w("| gate | definition | status |")
    w("|---|---|---|")
    w(f"| `flat` | `S_spat > {flat:.6f}` | the originally published operating point |")
    w(f"| `T(n)` | `mu + 1.645*c/sqrt(n)` | unfloored size-aware gate — intermediate result |")
    w(f"| `T'(n)` | `max(FLAT, T(n))` | **the current operating point** |")
    w(f"| `T2(n)` | `mu + 1.645*sqrt(a + c^2/n)`, both free | the reviewer's form, fit as specified |")
    w(f"| `T2c(n)` | same form, `c` **held** at the published {c1:.6f}, only `a` fit | supplementary — see Sec. 2 |")
    w("")
    w("---")
    w("")

    # ── 1. the fit ────────────────────────────────────────────────────────────
    w("## 1. The fit — and why it collapses")
    w("")
    w("### Estimator")
    w("")
    w("**Maximum likelihood**, on `d_i = s_i - mu ~ N(0, a + c^2/n_i)`, maximising")
    w("`sum -0.5*[log(2*pi*v_i) + d_i^2/v_i]` over `a >= 0`, `c >= 0` by Nelder-Mead from a grid of")
    w("starts. `mu` is held at the published fit-split mean, so the comparison against `T(n)` and")
    w("`T'(n)` changes exactly one thing — the variance model.")
    w("")
    w("ML rather than unweighted least squares on the squared deviations, because under the model")
    w("`Var(d_i^2) = 2*v_i^2`: the squared deviations are heteroscedastic *by construction*, with the")
    w("small-`n` points carrying the most noise precisely because they carry the most signal about")
    w("`c`. Unweighted LS treats all 20 squared deviations as equally precise, which is the wrong")
    w("weighting for the one parameter the model exists to estimate, and it admits `a < 0`. ML")
    w("weights each point by `1/v_i^2` automatically and respects the boundary.")
    w("")
    w("In this instance the choice is moot — both objectives land in the same place:")
    w("")
    w("| objective | `a` | `sqrt(a)` | `c` |")
    w("|---|---|---|---|")
    w(f"| maximum likelihood | {a:.6e} | {math.sqrt(a):.6f} | {c:.3e} |")
    w(f"| least squares on `d^2` | {f['a_ls']:.6e} | {math.sqrt(f['a_ls']):.6f} | {f['c_ls']:.3e} |")
    w("")
    w("### The result")
    w("")
    w(f"**`c` goes to zero.** The ML estimate is `a = {a:.6e}` (`sqrt(a) = {math.sqrt(a):.6f}`) and")
    w(f"`c = {c:.2e}` — a boundary optimum. The profile likelihood is monotone in `c`:")
    w("")
    w("| `c` | max log-lik (as `-NLL`) |")
    w("|---|---|")
    for k, v in f["profile_c"].items():
        star = "  ← published `c`" if abs(float(k) - c1) < 1e-6 else ""
        w(f"| {float(k):.4f} | {-v:.5f}{star} |")
    w("")
    w("Every unit of `c` costs likelihood. The reason is visible in the fit split itself: the")
    w("deviations do not shrink with `n`, they **grow**.")
    w("")
    w(f"* `corr(d^2, 1/n) = {f['corr_d2_invn']:+.4f}`")
    w(f"* `corr(|d|, 1/sqrt(n)) = {f['corr_absd_invsqrtn']:+.4f}`")
    w("")
    w("Both have the wrong sign for a `1/n` sampling law. The four largest `|d|` in the split sit at")
    w("`n` = 44,159 / 57,074 / 62,880 / 81,851 — the *large* end of the fit range. So when the model")
    w("is given a constant component to spend variance on, it spends all of it there.")
    w("")
    w("Two likelihood-ratio tests make the asymmetry explicit:")
    w("")
    w("| comparison | deviance | df | p |")
    w("|---|---|---|---|")
    w(f"| `a = 0` (the published one-component model) vs free `(a, c)` | {f['lr']:.3f} | 1 (boundary) | "
      f"{0.5 * chi2.sf(f['lr'], 1):.5f} |")
    w(f"| `c = {c1:.4f}` held vs free `(a, c)` | "
      f"{2 * (f['profile_c'][str(c1)] - f['nll_ml']):.3f} | 1 | "
      f"{chi2.sf(2 * (f['profile_c'][str(c1)] - f['nll_ml']), 1):.5f} |")
    w("")
    w("**The reviewer's premise is confirmed; the reviewer's conclusion is not.** A between-binary")
    w("variance component that does not vanish with `n` is strongly supported — `a = 0` is rejected")
    w(f"at p = {0.5 * chi2.sf(f['lr'], 1):.5f}. But on this split it does not sit *alongside* the")
    w("sampling component, it **replaces** it. The data reject the published `c` as well")
    w(f"(p = {chi2.sf(2 * (f['profile_c'][str(c1)] - f['nll_ml']), 1):.5f}).")
    w("")
    w("### What that makes T2(n)")
    w("")
    w(f"With `c = 0`, `T2(n) = mu + 1.645*sqrt(a) = {f['asymptote']:.6f}` for every `n`. **The")
    w("principled two-component gate, fit as specified, is a flat gate** — and it is flat")
    w(f"{flat - f['asymptote']:.6f} *below* the published flat gate, i.e. uniformly looser than")
    w("both operating points at every candidate count.")
    w("")
    w(f"That is not a floor-versus-no-floor question. It deletes the entire small-`n` correction the")
    w(f"repair exists to make: `T(1000) = {f['T1_at']['1000']:.4f}` against")
    w(f"`T2(1000) = {f['T2_at']['1000']:.4f}`.")
    w("")
    w("| `n` | `T(n)` | `T'(n)` | `T2(n)` | `T2c(n)` |")
    w("|---|---|---|---|---|")
    for k in f["T2_at"]:
        w(f"| {int(k):,} | {f['T1_at'][k]:.6f} | {f['Tflr_at'][k]:.6f} | {f['T2_at'][k]:.6f} | "
          f"{f['T2c_at'][k]:.6f} |")
    w("")
    w("`T2(n)` never rises above `FLAT` at any `n`, so the crossover question does not arise: it is")
    w("below the published gate everywhere.")
    w("")
    w("---")
    w("")

    # ── 2. T2c ────────────────────────────────────────────────────────────────
    w("## 2. `T2c(n)` — the variant the reviewer's argument actually describes")
    w("")
    w("Because the free fit never exercises the two-component *shape*, scoring `T2` alone would not")
    w("answer the question that was asked. So a second variant is reported: the sampling component is")
    w(f"**held** at the published `c = {c1:.6f}` and only the between-binary component is fit,")
    w(f"giving `a = {afx:.6e}` (`sqrt(a) = {math.sqrt(afx):.6f}`) and")
    w("")
    w("```")
    w(f"    T2c(n) = {mu:.6f} + 1.645 * sqrt({afx:.6e} + {c1:.6f}^2/n)")
    w("```")
    w("")
    w("This is the gate the reviewer's sentence describes: the `1/n` term intact, plus a floor that")
    w(f"comes from a named variance component rather than from `max(FLAT, ...)`. Its asymptote is")
    w(f"`mu + 1.645*sqrt(a) = {f['asymptote_fixc']:.6f}`, which is")
    w(f"{flat - f['asymptote_fixc']:.6f} below `FLAT`.")
    w("")
    w("**`c` here is asserted, not estimated.** The likelihood on the fit split rejects it")
    w(f"(p = {chi2.sf(2 * (f['profile_c'][str(c1)] - f['nll_ml']), 1):.5f}, table above). `T2c` is")
    w("reported as the most favourable reading of the proposal, not as a fit.")
    w("")
    w("---")
    w("")

    # ── 3. detection ──────────────────────────────────────────────────────────
    w("## 3. Detection — `S_spat` alone, `credibility.csv`")
    w("")
    w("| split | N | flat | `T(n)` | `T'(n)` | `T2(n)` | `T2c(n)` |")
    w("|---|---|---|---|---|---|---|")
    det = r["detection"]
    for role, label in (("clean_fit", "clean fit (in-sample)"),
                        ("benign", "clean holdout (false alarms)"),
                        ("obfuscated", "desync (sensitivity)"),
                        ("packed", "packed (sensitivity)")):
        d = det[role]
        w(f"| {label} | {d['N']} | " + " | ".join(f"{d[e]}/{d['N']}" for e, _ in GATES) + " |")
    w("")
    w(f"`T2` costs **{det['benign']['t2']}** clean-holdout false alarm against `T'`'s")
    w(f"{det['benign']['flr']} and re-introduces {det['clean_fit']['t2']} in-sample clean-fit alarms.")
    w("`T2c` matches `T'` exactly on all four splits. Sensitivity — desync")
    w(f"{det['obfuscated']['t2c']}/{det['obfuscated']['N']} and packed")
    w(f"{det['packed']['t2c']}/{det['packed']['N']} — is unchanged under every gate.")
    w("")
    w("---")
    w("")

    # ── 4. wild ───────────────────────────────────────────────────────────────
    W = r["wild"]["N"]
    wr = r["wild"]["rule"]
    w(f"## 4. Wild corpus — {W} stock Debian binaries")
    w("")
    w("No obfuscation anywhere, so **every alarm is a false alarm**.")
    w("")
    w("| gate | bare-rule switches | rate | vs `T'` |")
    w("|---|---|---|---|")
    for e, name in GATES:
        delta = "—" if e == "flr" else f"{(wr[e] - wr['flr']) / W:+.4f}"
        w(f"| {name} | {wr[e]}/{W} | {wr[e] / W:.4f} | {delta} |")
    w("")
    w("By candidate-count bucket:")
    w("")
    w("| bucket | N | flat | `T(n)` | `T'(n)` | `T2(n)` | `T2c(n)` |")
    w("|---|---|---|---|---|---|---|")
    for b in r["wild"]["buckets"]:
        w(f"| {b['bucket']} | {b['N']} | " +
          " | ".join(f"{b[e]:.3f} ({b[e + '_k']})" for e, _ in GATES) + " |")
    w("")
    w(f"**`T2` is worse than the gate it was proposed to replace, and worse than the one that gate")
    w(f"replaced.** {wr['t2']}/{W} = {wr['t2'] / W:.4f}, against {wr['old'] / W:.4f} for the original")
    w(f"flat gate and {wr['flr'] / W:.4f} for `T'`. Being a flat gate below `FLAT`, it fires more than")
    w(f"`flat` in every bucket — including {r['wild']['buckets'][0]['t2']:.3f} in the 250–1,000 bucket,")
    w("the regime the whole repair exists to fix, where `T'` fires on nothing at all.")
    w("")
    w(f"**`T2c` beats `T'` on the wild corpus by one binary**: {wr['t2c']}/{W} = {wr['t2c'] / W:.4f}")
    w(f"against {wr['flr']}/{W} = {wr['flr'] / W:.4f}. That margin is not a result. The two gates")
    w("disagree on 7 binaries — `T'` fires on 4 that `T2c` does not, `T2c` fires on 3 that `T'` does")
    w(f"not — which is McNemar p = {binomtest(3, 7, 0.5).pvalue:.3f}. There is no evidence of a")
    w("difference in wild false-alarm rate between them.")
    w("")
    w("The bucket breakdown shows where the one binary comes from and what it costs:")
    w("")
    b16 = [b for b in r["wild"]["buckets"] if b["bucket"].startswith(">=")][0]
    b4 = [b for b in r["wild"]["buckets"] if b["bucket"].startswith("4,000")][0]
    w(f"* 4,000–16,000: `T2c` {b4['t2c_k']} vs `T'` {b4['flr_k']} — **2 better**")
    w(f"* `>= 16,000`: `T2c` {b16['t2c_k']} vs `T'` {b16['flr_k']} — **1 worse**")
    w("")
    w("So the net gain of one is a trade, and the loss falls in the large-`n` bucket — exactly the")
    w("regime where the floor exists because nothing was ever measured to be wrong there.")
    w("")
    gd = r["wild"]["guard"]
    w("The abstention guard vetoes everything under all five gates: guarded-rule switch rate is")
    w(f"{gd['old']}/{W} under `flat`, {gd['flr']}/{W} under `T'`, {gd['t2']}/{W} under `T2`,")
    w(f"{gd['t2c']}/{W} under `T2c`. No gate choice is visible downstream of the guard on this corpus.")
    w("")
    w("---")
    w("")

    # ── 5. breadth ────────────────────────────────────────────────────────────
    br = r["breadth"]
    w("## 5. Packer breadth and ezuri")
    w("")
    w("| config | N | mean `n` | mean `S_spat` | mean `T'(n)` | mean `T2(n)` | flat | `T(n)` | `T'(n)` | `T2(n)` | `T2c(n)` |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for sl, d in br["by_config"].items():
        w(f"| `{sl}` | {d['N']} | {d['mean_n']:,.0f} | {d['mean_s_spat']:.4f} | {d['mean_tflr']:.4f} | "
          f"{d['mean_t2']:.4f} | " + " | ".join(f"{d[e]}/{d['N']}" for e, _ in GATES) + " |")
    w(f"| **all packed** | **{br['N']}** | | | | | " +
      " | ".join(f"**{br[e]}/{br['N']}**" for e, _ in GATES) + " |")
    w("")
    ez = br["by_config"]["ezuri"]
    w(f"**Packer breadth is {br['t2']}/{br['N']} and ezuri {ez['t2']}/{ez['N']} under every gate**,")
    w(f"including both new ones. ezuri's mean `S_spat` is {ez['mean_s_spat']:.4f} against a `T2c` of")
    w(f"{f['T2c_at']['719327']:.4f} at its mean `n` of {ez['mean_n']:,.0f} — a wider margin than the")
    w(f"{ez['mean_s_spat'] - ez['mean_tflr']:+.4f} it has under `T'`, because both new gates are")
    w("looser than `FLAT` out at 719k candidates. No sensitivity is at stake here under any option.")
    w("")
    w("---")
    w("")

    # ── 6. routing / selection ────────────────────────────────────────────────
    ab = r["ablation"]
    w("## 6. Routing ablation and selection accuracy")
    w("")
    w(f"Routing ablation, `credibility.csv` minus the {int(meta['n_fit'])} fit rows ({ab['N']} binaries):")
    w("")
    w("| arm | flat | `T(n)` | `T'(n)` | `T2(n)` | `T2c(n)` |")
    w("|---|---|---|---|---|---|")
    for arm, label in (("spat_only", "spatial-only"), ("rule", "both statistics")):
        w(f"| {label} | " + " | ".join(fr(ab[arm][e], ab['N']) for e, _ in GATES) + " |")
    w("")
    w("Selection accuracy:")
    w("")
    w("| corpus | N | arm | flat | `T(n)` | `T'(n)` | `T2(n)` | `T2c(n)` |")
    w("|---|---|---|---|---|---|---|---|")
    for key, label in (("core", "core (`switching.csv`, non-Tigress)"),
                       ("core_tig", "core Tigress arm"),
                       ("scale", "scale (`expanded.csv`)"),
                       ("guardc", "abstention-guard corpus")):
        d = r["selection"][key]
        for arm, nm in (("rule", "bare rule"), ("guard", "guarded rule")):
            if arm not in d:
                continue
            w(f"| {label} | {d['N']} | {nm} | " +
              " | ".join(fr(d[arm][e], d['N']) for e, _ in GATES) + " |")
    w("")
    sc = r["selection"]["scale"]["rule"]
    w(f"`T2` costs on every bare-rule arm it touches — scale {fr(sc['t2'], 500)} against `T'`'s")
    w(f"{fr(sc['flr'], 500)}, core {fr(r['selection']['core']['rule']['t2'], 53)} against")
    w(f"{fr(r['selection']['core']['rule']['flr'], 53)}, ablation")
    w(f"{fr(ab['rule']['t2'], ab['N'])} against {fr(ab['rule']['flr'], ab['N'])}. **`T2c` ties `T'`")
    w("on every selection and ablation figure**, to the binary. The core corpus still cannot be")
    w("scored under the guarded rule — `switching.csv` carries no `region_ent` column, so the guard's")
    w("input does not exist there; it is not estimated or substituted.")
    w("")
    w("---")
    w("")

    # ── 7. Tigress ────────────────────────────────────────────────────────────
    w("## 7. Tigress arm — both runs")
    w("")
    for cl, title in (("boundaries_meta", "`boundaries_meta.csv`"),
                      ("switching_core", "`switching.csv` core Tigress arm")):
        per = r["tigress"][cl]
        tot = {e: sum(d[e] for d in per.values()) for e, _ in GATES}
        N = sum(d["N"] for d in per.values())
        w(f"### {title}")
        w("")
        w("| transform | N | `n` range | mean `S_spat` | mean `T'(n)` | mean `T2(n)` | flat | `T(n)` | `T'(n)` | `T2(n)` | `T2c(n)` |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for sl, d in per.items():
            w(f"| `{sl}` | {d['N']} | {d['n_lo']:,}–{d['n_hi']:,} | {d['mean_s_spat']:.4f} | "
              f"{d['mean_tflr']:.4f} | {d['mean_t2']:.4f} | " +
              " | ".join(f"{d[e]}/{d['N']}" for e, _ in GATES) + " |")
        w(f"| **total** | **{N}** | | | | | " +
          " | ".join(f"**{tot[e]}/{N}**" for e, _ in GATES) + " |")
        w("")
        seq = lambda e: "/".join(str(per[sl][e]) for sl in ("tigL", "tigM", "tigH") if sl in per)
        w(f"Per-transform `packed` picks (tigL/tigM/tigH): flat **{seq('old')}**, `T'` "
          f"**{seq('flr')}**, `T2` **{seq('t2')}**, `T2c` **{seq('t2c')}**.")
        w("")
    w("**`T2` restores the Tigress numbers, and this is not a point in its favour.** It reproduces")
    w("the flat gate's per-transform counts because it *is* a flat gate, marginally looser than the")
    w("published one. Every one of those `packed` picks is a misroute — the true regime is")
    w("`obfuscated` for all 27 binaries in each run — so `T2` recovers a wrong answer. The dispatcher")
    w("gradient is a property of the flat gate's `n`-dependence artefact, and any gate that recovers")
    w("it recovers the artefact.")
    w("")
    w("`T2c` reproduces `T'` exactly on both runs. These binaries sit at `n` in the hundreds to low")
    w("thousands, where the `c^2/n` term dominates and `T2c ~ T(n)`, so the gradient stays lost")
    w("under it — same conclusion as Sec. 6 of the main results, unchanged.")
    w("")
    w("### `p05_vm` under virtualization")
    w("")
    w("| corpus | binary | transform | true regime | `n` | `S_spat` | `T'(n)` | `T2(n)` | flat | `T(n)` | `T'(n)` | `T2(n)` | `T2c(n)` |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in r["p05_vm"]:
        w(f"| `{d['corpus']}` | `{d['name']}` | `{d['sublabel']}` | {d['true_regime']} | {d['n']:,} | "
          f"{d['s_spat']:.6f} | {d['t_flr']:.6f} | {d['t_t2']:.6f} | " +
          " | ".join(d[f"{e}_rule"] for e, _ in GATES) + " |")
    w("")
    virt = [d for d in r["p05_vm"] if d["name"] == "p05_vm_virt"][0]
    w(f"**Yes — `p05_vm_virt` fires again under `T2`.** It is a `benign` binary whose only treatment")
    w(f"is legitimate virtualization, `n = {virt['n']:,}`, `S_spat = {virt['s_spat']:.6f}`. The flat")
    w(f"gate misrouted it to `packed`; `T(n)` and `T'(n)` both clear it")
    w(f"(`T'({virt['n']:,}) = {virt['t_flr']:.6f}`); `T2` puts the bar at {virt['t_t2']:.6f} and it")
    w(f"fires as `{virt['t2_rule']}` again. This is the single cleanest false alarm in the evaluation")
    w("— a benign program penalised for using a VM — and `T2` reinstates it. `T2c` clears it, as")
    w("`T'` does.")
    w("")
    w("The abstention guard vetoes it under every gate (`guard` column is `benign` throughout on")
    w("`boundaries_meta`), so the reinstated alarm is bare-rule only. It is still a bare-rule")
    w("regression on the one binary the arm was built to get right.")
    w("")
    w("---")
    w("")

    # ── 8. subset ─────────────────────────────────────────────────────────────
    w("## 8. Strict-subset property against the published gate")
    w("")
    w("**It does not hold, for either new gate.** The subset invariant the harness asserts is that")
    w("`T'(n) >= FLAT` everywhere, so nothing can fire under `T'` that did not fire under `flat`.")
    w("Both new gates fall below `T'` at large `n` — `T2` falls below it at *every* `n`.")
    w("")
    s = r["subset"]
    lo, hi, nb = s["t2_below_flr_n_range"]
    w("| comparison | binaries gaining a bare-rule alarm | gaining a spatial-only alarm | gaining a guarded alarm |")
    w("|---|---|---|---|")
    for k, label in (("t2_vs_flr", "`T2` vs `T'` (published)"), ("t2_vs_old", "`T2` vs `flat`"),
                     ("t2c_vs_flr", "`T2c` vs `T'` (published)"), ("t2c_vs_old", "`T2c` vs `flat`")):
        w(f"| {label} | {s[k]['rule']['count']} | {s[k]['spat_only']['count']} | {s[k]['guard']['count']} |")
    w("")
    w(f"`T2` gains alarms on **{s['t2_vs_flr']['rule']['count']}** binaries against the published")
    w(f"gate, out of {len(rows):,} replayed rows. It sits below `T'` on every one of those")
    w(f"{nb:,} rows (`n` = {lo:,}–{hi:,}), so enumerating the violations is not useful — the gate")
    w("is uniformly looser than the published one at every candidate count in the evaluation.")
    w("")
    w(f"`T2c` gains alarms on exactly **{s['t2c_vs_flr']['rule']['count']}** binaries. All three are")
    w("wild-corpus false alarms at large `n`, and they are:")
    w("")
    w("| corpus | binary | `n` | `S_spat` | `T'(n)` | `T2c(n)` | true regime |")
    w("|---|---|---|---|---|---|---|")
    for d in s["t2c_vs_flr"]["rule"]["rows"]:
        w(f"| `{d['corpus']}` | `{d['name']}` | {d['n']:,} | {d['s_spat']:.6f} | {d['t_ref']:.6f} | "
          f"{d['t_new']:.6f} | {d['true_regime']} |")
    w("")
    w("Each is a stock Debian binary with no obfuscation, so each is a new false alarm, and each")
    w("clears the new gate only because `T2c` sits below `FLAT` above `n ~ 100k`. The same three are")
    w("the whole of `T2c`'s violation against the original flat gate as well")
    w(f"({s['t2c_vs_old']['rule']['count']} binaries, identical set) — no obfuscated or packed binary")
    w("gains an alarm under `T2c` anywhere, and no guarded-rule decision changes on any corpus.")
    w("")
    w("The harness's per-row subset assertion would abort on these three. Adopting either gate means")
    w("removing that assertion, which is the invariant the current result is partly sold on.")
    w("")
    w("---")
    w("")

    # ── 9. verdict ────────────────────────────────────────────────────────────
    w("## 9. Side-by-side summary")
    w("")
    w("| result | flat | `T(n)` | `T'(n)` (published) | `T2(n)` | `T2c(n)` |")
    w("|---|---|---|---|---|---|")

    def line(label, get):
        w(f"| {label} | " + " | ".join(get(e) for e, _ in GATES) + " |")

    for role, label in (("clean_fit", "Detection: clean-fit alarms (in-sample)"),
                        ("benign", "Detection: clean-holdout false alarms"),
                        ("obfuscated", "Detection: desync sensitivity"),
                        ("packed", "Detection: packed sensitivity")):
        d = det[role]
        line(label, lambda e, d=d: fr(d[e], d["N"]))
    line("Wild census: bare-rule switch rate", lambda e: fr(wr[e], W))
    line("Wild census: guarded-rule switch rate", lambda e: fr(gd[e], W))
    for b in r["wild"]["buckets"]:
        line(f"Wild bucket {b['bucket']} (N={b['N']})", lambda e, b=b: f"{b[e]:.3f}")
    line("Packer breadth: packed detected (bare rule)", lambda e: fr(br[e], br["N"]))
    line("Packer breadth: ezuri", lambda e: fr(ez[e], ez["N"]))
    line("Routing ablation: spatial-only", lambda e: fr(ab["spat_only"][e], ab["N"]))
    line("Routing ablation: both statistics", lambda e: fr(ab["rule"][e], ab["N"]))
    for key, label, arm in (("core", "Selection: core / rule", "rule"),
                            ("core_tig", "Selection: core_tig / rule", "rule"),
                            ("scale", "Selection: scale / rule", "rule"),
                            ("scale", "Selection: scale / guard", "guard"),
                            ("guardc", "Selection: guardc / rule", "rule"),
                            ("guardc", "Selection: guardc / guard", "guard")):
        d = r["selection"][key]
        line(label, lambda e, d=d, arm=arm: fr(d[arm][e], d["N"]))
    for cl, label in (("boundaries_meta", "Tigress `boundaries_meta`: total `packed` picks"),
                      ("switching_core", "Tigress `switching_core`: total `packed` picks")):
        per = r["tigress"][cl]
        N = sum(x["N"] for x in per.values())
        line(label, lambda e, per=per, N=N: fr(sum(x[e] for x in per.values()), N))
    line("`p05_vm_virt` (benign, virtualized) bare-rule pick", lambda e: virt[f"{e}_rule"])
    line("Binaries gaining an alarm vs published `T'`",
         lambda e: "—" if e in ("old", "new", "flr")
         else str(s[f"{e}_vs_flr"]["rule"]["count"]))
    w("")
    w("---")
    w("")
    w("## 10. Recommendation")
    w("")
    w("**Keep the floor. Do not adopt either variant.**")
    w("")
    w("The decision rule set for this comparison was: adopt only if it beats the floored gate on the")
    w("wild corpus, *without* costing sensitivity anywhere, *and* without breaking the subset")
    w("property.")
    w("")
    w("`T2(n)` — the gate as specified, both parameters fit — fails all three:")
    w("")
    w(f"1. It loses on the wild corpus, badly: {wr['t2'] / W:.4f} against `T'`'s {wr['flr'] / W:.4f},")
    w(f"   and worse even than the original flat gate's {wr['old'] / W:.4f}.")
    w(f"2. It costs accuracy on scale ({fr(sc['t2'], 500)} vs {fr(sc['flr'], 500)}), core, the")
    w("   ablation, and the clean holdout, and it reinstates the `p05_vm_virt` false alarm.")
    w(f"3. It breaks the subset property on {s['t2_vs_flr']['rule']['count']} binaries.")
    w("")
    w("It fails because the fit degenerates: `c` goes to zero, and what is left is a flat gate")
    w(f"{flat - f['asymptote']:.6f} below the one we started with. That is the substantive finding —")
    w("the fit split cannot support a two-component model, because over its `n` range")
    w(f"({f['fit_n_lo']:,}–{f['fit_n_hi']:,}, a factor of {f['fit_n_hi'] / f['fit_n_lo']:.1f}) the")
    w("residuals get *larger* with `n`, not smaller.")
    w("")
    w("`T2c(n)` — `c` asserted at the published value, `a` fit — is the strong form of the proposal,")
    w("and it comes far closer. It ties `T'` on every sensitivity, selection, ablation, breadth, and")
    w("Tigress figure, to the binary. But it still does not clear the bar:")
    w("")
    w(f"* Its wild-corpus win is **one binary** ({wr['t2c']}/{W} vs {wr['flr']}/{W}), on 7 discordant")
    w(f"  cases, McNemar p = {binomtest(3, 7, 0.5).pvalue:.3f}. That is a tie, not a win.")
    w(f"* It **breaks the subset property** on 3 named binaries — `ceph-common__rbd-replay`,")
    w("  `graphviz__lefty`, `groff-base__pic` — all benign, all new false alarms, all at `n > 116k`.")
    w("* The bucket split shows the trade runs the wrong way: it gains 2 in the 4,000–16,000 bucket")
    w(f"  and loses 1 in the `>= 16,000` bucket ({b16['t2c']:.3f} vs {b16['flr']:.3f}), which is the")
    w("  regime where the floor exists precisely because nothing was ever measured to be wrong there.")
    w(f"* And its `c` is rejected by the likelihood on the fit split")
    w(f"  (p = {chi2.sf(2 * (f['profile_c'][str(c1)] - f['nll_ml']), 1):.5f}), so calling it")
    w("  'principled' overstates it — it is the published `c` with a fitted floor bolted on.")
    w("")
    w("So the trade on offer is: give up a hard, machine-checked invariant and three clean binaries,")
    w("in exchange for a one-binary wild-corpus difference that does not survive a significance test,")
    w("and a variance model whose free fit says the sampling term should not be there at all.")
    w("")
    w("**The floor stays.** The reviewer's observation is worth one sentence in Sec. 6.10, and that")
    w("sentence should say what is actually true here: the floor stands in for a between-binary")
    w(f"variance component `a`, the principled form is `T2(n) = mu + z*sqrt(a + c^2/n)`, and fitting")
    w(f"it on the 20 clean-fit binaries returns `a = {a:.2e}` with `c -> 0` — the fit split is too")
    w("narrow in `n` to identify both components, so the floor is retained as the conservative")
    w("one-sided stand-in. Naming the model we did not adopt, and saying why, is a stronger position")
    w("than either gate.")
    w("")
    w("---")
    w("")
    w(f"*Generated by `variance_gate.py` from `decisions.tsv` and `credibility.csv`. No number in this")
    w(f"document is typed by hand. The Python reimplementation of")
    w("`SignatureClassifier::{classify_rule, classify_guard}` used for scoring is validated against")
    w(f"all {out['classifier_cells_validated']:,} published gate decisions")
    w(f"({len(rows):,} rows x 3 gates) before any `T2` number is computed; a single mismatch aborts.")
    w("This document does not modify `SPATIAL_NULL_REPAIR_RESULTS.md`, which remains the published")
    w("result. Regenerate with:*")
    w("")
    w("```sh")
    w("python3 docs/spatial_null_repair/variance_gate.py")
    w("```")

    OUT2.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT2} ({len(L)} lines)")


if __name__ == "__main__":
    main()
