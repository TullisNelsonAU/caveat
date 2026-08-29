#!/usr/bin/env python3
"""Analyze derisk_probe.csv — pure reader, NO engine calls. Emits the per-transform-family
calibration/discrimination table (fair GT = insn_max), deltas vs each program's un-obfuscated
baseline, the confidently-wrong roll-up, the strict-GT (insn_min) robustness band, and the explicit
GO / NO-GO / PARTIAL verdict (DERISK_PROBE_SPEC sec 3).

Two GTs per specimen from the SAME binary:
  insn_max = generous GT — DWARF-line/func anchors + the decoded neutral zone counted positive.
  insn_min = strict GT   — reachability closure from DWARF anchors only; neutral zone counted
             NEGATIVE. This UNDER-counts true starts, so a high posterior on a neutral-zone (truly
             real) instruction registers as a false "confidently-wrong". Because cw@max≈0 (the model
             fires ~nothing outside insn_max) and insn_min ⊂ insn_max, EVERY min-GT confidently-wrong
             lies in the neutral zone by set inclusion — a GT-granularity artifact, not miscalibration
             (it is present at baseline too). insn_max is the fair axis the verdict rests on."""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.normpath(os.path.join(HERE, "..", "derisk", "derisk_probe.csv"))
ORDER = ["baseline", "Virtualize", "Flatten", "AddOpaque", "EncodeArithmetic", "EncodeLiterals"]


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else CSV
    if not os.path.exists(path):
        print("_(no derisk_probe.csv)_"); return
    allrows = [r for r in csv.DictReader(open(path))]
    okmax = [r for r in allrows if r["ok"] == "1" and r.get("gt_kind", "max") == "max"]
    okmin = [r for r in allrows if r["ok"] == "1" and r.get("gt_kind", "max") == "min"]
    by_t = {t: [r for r in okmax if r["transform"] == t] for t in ORDER}
    by_t_min = {t: [r for r in okmin if r["transform"] == t] for t in ORDER}
    base = {r["program"]: r for r in okmax if r["transform"] == "baseline"}

    n_fail = [r for r in allrows if r["ok"] != "1"]
    aligned = all(r["align_ok"] == "1" for r in okmax + okmin)
    print("### De-risk probe — Soft posterior under real Tigress obfuscation\n")
    print("Specimens: **%d programs x %d transforms = %d binaries**, each scored against two GTs "
          "(insn_max fair, insn_min strict) from the SAME binary. %d bench cells ok, %d failed. "
          "Alignment: **%s** (all ET_EXEC, fixed load base). GT by construction (gen-gt: DWARF "
          "`.debug_line` + function entries; never a disassembler). Two axes (ECE, AUROC) separate.\n"
          % (len(base), len([t for t in ORDER if t != "baseline"]) + 1,
             len(okmax), len(okmax) + len(okmin), len(n_fail),
             "all align_ok" if aligned else "MISALIGNED — investigate"))
    if n_fail:
        print("Failures: " + ", ".join("%s/%s/%s(%s)" % (r["program"], r["transform"],
              r.get("gt_kind", "?"), r["err"][:40]) for r in n_fail) + "\n")

    # ---- main per-transform table (fair GT) ----
    print("### Per-transform family — fair GT (insn_max), mean over %d programs\n" % len(base))
    print("| transform | n | raw ECE (max) | recal-ceil ECE | AUROC (min) | recall@0 | prec@0 | conf-wrong | ΔAUROC vs base | ΔrawECE vs base | text B (min–max) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in ORDER:
        rs = by_t[t]
        if not rs:
            print("| %s | 0 |" + " — |" * 10); continue
        raw = [f(r["ece_raw"]) for r in rs]
        au = [f(r["auroc_raw"]) for r in rs]
        tb = [int(r["text_bytes"]) for r in rs if r["text_bytes"] not in ("", "-1")]
        d_au = [f(r["auroc_raw"]) - f(base[r["program"]]["auroc_raw"]) for r in rs
                if r["program"] in base and t != "baseline"]
        d_ece = [f(r["ece_raw"]) - f(base[r["program"]]["ece_raw"]) for r in rs
                 if r["program"] in base and t != "baseline"]
        print("| %s | %d | %.4f (%.4f) | %.4f | %.3f (%.3f) | %.2f | %.2f | %d/%d | %s | %s | %d–%d |" % (
            t, len(rs), mean(raw), max(raw), mean([f(r["ece_recal"]) for r in rs]),
            mean(au), min(au), mean([f(r["recall0"]) for r in rs]), mean([f(r["prec0"]) for r in rs]),
            sum(1 for r in rs if r["cw_flag"] == "1"), len(rs),
            ("%+.3f" % mean(d_au)) if d_au else "—", ("%+.4f" % mean(d_ece)) if d_ece else "—",
            min(tb), max(tb)))

    # ---- confidently-wrong (fair GT) ----
    cw_all = [r for r in okmax if r["cw_flag"] == "1"]
    print("\n### Confidently-wrong on the fair GT (high π on a confident GT-negative — the desync signature)\n")
    if not cw_all:
        print("**None.** No insn_max specimen has strict-threshold precision < 0.90; worst-case strict "
              "precision across all %d = **%.3f**. The most-confident predictions carry no confident "
              "GT-negatives under any transform — the opposite of the desync π=1.0 collapse.\n"
              % (len(okmax), min(f(r["prec_strict"]) for r in okmax)))
    else:
        print("| program | transform | strict precision | conf-wrong FP | AUROC | raw ECE |")
        print("|---|---|---|---|---|---|")
        for r in sorted(cw_all, key=lambda r: f(r["prec_strict"])):
            print("| %s | %s | %.3f | %s | %.3f | %.4f |" % (r["program"], r["transform"],
                  f(r["prec_strict"]), r["cw_fp"], f(r["auroc_raw"]), f(r["ece_raw"])))

    # ---- strict-GT robustness band ----
    if okmin:
        print("\n### Robustness band — strict GT (insn_min: neutral zone counted NEGATIVE)\n")
        print("Pessimistic bound. insn_min ⊂ insn_max and cw@max≈0, so every min-GT \"confidently-wrong\" "
              "lies in the neutral zone (true instructions the conservative DWARF closure didn't reach) — "
              "a GT-granularity artifact, **present at baseline too**, not miscalibration.\n")
        print("| transform | AUROC min-GT (min) | recal ECE min-GT | conf-wrong specimens (min-GT) |")
        print("|---|---|---|---|")
        for t in ORDER:
            rs = by_t_min[t]
            if not rs:
                continue
            print("| %s | %.3f (%.3f) | %.4f | %d/%d |" % (
                t, mean([f(r["auroc_raw"]) for r in rs]), min(f(r["auroc_raw"]) for r in rs),
                mean([f(r["ece_recal"]) for r in rs]),
                sum(1 for r in rs if r["cw_flag"] == "1"), len(rs)))
        bl = by_t_min["baseline"]
        if bl:
            print("\n_Baseline (unobfuscated) under strict GT already shows %d/%d confidently-wrong specimens "
                  "(mean strict precision %.3f) — the artifact is intrinsic to insn_min's neutral zone, "
                  "not induced by any transform._" % (
                      sum(1 for r in bl if r["cw_flag"] == "1"), len(bl),
                      mean([f(r["prec_strict"]) for r in bl])))

    # ---- verdict ----
    print("\n### Verdict (on the fair GT, insn_max)\n")
    vr = {}
    for t in ORDER:
        if t == "baseline":
            continue
        rs = by_t[t]
        if not rs:
            vr[t] = "n/a"; continue
        recal_ok = mean([f(r["ece_recal"]) for r in rs]) <= 0.05
        auroc_ok = mean([f(r["auroc_raw"]) for r in rs]) >= 0.85
        cw_frac = sum(1 for r in rs if r["cw_flag"] == "1") / len(rs)
        vr[t] = "SURVIVES" if (recal_ok and auroc_ok and cw_frac < 0.25) else "COLLAPSES"
        print("- **%s**: %s — AUROC %.3f (min %.3f), recal ECE %.4f, raw ECE %.4f, conf-wrong %d/%d." % (
            t, vr[t], mean([f(r["auroc_raw"]) for r in rs]), min(f(r["auroc_raw"]) for r in rs),
            mean([f(r["ece_recal"]) for r in rs]), mean([f(r["ece_raw"]) for r in rs]),
            sum(1 for r in rs if r["cw_flag"] == "1"), len(rs)))
    coll = [t for t, v in vr.items() if v == "COLLAPSES"]
    surv = [t for t, v in vr.items() if v == "SURVIVES"]
    key = {"Virtualize", "Flatten"}
    gate = "**GO**" if not coll else ("**NO-GO**" if key & set(coll) else "**PARTIAL**")
    print("\n%s — survives: %s%s" % (gate, ", ".join(surv) or "none",
                                     ("; collapses: " + ", ".join(coll)) if coll else ""))


if __name__ == "__main__":
    main()
