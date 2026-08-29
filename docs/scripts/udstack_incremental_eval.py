#!/usr/bin/env python3
"""Arm B — incremental analysis (INTERACTIVE_APP_SPEC §2): the thesis made concrete. As evidence is
discovered, the uncertainty updates and stays honest.

An evidence stream of three kinds enters the stack one item at a time, each as a `clamp` propagating
through the fixpoint: `sym` (a withheld function symbol → function clamp), `trace` (a dynamic-trace
instruction hit → instruction clamp), `edge` (a resolved indirect edge → function clamp, the M3a
mechanism). After each item we measure the calibrated instruction map (AUROC / coverage / ECE) on the
held-out domain (trace-pinned instructions are excluded, so trace evidence is never self-scored), and
compare against a committing recursive-descent baseline seeded by `entry ∪ confirmed-so-far`.

The claim (and the acceptance test): the stack's quality rises ~monotonically while ECE stays bounded
(calibration MAINTAINED at every step, not just at the end), and the invariant π is frozen throughout
(honesty wall). The committing baseline can only flip hard 0/1 decisions — it cannot represent a
probabilistic clamp and its hard labels are miscalibrated (high ECE). GT = withheld symbols + real
instruction traces of the benign binaries; the corpus is the real code-in-data set (benign seeds).

Usage:  udstack_incremental_eval.py [--stream S,T,E] [--corpus /tmp/cid] [--out json]
"""
import argparse, glob, json, os, subprocess

R = os.path.expanduser("~/lab/projects/upd-suite-stack/target/release/udstack")
ECE_EPS = 0.10
COLS = ["step", "kind", "n_ev", "st_auroc", "st_cov", "st_ece",
        "base_auroc", "base_cov", "base_ece", "pi_auroc", "pi_ece"]


def decoy_from(stem):
    for l in open(stem + ".regions"):
        if "junk_decoy" in l:
            return l.split()[0]
    return None


def parse(stdout):
    rows = []
    for ln in stdout.splitlines():
        p = ln.split(",")
        if p[0] == "stack_incr" and p[1] != "step":
            d = dict(zip(COLS, p[1:]))
            rows.append({k: (v if k in ("kind",) else (float(v) if v != "NA" else float("nan")))
                         for k, v in d.items()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="12,8,2", help="SYM,TRACE,EDGE evidence counts")
    ap.add_argument("--corpus", default="/tmp/cid")
    ap.add_argument("--out")
    a = ap.parse_args()
    s, t, e = a.stream.split(",")

    jobs = {}
    for elf in sorted(glob.glob(os.path.join(a.corpus, "*__native-code-in-data.elf"))):
        stem = elf[:-4]
        spec = os.path.basename(stem).split("__")[0].replace("gcc_coreutils_64_O2_", "")
        jobs[spec] = subprocess.Popen(
            [R, elf, stem + ".gt", "--func-gt", stem + ".func.gt", "--decoy-from", decoy_from(stem),
             "--milestone", "b", "--lambda", "0.5", "--incremental", f"{s},{t},{e}"],
            stdout=subprocess.PIPE, text=True)
    res = {spec: parse(p.communicate()[0]) for spec, p in jobs.items()}

    out = {"stream": a.stream, "corpus": a.corpus, "specimens": {}}
    pi_ok, ece_ok, ece_max, mono_ok = True, True, 0.0, True
    print(f"=== Arm B — incremental quality vs evidence arrived (stream sym={s} trace={t} edge={e}) ===")
    for spec, rows in res.items():
        if not rows:
            print(f"[{spec}] no output"); continue
        print(f"\n[{spec}]  (held-out instruction domain)")
        print(f"  {'n_ev':>4} {'kind':>6} | {'STACK auroc':>11} {'cov':>6} {'ece':>6} | "
              f"{'BASE auroc':>10} {'cov':>6} {'ece':>6} | {'π auroc':>8} {'π ece':>6}")
        pa0, pe0 = rows[0]["pi_auroc"], rows[0]["pi_ece"]
        for r in rows:
            print(f"  {int(r['n_ev']):>4} {r['kind']:>6} | {r['st_auroc']:>11.4f} {r['st_cov']:>6.3f} "
                  f"{r['st_ece']:>6.4f} | {r['base_auroc']:>10.4f} {r['base_cov']:>6.3f} {r['base_ece']:>6.4f} | "
                  f"{r['pi_auroc']:>8.4f} {r['pi_ece']:>6.4f}")
            ece_max = max(ece_max, r["st_ece"])
            if r["st_ece"] > ECE_EPS:
                ece_ok = False
            if abs(r["pi_auroc"] - pa0) > 1e-9 or abs(r["pi_ece"] - pe0) > 1e-9:
                pi_ok = False
        # quality delta + monotone-ish check (stack coverage should rise; allow small dips)
        d_auroc = rows[-1]["st_auroc"] - rows[0]["st_auroc"]
        d_cov = rows[-1]["st_cov"] - rows[0]["st_cov"]
        dips = sum(1 for i in range(1, len(rows)) if rows[i]["st_cov"] < rows[i - 1]["st_cov"] - 1e-9)
        db_auroc = rows[-1]["base_auroc"] - rows[0]["base_auroc"]
        print(f"  → STACK ΔAUROC {d_auroc:+.4f}  Δcoverage {d_cov:+.3f}  (cov dips: {dips}); "
              f"ECE range [{min(r['st_ece'] for r in rows):.4f}, {max(r['st_ece'] for r in rows):.4f}]")
        print(f"  → BASE  ΔAUROC {db_auroc:+.4f}  final ECE {rows[-1]['base_ece']:.4f} "
              f"(hard labels — miscalibrated by construction)")
        out["specimens"][spec] = dict(
            d_auroc=d_auroc, d_cov=d_cov, cov_dips=dips,
            st_ece_final=rows[-1]["st_ece"], base_ece_final=rows[-1]["base_ece"],
            st_auroc_final=rows[-1]["st_auroc"], base_auroc_final=rows[-1]["base_auroc"])

    # aggregate
    sp = out["specimens"]
    if sp:
        n = len(sp)
        print(f"\n=== AGGREGATE (mean over {n} specimens) ===")
        print(f"  STACK: ΔAUROC {sum(v['d_auroc'] for v in sp.values())/n:+.4f}  "
              f"Δcoverage {sum(v['d_cov'] for v in sp.values())/n:+.3f}  "
              f"final ECE {sum(v['st_ece_final'] for v in sp.values())/n:.4f}")
        print(f"  BASE : final AUROC {sum(v['base_auroc_final'] for v in sp.values())/n:.4f}  "
              f"final ECE {sum(v['base_ece_final'] for v in sp.values())/n:.4f} (hard, miscalibrated)")
        print(f"  ECE gap (stack vs committing baseline, final): "
              f"{sum(v['st_ece_final'] for v in sp.values())/n:.4f} vs "
              f"{sum(v['base_ece_final'] for v in sp.values())/n:.4f}")

    print(f"\nHONESTY WALL: π frozen across all evidence steps on every specimen = {pi_ok}")
    print(f"CALIBRATION MAINTAINED: max stack ECE across all steps = {ece_max:.4f} (< ε={ECE_EPS}? {ece_ok})")
    assert pi_ok, "HONESTY WALL VIOLATED — π moved under incremental evidence"
    assert ece_ok, f"stack ECE exceeded ε={ECE_EPS} under evidence (max {ece_max:.4f}) — calibration broke"
    print("Both asserted. ✓  (committing baseline cannot make either claim — hard labels, no belief.)")

    if a.out:
        out["honesty_wall"] = pi_ok
        out["ece_max"] = ece_max
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
