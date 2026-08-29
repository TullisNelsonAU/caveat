#!/usr/bin/env python3
"""Arm A — the active-analysis PAYOFF (INTERACTIVE_APP_SPEC §1): turn the +243 demo into a result.

Task: "reach X% of the true instructions confirmed at P̂≥0.9 in as few analyst queries as possible."
A query = confirm/deny one candidate function head under a truthful symbol oracle (confirm a real head
⇒ inject q; deny a decoy ⇒ no positive evidence, a spent-but-wasted query). We compare four query
orderings on TWO corpora:

  * orderings: eig (F_h·ΔH, the calibrated value-of-information objective) vs three naive baselines —
    lowf (confirm the least-sure in-band head), highf (the most-sure), addr (arbitrary = lowest addr).
  * corpora: `real` = the stock code-in-data corpus (/tmp/cid, one small decoy block); `decoy-heavy`
    = /tmp/cid_heavy (decoy mass 1.5× .text, so the uncertain band is decoy-dominated).

The committing-tool baseline: a tool that commits to one disassembly has no belief to propagate a
confirmation through and no F_h to rank candidates by — the best it can do is an uninformed order
(= addr) and a hard re-decode per evidence step. So addr is the committing-tool proxy; EIG's margin
over it is the value of carrying calibrated uncertainty. (addr also gets a layout tailwind here: the
real .text precedes the appended decoy region, so lowest-address-first hits real heads early — we note
this, and EIG still dominates it.)

Payoff metrics: the effort curve (true instrs at P̂≥0.9 vs queries issued), queries-to-X%, and the
normalized area between EIG and the best naive ordering. Honesty wall (π frozen) and ECE<ε are asserted
at every step. EIG runs use the --query-cap shortlist; the naive orderings are cheap.

Usage:  udstack_active_payoff.py [--k 15] [--cap 16] [--out json]
"""
import argparse, glob, json, os, subprocess

R = os.path.expanduser("~/lab/projects/upd-suite-stack/target/release/udstack")
CORPORA = [("real", "/tmp/cid"), ("decoy-heavy", "/tmp/cid_heavy")]
STRATS = ["eig", "lowf", "highf", "addr"]
XPCTS = [0.30, 0.40, 0.50]
ECE_EPS = 0.05
COLS = ["strategy", "step", "head", "real", "f_prior", "eig", "entropy", "ece", "auroc",
        "pi_ece", "pi_auroc", "hi_mass", "mean_phat", "tp9", "fp9"]


def decoy_from(stem):
    for l in open(stem + ".regions"):
        if "junk_decoy" in l:
            return l.split()[0]
    return None


def gt_count(stem):
    return len(open(stem + ".gt").read().split())


def cmd(elf, strat, k, cap):
    stem = elf[:-4]
    c = [R, elf, stem + ".gt", "--func-gt", stem + ".func.gt", "--decoy-from", decoy_from(stem),
         "--milestone", "b", "--lambda", "0.5", "--active", f"{strat}:{k}"]
    if strat == "eig":
        c += ["--query-cap", str(cap)]
    return c


def parse(stdout):
    steps = []
    for ln in stdout.splitlines():
        p = ln.split(",")
        if p[0] == "stack_active" and p[1] != "strategy":
            d = dict(zip(COLS, p[1:]))
            steps.append({k: (v if k in ("strategy", "head") else float(v)) for k, v in d.items()})
    return steps


def queries_to(steps, target):
    """First query index whose tp9 ≥ target, else None (censored > K)."""
    for s in steps:
        if s["tp9"] >= target:
            return int(s["step"])
    return None


def area(steps, k):
    """Normalized area under the tp9-recovery curve: mean over queries of (tp9 - tp9_0)."""
    if len(steps) < 2:
        return 0.0
    base = steps[0]["tp9"]
    return sum(s["tp9"] - base for s in steps[1:]) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--cap", type=int, default=16)
    ap.add_argument("--out")
    a = ap.parse_args()

    # Launch every (corpus, specimen, strategy) run concurrently.
    jobs, meta = {}, {}
    for cname, cdir in CORPORA:
        for elf in sorted(glob.glob(os.path.join(cdir, "*__native-code-in-data.elf"))):
            stem = elf[:-4]
            spec = os.path.basename(stem).split("__")[0].replace("gcc_coreutils_64_O2_", "")
            meta[(cname, spec)] = dict(gt=gt_count(stem))
            for s in STRATS:
                jobs[(cname, spec, s)] = subprocess.Popen(
                    cmd(elf, s, a.k, a.cap), stdout=subprocess.PIPE, text=True)
    res = {key: parse(p.communicate()[0]) for key, p in jobs.items()}

    specs = sorted({(c, s) for (c, s, _) in res})
    out = {"k": a.k, "cap": a.cap, "corpora": {}}

    # ── honesty wall + ECE assertions across every run ──
    pi_ok, ece_ok, ece_max = True, True, 0.0
    for (c, spec, s), st in res.items():
        if not st:
            continue
        pe0, pa0 = st[0]["pi_ece"], st[0]["pi_auroc"]
        if any(abs(x["pi_ece"] - pe0) > 1e-9 or abs(x["pi_auroc"] - pa0) > 1e-9 for x in st):
            pi_ok = False
        m = max(x["ece"] for x in st)
        ece_max = max(ece_max, m)
        if m > ECE_EPS:
            ece_ok = False

    for cname, _ in CORPORA:
        print(f"\n{'='*78}\n=== corpus: {cname} — effort curve (true instrs at P̂≥0.9 vs queries) ===\n{'='*78}")
        rows = {}
        for (c, spec) in specs:
            if c != cname:
                continue
            g = meta[(c, spec)]["gt"]
            e = res.get((c, spec, "eig"), [])
            if not e:
                continue
            print(f"\n[{spec}]  |GT|={g}   base TP@.9={int(e[0]['tp9'])}")
            print(f"    {'query':>5}" + "".join(f"{s:>9}" for s in STRATS) + "     (TP@.9)")
            for i in range(a.k + 1):
                line = f"    {i:>5}"
                for s in STRATS:
                    st = res.get((c, spec, s), [])
                    v = int(st[i]["tp9"]) if i < len(st) else None
                    line += f"{(v if v is not None else '-'):>9}"
                print(line)
            # per-strategy summary
            srow = {}
            for s in STRATS:
                st = res.get((c, spec, s), [])
                if not st:
                    continue
                reals = sum(1 for x in st[1:] if x["real"] >= 0.5)
                srow[s] = dict(
                    tp_gain=int(st[-1]["tp9"] - st[0]["tp9"]),
                    reals=reals, area=round(area(st, a.k), 1),
                    q30=queries_to(st, XPCTS[0] * g), q40=queries_to(st, XPCTS[1] * g),
                    q50=queries_to(st, XPCTS[2] * g), ece_max=round(max(x["ece"] for x in st), 4))
            rows[spec] = srow
            print(f"    {'strat':>6} {'TP@.9 gain':>11} {'real/K':>7} {'area':>8} "
                  f"{'q→30%':>6} {'q→40%':>6} {'q→50%':>6} {'ECEmax':>7}")
            for s in STRATS:
                if s not in srow:
                    continue
                r = srow[s]
                fmt = lambda q: (str(q) if q is not None else f">{a.k}")
                print(f"    {s:>6} {r['tp_gain']:>11} {r['reals']:>4}/{a.k:<2} {r['area']:>8} "
                      f"{fmt(r['q30']):>6} {fmt(r['q40']):>6} {fmt(r['q50']):>6} {r['ece_max']:>7.4f}")
        out["corpora"][cname] = rows

    # ── aggregate: EIG vs best-naive, per corpus ──
    print(f"\n{'='*78}\n=== AGGREGATE: EIG vs best naive ordering (mean over specimens) ===\n{'='*78}")
    for cname, _ in CORPORA:
        rows = out["corpora"].get(cname, {})
        if not rows:
            continue
        agg = {s: dict(tp=[], area=[], q40=[]) for s in STRATS}
        for spec, srow in rows.items():
            for s in STRATS:
                if s in srow:
                    agg[s]["tp"].append(srow[s]["tp_gain"])
                    agg[s]["area"].append(srow[s]["area"])
                    agg[s]["q40"].append(srow[s]["q40"])
        print(f"\n  [{cname}]")
        print(f"    {'strat':>6} {'mean TP@.9 gain':>15} {'mean area':>10} {'q→40% (reached/total)':>24}")
        for s in STRATS:
            tp, ar, q = agg[s]["tp"], agg[s]["area"], agg[s]["q40"]
            if not tp:
                continue
            reached = [x for x in q if x is not None]
            qstr = f"{(sum(reached)/len(reached)):.1f} ({len(reached)}/{len(q)})" if reached else f"none/{len(q)}"
            print(f"    {s:>6} {sum(tp)/len(tp):>15.0f} {sum(ar)/len(ar):>10.1f} {qstr:>24}")
        eig_area = sum(agg["eig"]["area"]) / max(1, len(agg["eig"]["area"]))
        naive_best = max((sum(agg[s]["area"]) / max(1, len(agg[s]["area"])) for s in ("lowf", "highf", "addr")), default=0)
        if naive_best > 0:
            print(f"    → EIG recovers {eig_area / naive_best:.1f}× the confidence-per-query of the best naive ordering")

    print(f"\n{'='*78}")
    print(f"HONESTY WALL: π ECE/AUROC frozen across all queries on every run = {pi_ok}")
    print(f"CALIBRATION: max ECE across all runs/steps = {ece_max:.4f}  (< ε={ECE_EPS}? {ece_ok})")
    assert pi_ok, "HONESTY WALL VIOLATED — π moved during active querying"
    assert ece_ok, f"ECE exceeded ε={ECE_EPS} during active querying (max {ece_max:.4f})"
    print("Both asserted. ✓")

    if a.out:
        out["honesty_wall"] = pi_ok
        out["ece_max"] = ece_max
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
