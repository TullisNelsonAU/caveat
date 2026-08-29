#!/usr/bin/env python3
"""Sub-experiment 2c (SPEC sec 2c): Paper 3's EIG ordering vs. the certified greedy rule.

Paper 3 ranks active queries by  EIG = F_h . dH  (the head whose confirmation is expected to remove
the most instruction-map entropy). The (1-1/e) submodularity theorem instead certifies GREEDY ON MAX
CURRENT CONDITIONAL ENTROPY  H(X_o | E, X_queried)  -- for a function-head query the object's
conditional entropy is  h(F_o), maximised by the head whose reachedness posterior F_o is nearest 0.5.

We compare the two rules on the decoy-heavy corpus:
  * EIG      : engine-native, `udstack --active eig:K` (uses Stack::rank_queries = F_h.dH). Read-only.
  * CERTIFIED: harness-driven. Reconstruct each head's F_o from a `--active lowf:BIG` sweep, order by
               max h(F_o) (|F-0.5| ascending), and realise the effort curve by applying that order as
               sequential oracle-truthful `--clamp-func` (a real head is clamped q=1; a decoy query
               injects nothing -- identical honesty rule to the engine's active loop).
Reported: (a) selection agreement -- do the two rules pick the same heads / same order? (b) effort
curves -- recovered true instructions at Phat>=0.9 vs #queries. Agreement validates the heuristic
against the guarantee; a gap is a real finding (SPEC sec 2c, sec 4).

Read-only on the engine. One udstack process at a time (K+2 processes per specimen); no corpus held.

Usage: staircase_eig.py [--k 8] [--out docs/staircase/eig_vs_greedy.csv]
"""
import argparse, csv, glob, json, math, os, subprocess, sys

ROOT = os.path.expanduser("~/lab/projects")
UDSTACK = os.path.join(ROOT, "upd-suite-stack/target/release/udstack")
DECOY = os.path.join(ROOT, "upd-suite-sota/scratch/decoy-smoke")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import staircase_measure as S   # reuse func_gt_for / read_gt / read_regions


def run_active(elf, gt, func_gt, strat, k, clamps=None):
    cmd = [UDSTACK, elf, gt, "--func-gt", func_gt, "--active", "%s:%d" % (strat, k)]
    for a in (clamps or []):
        cmd += ["--clamp-func", "0x%x:1.0" % a]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    steps = []
    for ln in p.stdout.splitlines():
        if not ln.startswith("stack_active,"):
            continue
        f = ln.split(",")
        if f[1] == "strategy":            # header
            continue
        # f: stack_active,strategy,step,head,real,f_prior,eig,entropy,ece,auroc,pi_ece,pi_auroc,
        #    hi_mass,mean_phat,tp_at_0.9,fp_at_0.9  -> tp9=f[14], fp9=f[15]
        try:
            steps.append(dict(step=int(f[2]), head=(None if f[3] == "-" else int(f[3], 16)),
                              real=int(f[4]), f_prior=float(f[5]), eig=float(f[6]),
                              entropy=float(f[7]), tp9=int(f[14]), fp9=int(f[15])))
        except (ValueError, IndexError):
            pass
    return steps


def certified_order(elf, gt, func_gt, k):
    """Reconstruct per-head F from a lowf sweep (visits candidates in ascending F, reporting f_prior),
    then order by max conditional entropy h(F) = |F - 0.5| ascending. Returns [(head, F, real)]."""
    fg_set = S.read_gt(func_gt)
    sweep = run_active(elf, gt, func_gt, "lowf", 400)      # BIG k -> visit the whole candidate band
    seen = {}
    for st in sweep:
        if st["head"] is not None and st["head"] not in seen:
            seen[st["head"]] = (st["f_prior"], st["real"])
    order = sorted(seen.items(), key=lambda kv: abs(kv[1][0] - 0.5))
    return [(h, f, r) for (h, (f, r)) in order]


def certified_curve(elf, gt, func_gt, order, k):
    """Realise the certified ordering as sequential oracle-truthful clamps; effort curve = tp@0.9 at
    each step. A decoy pick injects nothing (no clamp) -- an honest wasted query, exactly as the engine."""
    clamps, seq, curve = [], [], []
    # step 0 baseline
    base = run_active(elf, gt, func_gt, "eig", 0)          # k=0 -> just the baseline report line
    curve.append(base[0]["tp9"] if base else None)
    for i in range(min(k, len(order))):
        head, f, real = order[i]
        seq.append(dict(step=i + 1, head=head, real=real, f_prior=f))
        if real:
            clamps.append(head)
        # measure recovered mass with the clamps applied so far (baseline of a k=0 active run)
        rep = run_active(elf, gt, func_gt, "eig", 0, clamps=clamps)
        curve.append(rep[0]["tp9"] if rep else None)
    return seq, curve


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="docs/staircase/eig_vs_greedy.csv")
    ap.add_argument("--summary", default="docs/staircase/eig_vs_greedy_summary.json")
    args = ap.parse_args()
    scratch = os.path.join(os.path.dirname(os.path.abspath(args.out)), "_scratch")

    rows, summary = [], []
    for elf in sorted(glob.glob(os.path.join(DECOY, "*.elf"))):
        stem = elf[:-4]
        st = os.path.basename(stem).split("field_")[-1]
        spec = dict(elf=elf, gt=stem + ".gt", func_gt=None, regions=stem + ".regions",
                    manifest=stem + ".manifest.json",
                    name=os.path.basename(stem).split("__")[0] + "_" + st, obf="decoy-heavy", struct=st)
        func_gt = S.func_gt_for(spec, scratch)
        if not func_gt:
            print("  SKIP %s (no func_gt)" % st); continue
        gt = spec["gt"]
        print("== %s ==" % st)

        eig = run_active(elf, gt, func_gt, "eig", args.k)
        eig_seq = [s for s in eig if s["step"] >= 1]
        eig_curve = [s["tp9"] for s in eig]                       # includes step 0 baseline
        eig_heads = [s["head"] for s in eig_seq]

        order = certified_order(elf, gt, func_gt, args.k)
        cert_seq, cert_curve = certified_curve(elf, gt, func_gt, order, args.k)
        cert_heads = [s["head"] for s in cert_seq]

        # agreement: set overlap of first-K picks, and exact-position matches
        jac = jaccard(eig_heads, cert_heads)
        pos_match = sum(1 for a, b in zip(eig_heads, cert_heads) if a == b)
        eig_final = eig_curve[-1] - eig_curve[0]                  # recovered mass over K queries
        cert_final = (cert_curve[-1] - cert_curve[0]) if cert_curve[-1] is not None else None
        eig_reals = sum(s["real"] for s in eig_seq)
        cert_reals = sum(s["real"] for s in cert_seq)

        for s in eig_seq:
            rows.append(dict(struct=st, rule="eig", **{k: s.get(k) for k in ("step", "head", "real", "f_prior", "eig")},
                             tp9=eig_curve[s["step"]]))
        for i, s in enumerate(cert_seq):
            rows.append(dict(struct=st, rule="certified", step=s["step"], head=s["head"], real=s["real"],
                             f_prior=s["f_prior"], eig=None, tp9=cert_curve[i + 1]))

        summary.append(dict(struct=st, k=args.k, jaccard=round(jac, 3), position_matches=pos_match,
                            eig_recovered=eig_final, cert_recovered=cert_final,
                            eig_real_hits=eig_reals, cert_real_hits=cert_reals,
                            eig_heads=["0x%x" % h for h in eig_heads],
                            cert_heads=["0x%x" % h for h in cert_heads]))
        print("   jaccard=%.2f pos_match=%d/%d  eig_recovered=%s cert_recovered=%s (eig_hits=%d cert_hits=%d)" % (
            jac, pos_match, len(eig_heads), eig_final, cert_final, eig_reals, cert_reals))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["struct", "rule", "step", "head", "real", "f_prior", "eig", "tp9"])
        w.writeheader()
        for r in rows:
            r = dict(r)
            if isinstance(r.get("head"), int):
                r["head"] = "0x%x" % r["head"]
            w.writerow(r)
    json.dump(summary, open(args.summary, "w"), indent=2)
    print("-> %s  (+ %s)" % (args.out, args.summary))


if __name__ == "__main__":
    main()
