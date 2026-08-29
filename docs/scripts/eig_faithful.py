#!/usr/bin/env python3
"""Follow-up 2 (STAIRCASE_FOLLOWUP_SPEC): the FAITHFUL EIG re-ranking loop -- adopt-vs-keep.

The first comparison (staircase_eig.py) found F_h.dH (Paper 3's EIG) != the certified max-conditional-
entropy greedy rule (Jaccard <=0.23), but it was NOT apples-to-apples: the certified curve used a STATIC
baseline-F ordering realised by an EXTERNAL clamp (q=1), while EIG used an INTERNAL re-ranking loop
(q=0.99). This runs BOTH rules through the SAME faithful loop and scores them on OUTCOME.

Both rules now live inside udstack's `run_active` (the one query loop), identical in everything except
the pick:
  * eig     : argmax_h F_h.dH  -- Stack::rank_queries, an exact frozen-relax what-if per candidate.
  * certent : argmax_h H(X_h | E, X_queried) = argmax_h h(F_h) = nearest-0.5 F_h  -- the certified rule
              the (1-1/e) submodularity theorem guarantees (LIMITS_HIERARCHY_PROOFS). The `f` map is the
              CURRENT post-relax confirmation, re-read every step => internal re-ranking, SAME q=0.99,
              SAME oracle-truthful clamp (a real head injects q; a decoy query injects nothing), SAME
              uncertain band [0.05,0.95] and body floor. So the loops differ ONLY in the selection rule.

Score (FU2 acceptance): recovered TRUE mass at P̂>=0.9 vs #queries (the effort curve; tp_at_0.9 column),
per-step selection overlap (reproduce the Jaccard under the fair loop), and the honesty-wall assertion
(pi frozen: stderr "pi invariant = held"). DECISION:
  * certent recovers >= eig  -> ADOPT the certified rule (it EARNS the (1-1/e) guarantee). Clean win.
  * eig recovers > certent   -> KEEP F_h.dH, report it as unguaranteed-but-better (it avoids wasting
                               queries on irreducible max-entropy objects; its objective is propagation
                               value, not total entropy). Also a clean, honest result.

Arms: decoy-heavy (coreutils, dense decoys -- where query-waste matters) and cid (code-in-data, small
real programs with proper symbol GT -- the lower-decoy "benign-ish" contrast the spec asks for).

Memory: ONE udstack process at a time (2 per specimen); one binary in memory; no corpus held. Read-only
on the engine (the honesty wall is asserted per run). Resumable by (arm, specimen).

Usage: eig_faithful.py [--k 8] [--out docs/staircase/eig_faithful.csv]
"""
import argparse, csv, glob, json, math, os, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import staircase_measure as S

UDSTACK = S.UDSTACK


def run_active(elf, gt, func_gt, strat, k, q=0.99):
    """One process. Returns (steps, pi_held). steps: list of dicts per active step (0..k)."""
    cmd = [UDSTACK, elf, gt, "--func-gt", func_gt, "--active", "%s:%d" % (strat, k), "--query-q", str(q)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    steps, pi_held = [], None
    for ln in p.stdout.splitlines():
        if not ln.startswith("stack_active,"):
            continue
        f = ln.split(",")
        if f[1] == "strategy":
            continue
        # stack_active,strat,step,head,real,f_prior,eig,entropy,ece,auroc,pi_ece,pi_auroc,hi_mass,
        #              mean_phat,tp_at_0.9,fp_at_0.9 -> tp9=f[14]
        try:
            steps.append(dict(step=int(f[2]), head=(None if f[3] == "-" else int(f[3], 16)),
                              real=int(f[4]), f_prior=float(f[5]), eig=float(f[6]),
                              entropy=float(f[7]), tp9=int(f[14])))
        except (ValueError, IndexError):
            pass
    for ln in p.stderr.splitlines():
        if "π invariant" in ln or "pi invariant" in ln or "invariant =" in ln:
            pi_held = ("held" in ln)
    return steps, pi_held


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def specs_for(arm):
    if arm == "decoy":
        for spec in S.decoy_specs():
            yield spec
    elif arm == "cid":
        for spec in S.cid_specs():
            yield spec


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                done.add((r["arm"], r["specimen"]))
    return done


CURVE_FIELDS = ["arm", "specimen", "rule", "step", "head", "real", "f_prior", "eig", "tp9"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--q", type=float, default=0.99)
    ap.add_argument("--arms", default="decoy,cid")
    ap.add_argument("--out", default="docs/staircase/eig_faithful.csv")
    ap.add_argument("--summary", default="docs/staircase/eig_faithful_summary.json")
    args = ap.parse_args()
    scratch = os.path.join(os.path.dirname(os.path.abspath(args.out)), "_scratch")

    done = load_done(args.summary + ".donekeys") if False else set()
    # resumable by (arm, specimen) via the curve CSV
    if os.path.exists(args.out):
        for r in csv.DictReader(open(args.out)):
            done.add((r["arm"], r["specimen"]))
    new_file = not os.path.exists(args.out)
    fout = open(args.out, "a", newline="")
    w = csv.DictWriter(fout, fieldnames=CURVE_FIELDS)
    if new_file:
        w.writeheader(); fout.flush()

    summary = json.load(open(args.summary)) if os.path.exists(args.summary) else []
    seen_summ = {(s["arm"], s["specimen"]) for s in summary}

    for arm in args.arms.split(","):
        for spec in specs_for(arm):
            if not os.path.exists(spec["elf"]) or not os.path.exists(spec["gt"]):
                continue
            key = (arm, spec["name"])
            if key in done:
                continue
            func_gt = S.func_gt_for(spec, scratch)
            if not func_gt:
                print("  SKIP %s/%s (no func_gt)" % key); continue
            print("== %s / %s ==" % key)

            eig, eig_pi = run_active(spec["elf"], spec["gt"], func_gt, "eig", args.k, args.q)
            cert, cert_pi = run_active(spec["elf"], spec["gt"], func_gt, "certent", args.k, args.q)
            if not eig or not cert:
                print("   no active steps, skipping"); continue

            eig_seq = [s for s in eig if s["step"] >= 1]
            cert_seq = [s for s in cert if s["step"] >= 1]
            eig_heads = [s["head"] for s in eig_seq]
            cert_heads = [s["head"] for s in cert_seq]

            for s in eig_seq:
                w.writerow(dict(arm=arm, specimen=spec["name"], rule="eig", step=s["step"],
                                head=("0x%x" % s["head"]) if s["head"] else "-", real=s["real"],
                                f_prior=round(s["f_prior"], 4), eig=round(s["eig"], 4), tp9=s["tp9"]))
            for s in cert_seq:
                w.writerow(dict(arm=arm, specimen=spec["name"], rule="certent", step=s["step"],
                                head=("0x%x" % s["head"]) if s["head"] else "-", real=s["real"],
                                f_prior=round(s["f_prior"], 4), eig=round(s["eig"], 4), tp9=s["tp9"]))
            fout.flush()

            eig_rec = eig[-1]["tp9"] - eig[0]["tp9"]
            cert_rec = cert[-1]["tp9"] - cert[0]["tp9"]
            jac = jaccard(eig_heads, cert_heads)
            pos = sum(1 for a, b in zip(eig_heads, cert_heads) if a == b)
            if cert_rec > eig_rec:
                decision = "adopt_certified"
            elif eig_rec > cert_rec:
                decision = "keep_eig"
            else:
                decision = "tie"
            rec = dict(arm=arm, specimen=spec["name"], k=args.k, q=args.q,
                       eig_recovered=eig_rec, cert_recovered=cert_rec,
                       eig_real_hits=sum(s["real"] for s in eig_seq),
                       cert_real_hits=sum(s["real"] for s in cert_seq),
                       jaccard=round(jac, 3), position_matches=pos, n_steps=len(eig_seq),
                       eig_pi_held=eig_pi, cert_pi_held=cert_pi, decision=decision,
                       eig_heads=["0x%x" % h for h in eig_heads],
                       cert_heads=["0x%x" % h for h in cert_heads])
            if key not in seen_summ:
                summary.append(rec)
            json.dump(summary, open(args.summary, "w"), indent=2)
            print("   eig_rec=%d cert_rec=%d jac=%.2f pos=%d/%d  pi(eig=%s,cert=%s) -> %s" % (
                eig_rec, cert_rec, jac, pos, len(eig_heads), eig_pi, cert_pi, decision))

    fout.close()
    print("-> %s  (+ %s)" % (args.out, args.summary))


if __name__ == "__main__":
    main()
