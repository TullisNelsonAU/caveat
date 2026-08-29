#!/usr/bin/env python3
"""Follow-up 1 (STAIRCASE_FOLLOWUP_SPEC): the A_k-RESTRICTED, dual-axis recoverability staircase.

The full-object staircase (staircase_measure.py) was shallow (~0.05 bits) by DILUTION -- the ambiguity
set A_k is a small fraction of all objects, so the near-zero-entropy pinned objects wash out the drop.
The value of evidence lives on A_k. This restates the staircase there, on BOTH axes, per rung per
obfuscation.

A_k BY CONSTRUCTION (never a threshold on q; FOLLOWUP_SPEC FU1):
  A_0 = the genuinely-confusable offsets = real instruction starts UNION decoy candidate starts
        (manifest.decoy_entries for decoy-heavy; the tiled real-code starts of the junk_decoy region for
        cid). Reused verbatim from staircase_ambiguity.decoy_candidate_starts -- same code path as 2b.
  A_k (k>=1) = A_0 minus the objects rung k PROVABLY PINS. The pin is the engine's DISCRETE
        reachability closure `recursive_descent(superset, seeds_k)` (udstack --dump-pins, machine line
        `pin_reach`), seeds_k = program entry  ∪  clamped heads (E4/E5 trace/oracle)  ∪  M3a resolved
        targets (E3). A closure is a membership FACT, not a posterior threshold:
          * reached  ⇒ determined code  ⇒ PINNED out of A_k;
          * decoy candidate NOT reached ⇒ determined junk ⇒ PINNED out of A_k;
          * real start NOT reached ⇒ still undetermined ⇒ STAYS in A_k;
          * decoy candidate REACHED ⇒ a LEAK the anchor failed to prune ⇒ STAYS in A_k (misclassified).
        So A_k = {real starts not yet reached} UNION {decoy candidates wrongly reached}. On disconnected
        decoys the closure never reaches them -> A_2 collapses -> leak -> 0. On interleaved/self-anchoring
        decoys the closure reaches them via fall-through -> they stay -> the predicted NON-MONOTONE leak.

TWO AXES per rung, restricted to A_k:
  1. entropy       U_k = mean binary entropy h(q_o) over o in A_k  (q = isotonic-self-calibrated rung
                   posterior -- the achievable ceiling, same PAV fit as staircase_measure).
  2. 0-1 / leak    n_misclass = |{o in A_k : MAP(o) != label(o)}|, MAP = rung_score >= 0.5; and the
                   decoy-leak = |{decoy candidates in A_k}| (all reached decoys are leaks). Count + rate.
The PINNED COMPLEMENT (A_0 \ A_k) mean entropy is reported alongside as a sanity check (should be ~flat,
near-zero: reached real code is high-confidence, unreached decoys are low-confidence).

E_0 tight-cell re-confirmation (FU1 acceptance): on A_0, beta_0 = |real in A_0| / |A_0|, and the
achiever U_0 (calibrated raw pi) is compared to the bound h(beta_0). It should survive the restriction.

Memory + integrity (unchanged, non-negotiable): ONE binary in memory at a time; udstack is single-binary
(`--jobs 1` inherent); every row appended+flushed; resumable by (binary, rung); NO corpus held. One plain
udstack run serves R0 (pi, full A_0, no pin) AND R2 (phat, entry-closure pin); R3/R4/R5 one run each
(resolve / half-heads / all-heads change both the posterior and the closure). GT from .gt/.func.gt/
.regions/manifest only; align_ok carried from the independent ELF load-base parse (the ET_DYN check).

Usage: staircase_ak.py [--out docs/staircase/staircase_ak.csv] [--corpora cid,decoy] [--rungs R0,R2,R3,R4,R5]
"""
import argparse, csv, glob, math, os, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import staircase_measure as S
from staircase_ambiguity import decoy_candidate_starts

UDSTACK = S.UDSTACK


def run_udstack_pins(elf, gt, func_gt=None, resolve=False, clamp=None):
    """One process, one binary. Returns (rows, reached_set). rows: (addr, phat, pi, label01) from
    instr_bel; reached_set: {addr} from pin_reach (the discrete E_k reachability closure)."""
    cmd = [UDSTACK, elf, gt, "--dump-instr", "--dump-pins"]
    if func_gt:
        cmd += ["--func-gt", func_gt]
    if resolve:
        cmd += ["--resolve-elf", elf]
    for a in (clamp or []):
        cmd += ["--clamp-func", "0x%x:1.0" % a]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    rows, reached = [], set()
    for ln in p.stdout.splitlines():
        if ln.startswith("instr_bel,"):
            f = ln.split(",")
            try:
                rows.append((int(f[1], 16), float(f[2]), float(f[3]), 1 if f[4] == "real" else 0))
            except (ValueError, IndexError):
                pass
        elif ln.startswith("pin_reach,"):
            f = ln.split(",")
            try:
                if int(f[2]) == 1:
                    reached.add(int(f[1], 16))
            except (ValueError, IndexError):
                pass
    return rows, reached


def a0_set(spec, gt_addrs, dumped):
    """A_0 by construction (same as staircase_ambiguity 2b): real starts UNION decoy candidate starts,
    intersected with the offsets udstack actually dumped. Returns (A0, real_in_A0, decoy_in_A0, prov)."""
    real = set(a for a in gt_addrs if a in dumped)
    decoy, prov = decoy_candidate_starts(spec, gt_addrs)
    decoy &= dumped
    return (real | decoy), real, decoy, prov


def ak_restrict(a0_real, a0_decoy, reached, apply_pin):
    """A_k = A_0 minus pinned_k. apply_pin False (R0/E0, pre-anchor) => A_k = A_0 (nothing pinned).
    Else: pinned = {reached} UNION {decoy not reached}; A_k = {real not reached} UNION {decoy reached}."""
    if not apply_pin:
        return set(a0_real) | set(a0_decoy)
    real_unreached = {a for a in a0_real if a not in reached}
    decoy_leaked = {a for a in a0_decoy if a in reached}       # wrongly reached => still ambiguous
    return real_unreached | decoy_leaked


def metrics_on(A, q, lab, score, a0_decoy):
    """Dual-axis metrics restricted to set A. q: calibrated posterior map; score: rung raw score map;
    lab: 0/1 map; a0_decoy: the decoy-candidate set (for the leak count)."""
    A = [a for a in A if a in q]
    if not A:
        return None
    U_H = sum(S.h_bin(q[a]) for a in A) / len(A)
    U_bayes = sum(min(q[a], 1 - q[a]) for a in A) / len(A)
    n_mis = sum(1 for a in A if (1 if score[a] >= 0.5 else 0) != lab[a])
    n_decoy = sum(1 for a in A if a in a0_decoy)
    return dict(n=len(A), U_entropy=U_H, U_bayes=U_bayes, n_misclass=n_mis,
                leak_rate=round(n_mis / len(A), 4), n_decoy=n_decoy)


FIELDS = ["binary", "obf", "struct", "rung", "n_A0", "n_Ak", "beta_A0", "h_beta_A0", "U0_A0_achiever",
          "tight_verdict", "U_entropy_Ak", "U_bayes_Ak", "n_misclass_Ak", "leak_rate_Ak", "n_decoy_Ak",
          "pinned_n", "pinned_mean_H", "reached_n", "align_ok", "e_type", "ambig_provenance", "status"]


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                done.add((r["binary"], r["rung"]))
    return done


# per-rung engine plan: (apply_pin, resolve, clamp_kind). clamp_kind in {None,'half','all'}.
RUNG_PLAN = {"R0": (False, False, None), "R2": (True, False, None), "R3": (True, True, None),
             "R4": (True, True, "half"), "R5": (True, True, "all")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/staircase/staircase_ak.csv")
    ap.add_argument("--corpora", default="cid,decoy")
    ap.add_argument("--rungs", default="R0,R2,R3,R4,R5")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    scratch = os.path.join(os.path.dirname(os.path.abspath(args.out)), "_scratch")
    rungs = args.rungs.split(",")
    done = load_done(args.out)
    new_file = not os.path.exists(args.out)
    fout = None if args.dry_run else open(args.out, "a", newline="")
    w = None
    if fout:
        w = csv.DictWriter(fout, fieldnames=FIELDS)
        if new_file:
            w.writeheader(); fout.flush()

    specs = []
    for c in args.corpora.split(","):
        specs += list(S.CORPORA[c]())

    print("# %d specimens, rungs=%s, %d already done" % (len(specs), rungs, len(done)))
    for spec in specs:
        if not spec.get("gt") or not os.path.exists(spec["gt"]) or not os.path.exists(spec["elf"]):
            print("  SKIP %s (no gt/elf)" % spec["name"]); continue
        gt_addrs = S.read_gt(spec["gt"])
        align_ok, info = S.align_audit(spec["elf"], gt_addrs)
        func_gt = S.func_gt_for(spec, scratch)

        todo = [rk for rk in rungs if (spec["name"], rk) not in done]
        if not todo:
            continue
        print("== %s (%s/%s) todo=%s ==" % (spec["name"], spec["obf"], spec["struct"], todo))

        # cache the plain run (serves R0 + R2) so we never load the binary twice for it.
        plain = None

        def get_run(rung):
            nonlocal plain
            apply_pin, resolve, ck = RUNG_PLAN[rung]
            heads = None
            if ck:
                if not func_gt:
                    return None, None, None  # no oracle -> rung n/a
                hs = sorted(S.read_gt(func_gt))
                heads = hs[::2] if ck == "half" else hs
            if not resolve and not heads:            # R0/R2 share one plain process
                if plain is None:
                    plain = run_udstack_pins(spec["elf"], spec["gt"], func_gt=func_gt)
                return plain[0], plain[1], heads
            rows, reached = run_udstack_pins(spec["elf"], spec["gt"], func_gt=func_gt,
                                             resolve=resolve, clamp=heads)
            return rows, reached, heads

        for rung in todo:
            apply_pin, resolve, ck = RUNG_PLAN[rung]
            if args.dry_run:
                print("  would run", spec["name"], rung, "pin=%s resolve=%s clamp=%s" % (apply_pin, resolve, ck))
                continue
            base = dict(binary=spec["name"], obf=spec["obf"], struct=spec["struct"], rung=rung,
                        align_ok=align_ok, e_type=(info or {}).get("e_type"))
            try:
                rows, reached, heads = get_run(rung)
                if rows is None:
                    base["status"] = "na_no_func_gt"; emit(w, fout, base); continue
                if not rows:
                    raise RuntimeError("no instr_bel rows")
            except Exception as ex:
                base["status"] = "ERR:%s" % str(ex)[:40]; emit(w, fout, base); continue

            dumped = set(r[0] for r in rows)
            lab = {r[0]: r[3] for r in rows}
            score = {r[0]: (r[2] if rung == "R0" else r[1]) for r in rows}   # R0 uses pi, else phat
            cal = S.isotonic_fit([score[r[0]] for r in rows], [lab[r[0]] for r in rows])
            q = {a: S.isotonic_apply(cal, score[a]) for a in dumped}

            A0, a0_real, a0_decoy, prov = a0_set(spec, gt_addrs, dumped)
            beta0 = (len(a0_real) / len(A0)) if A0 else None
            hb0 = S.h_bin(beta0) if beta0 is not None else None
            U0_ach = (sum(S.h_bin(q[a]) for a in A0) / len(A0)) if A0 else None
            tv = tight_verdict(U0_ach, hb0) if rung == "R0" else ""

            Ak = ak_restrict(a0_real, a0_decoy, reached, apply_pin)
            m = metrics_on(Ak, q, lab, score, a0_decoy)
            pinned = A0 - Ak
            pinned_H = (sum(S.h_bin(q[a]) for a in pinned) / len(pinned)) if pinned else None

            base.update(dict(n_A0=len(A0), beta_A0=(round(beta0, 4) if beta0 is not None else None),
                             h_beta_A0=(round(hb0, 4) if hb0 is not None else None),
                             U0_A0_achiever=(round(U0_ach, 4) if U0_ach is not None else None),
                             tight_verdict=tv, ambig_provenance=prov, reached_n=len(reached),
                             pinned_n=len(pinned),
                             pinned_mean_H=(round(pinned_H, 4) if pinned_H is not None else None),
                             status="ok" if m else "empty_Ak"))
            if m:
                base.update(dict(n_Ak=m["n"], U_entropy_Ak=round(m["U_entropy"], 4),
                                 U_bayes_Ak=round(m["U_bayes"], 4), n_misclass_Ak=m["n_misclass"],
                                 leak_rate_Ak=m["leak_rate"], n_decoy_Ak=m["n_decoy"]))
            else:
                base["status"] = "corpus-limited(empty_Ak)"
            emit(w, fout, base)
    if fout:
        fout.close()
    print("# done ->", args.out)


def tight_verdict(u, hb):
    if u is None or hb is None:
        return "n/a"
    d = u - hb
    if abs(d) <= 0.05:
        return "TIGHT(%+.3f)" % d
    if d < -0.05:
        return "below_bound(%+.3f)" % d
    return "gap(%+.3f)" % d


def emit(w, fout, row):
    if not w:
        return
    w.writerow({k: row.get(k) for k in FIELDS}); fout.flush()
    print("  %-30s %-3s A0=%s Ak=%s U_Ak=%s leak=%s/%s pinnedH=%s %s" % (
        row["binary"], row["rung"], row.get("n_A0"), row.get("n_Ak"),
        row.get("U_entropy_Ak"), row.get("n_decoy_Ak"), row.get("n_Ak"),
        row.get("pinned_mean_H"), row["status"]))


if __name__ == "__main__":
    main()
