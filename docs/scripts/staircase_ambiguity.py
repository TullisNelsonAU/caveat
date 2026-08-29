#!/usr/bin/env python3
"""Sub-experiment 2b (SPEC sec 2b): the tight-cell hunt.

Find any cell where the measured achiever U_0 (mean binary entropy of the *calibrated* raw posterior
over the ambiguity set A_0) MEETS the computed lower bound h(beta_0). A tight cell = "fundamental
limit pinned, our method proven optimal there" -- so the spec demands it be AUDITED THREE WAYS before
belief. This script is a DELIBERATELY SEPARATE code path from staircase_measure.py: the achiever and
the bound must be computed by different code, or a shared bug could manufacture a false tight cell.

Lower bound (LIMITS_E0_PROOF.md): on the ambiguity set A_0 the E0 posterior is provably flat,
P(X*_o=1 | E0) = beta_0(o), so U_0(o) >= h(beta_0), h(b) = -b log2 b - (1-b) log2(1-b). beta_0 is the
prior real-mass of the confusable class = |real in A_0| / |A_0|.

A_0 by construction, computed THREE WAYS (the mandated audit):
  A1 full        : real GT starts UNION decoy candidate starts. decoy-heavy: exact manifest
                   decoy_entries; cid: real-code tiling reconstructed into the junk_decoy region.
                   beta_0 = |real| / (|real| + |decoy|).
  A2 confusable  : the genuinely byte-confusable core -- the decoy starts plus an EQUAL number of the
                   hardest (lowest raw-pi) real starts. This drives beta_0 toward 0.5 (h -> 1 bit): if
                   the achiever meets h here, the method is optimal on the maximally-ambiguous core.
  A3 empirical   : offsets whose raw pi lies in the uncertain band [0.2, 0.8]; beta_0 = real fraction
                   there. An engine-driven cross-check that does not use the construction labels.
Achiever: self-calibrate raw pi against the 0/1 GT (isotonic PAV -- the achievable ceiling), U_0 =
mean h(q_o) over each A_0 variant. A cell is TIGHT if |U_0 - h(beta_0)| is small in a variant AND the
verdict is stable across the three (the audit).

Read-only: one udstack --dump-instr per specimen (R0 raw pi). One binary in memory at a time.

Usage: staircase_ambiguity.py [--out docs/staircase/tight_cell.csv]
"""
import argparse, csv, glob, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import staircase_measure as S      # reuse run_udstack / isotonic / h_bin / read_gt / read_regions

ROOT = S.ROOT


def decoy_candidate_starts(spec, gt_addrs):
    """Return (decoy_start_set, provenance). decoy-heavy: exact manifest decoy_entries. cid: tile the
    real-code instruction starts across the junk_decoy region (the decoy IS tiled real code, so its
    internal starts are the real-start pattern repeated -- by construction, not by decode)."""
    if spec.get("manifest") and os.path.exists(spec["manifest"]):
        man = json.load(open(spec["manifest"]))
        de = man.get("params", {}).get("decoy_entries")
        if isinstance(de, list) and de:
            return set(de), "manifest.decoy_entries"
    regions = S.read_regions(spec.get("regions"))
    real_spans = [(s, e) for (s, e, l, k) in regions if k == "real_code"]
    decoy_spans = [(s, e) for (s, e, l, k) in regions if k == "junk_decoy"]
    if real_spans and decoy_spans:
        rs, re_ = real_spans[0]
        L = re_ - rs
        rel = sorted((a - rs) for a in gt_addrs if rs <= a < re_)   # real starts, region-relative
        decoy = set()
        for (ds, de_) in decoy_spans:
            t = 0
            while ds + t * L < de_:                                  # each tile of the real block
                for r in rel:
                    off = ds + t * L + r
                    if off < de_:
                        decoy.add(off)
                t += 1
        return decoy, "tiled_real_starts"
    return set(), "none"


def h(b):
    return S.h_bin(b)


def verdict(u, hb):
    if hb is None or u is None:
        return "n/a"
    d = u - hb
    if abs(d) <= 0.05:
        return "TIGHT"
    if d < -0.05:
        return "below_bound(check)"   # achiever below a valid lower bound => set/label problem, flag it
    return "gap+%.2f" % d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/staircase/tight_cell.csv")
    args = ap.parse_args()

    specs = list(S.cid_specs()) + list(S.decoy_specs())
    fields = ["binary", "obf", "struct", "variant", "n_ambig", "n_real", "beta0",
              "h_beta0", "U0_achiever", "gap", "verdict", "provenance"]
    out = open(args.out, "w", newline="")
    w = csv.DictWriter(out, fieldnames=fields); w.writeheader()

    for spec in specs:
        if not os.path.exists(spec["elf"]) or not os.path.exists(spec["gt"]):
            continue
        gt_addrs = S.read_gt(spec["gt"])
        rows, _ = S.run_udstack(spec["elf"], spec["gt"])       # R0: pi in col 3
        if not rows:
            print("  SKIP %s (no dump)" % spec["name"]); continue
        addr = [r[0] for r in rows]
        pi = {r[0]: r[2] for r in rows}
        lab = {r[0]: r[3] for r in rows}
        # self-calibrate raw pi against GT (achievable ceiling)
        cal = S.isotonic_fit([r[2] for r in rows], [r[3] for r in rows])
        q = {a: S.isotonic_apply(cal, pi[a]) for a in addr}

        real = set(a for a in addr if lab[a] == 1)
        decoy, prov = decoy_candidate_starts(spec, gt_addrs)
        decoy &= set(addr)                                     # keep only offsets actually in .text dump
        real_in = real                                         # all real starts are candidate code starts

        def cell(name, ambig_real, ambig_decoy):
            A = list(ambig_real) + list(ambig_decoy)
            if not A:
                return None
            nr = len(ambig_real)
            beta = nr / len(A)
            hb = h(beta)
            u = sum(S.h_bin(q[a]) for a in A) / len(A)
            return dict(binary=spec["name"], obf=spec["obf"], struct=spec["struct"], variant=name,
                        n_ambig=len(A), n_real=nr, beta0=round(beta, 4), h_beta0=round(hb, 4),
                        U0_achiever=round(u, 4), gap=round(u - hb, 4), verdict=verdict(u, hb),
                        provenance=prov)

        # A1 full: all real + all decoy candidates
        c1 = cell("A1_full", real_in, decoy)
        # A2 confusable core: decoy + equal count of hardest (lowest calibrated q) real starts
        hard_real = sorted(real_in, key=lambda a: q[a])[:len(decoy)] if decoy else []
        c2 = cell("A2_confusable", hard_real, decoy)
        # A3 empirical uncertain band pi in [0.2,0.8]
        band = [a for a in addr if 0.2 <= pi[a] <= 0.8]
        c3 = cell("A3_band", [a for a in band if lab[a] == 1], [a for a in band if lab[a] == 0])

        for c in (c1, c2, c3):
            if c:
                w.writerow(c); out.flush()
                print("  %-34s %-14s beta0=%.3f h=%.3f U0=%.3f gap=%+.3f  %s" % (
                    spec["name"], c["variant"], c["beta0"], c["h_beta0"], c["U0_achiever"],
                    c["gap"], c["verdict"]))
    out.close()
    print("-> %s" % args.out)


if __name__ == "__main__":
    main()
