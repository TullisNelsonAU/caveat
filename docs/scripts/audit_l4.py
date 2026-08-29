#!/usr/bin/env python3
"""Three-way audit of the L4 (module/SCC) function-level discrimination result.

Same discipline as `audit_cid.py`: any surprising discrimination number gets audited on three axes
before it goes in the doc. Here the claim under audit is "adding the L4 module layer raises
function-level AUROC by pruning decoy-component functions." The three axes:

  1. GT PROVENANCE  — the FUNC GT is the benign seed's .symtab (objdump -t), never a disassembler's
                      output; the decoy region comes from the gauntlet .regions; and no decoy head is
                      labelled real (a polluted GT would fabricate the win). Module GT is derived from
                      FUNC GT purely by set membership (a component is real iff a member head is).
  2. MECHANISM      — the win must be *reachability*: decoy heads (real=0, in the decoy region) must
                      land in low-F_c (disconnected) components, real heads in high-F_c ones. F_c is
                      computed with NO labels (component prior 0, entry pinned 1, edges from the
                      confirmed call graph), so any real/decoy separation in F_c is genuine structure,
                      not label leakage. We report AUROC(F_c ; real-vs-decoy heads).
  3. ALIGNMENT      — the sign is honest: report the false-prune rate (real heads stranded at low F_c),
                      which is exactly the ls regression, so a per-binary loss can't hide inside a mean.

Input: the `stack_headmod` dump (`udstack --layers 3 --dump-modules`), one CSV per specimen:
  stack_headmod,head,f_h,comp,f_c,real,in_decoy
Usage: audit_l4.py <headmod_dir>
"""
import sys
import glob
import os


def auroc(scores_pos, scores_neg):
    """Tie-averaged Mann-Whitney AUROC — matches evalkit::auroc."""
    if not scores_pos or not scores_neg:
        return float("nan")
    pairs = [(s, 1) for s in scores_pos] + [(s, 0) for s in scores_neg]
    pairs.sort(key=lambda x: x[0])
    # Rank with ties averaged.
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    sum_pos = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load(path):
    rows = []
    for line in open(path):
        if not line.startswith("stack_headmod,head"):
            if line.startswith("stack_headmod,"):
                _, h, f_h, comp, f_c, real, in_decoy = line.strip().split(",")
                rows.append(dict(head=int(h, 16), f_h=float(f_h), comp=int(comp, 16),
                                 f_c=float(f_c), real=int(real), in_decoy=int(in_decoy)))
    return rows


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(d, "headmod_*.csv")))
    if not files:
        print(f"no headmod_*.csv in {d}", file=sys.stderr)
        sys.exit(2)

    ok = True
    print(f"{'specimen':<10} {'heads':>5} {'real':>4} {'decoy':>5} "
          f"{'F_c AUROC':>9} {'med F_c real':>12} {'med F_c decoy':>13} "
          f"{'decoy pruned':>12} {'real stranded':>13}")
    for f in files:
        name = os.path.basename(f)[len("headmod_"):-len(".csv")]
        rows = load(f)
        # A decoy head = sits in the decoy region and is not a real function.
        decoy = [r for r in rows if r["in_decoy"] and not r["real"]]
        real = [r for r in rows if r["real"]]

        # (1) GT PROVENANCE: no decoy-region head may be labelled real (would poison the win).
        poison = [r for r in rows if r["in_decoy"] and r["real"]]
        if poison:
            ok = False
            print(f"  {name}: FAIL provenance — {len(poison)} decoy-region heads labelled real")

        # (2) MECHANISM: F_c separates real from decoy heads (label-free structural signal).
        au = auroc([r["f_c"] for r in real], [r["f_c"] for r in decoy])
        med_r = median([r["f_c"] for r in real])
        med_d = median([r["f_c"] for r in decoy])
        # decoy pruned = decoy heads correctly stranded at F_c < 0.5 (disconnected).
        pruned = sum(1 for r in decoy if r["f_c"] < 0.5) / max(1, len(decoy))
        # (3) ALIGNMENT: real heads wrongly stranded at low F_c — the honest failure mode (ls).
        stranded = sum(1 for r in real if r["f_c"] < 0.5) / max(1, len(real))

        mech_ok = (med_d < med_r) and (au > 0.5)
        if not mech_ok:
            ok = False
        print(f"{name:<10} {len(rows):>5} {len(real):>4} {len(decoy):>5} "
              f"{au:>9.3f} {med_r:>12.3f} {med_d:>13.3f} "
              f"{pruned:>11.1%} {stranded:>12.1%}  {'' if mech_ok else '<-- MECH FAIL'}")

    print()
    print("AUDIT", "PASS" if ok else "FAIL",
          "— F_c is a label-free structural score; a real>decoy separation is genuine reachability.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
