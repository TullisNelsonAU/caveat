#!/usr/bin/env python3
"""Analyze the A_k-restricted dual-axis staircase (staircase_ak.csv) -> markdown tables + predictions.
Pure CSV reader; NO engine calls. Emits the corrected headline figure's numbers (FOLLOWUP_SPEC FU1).
"""
import csv, json, os, statistics as st

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "staircase")
RUNGS = ["R0", "R2", "R3", "R4", "R5"]
LAB = {"R0": "E0 raw", "R2": "E2 confirm", "R3": "E3 resolve", "R4": "E4 trace", "R5": "E5 oracle"}


def load(name):
    p = os.path.join(D, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def inum(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def by_cell(rows):
    g = {}
    for r in rows:
        g.setdefault((r["obf"], r["struct"]), {})[r["rung"]] = r
    return g


def dual_axis_tables(rows):
    g = by_cell(rows)
    out = []
    out.append("#### Entropy axis — U_k = mean h(q) over A_k (bits)\n")
    out.append("| obfuscation | structure | " + " | ".join(LAB[r] for r in RUNGS) + " |")
    out.append("|" + "---|" * (len(RUNGS) + 2))
    for cell in sorted(g):
        row = [cell[0], cell[1]]
        for rk in RUNGS:
            r = g[cell].get(rk)
            v = fnum(r["U_entropy_Ak"]) if r else None
            row.append("%.3f" % v if v is not None else ("cl" if r and "limited" in (r["status"] or "") else "-"))
        out.append("| " + " | ".join(row) + " |")

    out.append("\n#### Decode-leak axis — decoy candidates surviving in A_k (count)\n")
    out.append("| obfuscation | structure | " + " | ".join(LAB[r] for r in RUNGS) + " | leak E0→E2 |")
    out.append("|" + "---|" * (len(RUNGS) + 3))
    for cell in sorted(g):
        row = [cell[0], cell[1]]
        vals = {}
        for rk in RUNGS:
            r = g[cell].get(rk)
            v = inum(r["n_decoy_Ak"]) if r else None
            vals[rk] = v
            row.append(str(v) if v is not None else "-")
        drop = "%s→%s" % (vals.get("R0"), vals.get("R2")) if vals.get("R0") is not None else "-"
        row.append(drop)
        out.append("| " + " | ".join(row) + " |")

    out.append("\n#### |A_k| (set size — pins shrink it) and pinned-complement mean entropy (sanity ≈flat)\n")
    out.append("| obfuscation | structure | " + " | ".join("|A|%s" % LAB[r].split()[0] for r in RUNGS) +
               " | pinned mean H (E2..E5) |")
    out.append("|" + "---|" * (len(RUNGS) + 3))
    for cell in sorted(g):
        row = [cell[0], cell[1]]
        for rk in RUNGS:
            r = g[cell].get(rk)
            row.append(str(inum(r["n_Ak"])) if r and inum(r["n_Ak"]) is not None else "-")
        ph = [fnum(g[cell][rk]["pinned_mean_H"]) for rk in RUNGS[1:] if g[cell].get(rk) and fnum(g[cell][rk]["pinned_mean_H"]) is not None]
        row.append(", ".join("%.3f" % x for x in ph) if ph else "-")
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def predictions(rows):
    g = by_cell(rows)
    out = []
    # (1) leak axis steep where full-object entropy was flat: decoy leak E0->E2 drop, per cell
    out.append("**P1 — decode-leak axis is steep where the full-object entropy staircase was flat (~0.05 bits):**")
    for cell in sorted(g):
        r0, r2 = g[cell].get("R0"), g[cell].get("R2")
        if r0 and r2:
            d0, d2 = inum(r0["n_decoy_Ak"]), inum(r2["n_decoy_Ak"])
            if d0 is not None and d2 is not None:
                tag = "→0 (anchor prunes)" if d2 == 0 else ("NON-MONOTONE leak survives" if d2 >= d0 * 0.5 else "partial")
                out.append("  - %s/%s: decoy-leak %d → %d  **%s**" % (cell[0], cell[1], d0, d2, tag))
    # (2) anchored vs anchorless within decoy-heavy
    dec = {c[1]: g[c] for c in g if c[0] == "decoy-heavy"}
    out.append("\n**P2 — anchored (disconnected) vs anchorless (self-anchoring / interleaved):**")
    for stru in sorted(dec):
        r0, r2 = dec[stru].get("R0"), dec[stru].get("R2")
        if r0 and r2:
            d0, d2 = inum(r0["n_decoy_Ak"]), inum(r2["n_decoy_Ak"])
            monot = "prunes→0" if (d2 == 0) else "leak persists (non-monotone)"
            out.append("  - %s: %s → %s  (%s)" % (stru, d0, d2, monot))
    # (3) E0 tight cell on A_0
    out.append("\n**P3 — E0 tight cell on A_0 (achiever U0 vs h(β0)):**")
    for cell in sorted(g):
        r0 = g[cell].get("R0")
        if r0 and r0.get("tight_verdict"):
            out.append("  - %s/%s: β0=%s h(β0)=%s U0=%s  **%s**" % (
                cell[0], cell[1], r0["beta_A0"], r0["h_beta_A0"], r0["U0_A0_achiever"], r0["tight_verdict"]))
    # (4) corpus-limited
    cl = [(r["obf"], r["struct"], r["rung"]) for r in rows if "limited" in (r["status"] or "")]
    if cl:
        out.append("\n**P4 — corpus-limited rungs (A_k emptied — no drop forced):** " +
                   ", ".join("%s/%s@%s" % c for c in cl))
    return "\n".join(out)


def main():
    rows = load("staircase_ak.csv")
    if not rows:
        print("_(staircase_ak.csv not present)_"); return
    print("### FU1 — A_k-restricted dual-axis staircase\n")
    print(dual_axis_tables(rows))
    print("\n### FU1 predictions\n")
    print(predictions(rows))
    json.dump({"n_rows": len(rows)}, open(os.path.join(D, "ak_analysis_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
