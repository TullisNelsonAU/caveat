#!/usr/bin/env python3
"""Analyze the staircase CSVs and emit the machine summary + markdown fragments for STAIRCASE_RESULTS.md.
Pure CSV reader -- NO engine calls, safe to run any time after the measurement completes.

2a  staircase U_0 >= U_2 >= U_3 >= U_4 >= U_5 per class x obfuscation, and the predictions:
    - dU_{0->2} (reachability-confirmation) large on ANCHORED (decoy 'disconnected'), ~0 on ANCHORLESS
      ('self-anchoring'); steepest on decoy-heavy overall.
2b  tight-cell verdict table (from tight_cell.csv).
2c  EIG vs certified greedy (from eig_vs_greedy_summary.json).
"""
import csv, json, os, statistics as st, sys

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "staircase")
RUNGS = ["R0", "R1", "R2", "R3", "R4", "R5"]
RUNG_LABEL = {"R0": "E0 raw pi", "R1": "E1 cover", "R2": "E2 confirm", "R3": "E3 resolve",
              "R4": "E4 trace-clamp", "R5": "E5 oracle-clamp"}


def load(name):
    p = os.path.join(D, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def staircase_2a(rows):
    ins = [r for r in rows if r["obj_class"] == "instruction-start" and r["status"] == "ok"]
    groups = {}  # (obf, struct) -> rung -> [U_entropy]
    for r in ins:
        g = (r["obf"], r["struct"])
        groups.setdefault(g, {}).setdefault(r["rung"], []).append(fnum(r["U_entropy"]))
    lines = ["| obfuscation | structure | " + " | ".join(RUNG_LABEL[x] for x in RUNGS) +
             " | dU(0->2) | dU(0->5) |", "|" + "---|" * (len(RUNGS) + 4)]
    summary = []
    for g in sorted(groups):
        cells = {rk: mean(v) for rk, v in groups[g].items()}
        u0, u2, u5 = cells.get("R0"), cells.get("R2"), cells.get("R5")
        d02 = (u0 - u2) if (u0 is not None and u2 is not None) else None
        d05 = (u0 - u5) if (u0 is not None and u5 is not None) else None
        row = [g[0], g[1]] + ["%.3f" % cells[rk] if cells.get(rk) is not None else "-" for rk in RUNGS]
        row += ["%.3f" % d02 if d02 is not None else "-", "%.3f" % d05 if d05 is not None else "-"]
        lines.append("| " + " | ".join(row) + " |")
        summary.append(dict(obf=g[0], struct=g[1], U=cells, dU_0_2=d02, dU_0_5=d05))
    return "\n".join(lines), summary


def predictions(summary):
    out = []
    # anchored vs anchorless within decoy-heavy
    dec = {s["struct"]: s for s in summary if s["obf"] == "decoy-heavy"}
    if "disconnected" in dec and "self-anchoring" in dec:
        a = dec["disconnected"]["dU_0_2"]; b = dec["self-anchoring"]["dU_0_2"]
        out.append(("anchored(disconnected) dU_0_2=%.3f  vs  anchorless(self-anchoring) dU_0_2=%.3f" %
                    (a or 0, b or 0),
                    "PASS" if (a is not None and b is not None and a > b) else "CHECK"))
    # steepest on decoy-heavy
    steep = {s["obf"]: mean([x["dU_0_5"] for x in summary if x["obf"] == s["obf"]])
             for s in summary}
    if steep:
        top = max(steep, key=lambda k: steep[k] if steep[k] is not None else -9)
        out.append(("mean dU_0_5 by obfuscation: " +
                    ", ".join("%s=%.3f" % (k, v) for k, v in sorted(steep.items(), key=lambda kv: -(kv[1] or 0)) if v is not None),
                    "steepest=%s %s" % (top, "PASS" if top == "decoy-heavy" else "note")))
    # monotonicity
    viol = []
    for s in summary:
        seq = [s["U"].get(r) for r in ["R0", "R2", "R3", "R5"]]
        seq = [x for x in seq if x is not None]
        if any(seq[i + 1] > seq[i] + 1e-6 for i in range(len(seq) - 1)):
            viol.append("%s/%s" % (s["obf"], s["struct"]))
    out.append(("monotone U0>=U2>=U3>=U5 across cells", "PASS" if not viol else "violations: " + ",".join(viol)))
    return out


def tight_2b(rows):
    if not rows:
        return "_(tight_cell.csv not present yet)_", []
    lines = ["| binary | variant | beta0 | h(beta0) | U0 achiever | gap | verdict |",
             "|---|---|---|---|---|---|---|"]
    tights = []
    for r in rows:
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            r["binary"], r["variant"], r["beta0"], r["h_beta0"], r["U0_achiever"], r["gap"], r["verdict"]))
        if r["verdict"] == "TIGHT":
            tights.append(r)
    return "\n".join(lines), tights


def eig_2c(summary):
    if not summary:
        return "_(eig_vs_greedy_summary.json not present yet)_"
    lines = ["| structure | Jaccard | pos-match | EIG recovered | certified recovered | EIG hits | cert hits |",
             "|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append("| %s | %.2f | %d/%d | %s | %s | %s | %s |" % (
            s["struct"], s["jaccard"], s["position_matches"], s["k"],
            s["eig_recovered"], s["cert_recovered"], s["eig_real_hits"], s["cert_real_hits"]))
    return "\n".join(lines)


def main():
    rows = load("staircase_raw.csv")
    t2a, summary = staircase_2a(rows)
    preds = predictions(summary)
    t2b, tights = tight_2b(load("tight_cell.csv"))
    eig_sum = json.load(open(os.path.join(D, "eig_vs_greedy_summary.json"))) if os.path.exists(
        os.path.join(D, "eig_vs_greedy_summary.json")) else []
    t2c = eig_2c(eig_sum)

    print("### 2a staircase (mean U_entropy over binaries, instruction-start axis)\n")
    print(t2a)
    print("\n### 2a predictions\n")
    for msg, res in preds:
        print("- %s  -> **%s**" % (msg, res))
    print("\n### 2b tight cell\n")
    print(t2b)
    print("\nTIGHT cells: %d" % len(tights))
    print("\n### 2c EIG vs certified greedy\n")
    print(t2c)

    json.dump(dict(staircase=summary, predictions=preds, n_tight=len(tights)),
              open(os.path.join(D, "analysis_summary.json"), "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
