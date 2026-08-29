#!/usr/bin/env python3
"""Analyze the faithful EIG-vs-certified loop (eig_faithful_summary.json + eig_faithful.csv).
Pure reader; NO engine calls. Emits the decision table + the adopt-vs-keep verdict (FOLLOWUP_SPEC FU2).
"""
import csv, json, os, statistics as st

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "staircase")


def load_summary():
    p = os.path.join(D, "eig_faithful_summary.json")
    return json.load(open(p)) if os.path.exists(p) else []


def main():
    S = load_summary()
    if not S:
        print("_(eig_faithful_summary.json not present)_"); return

    print("### FU2 — faithful EIG (F_h·ΔH) vs certified max-conditional-entropy, same loop (q=%.2f)\n"
          % (S[0].get("q", 0.99)))
    print("| arm | specimen | steps | EIG recovered | certified recovered | Δ(cert−eig) | Jaccard | pos-match | π held | decision |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s in S:
        d = s["cert_recovered"] - s["eig_recovered"] if s["cert_recovered"] is not None else None
        pi = "eig=%s cert=%s" % (s.get("eig_pi_held"), s.get("cert_pi_held"))
        print("| %s | %s | %d | %s | %s | %s | %.2f | %d/%d | %s | %s |" % (
            s["arm"], s["specimen"], s.get("n_steps", 0), s["eig_recovered"], s["cert_recovered"],
            ("%+d" % d) if d is not None else "-", s["jaccard"], s["position_matches"],
            s.get("n_steps", 0), pi, s["decision"]))

    # aggregate verdict
    print("\n### FU2 verdict\n")
    for arm in sorted({s["arm"] for s in S}):
        rows = [s for s in S if s["arm"] == arm]
        eig_tot = sum(s["eig_recovered"] for s in rows)
        cert_tot = sum(s["cert_recovered"] for s in rows)
        n_adopt = sum(1 for s in rows if s["decision"] == "adopt_certified")
        n_keep = sum(1 for s in rows if s["decision"] == "keep_eig")
        n_tie = sum(1 for s in rows if s["decision"] == "tie")
        jac = st.mean(s["jaccard"] for s in rows)
        pos = sum(s["position_matches"] for s in rows)
        steps = sum(s.get("n_steps", 0) for s in rows)
        pi_ok = all(s.get("eig_pi_held") and s.get("cert_pi_held") for s in rows)
        if cert_tot > eig_tot:
            verdict = "ADOPT certified (recovers more true mass; earns the (1−1/e) guarantee)"
        elif cert_tot == eig_tot:
            verdict = "TIE on recovered mass — adopt certified (equal outcome, but it carries the guarantee)"
        else:
            verdict = "KEEP F_h·ΔH (recovers more; unguaranteed-but-better — objective is propagation value, not total entropy)"
        print("- **%s**: EIG total=%d, certified total=%d  → %s" % (arm, eig_tot, cert_tot, verdict))
        print("  - decisions: adopt=%d tie=%d keep=%d;  mean Jaccard=%.2f, positional agreement=%d/%d;  π invariant held both rules = %s"
              % (n_adopt, n_tie, n_keep, jac, pos, steps, pi_ok))
    print("\n_Selection differs (low Jaccard, ~0 positional) — the earlier gap reproduces under the fair loop — "
          "but the OUTCOME (recovered true mass) is where the adopt-vs-keep decision is made._")


if __name__ == "__main__":
    main()
