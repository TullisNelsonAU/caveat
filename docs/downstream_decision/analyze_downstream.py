#!/usr/bin/env python3
"""Turn `decisions.csv` into `DOWNSTREAM_DECISION_RESULTS.md`.

Every number in the report is computed here from the raw per-binary counts. Nothing is hand-typed,
so the doc cannot drift from the run that produced it — re-run this after any re-run of
`run_downstream.sh` and the tables regenerate.

Aggregation is micro (pool tp/fp/fn across the regime's held-out binaries) because that is what the
analyst actually experiences: they work a pile of accepted addresses, and a bigger binary
contributes more of that pile. Per-binary macro means are carried in the appendix so a reader can
see whether the micro headline is being driven by one large specimen.

Two cells are deliberately empty rather than filled with a convenient number:
  * packed recall — the packed GT is UPX's own b_info chain, which proves a window is compressed
    *data*. It yields negatives only. There are no known packed positives, so recall has no
    denominator and we do not invent one.
  * precision where an arm accepted nothing adjudicable — an empty accept-set has no precision.
"""
import csv
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["stale", "switch_rule", "switch_guard", "oracle"]
ARM_LABEL = {
    "stale": "stale (always-benign)",
    "switch_rule": "switch (rule)",
    "switch_guard": "switch (guarded)",
    "oracle": "oracle (labels)",
}
GROUPS = ["benign", "packed", "desync"]
GROUP_LABEL = {
    "benign": "benign (clean coreutils)",
    "packed": "packed (UPX NRV + LZMA)",
    "desync": "desync (junk-insertion obfuscation)",
    "tigress": "tigress (semantic obfuscation — held-out blind spot)",
    "legit_vm": "legit-VM (false-positive gate; true regime benign)",
}


def group_of(row):
    if row["sublabel"].startswith("tig"):
        return "tigress"
    if row["sublabel"].startswith("vm"):
        return "legit_vm"
    return {"benign": "benign", "packed": "packed", "obfuscated": "desync"}[row["regime"]]


def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


class Cell:
    def __init__(self):
        self.n_bins = 0
        self.tp = self.fp = self.fn = 0
        self.n_accept = self.n_pos = 0
        self.n_window = self.win_accept = 0
        self.macro_prec = []
        self.macro_rec = []
        self.picks = defaultdict(int)

    def add(self, r):
        self.n_bins += 1
        for f in ("tp", "fp", "fn", "n_accept", "n_pos", "n_window", "win_accept"):
            setattr(self, f, getattr(self, f) + int(r[f]))
        tp, fp, n_pos = int(r["tp"]), int(r["fp"]), int(r["n_pos"])
        if tp + fp:
            self.macro_prec.append(tp / (tp + fp))
        if n_pos:
            self.macro_rec.append(tp / n_pos)
        self.picks[r["pick"]] += 1

    @property
    def precision(self):
        d = self.tp + self.fp
        return self.tp / d if d else None

    @property
    def recall(self):
        return self.tp / self.n_pos if self.n_pos else None

    @property
    def window_fa(self):
        return self.win_accept / self.n_window if self.n_window else None


def fmt(v, pct=False):
    if v is None:
        return "—"
    return f"{100 * v:.1f}%" if pct else f"{v:.4f}"


def bucket(rows):
    cells = defaultdict(Cell)
    for r in rows:
        cells[(group_of(r), r["arm"], float(r["tau"]))].add(r)
    return cells


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "decisions.csv")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "DOWNSTREAM_DECISION_RESULTS.md")
    rows = load(csv_path)
    if not rows:
        sys.exit(f"no rows in {csv_path}")
    cells = bucket(rows)
    taus = sorted({float(r["tau"]) for r in rows})
    hi = max(taus)

    meta_path = os.path.join(os.path.dirname(csv_path), "decisions_meta.csv")
    meta = load(meta_path) if os.path.exists(meta_path) else []

    L = []
    w = L.append

    w("# Result — Stale calibration corrupts the analyst's accept-as-code decision; the switch repairs it")
    w("")
    w("> The switching probe reported ECE. A fair reviewer asks: the number moved, so what? This is the")
    w("> answer in the currency that matters downstream. An analyst treats the calibrated confidence as a")
    w("> probability and accepts every address at **confidence ≥ τ** as real code — reading τ = 0.9 as")
    w("> \"I'll be right about 90% of the time.\" That reading is licensed only if the calibration map fits")
    w("> the binary in front of them. Under a stale map it does not, and the threshold quietly stops")
    w("> meaning what it says. Same corpus, same seed, same split, same bank as")
    w("> `docs/consistency_switching/` and `docs/abstention_guard/` — this is a re-reading of the")
    w("> published calibration numbers as a decision, not a new experiment.")
    w("")

    # ── Headline ──
    w("## 1. Headline — what \"confidence ≥ %.1f\" actually buys" % hi)
    w("")
    w("Micro-averaged precision of the accept-as-code decision at τ = %.1f, by regime and arm." % hi)
    w("")
    w("| regime | analyst expects | stale (always-benign) | switch (guarded, GT-free) | oracle (labels) |")
    w("|---|---|---|---|---|")
    for g in GROUPS:
        s, sw, o = (cells.get((g, a, hi)) for a in ("stale", "switch_guard", "oracle"))
        if not (s and sw and o):
            continue
        w("| %s | %.0f%% | %s | %s | %s |" % (
            GROUP_LABEL[g], 100 * hi, fmt(s.precision, pct=True), fmt(sw.precision, pct=True),
            fmt(o.precision, pct=True)))
    w("")
    # Packed precision is 0 by construction whenever anything in the provable-data window is accepted
    # — there are no provable packed positives to offset it — so that row is the same number for every
    # arm and says nothing about which arm is better. The discriminating quantity is how much provable
    # data each arm let through; give it its own table rather than let the degenerate cell mislead.
    if any((("packed", a, hi) in cells) for a in ARMS):
        w("Packed needs its own row: with negatives-only ground truth, precision there is 0% for any")
        w("arm that accepts anything at all, so it cannot separate the arms. What separates them is how")
        w("much provably-compressed data each one admits as code.")
        w("")
        w("| arm | provable-data window admitted as code (τ = %.1f) |" % hi)
        w("|---|---|")
        for a in ARMS:
            c = cells.get(("packed", a, hi))
            if c:
                w("| %s | %s |" % (ARM_LABEL[a], fmt(c.window_fa, pct=True)))
        w("")

    # ── Full tables ──
    w("## 2. Three-arm precision / recall, per regime and threshold")
    w("")
    w("`accepts` is the size of the accept-set the analyst would have to work through. Precision is")
    w("micro-averaged over the regime's held-out binaries and computed over adjudicable acceptances")
    w("(`tp + fp`); recall is over candidate positives.")
    w("")
    for g in GROUPS + ["tigress", "legit_vm"]:
        present = [(t, a) for t in taus for a in ARMS if (g, a, t) in cells]
        if not present:
            continue
        n_bins = max(cells[(g, a, t)].n_bins for t, a in present)
        w("### %s — n = %d held-out binaries" % (GROUP_LABEL[g], n_bins))
        w("")
        if g == "packed":
            w("| τ | arm | precision | recall | accepts | provable-data window admitted as code |")
            w("|---|---|---|---|---|---|")
        else:
            w("| τ | arm | precision | recall | accepts | routed to |")
            w("|---|---|---|---|---|---|")
        for t in taus:
            for a in ARMS:
                c = cells.get((g, a, t))
                if not c:
                    continue
                last = " ".join("%s×%d" % (k, v) for k, v in sorted(c.picks.items()))
                tail = fmt(c.window_fa, pct=True) if g == "packed" else last
                w("| %.1f | %s | %s | %s | %d | %s |" % (
                    t, ARM_LABEL[a], fmt(c.precision, pct=True), fmt(c.recall, pct=True),
                    c.n_accept, tail))
        w("")

    # ── Consequence, computed not asserted ──
    w("## 3. The security consequence")
    w("")
    lines = []
    for g in GROUPS:
        s, sw, o = (cells.get((g, a, hi)) for a in ("stale", "switch_guard", "oracle"))
        if not (s and sw and o):
            continue
        if g == "packed":
            fa_s, fa_w = s.window_fa, sw.window_fa
            if fa_s is not None and fa_w is not None:
                lines.append(
                    "- **packed**: at τ = %.1f the stale map admits **%s** of the bytes UPX's own "
                    "`b_info` chain proves are compressed *data* as if they were code "
                    "(precision %s, not %.0f%%); the switch admits **%s** (oracle **%s**)." % (
                        hi, fmt(fa_s, pct=True), fmt(s.precision, pct=True), 100 * hi,
                        fmt(fa_w, pct=True), fmt(o.window_fa, pct=True)))
        else:
            if s.precision is None or sw.precision is None:
                continue
            dp = sw.precision - s.precision
            dr = (sw.recall - s.recall) if (s.recall is not None and sw.recall is not None) else None
            lines.append(
                "- **%s**: under the stale map the threshold advertises %.0f%% and delivers %s — %s. "
                "The switch moves precision to %s (%s) and recall to %s (%s), against an oracle of "
                "%s / %s." % (
                    g, 100 * hi, fmt(s.precision, pct=True), _gap_phrase(hi, s.precision),
                    fmt(sw.precision, pct=True), _delta_phrase(dp),
                    fmt(sw.recall, pct=True), _delta_phrase(dr),
                    fmt(o.precision, pct=True), fmt(o.recall, pct=True)))
    L.extend(lines)
    w("")

    # The one-line honest statement the spec asks for, assembled from the same numbers.
    s_p = cells.get(("packed", "stale", hi))
    d_s = cells.get(("desync", "stale", hi))
    d_w = cells.get(("desync", "switch_guard", hi))
    w("**One line, honestly:** ")
    w(_one_liner(hi, s_p, d_s, d_w, cells))
    w("")

    # ── Honest limits ──
    w("## 4. Where this does not hold")
    w("")
    tig = cells.get(("tigress", "stale", hi)), cells.get(("tigress", "switch_guard", hi))
    if all(tig):
        ts, tw = tig
        # Whether the switch did anything here is a measured fact: how often it left the benign route,
        # and whether the precision actually moved. Do not assert the blind spot — show it.
        stayed = tw.picks.get("benign", 0)
        total = sum(tw.picks.values())
        d = (tw.precision - ts.precision) if (ts.precision is not None and tw.precision is not None) else None
        w("- **Tigress (semantic obfuscation).** The consistency signature is not expected to fire on\n"
          "  obfuscation that preserves clean decoding. Measured at τ = %.1f: the guarded switch left\n"
          "  %d of %d of these binaries on the benign route, and precision went %s stale → %s switched\n"
          "  (%s). Where the stale map is already %s, there is nothing for the switch to repair; where\n"
          "  it is not, this is the honest limit — the decision is not rescued because the *detector*\n"
          "  never fires, the same blind spot the switching probe reported, in decision currency." % (
              hi, stayed, total, fmt(ts.precision, pct=True), fmt(tw.precision, pct=True),
              _delta_phrase(d), _gap_phrase(hi, ts.precision)))
    vm = cells.get(("legit_vm", "stale", hi)), cells.get(("legit_vm", "switch_guard", hi))
    if all(vm):
        vs, vw = vm
        abstained = vw.picks.get("benign", 0)
        total = sum(vw.picks.values())
        d = (vw.precision - vs.precision) if (vs.precision is not None and vw.precision is not None) else None
        verdict = ("the guard does not damage the decision here"
                   if d is not None and d >= -0.005 else
                   "**the guard costs precision here** — a real false-positive cost, reported straight")
        w("- **Legitimate VMs (false-positive gate).** Benign binaries whose dispatch loops trip the\n"
          "  spatial statistic; the correct action is to abstain and stay benign. At τ = %.1f the guard\n"
          "  abstained on %d of %d, and precision went %s → %s (%s): %s." % (
              hi, abstained, total, fmt(vs.precision, pct=True), fmt(vw.precision, pct=True),
              _delta_phrase(d), verdict))
    w("- **Packed recall is undefined, not zero.** UPX's `b_info` chain proves data, never code, so the")
    w("  packed rows have negatives only. We report the fraction of provable-data candidates admitted as")
    w("  code and leave recall empty rather than invent packed positives.")
    w("- **Precision is over adjudicable acceptances.** On packed that is the provable-data window only;")
    w("  acceptances elsewhere in the packed region (the UPX stub, which is real code) are unlabelled and")
    w("  excluded from both numerator and denominator.")
    w("")

    # ── Appendix ──
    w("## 5. Appendix — macro means and the arm's routing")
    w("")
    w("Per-binary means, to check the micro headline isn't one large specimen talking.")
    w("")
    w("| regime | τ | arm | micro precision | macro precision | micro recall | macro recall | binaries with empty accept-set |")
    w("|---|---|---|---|---|---|---|---|")
    for g in GROUPS:
        for t in taus:
            for a in ARMS:
                c = cells.get((g, a, t))
                if not c:
                    continue
                mp = sum(c.macro_prec) / len(c.macro_prec) if c.macro_prec else None
                mr = sum(c.macro_rec) / len(c.macro_rec) if c.macro_rec else None
                w("| %s | %.1f | %s | %s | %s | %s | %s | %d |" % (
                    g, t, ARM_LABEL[a], fmt(c.precision, pct=True), fmt(mp, pct=True),
                    fmt(c.recall, pct=True), fmt(mr, pct=True),
                    c.n_bins - len(c.macro_prec)))
    w("")
    if meta:
        w("Selection accuracy of the GT-free rules on this split (from `decisions_meta.csv`):")
        w("")
        core = [m for m in meta if not m["sublabel"].startswith(("tig", "vm"))]
        w("| rule | picks true regime |")
        w("|---|---|")
        for col, label in (("rule_pick", "threshold rule"), ("guard_pick", "guarded rule"),
                           ("clf_pick", "nearest-centroid"), ("mmae_pick", "MMAE (S_glob)")):
            acc = sum(1 for m in core if m[col] == m["regime"]) / len(core) if core else 0.0
            w("| %s | %.0f%% (%d/%d) |" % (label, 100 * acc,
                                           sum(1 for m in core if m[col] == m["regime"]), len(core)))
        w("")

    w("## 6. Reproduce")
    w("")
    w("```")
    w("./docs/downstream_decision/run_downstream.sh          # ~1–2 h, serial, resumable")
    w("python3 docs/downstream_decision/analyze_downstream.py # regenerates this file")
    w("```")
    w("")
    w("The shared bank/classifier module is checked against `bin/switching.rs` numerically:")
    w("")
    w("```")
    w("AB_BASELINE=<pre-refactor switching binary> ./docs/downstream_decision/verify_ab.sh")
    w("```")
    w("")

    with open(out_path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote %s (%d lines) from %d decision rows" % (out_path, len(L), len(rows)))


def _gap_phrase(tau, prec, tol=0.005):
    """State which side of the advertised threshold the measured precision fell on.

    The report must read the same whichever way the run comes out. A stale map that happens to be
    *conservative* on a regime (precision above τ) is not a finding to be dressed up as a failure,
    and a tiny gap is not a scandal — so the wording is chosen from the measured sign and size, never
    assumed in advance.
    """
    if prec is None:
        return "no adjudicable acceptances, so the threshold buys nothing measurable"
    d = prec - tau
    if abs(d) <= tol:
        return "matching what it advertises, to within %.1f pts" % (100 * tol)
    if d > 0:
        return "**over**-delivering by %.1f pts (the threshold is conservative here, not unsafe)" % (100 * d)
    return "**short by %.1f pts** — the analyst is over-trusting the number" % (100 * -d)


def _delta_phrase(d, tol=0.005):
    """Signed change, worded so a null result reads as a null result."""
    if d is None:
        return "—"
    if abs(d) <= tol:
        return "essentially unchanged"
    return "%+.1f pts" % (100 * d)


def _one_liner(hi, packed_stale, desync_stale, desync_switch, cells):
    """The single honest sentence the spec asks for, built from the measured numbers."""
    bits = []
    if packed_stale is not None and packed_stale.window_fa is not None:
        bits.append("on packed binaries a stale always-benign calibration map turns \"confidence ≥ %.1f\" "
                    "into a decision that admits %s of provably-compressed data as executable code"
                    % (hi, fmt(packed_stale.window_fa, pct=True)))
    if desync_stale is not None and desync_stale.precision is not None:
        d = desync_stale.precision - hi
        if d < -0.005:
            bits.append("and on junk-inserted code it delivers %s precision where the number promised "
                        "%.0f%%" % (fmt(desync_stale.precision, pct=True), 100 * hi))
        else:
            bits.append("while on junk-inserted code the same stale map happens to stay at %s, at or "
                        "above the %.0f%% promised — the damage is regime-dependent, not uniform"
                        % (fmt(desync_stale.precision, pct=True), 100 * hi))
    if (desync_switch is not None and desync_switch.precision is not None
            and desync_stale is not None and desync_stale.precision is not None):
        oracle = cells.get(("desync", "oracle", hi))
        near = (oracle is not None and oracle.precision is not None
                and abs(desync_switch.precision - oracle.precision) <= 0.01)
        bits.append("— the ground-truth-free switch moves it to %s%s"
                    % (fmt(desync_switch.precision, pct=True),
                       ", matching the label-using oracle" if near else
                       (", against an oracle of %s" % fmt(oracle.precision, pct=True) if oracle else "")))
    if not bits:
        return "_(insufficient data to state the consequence — check the run completed.)_"
    return (" ".join(bits) +
            ", so a confidence threshold carried over from clean binaries is not a safety margin: "
            "what it actually buys depends on a regime the analyst cannot read off the number itself.")


if __name__ == "__main__":
    main()
