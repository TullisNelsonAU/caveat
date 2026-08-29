#!/usr/bin/env python3
"""Turn `boundaries.csv` into `DOWNSTREAM_RECOVERY_RESULTS.md`.

Every number in the report is computed here from the raw per-binary counts. Nothing is hand-typed,
so the doc cannot drift from the run that produced it — re-run this after any re-run of
`run_downstream.sh` and the tables regenerate.

Aggregation is micro (pool tp/fp/fn across the regime's held-out binaries) because that is what an
analyst actually experiences: they work one pile of recovered function heads, and a bigger binary
contributes more of that pile. Per-binary macro means are carried in the appendix so a reader can
see whether the micro headline is being driven by one large specimen.

Three cells are deliberately empty rather than filled with a convenient number:
  * packed recall and F1 — the packed GT is UPX's own b_info chain, which proves a window is
    compressed *data*. It yields negatives only. There are no known packed function heads, so recall
    has no denominator and we do not invent one. What separates the arms on packed is how much
    provable data each one mistook for a function, and that gets its own table.
  * precision where an arm recovered nothing — an empty head set has no precision.
  * tigress, when the graded corpus is not on disk. Absent is reported as absent, never as passing.
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
GROUPS = ["benign", "packed", "desync", "tigress", "legit_vm"]
GROUP_LABEL = {
    "benign": "benign (clean coreutils)",
    "packed": "packed (UPX NRV + LZMA)",
    "desync": "desync (junk-insertion obfuscation)",
    "tigress": "tigress (semantic obfuscation — held-out blind spot)",
    "legit_vm": "legit-VM (false-positive gate; true regime benign)",
}
# The arm the prose calls "the switch". The guarded rule is the one that ships.
SWITCH = "switch_guard"


def group_of(row):
    if row["sublabel"].startswith("tig"):
        return "tigress"
    if row["sublabel"].startswith("vm"):
        return "legit_vm"
    return {"benign": "benign", "packed": "packed", "obfuscated": "desync"}[row["regime"]]


def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def harmonic(p, r):
    if p is None or r is None:
        return None
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


class Cell:
    """Pooled counts for one (group, arm, τ)."""

    FIELDS = ("tp", "fp", "fn", "n_pred", "n_gt", "n_reach", "n_window", "win_pred")

    def __init__(self):
        self.n_bins = 0
        for f in self.FIELDS:
            setattr(self, f, 0)
        self.macro_f1 = []
        self.n_empty = 0
        self.picks = defaultdict(int)

    def add(self, r):
        self.n_bins += 1
        for f in self.FIELDS:
            setattr(self, f, getattr(self, f) + int(r[f]))
        tp, fp, n_gt = int(r["tp"]), int(r["fp"]), int(r["n_gt"])
        p = tp / (tp + fp) if tp + fp else None
        rec = tp / n_gt if n_gt else None
        f1 = harmonic(p, rec)
        if f1 is not None:
            self.macro_f1.append(f1)
        if int(r["n_pred"]) == 0:
            self.n_empty += 1
        self.picks[r["pick"]] += 1

    @property
    def precision(self):
        d = self.tp + self.fp
        return self.tp / d if d else None

    @property
    def recall(self):
        return self.tp / self.n_gt if self.n_gt else None

    @property
    def recall_reach(self):
        return self.tp / self.n_reach if self.n_reach else None

    @property
    def f1(self):
        return harmonic(self.precision, self.recall)

    @property
    def f1_reach(self):
        return harmonic(self.precision, self.recall_reach)

    @property
    def window_fa(self):
        return self.win_pred / self.n_window if self.n_window else None


def fmt(v, pct=False):
    if v is None:
        return "—"
    return f"{100 * v:.1f}%" if pct else f"{v:.4f}"


def signed(a, b):
    """a − b, formatted, or an em dash when either side is undefined."""
    if a is None or b is None:
        return "—"
    return f"{a - b:+.4f}"


def bucket(rows):
    cells = defaultdict(Cell)
    for r in rows:
        cells[(group_of(r), r["arm"], float(r["tau"]))].add(r)
    return cells


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "boundaries.csv")
    rows = load(csv_path)
    if not rows:
        sys.exit(f"no rows in {csv_path}")
    cells = bucket(rows)
    taus = sorted({float(r["tau"]) for r in rows})
    hi = max(taus)

    base = os.path.dirname(csv_path) or "."
    meta_path = os.path.join(base, "boundaries_meta.csv")
    metas = load(meta_path) if os.path.exists(meta_path) else []
    sum_path = os.path.join(base, "boundaries_summary.json")
    summary = json.load(open(sum_path)) if os.path.exists(sum_path) else {}
    null = summary.get("clean_fit_null", {})
    glob_hi, spat_hi = null.get("s_glob_hi"), null.get("s_spat_hi")

    present = [g for g in GROUPS if any(k[0] == g for k in cells)]
    out = []
    w = out.append

    w("# Downstream recovery: function boundaries under a stale calibration map")
    w("")
    w("Generated by `analyze_recovery.py` from `boundaries.csv`. Do not hand-edit.")
    w("")
    w("The task is function-boundary recovery — the first thing any tool must do to a stripped")
    w("binary. A head is recovered when a direct call reaches it and **both** ends of that call clear")
    w("τ under the arm's calibrated posterior, so the recovered set is a function of the calibration")
    w("rather than of the decoder. Ground truth is the unstripped original's `.symtab` `STT_FUNC`")
    w("entries — symbol-table rows, never a disassembly.")
    w("")

    # ── 1. The headline ────────────────────────────────────────────────────────
    w(f"## 1. Corrupt → recover: boundary F1 at τ = {hi}")
    w("")
    w("| regime | stale | switch | oracle | drop (stale − oracle) | recovery (switch − stale) |")
    w("|---|---|---|---|---|---|")
    for g in present:
        s, sw, o = (cells.get((g, a, hi)) for a in ("stale", SWITCH, "oracle"))
        if not (s and sw and o):
            continue
        w("| %s | %s | %s | %s | %s | %s |" % (
            GROUP_LABEL[g], fmt(s.f1), fmt(sw.f1), fmt(o.f1),
            signed(s.f1, o.f1), signed(sw.f1, s.f1)))
    w("")
    w("Packed is dashed by construction, not by omission: with negatives-only ground truth there are")
    w("no known heads, so recall and F1 have no denominator. Section 4 reports what does separate the")
    w("arms there.")
    w("")

    # ── 2. The detector ───────────────────────────────────────────────────────
    if summary.get("detect"):
        w("## 2. The detector, read against the clean-fit null")
        w("")
        if glob_hi is not None:
            w("The null is fit on the benign FIT split alone and never sees a held-out binary or a label:")
            w(f"`S_glob > {glob_hi:.4f}` ⇒ obfuscated, else `S_spat > {spat_hi:.4f}` ⇒ packed, else benign.")
            w("")
        w("| group | n | mean S_glob | mean S_spat | S_glob fires | S_spat fires | rule acc | guard acc |")
        w("|---|---|---|---|---|---|---|---|")
        for d in summary["detect"]:
            w("| %s | %d | %.4f | %.4f | %d/%d | %d/%d | %s | %s |" % (
                GROUP_LABEL.get(d["group"], d["group"]), d["n"],
                d["mean_s_glob"], d["mean_s_spat"],
                d["n_glob_fire"], d["n"], d["n_spat_fire"], d["n"],
                fmt(d["rule_accuracy"], pct=True), fmt(d["guard_accuracy"], pct=True)))
        w("")
        tot = sum(d["n"] for d in summary["detect"])
        if tot:
            gacc = sum(d["guard_accuracy"] * d["n"] for d in summary["detect"]) / tot
            racc = sum(d["rule_accuracy"] * d["n"] for d in summary["detect"]) / tot
            w("Pooled GT-free selection accuracy over all %d held-out binaries: rule **%s**, guarded **%s**."
              % (tot, fmt(racc, pct=True), fmt(gacc, pct=True)))
            w("")

    # ── 3. The full table ─────────────────────────────────────────────────────
    w("## 3. Boundary precision / recall / F1, per regime and threshold")
    w("")
    w("`F1_reach` restricts recall to heads that are the target of some direct call in the superset —")
    w("the structural ceiling of this task rule, identical across arms. It cannot manufacture the arm")
    w("differences in section 1; it is here so the absolute numbers are readable.")
    w("")
    for g in present:
        if not any((g, a, t) in cells for t in taus for a in ARMS):
            continue
        w(f"### {GROUP_LABEL[g]}")
        w("")
        w("| τ | arm | precision | recall | F1 | F1_reach | heads recovered | routed to |")
        w("|---|---|---|---|---|---|---|---|")
        for t in taus:
            for a in ARMS:
                c = cells.get((g, a, t))
                if not c:
                    continue
                picks = ", ".join(f"{k}×{v}" for k, v in sorted(c.picks.items(), key=lambda kv: -kv[1]))
                w("| %.1f | %s | %s | %s | %s | %s | %d | %s |" % (
                    t, ARM_LABEL[a], fmt(c.precision, pct=True), fmt(c.recall, pct=True),
                    fmt(c.f1), fmt(c.f1_reach), c.n_pred, picks))
        w("")

    # ── 4. Packed ─────────────────────────────────────────────────────────────
    if any(k[0] == "packed" for k in cells):
        w("## 4. Packed: what separates the arms when there are no positives")
        w("")
        w("Every function head nominated inside UPX's provable-data window is false by construction.")
        w("The denominator is the call targets that land in that window at all.")
        w("")
        w(f"| arm | provable-data window nominated as function heads (τ = {hi}) |")
        w("|---|---|")
        for a in ARMS:
            c = cells.get(("packed", a, hi))
            if c:
                w("| %s | %s (%d / %d) |" % (
                    ARM_LABEL[a], fmt(c.window_fa, pct=True), c.win_pred, c.n_window))
        w("")

    # ── 5. Appendix ───────────────────────────────────────────────────────────
    w("## 5. Appendix")
    w("")
    w("### 5.1 Macro (per-binary mean) F1, against the micro headline")
    w("")
    w("| regime | τ | arm | micro F1 | macro F1 | binaries | recovered nothing |")
    w("|---|---|---|---|---|---|---|")
    for g in present:
        for t in taus:
            for a in ARMS:
                c = cells.get((g, a, t))
                if not c:
                    continue
                macro = sum(c.macro_f1) / len(c.macro_f1) if c.macro_f1 else None
                w("| %s | %.1f | %s | %s | %s | %d | %d |" % (
                    g, t, ARM_LABEL[a], fmt(c.f1), fmt(macro), c.n_bins, c.n_empty))
    w("")

    missing = [g for g in GROUPS if g not in present]
    if missing:
        w("### 5.2 Groups absent from this run")
        w("")
        for g in missing:
            w(f"* **{GROUP_LABEL[g]}** — no held-out binaries; the corpus was not on disk for this run.")
        w("")

    if metas:
        w("### 5.3 Calibration bookkeeping (drift guard against `switching`)")
        w("")
        w("| regime | binaries | mean ECE always-benign | mean ECE oracle |")
        w("|---|---|---|---|")
        by = defaultdict(list)
        for m in metas:
            by[group_of(m)].append(m)
        for g in present:
            ms = by.get(g)
            if not ms:
                continue
            ab = sum(float(m["ece_always_benign"]) for m in ms) / len(ms)
            orc = sum(float(m["ece_oracle"]) for m in ms) / len(ms)
            w("| %s | %d | %.4f | %.4f |" % (GROUP_LABEL[g], len(ms), ab, orc))
        w("")

    dst = os.path.join(base, "DOWNSTREAM_RECOVERY_RESULTS.md")
    with open(dst, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"wrote {dst} ({len(out)} lines) from {len(rows)} rows")


if __name__ == "__main__":
    main()
