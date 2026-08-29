#!/usr/bin/env python3
"""OLLVM staleness probe — per-transform analysis from optregime.csv.

The question here is NARROWER than the opt-regime probe. We are not asking whether the
signature can *steer* a bank (Phase B); we are asking, per transform, the prior question a
new bank regime has to earn first:

    Under the benign (clean OLLVM-clang) calibration map, does the transform push
    instruction-level ECE MATERIALLY above the benign held-out baseline?

That is calibration *staleness*: the benign map, fit on clean code, no longer keeps the
posterior honest on the transformed code. If a transform decodes cleanly and stays
well-calibrated under the benign map (ECE ≈ benign baseline, oracle ≈ default), there is
nothing for a switch to repair and we do NOT claim it as a bank regime. If it induces real
staleness (ECE materially up, and a transform-specific oracle map recovers it), it is a
candidate regime.

Reads the per-binary CSV the (unmodified) optregime harness emits. `--default benign`, so
`ece_default` = the benign map applied to every regime = the staleness measurement.

Gate, per transform T (held-out means):
  * degradation  = ece_default(T) - ece_default(benign)           (drift vs baseline)
  * ratio        = ece_default(T) / ece_default(benign)
  * repairable   = ece_default(T) - ece_oracle(T)                 (what a T-map could recover)
  STALE (candidate regime) iff  ratio >= 1.5x OR degradation >= 0.010 ECE,
  AND the gap is repairable (oracle materially below default). Else CLEAN (no-go).
"""
import csv
import sys
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else "optregime.csv"
DEFAULT = sys.argv[2] if len(sys.argv) > 2 else "benign"

# Gate thresholds (same "material" bar as the opt-regime Gate A).
RATIO_GATE = 1.5
ABS_GATE = 0.010
REPAIR_GATE = 0.005  # oracle must sit at least this far below default for a switch to have a job


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            def fnum(k):
                v = r[k]
                return None if v == "NA" else float(v)
            rows.append(dict(
                stem=r["stem"], level=r["level"], split=r["split"],
                n=int(r["n"]), code_bytes=int(r["code_bytes"]),
                entropy=float(r["entropy"]), base_rate=float(r["base_rate"]),
                s_glob=float(r["s_glob"]), s_spat=float(r["s_spat"]),
                ece_raw=fnum("ece_raw"), ece_default=fnum("ece_default"),
                ece_oracle=fnum("ece_oracle"), ece_sig=fnum("ece_sig"),
                ece_se=fnum("ece_se"), sig_pick=r["sig_pick"], se_pick=r["se_pick"],
            ))
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def sem(xs):
    xs = [x for x in xs if x is not None]
    return float(np.std(xs, ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else float("nan")


def main():
    rows = load(CSV)
    # Transform order: benign first, then the rest as they appear.
    order = []
    for r in rows:
        if r["level"] not in order:
            order.append(r["level"])
    if DEFAULT in order:
        order = [DEFAULT] + [l for l in order if l != DEFAULT]

    allr = rows
    hold = [r for r in rows if r["split"] == "holdout"]
    nfit = len(set(r["stem"] for r in rows if r["split"] == "fit"))
    nhold = len(set(r["stem"] for r in hold))

    print(f"# OLLVM staleness probe — analysis of `{CSV}`\n")
    print(f"- transforms: {order} | benign (default) map fit on the clean OLLVM-clang build")
    print(f"- binaries: {len(rows)} total, {len(hold)} held-out "
          f"({nfit} fit programs, {nhold} holdout programs); split by program (all 4 builds share a fate)\n")

    base_default = mean([r["ece_default"] for r in hold if r["level"] == DEFAULT])
    base_oracle = mean([r["ece_oracle"] for r in hold if r["level"] == DEFAULT])

    # ── Main table: per-transform staleness under the benign map + cavity surprise ──
    print("## Per-transform staleness (benign calibration map) + cavity surprise\n")
    print("Held-out means. `raw` = uncalibrated ECE; **`benign-map ECE`** = the benign map applied "
          "unchanged (the staleness measurement); `oracle ECE` = the transform's own map (the repair "
          "ceiling). `S_glob` = mean cavity surprise, `S_spat` = Moran's I of the standardized residual.\n")
    print("| transform | n | raw ECE | **benign-map ECE** | oracle ECE | drift vs benign | ratio | repairable | S_glob | S_spat |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    verdict = {}
    for lv in order:
        rs = [r for r in hold if r["level"] == lv]
        if not rs:
            continue
        raw = mean([r["ece_raw"] for r in rs])
        dfl = mean([r["ece_default"] for r in rs])
        ora = mean([r["ece_oracle"] for r in rs])
        sg = mean([r["s_glob"] for r in rs])
        sp = mean([r["s_spat"] for r in rs])
        drift = dfl - base_default
        ratio = dfl / base_default if base_default > 0 else float("nan")
        repair = dfl - ora
        star = " (baseline)" if lv == DEFAULT else ""
        print(f"| {lv}{star} | {len(rs)} | {raw:.4f} | {dfl:.4f} | {ora:.4f} | {drift:+.4f} | "
              f"{ratio:.2f}× | {repair:+.4f} | {sg:.3f} | {sp:.3f} |")
        verdict[lv] = dict(raw=raw, dfl=dfl, ora=ora, drift=drift, ratio=ratio, repair=repair,
                           sg=sg, sp=sp, n=len(rs))
    print(f"\n*Benign held-out baseline: benign-map ECE = {base_default:.4f}, oracle = {base_oracle:.4f}.*\n")

    # ── Full-corpus cavity signature (fit+holdout) for context: does surprise even move? ──
    print("## Cavity signature across the whole corpus (fit + holdout)\n")
    print("| transform | n | S_glob mean±sem | S_spat mean±sem | .text KB (mean) | byte-entropy |")
    print("|---|---|---|---|---|---|")
    for lv in order:
        rs = [r for r in allr if r["level"] == lv]
        if not rs:
            continue
        sg = [r["s_glob"] for r in rs]
        sp = [r["s_spat"] for r in rs]
        kb = mean([r["code_bytes"] for r in rs]) / 1024.0
        ent = mean([r["entropy"] for r in rs])
        print(f"| {lv} | {len(rs)} | {mean(sg):.3f} ± {sem(sg):.3f} | {mean(sp):.3f} ± {sem(sp):.3f} "
              f"| {kb:.1f} | {ent:.3f} |")
    print()

    # ── Per-transform go/no-go ──
    print("## Per-transform go / no-go\n")
    print(f"Gate: STALE iff (ratio ≥ {RATIO_GATE:.1f}× OR drift ≥ {ABS_GATE:.3f} ECE) AND "
          f"repairable ≥ {REPAIR_GATE:.3f} (a transform-specific map actually recovers the gap).\n")
    print("| transform | benign-map ECE | drift | ratio | repairable | verdict |")
    print("|---|---|---|---|---|---|")
    for lv in order:
        if lv == DEFAULT or lv not in verdict:
            continue
        v = verdict[lv]
        stale = (v["ratio"] >= RATIO_GATE or v["drift"] >= ABS_GATE) and v["repair"] >= REPAIR_GATE
        tag = "**STALE → candidate bank regime**" if stale else "CLEAN → no-go (do not claim)"
        print(f"| {lv} | {v['dfl']:.4f} | {v['drift']:+.4f} | {v['ratio']:.2f}× | {v['repair']:+.4f} | {tag} |")
    print()
    print("_Benign is the baseline, not gated._\n")


if __name__ == "__main__":
    main()
