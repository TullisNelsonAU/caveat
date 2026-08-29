#!/usr/bin/env python3
"""Complementary decode-side staircase (SPEC sec 2a) -- the decoy-pruning magnitude that the calibrated
posterior's entropy UNDER-states.

The honesty wall means bench's raw per-byte posterior is byte-identical E0..E3, and self-calibration at
E0 already extracts most instruction-start RESOLUTION, so the entropy staircase U_k drops only modestly.
The reachability-confirmation win on decoy-heavy shows up instead in the DECODE: the decoy-leak (junk
starts selected) collapses when the true entry anchors a confirmation closure. This script measures that
with `bench` (read-only): decoy-leak floor and precision-at-matched-recall for cover (E1) vs
confirm-soft (E2). The anchored-vs-anchorless prediction is tested across decoy STRUCTURE -- disconnected
(decoys off every path from entry -> anchor prunes -> leak->0) vs self-anchoring (decoys form their own
reachable islands -> the entry anchor cannot prune them).

One bench process per (binary, mode). One binary in memory at a time.

Usage: staircase_decode.py [--out docs/staircase/decode_leak.csv]
"""
import argparse, csv, glob, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import staircase_measure as S

ROOT = S.ROOT
BENCH = os.path.join(ROOT, "upd-suite/target/release/bench")
BIASES = ",".join("%.2f" % (b / 4) for b in range(-80, 81))


def decoy_from(spec):
    for (s, e, lab, kind) in S.read_regions(spec.get("regions")):
        if kind == "junk_decoy":
            return s
    return None


def run(elf, gt, dfrom, extra):
    cmd = [BENCH, elf, gt, "--biases", BIASES]
    if dfrom is not None:
        cmd += ["--decoy-from", "0x%x" % dfrom]
    cmd += extra
    o = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    rows = []
    for ln in o.stdout.splitlines():
        p = ln.split(",")
        if len(p) == 7 and p[0] not in ("threshold", "bias", "calibration"):
            try:
                rows.append((float(p[3]), float(p[4]), int(p[6])))   # recall, precision, decoy_leak
            except ValueError:
                pass
    return rows


def leak_floor(rows, min_recall=0.30):
    c = [r for r in rows if r[0] >= min_recall]
    return min((r[2] for r in c), default=None)


def prec_at(rows, t=0.50):
    c = [r for r in rows if r[0] >= t]
    return max((r[1] for r in c), default=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/staircase/decode_leak.csv")
    args = ap.parse_args()
    specs = list(S.cid_specs()) + list(S.decoy_specs())
    modes = {"E1_cover": ["--cover"], "E2_confirm": ["--confirm-soft"]}
    out = open(args.out, "w", newline="")
    w = csv.DictWriter(out, fieldnames=["binary", "obf", "struct", "mode", "leak_floor",
                                        "prec_at_0_5", "decoy_from"])
    w.writeheader()
    for spec in specs:
        if not os.path.exists(spec["elf"]) or not os.path.exists(spec["gt"]):
            continue
        d = decoy_from(spec)
        if d is None:
            print("  SKIP %s (no junk_decoy region)" % spec["name"]); continue
        for m, extra in modes.items():
            try:
                rows = run(spec["elf"], spec["gt"], d, extra)
                lf = leak_floor(rows); pr = prec_at(rows)
                status = "ok" if rows else "no_rows"
            except Exception as ex:
                lf = pr = None; status = "ERR:%s" % str(ex)[:30]
            w.writerow(dict(binary=spec["name"], obf=spec["obf"], struct=spec["struct"], mode=m,
                            leak_floor=lf, prec_at_0_5=pr, decoy_from="0x%x" % d))
            out.flush()
            print("  %-34s %-11s leak_floor=%s prec@0.5=%s %s" % (
                spec["name"], m, lf, ("%.3f" % pr) if pr is not None else "-", status))
    out.close()
    print("-> %s" % args.out)


if __name__ == "__main__":
    main()
