#!/usr/bin/env python3
"""Per-packer three-arm + signature-routing summary for the packer-breadth run.

Reads breadth_main.csv (+ optional breadth_ezuri.csv), groups held-out packed binaries by their
packer sublabel, and reports:
  - stale (ece_always_benign) / oracle (ece_oracle) / switch (ece_rule) ECE,
  - the GT-free signature routing (rule_pick distribution) + selection accuracy vs true=packed,
  - analyzed-region entropy and the (S_glob, S_spat) benign-engine signature.

ECE validity is packer-dependent and stated per row:
  in-band  (upxnrv/upxlzma = compressed, kite = RC4 tail) → provable-data oracle, ECE meaningful.
  out-of-band (kiten loader, ezuri Go .text) → analyzed region is genuine code, no in-band data
             oracle; ECE columns are N/A and only routing/signature are reported.
"""
import csv, sys, statistics as st
from pathlib import Path

HERE = Path(__file__).parent
IN_BAND = {"upxnrv", "upxlzma", "kite"}       # provable-data oracle exists
OUT_BAND = {"kiten", "ezuri"}                 # analyzed region = genuine code; ECE N/A

DESC = {
    "upxnrv":  "UPX NRV2 (compression, in-band)",
    "upxlzma": "UPX LZMA (compression, in-band)",
    "kite":    "kiteshield RC4 per-fn (in-band, encrypted tail)",
    "kiten":   "kiteshield -n (loader stub only, out-of-band)",
    "ezuri":   "Ezuri AES-CFB overlay + memfd (out-of-band)",
}

def load(paths):
    rows = []
    for p in paths:
        if not Path(p).exists():
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                if r["regime"] == "packed":
                    rows.append(r)
    return rows

def mean(xs): return sum(xs)/len(xs) if xs else float("nan")

def main():
    rows = load([HERE/"breadth_main.csv", HERE/"breadth_ezuri.csv"])
    by = {}
    for r in rows:
        by.setdefault(r["sublabel"], []).append(r)

    order = ["upxnrv", "upxlzma", "kite", "kiten", "ezuri"]
    print("# Per-packer three-arm ECE + signature routing (true regime = packed)\n")
    print(f"{'packer':8} {'n':>2} {'band':9} {'stale':>7} {'oracle':>7} {'switch':>7} "
          f"{'route→packed':>12} {'S_glob':>7} {'S_spat':>7} {'regionH':>7}")
    print("-"*92)
    summary = {}
    for lab in order:
        rs = by.get(lab, [])
        if not rs: continue
        n = len(rs)
        stale  = mean([float(r["ece_always_benign"]) for r in rs])
        oracle = mean([float(r["ece_oracle"]) for r in rs])
        switch = mean([float(r["ece_rule"]) for r in rs])
        n_packed = sum(1 for r in rs if r["rule_pick"] == "packed")
        sg = mean([float(r["s_glob_benign_eng"]) for r in rs])
        ss = mean([float(r["s_spat_benign_eng"]) for r in rs])
        he = mean([float(r["region_ent"]) for r in rs])
        band = "in-band" if lab in IN_BAND else "OUT-band"
        if lab in IN_BAND:
            se, oe, we = f"{stale:.3f}", f"{oracle:.3f}", f"{switch:.3f}"
        else:
            se = oe = we = "  N/A"   # no in-band data oracle
        print(f"{lab:8} {n:>2} {band:9} {se:>7} {oe:>7} {we:>7} "
              f"{n_packed:>5}/{n:<6} {sg:>7.2f} {ss:>7.3f} {he:>7.3f}")
        summary[lab] = dict(n=n, stale=stale, oracle=oracle, switch=switch,
                            route_packed=n_packed, s_glob=sg, s_spat=ss, region_ent=he)
    print()
    # Routing detail: where does the rule send each packer's binaries?
    print("# Routing detail (rule_pick distribution per packer)")
    for lab in order:
        rs = by.get(lab, [])
        if not rs: continue
        dist = {}
        for r in rs: dist[r["rule_pick"]] = dist.get(r["rule_pick"], 0) + 1
        print(f"  {lab:8} ({DESC[lab]}): {dict(sorted(dist.items()))}")
    return summary

if __name__ == "__main__":
    main()
