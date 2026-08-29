#!/usr/bin/env python3
"""Desync control: reach must NOT beat cover (junk rides live fall-through -> not suppressed)."""
import subprocess, os

BENCH = os.path.expanduser("~/lab/projects/upd-suite/target/release/bench")
D = os.path.expanduser("~/lab/projects/probablistic/corpus/desync-pilot/unstripped")
DGT = os.path.expanduser("~/lab/projects/probablistic/corpus/desync-gt")

def run(elf, gt, extra):
    cmd = [BENCH, elf, gt, "--biases", ",".join(f"{b/2:.2f}" for b in range(-24, 25))] + extra
    out = subprocess.run(cmd, capture_output=True, text=True)
    rows, cal = [], None
    for ln in out.stdout.splitlines():
        p = ln.split(",")
        if p[0] == "calibration":
            cal = ln
        elif len(p) == 6 and p[0] != "bias":
            try:
                rows.append(dict(bias=float(p[0]), n=int(p[1]), tp=int(p[2]),
                                 rec=float(p[3]), prec=float(p[4]), f1=float(p[5])))
            except ValueError:
                pass
    return rows, cal

def best_f1(rows): return max(rows, key=lambda r: r["f1"])
def prec_at(rows, t):
    c=[r for r in rows if r["rec"]>=t-1e-9] or rows
    return max(c, key=lambda r: r["prec"])

for name in ["ls","cat","sort"]:
    elf=f"{D}/desync_coreutils_64_O0_{name}"; gt=f"{DGT}/desync_coreutils_64_O0_{name}.gt"
    cov,cc=run(elf,gt,["--cover"]); rch,rc=run(elf,gt,["--reach"])
    cb=best_f1(cov); rb=best_f1(rch); mr=prec_at(rch,cb["rec"])
    print(f"{name:<6} cover bestF1  R={cb['rec']:.4f} P={cb['prec']:.4f} F1={cb['f1']:.4f}")
    print(f"{'':<6} reach bestF1  R={rb['rec']:.4f} P={rb['prec']:.4f} F1={rb['f1']:.4f}  "
          f"matchR_P={mr['prec']:.4f}")
    print(f"{'':<6} honesty cover: {cc}")
    print(f"{'':<6} honesty reach: {rc}   (identical => honesty wall holds)")
    print()
