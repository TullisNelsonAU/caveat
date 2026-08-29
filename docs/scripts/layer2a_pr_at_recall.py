#!/usr/bin/env python3
"""Precision + decoy-leak at fixed matched recall: the honest P/R-dominance test."""
import subprocess, glob, os

BENCH = os.path.expanduser("~/lab/projects/upd-suite/target/release/bench")
TARGETS = [0.70, 0.80, 0.90, 0.95]

def decoy_from(stem):
    for l in open(stem+".regions"):
        if "junk_decoy" in l: return int(l.split()[0],16)

def run(elf, gt, d, extra):
    cmd=[BENCH,elf,gt,"--decoy-from",hex(d),
         "--biases",",".join(f"{b/4:.2f}" for b in range(-60,61))]+extra
    o=subprocess.run(cmd,capture_output=True,text=True)
    rows=[]
    for ln in o.stdout.splitlines():
        p=ln.split(",")
        if len(p)==7 and p[0] not in("bias","calibration"):
            try: rows.append((float(p[3]),float(p[4]),int(p[6])))  # rec,prec,leak
            except ValueError: pass
    return rows

def at_recall(rows,t):
    """highest-precision row with recall>=t (feasible operating point at that recall)."""
    c=[r for r in rows if r[0]>=t-1e-9]
    if not c: return None
    return max(c,key=lambda r:r[1])

modes={"cover":["--cover"],
       "reach g4":["--reach","--gamma","4"],
       "reach g8":["--reach","--gamma","8"],
       "reach g16":["--reach","--gamma","16"]}
agg={m:{t:[] for t in TARGETS} for m in modes}
for elf in sorted(glob.glob("/tmp/cid/*__native-code-in-data.elf")):
    stem=elf[:-4]; gt=stem+".gt"; d=decoy_from(stem)
    for m,ex in modes.items():
        rows=run(elf,gt,d,ex)
        for t in TARGETS:
            r=at_recall(rows,t)
            if r: agg[m][t].append(r)
print("mean precision (and decoy-leak) at fixed recall, across 5 specimens")
print(f"{'recall':>7} | "+" | ".join(f"{m:>16}" for m in modes))
for t in TARGETS:
    cells=[]
    for m in modes:
        v=agg[m][t]
        if not v:
            cells.append("unreachable"); continue
        p=sum(x[1] for x in v)/len(v); lk=sum(x[2] for x in v)/len(v)
        cells.append(f"P={p:.3f} lk={lk:5.0f} (n={len(v)})")
    print(f"{t:>7.2f} | "+" | ".join(f"{c:>16}" for c in cells))
