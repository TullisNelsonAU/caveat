#!/usr/bin/env python3
"""Phase-2 convergence / contraction sweep for the udstack fixpoint.

For each specimen, K ∈ {2, 3}, and λ ∈ {0.1 … 1.0}, run the coupled relaxation to a tight ε and record
the empirical contraction ratio ρ ≈ limsup ‖Δ^{t+1}‖/‖Δ^{t}‖ (log-odds space) and the iteration count.
The damped log-odds update is `logit(bel)^{t+1} = (1−λ)·logit^t + λ·(S1 message)`, so near the fixpoint
it is the affine map `Δ^{t+1} = [(1−λ)I + λ M] Δ^t` where `M` is the coupled message Jacobian; ρ is the
spectral radius of that map. The question this answers: for what λ, K is the sweep a contraction, and
how fast.

Emits a ρ(λ,K) table + the iterations-to-ε per λ, and flags any non-contracting regime (ρ ≥ 1). Uses
the existing corpora only (code-in-data + desync coreutils).

Usage: udstack_converge.py            # runs the default corpus subset
       udstack_converge.py --full     # all 5 code-in-data + 5 desync
"""
import subprocess
import sys
import os
import glob

BIN = os.path.expanduser("~/lab/projects/upd-suite-stack/target/release/udstack")
CID = "/tmp/cid"
DES = os.path.expanduser("~/lab/projects/probablistic/corpus/desync-pilot/unstripped")
DGT = os.path.expanduser("~/lab/projects/probablistic/corpus/desync-gt")
# Desync FUNC GT is generated once from the unstripped .symtab (objdump -t) — same rule as gen_func_gt.
DES_FGT = "/tmp/udstack_desync_funcgt"

LAMBDAS = [round(0.1 * i, 1) for i in range(1, 11)]
EPS = 1e-8
MAX_SWEEPS = 300


def desync_func_gt(binpath, stem):
    os.makedirs(DES_FGT, exist_ok=True)
    out = os.path.join(DES_FGT, stem + ".func.gt")
    if not os.path.exists(out):
        t = subprocess.run(["objdump", "-t", binpath], capture_output=True, text=True).stdout
        addrs = sorted({int(ln.split()[0], 16) for ln in t.splitlines() if " F .text\t" in ln})
        open(out, "w").write("".join(f"0x{a:x}\n" for a in addrs))
    return out


def run(elf, gt, fgt, k, lam):
    r = subprocess.run(
        [BIN, elf, gt, "--func-gt", fgt, "--milestone", "b", "--layers", str(k),
         "--lambda", str(lam), "--eps", str(EPS), "--max-sweeps", str(MAX_SWEEPS), "--trace"],
        capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        if ln.startswith("stack_rho,"):
            _, l, kk, iters, conv, rho, final = ln.split(",")
            return dict(iters=int(iters), converged=conv == "true", rho=float(rho))
    return None


def specimens(full):
    cid = sorted(glob.glob(f"{CID}/*__native-code-in-data.elf"))
    if not full:
        cid = cid[:4]
    out = []
    for e in cid:
        s = e[:-4]
        out.append((os.path.basename(s).replace("gcc_coreutils_64_O2_", "").replace("__native-code-in-data", ""),
                    e, s + ".gt", s + ".func.gt"))
    des = sorted(glob.glob(f"{DES}/desync_coreutils_64_O0_*"))
    des = [d for d in des if "[" not in d]
    if not full:
        des = des[:3]
    for d in des:
        stem = os.path.basename(d)
        out.append((stem.replace("desync_coreutils_64_O0_", "des_"),
                    d, os.path.join(DGT, stem + ".gt"), desync_func_gt(d, stem)))
    return out


def main():
    full = "--full" in sys.argv
    specs = specimens(full)
    # rows[(k, lam)] = list of (rho, iters, converged)
    rows = {}
    for name, elf, gt, fgt in specs:
        for k in (2, 3):
            for lam in LAMBDAS:
                res = run(elf, gt, fgt, k, lam)
                if res:
                    rows.setdefault((k, lam), []).append(res)
        sys.stderr.write(f"  done {name}\n")

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"# ρ(λ,K) contraction sweep — {len(specs)} binaries, ε={EPS}, mean over binaries")
    print(f"{'λ':>4} | {'K=2 ρ':>7} {'K=2 iters':>9} {'K=2 conv':>8} | {'K=3 ρ':>7} {'K=3 iters':>9} {'K=3 conv':>8}")
    worst = 0.0
    for lam in LAMBDAS:
        c = []
        for k in (2, 3):
            r = rows.get((k, lam), [])
            c.append((mean([x["rho"] for x in r]), mean([x["iters"] for x in r]),
                      sum(x["converged"] for x in r), len(r)))
            worst = max(worst, max((x["rho"] for x in r), default=0.0))
        print(f"{lam:>4} | {c[0][0]:>7.3f} {c[0][1]:>9.1f} {c[0][2]:>4}/{c[0][3]:<3} | "
              f"{c[1][0]:>7.3f} {c[1][1]:>9.1f} {c[1][2]:>4}/{c[1][3]:<3}")
    print(f"\nworst-case ρ over all (binary, λ, K) = {worst:.4f}  "
          f"({'all contract (ρ<1)' if worst < 1.0 else 'NON-CONTRACTING regime present'})")


if __name__ == "__main__":
    main()
