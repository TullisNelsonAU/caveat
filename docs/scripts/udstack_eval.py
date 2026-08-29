#!/usr/bin/env python3
"""Layer-3 udstack evaluation (UDSTACK_BUILD_SPEC / LAYER3_STACK_DESIGN §7) on the code-in-data corpus.

Answers:
  Milestone A  does the two-layer stack (bottom-up only) reproduce M2's `bench --fuse` P̂?
  §7 joint-vs-parts  L1-only (π) vs L2-only (R) vs joint stack P̂ — does coupling improve instruction
                     discrimination *and* hold calibration, and is F_h calibrated (function axis)?
  Milestone B  top-down coupled relaxation: convergence (sweeps, λ) and joint P̂ at the fixpoint (Thm 4).
  §7 online    clamp a confirmed function (uncalled-tail real head) → F rises, marginals re-converge.
  §7 ablation  damping λ sweep + with/without (S3) exclusion — convergence behavior.
"""
import glob, os, re, subprocess, sys

ROOT = os.path.expanduser("~/lab/projects/upd-suite-stack/target/release")
UD, BENCH = f"{ROOT}/udstack", f"{ROOT}/bench"
SPECS = sorted(glob.glob("/tmp/cid/*__native-code-in-data.elf"))
TAUS = "0.1,0.3,0.5,0.7,0.9"


def decoy_from(stem):
    for l in open(stem + ".regions"):
        if "junk_decoy" in l:
            return l.split()[0]
    raise SystemExit("no junk_decoy in " + stem)


def run(exe, args):
    return subprocess.run([exe] + args, capture_output=True, text=True)


def name_of(stem):
    return os.path.basename(stem).split("__")[0].replace("gcc_coreutils_64_O2_", "")


def parse_ud(o):
    d = {"phat": []}
    for ln in o.stdout.splitlines():
        p = ln.split(",")
        if p[0] == "stack_instr":
            d["si_" + p[1]] = (float(p[2]), p[3])       # ece, auroc
        elif p[0] == "stack_func":
            d["sf_" + p[1]] = (float(p[2]), p[3], float(p[4]))
        elif p[0] == "stack_converge":
            d["iters"] = int(p[1]); d["converged"] = p[2] == "true"; d["final_delta"] = float(p[3])
    m = re.search(r"CONVERGE: iters=(\d+) converged=(\w+)", o.stderr)
    return d


def bench_phat(elf, gt, dfrom):
    o = run(BENCH, [elf, gt, "--fuse", "--decoy-from", dfrom])
    for ln in o.stdout.splitlines():
        if ln.startswith("fuse_calib,phat"):
            p = ln.split(","); return (float(p[2]), p[3])
    return None


rows = []
for elf in SPECS:
    stem, name = elf[:-4], name_of(elf[:-4])
    gt, fgt, dfrom = stem + ".gt", stem + ".func.gt", decoy_from(stem)
    print(f"[{name}] running…", file=sys.stderr, flush=True)
    a = parse_ud(run(UD, [elf, gt, "--func-gt", fgt, "--decoy-from", dfrom, "--milestone", "a"]))
    b = parse_ud(run(UD, [elf, gt, "--func-gt", fgt, "--decoy-from", dfrom, "--milestone", "b", "--lambda", "0.5"]))
    bp = bench_phat(elf, gt, dfrom)
    rows.append(dict(name=name, a=a, b=b, bench_phat=bp))

# ── Milestone A: reproduce M2 ─────────────────────────────────────────────────────────────────────
print("=== Milestone A — two-layer stack (bottom-up) reproduces bench --fuse ===")
print(f"{'spec':>10} | {'bench P̂ ECE/AUROC':>18} | {'stack P̂ ECE/AUROC':>18} | match")
allok = True
for r in rows:
    sp = r["a"]["si_phat"]; bp = r["bench_phat"]
    ok = abs(sp[0] - bp[0]) < 5e-4 and sp[1] == bp[1]
    allok &= ok
    print(f"{r['name']:>10} | {bp[0]:6.4f}/{bp[1]:>6}      | {sp[0]:6.4f}/{sp[1]:>6}      | {'YES' if ok else 'NO'}")
print(f"{'':>10}   Milestone A reproduction: {'ALL MATCH' if allok else 'MISMATCH'}")

# ── §7 joint-beats-parts (Milestone A marginals) ──────────────────────────────────────────────────
print("\n=== §7 joint-beats-parts (instruction axis: L1 π vs L2 R vs joint P̂) + function F_h ===")
print(f"{'spec':>10} | {'L1 π ECE/AUROC':>15} | {'L2 R ECE/AUROC':>15} | {'joint P̂ ECE/AUROC':>17} | {'ΔAUROC':>7} | {'F_h ECE/AUROC':>14}")
def fmt(t): return f"{t[0]:6.4f}/{t[1]:>6}"
for r in rows:
    a = r["a"]; pi, R, ph = a["si_pi"], a["si_R"], a["si_phat"]; F = a.get("sf_F", (0,"NA",0))
    dauroc = float(ph[1]) - float(pi[1])
    print(f"{r['name']:>10} | {fmt(pi):>15} | {fmt(R):>15} | {fmt(ph):>17} | {dauroc:+7.4f} | {F[0]:6.4f}/{F[1]:>6}")
def mean(f):
    vs=[f(r) for r in rows]; return sum(vs)/len(vs)
print(f"{'MEAN':>10} | {mean(lambda r:r['a']['si_pi'][0]):6.4f}/{mean(lambda r:float(r['a']['si_pi'][1])):.4f} "
      f"| {mean(lambda r:r['a']['si_R'][0]):6.4f}/{mean(lambda r:float(r['a']['si_R'][1])):.4f} "
      f"| {mean(lambda r:r['a']['si_phat'][0]):6.4f}/{mean(lambda r:float(r['a']['si_phat'][1])):.4f}   "
      f"| {mean(lambda r:float(r['a']['si_phat'][1])-float(r['a']['si_pi'][1])):+7.4f} "
      f"| {mean(lambda r:r['a']['sf_F'][0]):6.4f}/{mean(lambda r:float(r['a']['sf_F'][1])):.4f}")

# ── Milestone B: coupled fixpoint ─────────────────────────────────────────────────────────────────
print("\n=== Milestone B — coupled top-down relaxation to a fixpoint (λ=0.5, Theorem 4) ===")
print(f"{'spec':>10} | {'sweeps':>6} | {'converged':>9} | {'final‖Δ‖∞':>10} | {'joint P̂ ECE/AUROC':>17} | {'ΔAUROC vs L1':>12}")
for r in rows:
    b = r["b"]; ph = b["si_phat"]; pi = b["si_pi"]
    print(f"{r['name']:>10} | {b.get('iters','-'):>6} | {str(b.get('converged','-')):>9} | {b.get('final_delta',0):10.2e} "
          f"| {fmt(ph):>17} | {float(ph[1])-float(pi[1]):+12.4f}")
print(f"{'MEAN':>10} | {mean(lambda r:r['b'].get('iters',0)):6.1f} |           |            "
      f"| {mean(lambda r:r['b']['si_phat'][0]):6.4f}/{mean(lambda r:float(r['b']['si_phat'][1])):.4f}   "
      f"| {mean(lambda r:float(r['b']['si_phat'][1])-float(r['b']['si_pi'][1])):+12.4f}")

# ── §7 ablation: damping λ ─────────────────────────────────────────────────────────────────────────
print("\n=== §7 ablation — damping λ (convergence + joint P̂ AUROC), on cat ===")
cat = next(e for e in SPECS if "cat" in e); stem = cat[:-4]; dfrom = decoy_from(stem)
print(f"{'λ':>5} | {'sweeps':>6} | {'converged':>9} | {'final‖Δ‖∞':>10} | {'joint P̂ ECE/AUROC':>17}")
for lam in ["0.3", "0.5", "0.8", "1.0"]:
    d = parse_ud(run(UD, [cat, stem + ".gt", "--decoy-from", dfrom, "--milestone", "b", "--lambda", lam]))
    ph = d["si_phat"]
    print(f"{lam:>5} | {d.get('iters','-'):>6} | {str(d.get('converged','-')):>9} | {d.get('final_delta',0):10.2e} | {fmt(ph):>17}")

# ── §7 online update: clamp a confirmed function in the uncalled tail ──────────────────────────────
print("\n=== §7 online update — clamp a real uncalled-tail function (F rises, body recovers), on cat ===")
fgt = stem + ".func.gt"
# Dump heads, pick the real FUNC-GT head the coupled stack leaves LOWEST in the uncalled tail.
dump = run(UD, [cat, stem + ".gt", "--func-gt", fgt, "--decoy-from", dfrom, "--milestone", "b", "--lambda", "0.5", "--dump-heads"])
tail = []
for ln in dump.stdout.splitlines():
    p = ln.split(",")
    if p[0] == "stack_head" and p[3] == "real" and float(p[2]) < 0.5:
        tail.append((int(p[1], 16), float(p[2])))
tail.sort(key=lambda x: x[1])
if not tail:
    print("  (no real uncalled-tail head found — nothing to clamp)")
else:
    h = hex(tail[0][0])
    def report(extra):
        o = run(UD, [cat, stem + ".gt", "--func-gt", fgt, "--decoy-from", dfrom, "--milestone", "b",
                     "--lambda", "0.5", "--report-head", h] + extra)
        for ln in o.stdout.splitlines():
            p = ln.split(",")
            if p[0] == "stack_report_head":
                return dict(F=float(p[2]), body=int(p[3]), hi=int(p[4]), mean=float(p[5]))
        return {}
    base = report([])
    clamped = report(["--clamp-func", h + ":0.99"])
    print(f"clamped real tail head {h} at q=0.99  (body {base['body']} instructions)")
    print(f"{'metric':>18} | {'base → clamp':>16}")
    print(f"{'F_h':>18} | {base['F']:.4f} → {clamped['F']:.4f}")
    print(f"{'body P̂≥0.9':>18} | {base['hi']:>3d}  → {clamped['hi']:>3d}")
    print(f"{'body mean P̂':>18} | {base['mean']:.4f} → {clamped['mean']:.4f}")
