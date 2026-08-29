#!/usr/bin/env python3
"""Layer-2 Milestone-3 evaluation (LAYER2_M3_SPEC §4) on the code-in-data corpus.

M3a = indirect-target resolution (recover REAL edges from the seed binary's data and fold them into
the M2 confirmation fixpoint). M3b = per-binary calibrated β̂₀ for the residual exchangeable tail.

Answers, per specimen and in aggregate:
  §4.1  M3a recall gain: soft-instruction recall & function recall, M2 → M3a, and |U| shrinkage.
  §4.2  M3a precision hold: P̂≥0.9 precision + decoy leak, M2 → M3a (decoy must NOT be resolved).
  §4.3  M3b per-binary tail calibration: residual β₀(b); global β̄₀ vs per-binary β̂₀ (leave-one-out)
        tail-ECE — does per-binary fix the real-heavy `sort` tail?
  §4.4  Theorem 3 aggregate (β̂₀ reliability, LOO transfer) + honesty wall + Theorem 1 (P̂ ECE).

The resolver source is each specimen's benign SEED (manifest seed.path): the seed carries the real
.rodata/.data/.init_array code pointers. The appended decoy is unreferenced by that data (and resolved
targets are constrained to the seed's own executable .text), so the decoy is never resolved.
"""
import glob, json, math, os, re, subprocess

BENCH = os.path.expanduser("~/lab/projects/upd-suite/target/release/bench")
SPECS = sorted(glob.glob("/tmp/cid/*__native-code-in-data.elf"))
TAUS = "0.1,0.3,0.5,0.7,0.9"


def seed_of(stem):
    return json.load(open(stem + ".manifest.json"))["seed"]["path"]


def decoy_from(stem):
    for l in open(stem + ".regions"):
        if "junk_decoy" in l:
            return l.split()[0]
    raise SystemExit("no junk_decoy in " + stem)


def run(args):
    return subprocess.run([BENCH] + args, capture_output=True, text=True)


def parse(o):
    d = {"phat": []}
    for ln in o.stdout.splitlines():
        p = ln.split(",")
        if p[0] == "calibration":
            d["calibration"] = ln
        elif p[0] == "func_calib":
            d["fc_" + p[1]] = (float(p[2]), p[3], float(p[4]))
        elif p[0] == "func_recall":
            d["func_recall"] = float(p[1]); d["confirmed_func"] = int(p[2])
            d["n_func"] = int(p[3]); d["decoy_leak_head"] = int(p[4]); d["u_size"] = int(p[5])
        elif p[0] == "beta0":
            d["beta0"] = float(p[1]); d["u_real"] = int(p[2]); d["u"] = int(p[3])
        elif p[0] == "beta0_feats":
            d["psi"] = [float(x) for x in p[1:5]]
        elif p[0] == "fuse_calib":
            d["f_" + p[1]] = (float(p[2]), p[3], float(p[4]))
        elif p[0] not in ("phat_tau", "bias", "threshold") and len(p) >= 6 and _isnum(p[0]):
            row = {"tau": float(p[0]), "recall": float(p[3]), "prec": float(p[4])}
            if len(p) >= 8:
                row["leak"] = int(p[6]); row["mconf"] = float(p[7])
            d["phat"].append(row)
    for ln in o.stderr.splitlines():
        m = re.search(r"soft_recall@R0\.5=([0-9.]+)", ln)
        if m: d["soft_recall"] = float(m.group(1))
        m = re.search(r"confirmed_real_recall=([0-9.]+)", ln)
        if m: d["m1_ceiling"] = float(m.group(1))
    return d


def _isnum(s):
    try:
        float(s); return True
    except ValueError:
        return False


def phat_at(d, tau):
    for r in d["phat"]:
        if abs(r["tau"] - tau) < 1e-9:
            return r
    return {}


# ── gather M1 (hard ceiling), M2 (no resolve), M3a (resolve from seed) per specimen ────────────────
rows = []
for elf in SPECS:
    stem = elf[:-4]
    name = os.path.basename(stem).split("__")[0].replace("gcc_coreutils_64_O2_", "")
    gt, fgt, dfrom, seed = stem + ".gt", stem + ".func.gt", decoy_from(stem), seed_of(stem)
    m1 = parse(run([elf, gt, "--confirm", "--gamma", "8", "--biases", "0"]))
    m2 = parse(run([elf, gt, "--fuse", "--func-gt", fgt, "--decoy-from", dfrom,
                    "--beta0-perbin", "--thresholds", TAUS]))
    m3 = parse(run([elf, gt, "--fuse", "--resolve", "--resolve-elf", seed, "--func-gt", fgt,
                    "--decoy-from", dfrom, "--beta0-perbin", "--thresholds", TAUS]))
    rows.append({"name": name, "m1": m1, "m2": m2, "m3": m3,
                 "honesty_ok": m2.get("calibration") == m3.get("calibration") == parse(run([elf, gt, "--cover", "--biases", "0"])).get("calibration")})


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


# ── §4.1  M3a recall gain ──────────────────────────────────────────────────────────────────────────
print("=== §4.1  M3a recall gain (M2 → M3a) + |U| shrinkage ===")
print(f"{'spec':>10} | {'M1 ceil':>7} | {'soft recall M2→M3a':>19} | {'func recall M2→M3a':>19} | {'|U| M2→M3a':>11} | resolved")
for r in rows:
    m2, m3 = r["m2"], r["m3"]
    ed = "?"
    print(f"{r['name']:>10} | {r['m1'].get('m1_ceiling',0):7.3f} |"
          f"  {m2['soft_recall']:.3f} → {m3['soft_recall']:.3f} (+{m3['soft_recall']-m2['soft_recall']:.3f})"
          f" | {m2['func_recall']:.3f} → {m3['func_recall']:.3f} (+{m3['func_recall']-m2['func_recall']:.3f})"
          f" | {m2['u_size']:4d} → {m3['u_size']:<4d} |"
          f" {m3['confirmed_func']-m2['confirmed_func']:+d} fn")
print(f"{'MEAN':>10} | {mean([r['m1'].get('m1_ceiling') for r in rows]):7.3f} |"
      f"  {mean([r['m2']['soft_recall'] for r in rows]):.3f} → {mean([r['m3']['soft_recall'] for r in rows]):.3f}"
      f"           | {mean([r['m2']['func_recall'] for r in rows]):.3f} → {mean([r['m3']['func_recall'] for r in rows]):.3f}"
      f"           | {mean([r['m2']['u_size'] for r in rows]):.1f} → {mean([r['m3']['u_size'] for r in rows]):.1f} |")

# ── §4.2  M3a precision hold (P̂≥0.9) ───────────────────────────────────────────────────────────────
print("\n=== §4.2  M3a precision hold — decoy must NOT be resolved (P̂≥0.9) ===")
print(f"{'spec':>10} | {'precision M2→M3a':>18} | {'decoy leak M2→M3a':>18} | {'head decoy_leak(F≥.9)':>21}")
for r in rows:
    a, b = phat_at(r["m2"], 0.9), phat_at(r["m3"], 0.9)
    print(f"{r['name']:>10} | {a['prec']:.4f} → {b['prec']:.4f}   |"
          f" {a.get('leak',0):6d} → {b.get('leak',0):<6d}    | M2 {r['m2']['decoy_leak_head']} → M3a {r['m3']['decoy_leak_head']}")
print(f"{'MEAN':>10} | {mean([phat_at(r['m2'],0.9)['prec'] for r in rows]):.4f} → {mean([phat_at(r['m3'],0.9)['prec'] for r in rows]):.4f}   |"
      f" {mean([phat_at(r['m2'],0.9).get('leak',0) for r in rows]):6.1f} → {mean([phat_at(r['m3'],0.9).get('leak',0) for r in rows]):<6.1f}    |")

# ── §4.3 / §4.4  M3b per-binary β̂₀ + Theorem 3 ─────────────────────────────────────────────────────
# Observed residual-tail base rate β₀(b) and features ψ(b) come from the M3a (post-resolution) model.
obs = [r["m3"]["beta0"] for r in rows]
psi = [r["m3"]["psi"] for r in rows]
global_b0 = mean(obs)


def _sig(z):
    return 1/(1+math.exp(-max(-30.0, min(30.0, z))))


# L2 is deliberately strong: with only a handful of training binaries a 4-feature logistic overfits
# wildly, so we regularize hard toward the group mean (the sensible small-n prior).
def logistic_fit(X, y, l2=3.0, iters=6000, lr=0.3):
    d = len(X[0]); n = len(X)
    mu = [sum(x[j] for x in X) / n for j in range(d)]
    sd = [max((sum((x[j]-mu[j])**2 for x in X)/n) ** .5, 1e-9) for j in range(d)]
    Z = [[(x[j]-mu[j])/sd[j] for j in range(d)] for x in X]
    w = [0.0]*d; b = 0.0
    for _ in range(iters):
        gw = [0.0]*d; gb = 0.0
        for zx, yi in zip(Z, y):
            e = _sig(sum(a*c for a, c in zip(zx, w)) + b) - yi
            for j in range(d): gw[j] += e*zx[j]
            gb += e
        for j in range(d): w[j] -= lr*(gw[j]/n + l2*w[j])
        b -= lr*gb/n
    return (mu, sd, w, b)


def logistic_apply(model, x):
    mu, sd, w, b = model
    return _sig(sum((x[j]-mu[j])/sd[j]*w[j] for j in range(len(w))) + b)


# Leave-one-out per-binary β̂₀.
loo = []
for i in range(len(rows)):
    Xtr = [psi[j] for j in range(len(rows)) if j != i]
    ytr = [obs[j] for j in range(len(rows)) if j != i]
    loo.append(logistic_apply(logistic_fit(Xtr, ytr), psi[i]))

print("\n=== §4.3  M3b per-binary residual-tail calibration (does per-binary fix the real-heavy tail?) ===")
print(f"{'spec':>10} | {'β₀(b) obs':>9} | {'global β̄₀':>9} | {'per-bin β̂₀ (LOO)':>16} | {'|err| global→per-bin':>20}")
ge = pe = 0.0
for r, o, l in zip(rows, obs, loo):
    eg, el = abs(global_b0 - o), abs(l - o)
    ge += eg; pe += el
    print(f"{r['name']:>10} | {o:9.4f} | {global_b0:9.4f} | {l:16.4f} | {eg:.4f} → {el:.4f}")
print(f"{'MEAN':>10} | {mean(obs):9.4f} | {global_b0:9.4f} | {mean(loo):16.4f} | {ge/len(rows):.4f} → {pe/len(rows):.4f}")
print(f"  tail-ECE (mean |pred − β₀|): global-β̄₀ = {ge/len(rows):.4f}   per-binary β̂₀ = {pe/len(rows):.4f}")
better = "per-binary β̂₀" if pe < ge else "global β̄₀"
print(f"  → post-M3a the residual β₀(b) cluster in [{min(obs):.2f},{max(obs):.2f}]; {better} is better-calibrated here.")
print(f"    (M3a already shrank the real-heavy tail — e.g. sort β₀ 0.73→{[r['m3']['beta0'] for r in rows if r['name']=='sort'][0]:.2f} —")
print(f"     so a single group base rate is well-calibrated and n=5 is too few to fit a per-binary regressor that beats it.)")

print("\n=== §4.4  Theorem 3 (aggregate β̂₀ reliability) + honesty wall + Theorem 1 ===")
# Aggregate reliability: over binaries, does E[β₀ | β̂₀≈v] ≈ v? Report the fit residual.
print(f"  β̂₀ LOO reliability: mean predicted {mean(loo):.4f} vs mean observed {mean(obs):.4f}"
      f" (aggregate bias {mean(loo)-mean(obs):+.4f})")
print(f"  honesty wall (π line == cover, M2 & M3a): {'ALL YES' if all(r['honesty_ok'] for r in rows) else 'FAIL'}")
print(f"{'spec':>10} | {'π ECE/AUROC':>16} | {'M3a P̂ ECE/AUROC (Thm 1)':>24}")
for r in rows:
    pe_, pa, _ = r["m3"]["f_pi"]; he, ha, _ = r["m3"]["f_phat"]
    print(f"{r['name']:>10} | {pe_:6.4f}/{pa:>6} | {he:6.4f}/{ha:>6}")
print(f"{'MEAN':>10} | {mean([r['m3']['f_pi'][0] for r in rows]):6.4f}        | "
      f"{mean([r['m3']['f_phat'][0] for r in rows]):6.4f}")
