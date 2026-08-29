#!/usr/bin/env python3
"""Layer-2 Milestone-2 evaluation (LAYER2_M2_SPEC §9) on the code-in-data corpus.

Answers, per specimen and in aggregate:
  §9.1  is F_h calibrated vs FUNC-symbol GT? does the core sit near 1 and the tail near β₀?
  §9.2  confirmed-core precision (high P̂) — leak vs the M1 hard gate (leak 0).
  §9.3  graceful tail: does soft recall recover past M1's ceiling, and is admitted decoy FLAGGED
        (moderate P̂) rather than asserted?
  §9.4  honesty: mode (a) π line identical to cover; mode (b) P̂ isotonic (ECE↓, AUROC= vs raw
        fusion — Theorem 1).
  §9.5  transfer: fit fusion on held-out binaries, test P̂ ECE.
  §9.6  β₀ (Theorem 2) — the uncalled-tail base rate.
"""
import glob, os, re, subprocess

BENCH = os.path.expanduser("~/lab/projects/upd-suite/target/release/bench")
SPECS = sorted(glob.glob("/tmp/cid/*__native-code-in-data.elf"))
TAUS = "0.1,0.3,0.5,0.7,0.9"


def decoy_from(stem):
    for l in open(stem + ".regions"):
        if "junk_decoy" in l:
            return l.split()[0]
    raise SystemExit("no junk_decoy in " + stem)


def run(args):
    return subprocess.run([BENCH] + args, capture_output=True, text=True)


def parse(o):
    """Pull the machine-readable lines + the two stderr summaries into one dict."""
    d = {"phat": []}
    for ln in o.stdout.splitlines():
        p = ln.split(",")
        if p[0] == "calibration":
            d["calibration"] = ln
        elif p[0] == "func_calib":
            d["fc_" + p[1]] = (float(p[2]), p[3], float(p[4]))   # ece, auroc, base_rate
        elif p[0] == "beta0":
            d["beta0"] = float(p[1]); d["u_real"] = int(p[2]); d["u"] = int(p[3])
        elif p[0] == "fuse_calib":
            d["f_" + p[1]] = (float(p[2]), p[3], float(p[4]))
        elif p[0] not in ("phat_tau", "bias", "threshold") and len(p) >= 6 and _isnum(p[0]):
            # phat_tau row: tau,n_pred,tp,recall,precision,f1[,leak,meanconf]
            row = {"tau": float(p[0]), "recall": float(p[3]), "prec": float(p[4])}
            if len(p) >= 8:
                row["leak"] = int(p[6]); row["mconf"] = float(p[7])
            d["phat"].append(row)
    for ln in o.stderr.splitlines():
        m = re.search(r"soft_recall@R0\.5=([0-9.]+)", ln)
        if m:
            d["soft_recall"] = float(m.group(1))
        m = re.search(r"confirmed_real_recall=([0-9.]+)", ln)
        if m:
            d["m1_ceiling"] = float(m.group(1))
        m = re.search(r"core n=\d+ realFUNC=\d+ meanF=([0-9.]+).*tail n=\d+ realFUNC=\d+ meanF=([0-9.]+)", ln)
        if m:
            d["core_meanF"], d["tail_meanF"] = float(m.group(1)), float(m.group(2))
    return d


def _isnum(s):
    try:
        float(s); return True
    except ValueError:
        return False


rows = []
for elf in SPECS:
    stem = elf[:-4]
    gt, fgt, dfrom = stem + ".gt", stem + ".func.gt", decoy_from(stem)
    name = os.path.basename(stem).split("__")[0].replace("gcc_coreutils_64_O2_", "")
    m1 = parse(run([elf, gt, "--confirm", "--gamma", "8", "--biases", "0"]))
    fu = parse(run([elf, gt, "--fuse", "--func-gt", fgt, "--decoy-from", dfrom, "--thresholds", TAUS]))
    # §9.5 transfer: fit the fusion on a DIFFERENT specimen, test P̂ ECE on this one.
    other = next(e for e in SPECS if e != elf)
    tr = parse(run([elf, gt, "--fuse", "--fit-elf", other, "--fit-gt", other[:-4] + ".gt",
                    "--func-gt", fgt, "--decoy-from", dfrom, "--thresholds", TAUS]))
    fu["m1_ceiling"] = m1.get("m1_ceiling")
    fu["transfer_phat"] = tr.get("f_phat")
    fu["name"] = name
    fu["honesty_ok"] = fu.get("calibration") == m1.get("calibration")
    rows.append(fu)

# ── §9.1  F calibration ──────────────────────────────────────────────────────────────────────────
print("=== §9.1  function-confirmation calibration (F_h vs FUNC-symbol GT) ===")
print(f"{'spec':>10} | {'base':>5} | {'prior ECE/AUROC':>16} | {'F ECE/AUROC':>16} | {'F-isoceil':>9} | core/tail meanF")
for r in rows:
    pe, pa, _ = r["fc_prior"]; fe, fa, _ = r["fc_F"]; ie, _, _ = r["fc_F_isoceil"]; _, _, base = r["fc_F"]
    print(f"{r['name']:>10} | {base:5.3f} | {pe:6.4f}/{pa:>6} | {fe:6.4f}/{fa:>6} | {ie:9.4f} |"
          f" {r.get('core_meanF',0):.3f}/{r.get('tail_meanF',0):.3f}")


def mean(key, sub=None):
    vs = [(_g(r, key) if sub is None else _g(r, key)[sub]) for r in rows if _g(r, key) is not None]
    vs = [v for v in vs if v is not None]
    return sum(vs) / len(vs) if vs else float("nan")


def _g(r, key):
    return r.get(key)


print(f"{'MEAN':>10} |       | {mean('fc_prior',0):6.4f}        | {mean('fc_F',0):6.4f}        |"
      f" {mean('fc_F_isoceil',0):9.4f} | {mean('core_meanF'):.3f}/{mean('tail_meanF'):.3f}")

# ── §9.3  recall recovery + §9.6 β₀ ────────────────────────────────────────────────────────────────
print("\n=== §9.3 / §9.6  recall recovery past the M1 ceiling, and β₀ ===")
print(f"{'spec':>10} | {'M1 hard ceiling':>15} | {'soft recall@R0.5':>16} | {'Δrecall':>8} | {'β₀':>6} | tail real/total")
for r in rows:
    c, s = r.get("m1_ceiling"), r.get("soft_recall")
    print(f"{r['name']:>10} | {c:15.4f} | {s:16.4f} | {s-c:+8.4f} | {r['beta0']:6.4f} | {r['u_real']}/{r['u']}")
print(f"{'MEAN':>10} | {mean('m1_ceiling'):15.4f} | {mean('soft_recall'):16.4f} |"
      f" {mean('soft_recall')-mean('m1_ceiling'):+8.4f} | {mean('beta0'):6.4f} |")

# ── §9.2 / §9.3  P̂ risk–coverage: precision + FLAGGING of admitted decoy ───────────────────────────
print("\n=== §9.2 / §9.3  P̂ risk–coverage (precision, decoy leak, mean decoy confidence) ===")
taus = sorted({row["tau"] for r in rows for row in r["phat"]}, reverse=True)
print(f"{'P̂≥τ':>6} | {'mean recall':>11} | {'mean prec':>9} | {'mean decoy leak':>15} | {'mean decoy conf':>15}")
for t in taus:
    rec = pre = leak = mc = n = 0
    for r in rows:
        for row in r["phat"]:
            if abs(row["tau"] - t) < 1e-9:
                rec += row["recall"]; pre += row["prec"]; leak += row.get("leak", 0); mc += row.get("mconf", 0); n += 1
    print(f"{t:6.2f} | {rec/n:11.4f} | {pre/n:9.4f} | {leak/n:15.1f} | {mc/n:15.4f}")

# ── §9.4  honesty + Theorem 1 ──────────────────────────────────────────────────────────────────────
print("\n=== §9.4  honesty wall + Theorem 1 (P̂ isotonic: ECE↓, AUROC≈ vs raw fusion) ===")
print(f"{'spec':>10} | {'π line=cover':>12} | {'π ECE/AUROC':>16} | {'fusion ECE/AUROC':>16} | {'P̂ ECE/AUROC':>16}")
for r in rows:
    pe, pa, _ = r["f_pi"]; ge, ga, _ = r["f_fusion"]; he, ha, _ = r["f_phat"]
    print(f"{r['name']:>10} | {'YES' if r['honesty_ok'] else 'NO':>12} |"
          f" {pe:6.4f}/{pa:>6} | {ge:6.4f}/{ga:>6} | {he:6.4f}/{ha:>6}")
print(f"{'MEAN':>10} | {'ALL YES' if all(r['honesty_ok'] for r in rows) else 'FAIL':>12} |"
      f" {mean('f_pi',0):6.4f}        | {mean('f_fusion',0):6.4f}        | {mean('f_phat',0):6.4f}")

# ── §9.5  transfer ─────────────────────────────────────────────────────────────────────────────────
print("\n=== §9.5  transfer: fusion fit on a HELD-OUT binary, P̂ ECE/AUROC on target ===")
print(f"{'spec':>10} | {'self-fit P̂ ECE/AUROC':>21} | {'transfer P̂ ECE/AUROC':>21}")
for r in rows:
    he, ha, _ = r["f_phat"]; te, ta, _ = r["transfer_phat"]
    print(f"{r['name']:>10} | {he:6.4f}/{ha:>6}         | {te:6.4f}/{ta:>6}")
print(f"{'MEAN':>10} | {mean('f_phat',0):6.4f}                | "
      f"{sum(r['transfer_phat'][0] for r in rows)/len(rows):6.4f}")
