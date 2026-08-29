#!/usr/bin/env python3
"""Layer-2a code-in-data evaluation: cover vs reach precision/leak at matched recall."""
import subprocess, struct, glob, os, sys

BENCH = os.path.expanduser("~/lab/projects/upd-suite/target/release/bench")

def decoy_from(stem):
    for line in open(stem + ".regions"):
        if "junk_decoy" in line:
            return int(line.split()[0], 16)
    raise SystemExit("no decoy region in " + stem)

def run(elf, gt, decoy, extra):
    cmd = [BENCH, elf, gt, "--decoy-from", hex(decoy),
           "--biases", ",".join(f"{b/2:.2f}" for b in range(-24, 25))] + extra
    out = subprocess.run(cmd, capture_output=True, text=True)
    rows = []
    for ln in out.stdout.splitlines():
        p = ln.split(",")
        if len(p) == 7 and p[0] not in ("bias", "calibration"):
            try:
                bias, n, tp, rec, prec, f1, leak = (float(p[0]), int(p[1]), int(p[2]),
                                                    float(p[3]), float(p[4]), float(p[5]), int(p[6]))
                rows.append(dict(bias=bias, n=n, tp=tp, rec=rec, prec=prec, f1=f1, leak=leak))
            except ValueError:
                pass
    return rows

def best_f1(rows):
    return max(rows, key=lambda r: r["f1"])

def prec_at_recall(rows, target):
    """Highest-precision row whose recall >= target (matched-recall precision)."""
    cand = [r for r in rows if r["rec"] >= target - 1e-9]
    if not cand:
        cand = rows
    return max(cand, key=lambda r: r["prec"])

def main():
    specs = sorted(glob.glob("/tmp/cid/*__native-code-in-data.elf"))
    configs = {
        "reach g4 fd.9 calls":  ["--reach", "--gamma", "4",  "--fall-decay", "0.9"],
        "reach g4 fd.8 calls":  ["--reach", "--gamma", "4",  "--fall-decay", "0.8"],
        "reach g4 fd.95 calls": ["--reach", "--gamma", "4",  "--fall-decay", "0.95"],
        "reach g8 fd.9 calls":  ["--reach", "--gamma", "8",  "--fall-decay", "0.9"],
        "reach g4 fd.9 ALL":    ["--reach", "--gamma", "4",  "--fall-decay", "0.9", "--anchors-all"],
    }
    agg = {k: [] for k in ["cover"] + list(configs)}
    print(f"{'specimen':<26} {'mode':<20} {'bestF1_R':>8} {'bestF1_P':>8} {'leak':>6}  "
          f"{'matchR_P':>8} {'matchR_leak':>11}")
    for elf in specs:
        stem = elf[:-4]
        gt = stem + ".gt"
        d = decoy_from(stem)
        name = os.path.basename(stem).replace("gcc_coreutils_64_O2_", "").replace("__native-code-in-data", "")
        cover = run(elf, gt, d, ["--cover"])
        cb = best_f1(cover)
        target = cb["rec"]
        agg["cover"].append((cb, cb))
        print(f"{name:<26} {'cover':<20} {cb['rec']:>8.4f} {cb['prec']:>8.4f} {cb['leak']:>6} "
              f"{'--':>8} {'--':>11}")
        for cfgname, extra in configs.items():
            rows = run(elf, gt, d, extra)
            rb = best_f1(rows)
            mr = prec_at_recall(rows, target)
            agg[cfgname].append((rb, mr))
            print(f"{'':<26} {cfgname:<20} {rb['rec']:>8.4f} {rb['prec']:>8.4f} {rb['leak']:>6} "
                  f"{mr['prec']:>8.4f} {mr['leak']:>11}")
        print()

    print("=== MEANS across specimens ===")
    print(f"{'mode':<20} {'bestF1_R':>8} {'bestF1_P':>8} {'bestF1_leak':>11}  {'matchR_P':>8} {'matchR_leak':>11}")
    for k, lst in agg.items():
        br = sum(x[0]["rec"] for x in lst)/len(lst)
        bp = sum(x[0]["prec"] for x in lst)/len(lst)
        bl = sum(x[0]["leak"] for x in lst)/len(lst)
        mp = sum(x[1]["prec"] for x in lst)/len(lst)
        ml = sum(x[1]["leak"] for x in lst)/len(lst)
        print(f"{k:<20} {br:>8.4f} {bp:>8.4f} {bl:>11.1f}  {mp:>8.4f} {ml:>11.1f}")

main()
