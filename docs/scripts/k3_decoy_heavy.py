#!/usr/bin/env python3
"""K=3 module-layer discrimination on the decoy-heavy corpus (K3_DECOY_HEAVY_SPEC).

The last depth lever for Paper 3: re-run the L4 module layer (K=2 vs K=3) on the now-existing
decoy-heavy corpus, but *informed* by the staircase follow-ups. Decoy-heavy is a MIX --- disconnected
decoys (a component the entry closure never reaches; the module layer should crush these) and
self-anchoring / interleaved decoys (the closure structurally reaches them --- the FU1 residue that
persisted 1170->187->194 through E5, and the FU2 drain). The real result is whether the K=3 win splits
along exactly that line: big on disconnected, ~none on reached.

Memory rules, unchanged (a corpus build crashed the box once): ONE binary in memory at a time; `--jobs 1`
(we never fan out); every specimen streamed+flushed to CSV; resumable (skip specimens already in the CSV);
NO corpus build (decoy-heavy already exists under upd-suite-sota/scratch/decoy-smoke). Read-only on the
engine and on probdisasm/probcfg --- the ONLY engine surface is the existing default-off dump flags.

Two udstack runs per specimen (K=2 then K=3), one process each:
  K=2:  --layers 2 --dump-heads  --dump-instr           (F_h per head, joint P̂+π per offset)
  K=3:  --layers 3 --dump-modules --dump-instr --dump-pins   (F_h/F_c/comp per head, π per offset, reach)
Partition is by CONSTRUCTION, never a threshold:
  - decoy head  := head whose vaddr lands in a `junk_decoy` region span (regions sidecar; interleaved
                   alternates real/junk, so a single --decoy-from threshold is WRONG --- we use spans).
  - real head   := head in the FUNC-symbol GT (derived from the benign seed, restricted to real_code).
  - reached     := `pin_reach`==1 (recursive_descent closure from the entry --- a membership fact).
  disconnected decoys = decoy & not reached ; self-anchoring/interleaved decoys = decoy & reached.
"""
import argparse
import csv
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import staircase_measure as S  # func_gt_for, read_regions, align_audit, auroc, UDSTACK, decoy_specs

SCRATCH = os.path.join(S.ROOT, "upd-suite/docs/staircase/_k3scratch")
FIELDS = [
    "obf", "struct", "name", "align_ok", "e_type", "load_base",
    "n_real", "n_decoy", "n_decoy_disc", "n_decoy_reach",
    # engine's own global axis (the paper's 0.889->0.925 line): FUNC-CAL over all func marginals
    "func_auroc_k2", "func_auroc_k3", "d_func_auroc",
    "module_auroc_k3", "module_ece_k3",
    "func_ece_k2", "func_ece_k3", "instr_ece_k3", "instr_auroc_k3",
    # real-vs-decoy discrimination we compute from the head dumps, split by reachability
    "rd_auroc_k2", "rd_auroc_k3", "d_rd_auroc",              # all decoys
    "rd_disc_auroc_k2", "rd_disc_auroc_k3", "d_rd_disc",     # disconnected decoys only  (expect big win)
    "rd_reach_auroc_k2", "rd_reach_auroc_k3", "d_rd_reach",  # reached decoys only        (expect ~none)
    "fc_disc_auroc", "fc_reach_auroc",                       # module belief F_c separation, per subset
    "fc_mean_real", "fc_mean_disc", "fc_mean_reach",         # where does F_c actually pin?
    # honesty wall: ‖π^K3 − π^baseline‖_∞ over every dumped offset, MUST be 0.0
    "pi_linf", "n_offsets",
]


def run(cmd):
    """One udstack process. Returns (stdout_lines, stderr_text). One binary in memory, freed on return."""
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        sys.stderr.write("  !! udstack rc=%d: %s\n" % (p.returncode, p.stderr[-400:]))
    return p.stdout.splitlines(), p.stderr


def parse_common(lines):
    """Pull the calibration axis lines + the π column (from --dump-instr) out of a udstack stdout."""
    cal = {}          # (obj,which) -> (ece, auroc, base)
    pi = {}           # addr -> π_a   (the invariant Layer-1 posterior)
    for ln in lines:
        if ln.startswith("stack_instr,") or ln.startswith("stack_func,") or ln.startswith("stack_module,"):
            t, which, ece, au, base = ln.split(",")
            cal[(t, which)] = (float(ece), float(au) if au != "nan" else float("nan"), float(base))
        elif ln.startswith("instr_bel,"):
            _, a, _phat, p_pi, _r = ln.split(",")
            pi[int(a, 16)] = float(p_pi)
    return cal, pi


def in_spans(addr, spans):
    return any(s <= addr < e for (s, e) in spans)


def measure(spec):
    regions = S.read_regions(spec["regions"])
    junk = [(s, e) for (s, e, lab, kind) in regions if kind == "junk_decoy"]
    func_gt_path = S.func_gt_for(spec, SCRATCH)
    if not func_gt_path:
        sys.stderr.write("  skip %s: no func GT derivable\n" % spec["name"])
        return None
    real_heads = S.read_gt(func_gt_path)
    align_ok, info = S.align_audit(spec["elf"], real_heads)

    # ---- K=2 baseline: F_h per head (--dump-heads) + π (--dump-instr) ----
    k2_out, _ = run([S.UDSTACK, spec["elf"], spec["gt"], "--func-gt", func_gt_path,
                     "--layers", "2", "--dump-heads", "--dump-instr"])
    cal2, pi2 = parse_common(k2_out)
    fh_k2, belf_k2 = {}, {}
    for ln in k2_out:
        if ln.startswith("stack_head,"):
            _, a, f, _lab, bel_f = ln.split(",")
            fh_k2[int(a, 16)] = float(f)          # raw confirmation (pre-fusion)
            belf_k2[int(a, 16)] = float(bel_f)    # calibrated fused func marginal (the paper's axis)

    # ---- K=3: F_h/F_c/comp per head (--dump-modules) + π + reachability closure (--dump-pins) ----
    k3_out, _ = run([S.UDSTACK, spec["elf"], spec["gt"], "--func-gt", func_gt_path,
                     "--layers", "3", "--dump-modules", "--dump-instr", "--dump-pins"])
    cal3, pi3 = parse_common(k3_out)
    fh_k3, fc_k3, belf_k3 = {}, {}, {}
    reach = {}
    for ln in k3_out:
        if ln.startswith("stack_headmod,0x"):
            _, a, f_h, comp, f_c, real, in_dec, bel_f = ln.split(",")
            fh_k3[int(a, 16)] = float(f_h)
            fc_k3[int(a, 16)] = float(f_c)
            belf_k3[int(a, 16)] = float(bel_f)    # calibrated fused func marginal at K=3
        elif ln.startswith("pin_reach,"):
            _, a, r = ln.split(",")
            reach[int(a, 16)] = (r == "1")

    # ---- honesty wall: ‖π^K3 − π^baseline‖_∞ over the shared offset domain ----
    common = pi2.keys() & pi3.keys()
    pi_linf = max((abs(pi2[a] - pi3[a]) for a in common), default=float("nan"))

    # ---- per-head classification by CONSTRUCTION (regions + closure), then split AUROCs ----
    heads = sorted(fh_k3.keys() & fh_k2.keys())
    real_lab, decoy_lab, reach_lab = {}, {}, {}
    for h in heads:
        is_real = h in real_heads
        is_decoy = in_spans(h, junk) and not is_real
        real_lab[h] = is_real
        decoy_lab[h] = is_decoy
        reach_lab[h] = reach.get(h, False)

    def rd_auroc(fmap, decoy_pred):
        """real-vs-decoy AUROC: positives = real heads, negatives = decoy heads matching decoy_pred."""
        sc, lb = [], []
        for h in heads:
            if real_lab[h]:
                sc.append(fmap[h]); lb.append(1)
            elif decoy_lab[h] and decoy_pred(h):
                sc.append(fmap[h]); lb.append(0)
        return S.auroc(sc, lb)

    any_decoy = lambda h: True
    disc = lambda h: not reach_lab[h]          # disconnected: closure never reaches it
    reached = lambda h: reach_lab[h]           # self-anchoring / interleaved: closure reaches it

    disc_heads = [h for h in heads if decoy_lab[h] and disc(h)]
    reach_heads = [h for h in heads if decoy_lab[h] and reached(h)]
    real_hset = [h for h in heads if real_lab[h]]

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    fa2 = cal2.get(("stack_func", "F"), (float("nan"),) * 3)[1]
    fa3 = cal3.get(("stack_func", "F"), (float("nan"),) * 3)[1]
    # Split AUROCs on the CALIBRATED FUSED func marginal `bel_f` — the same axis as the engine's global
    # func AUROC (the 0.889→0.925 line), which is what actually carries the K=3 module message. (Raw
    # `f_h`=confirmation_map is pre-fusion and does not move with depth.)
    rd2 = rd_auroc(belf_k2, any_decoy)
    rd3 = rd_auroc(belf_k3, any_decoy)
    rdd2 = rd_auroc(belf_k2, disc)
    rdd3 = rd_auroc(belf_k3, disc)
    rdr2 = rd_auroc(belf_k2, reached)
    rdr3 = rd_auroc(belf_k3, reached)

    def d(a, b):
        return (b - a) if (a == a and b == b) else float("nan")  # nan-safe

    return {
        "obf": spec["obf"], "struct": spec["struct"], "name": spec["name"],
        "align_ok": align_ok, "e_type": info["e_type"] if info else "",
        "load_base": ("0x%x" % info["exec_vaddr"]) if info and info["exec_vaddr"] else "",
        "n_real": len(real_hset), "n_decoy": len(disc_heads) + len(reach_heads),
        "n_decoy_disc": len(disc_heads), "n_decoy_reach": len(reach_heads),
        "func_auroc_k2": fa2, "func_auroc_k3": fa3, "d_func_auroc": d(fa2, fa3),
        "module_auroc_k3": cal3.get(("stack_module", "F"), (float("nan"),) * 3)[1],
        "module_ece_k3": cal3.get(("stack_module", "F"), (float("nan"),) * 3)[0],
        "func_ece_k2": cal2.get(("stack_func", "F"), (float("nan"),) * 3)[0],
        "func_ece_k3": cal3.get(("stack_func", "F"), (float("nan"),) * 3)[0],
        "instr_ece_k3": cal3.get(("stack_instr", "phat"), (float("nan"),) * 3)[0],
        "instr_auroc_k3": cal3.get(("stack_instr", "phat"), (float("nan"),) * 3)[1],
        "rd_auroc_k2": rd2, "rd_auroc_k3": rd3, "d_rd_auroc": d(rd2, rd3),
        "rd_disc_auroc_k2": rdd2, "rd_disc_auroc_k3": rdd3, "d_rd_disc": d(rdd2, rdd3),
        "rd_reach_auroc_k2": rdr2, "rd_reach_auroc_k3": rdr3, "d_rd_reach": d(rdr2, rdr3),
        "fc_disc_auroc": rd_auroc(fc_k3, disc), "fc_reach_auroc": rd_auroc(fc_k3, reached),
        "fc_mean_real": mean([fc_k3[h] for h in real_hset]),
        "fc_mean_disc": mean([fc_k3[h] for h in disc_heads]),
        "fc_mean_reach": mean([fc_k3[h] for h in reach_heads]),
        "pi_linf": pi_linf, "n_offsets": len(common),
    }


def load_done(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {r["name"] for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(S.ROOT, "upd-suite/docs/staircase/k3_decoy_heavy.csv"))
    ap.add_argument("--only", default=None, help="comma struct filter (debug)")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(SCRATCH, exist_ok=True)

    done = load_done(args.out)
    new = not os.path.exists(args.out)
    f = open(args.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader(); f.flush()

    only = set(args.only.split(",")) if args.only else None
    for spec in S.decoy_specs():
        if only and spec["struct"] not in only:
            continue
        if spec["name"] in done:
            sys.stderr.write("resume: skip %s (in CSV)\n" % spec["name"]); continue
        sys.stderr.write("== %s (%s) ==\n" % (spec["name"], spec["struct"]))
        row = measure(spec)          # exactly two udstack processes; one binary in memory at a time
        if row is None:
            continue
        w.writerow(row); f.flush()   # stream+flush -> resumable, memory-safe
        sys.stderr.write("   func AUROC K2=%.3f K3=%.3f (Δ%+.3f) | disc Δ%+.3f reach Δ%+.3f | F_c disc=%.3f reach=%.3f | πLinf=%.1e\n" % (
            row["func_auroc_k2"], row["func_auroc_k3"], row["d_func_auroc"],
            row["d_rd_disc"], row["d_rd_reach"], row["fc_mean_disc"], row["fc_mean_reach"], row["pi_linf"]))
    f.close()


if __name__ == "__main__":
    main()
