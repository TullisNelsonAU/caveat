#!/usr/bin/env python3
"""Layer-2 Milestone-5 evaluation: M4 code-anchored resolvers on a PIE corpus.

The M4 null on non-PIE C coreutils was structural: pie_rel/vtable idioms were absent, and computed-goto
tables were 8-byte-absolute (already in M3's blind scan). M5 rebuilds the corpus as x86_64 PIE (ET_DYN)
so the idioms actually occur — 4-byte-relative switch jump tables (pie_rel, invisible to the 8-byte
scan) and C++ `.data.rel.ro` vtables — and re-runs the SAME eval to measure per-idiom coverage.

Per specimen and in aggregate:
  §1  Coverage by kind (M4) + net-new resolved targets over M3, split into net-new INSTRUCTIONS
      (target in instruction GT) and net-new real FUNCTIONS (target in function GT).
  §2  Flagship: real (in-GT) functions confirmed (F>=0.9) M3 -> M4 — the number that was 0 on non-PIE C.
  §3  Decoy-discipline audit: no resolved edge (M3 or M4) targets [decoy_from, end).
  §4  Both axes: function F[AUROC] and recalibrated instruction P̂[ECE] — M3 -> M4.

Resolver source = each specimen's PIE seed (manifest seed.path). GT is construction-based (gen-gt
instruction starts + objdump FUNC symbols); tools are never GT. Run audit_cid_pie.py FIRST.
"""
import glob, json, os, subprocess, re

BENCH = os.path.expanduser("~/lab/projects/upd-suite-stack/target/release/bench")
CID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "corpus_pie", "cid")
CODE_KINDS = ("computed_goto", "pie_rel", "vtable")
# idiom family each specimen is built to exercise (for per-idiom grouping)
FAMILY = {"switch_dense": "pie_rel", "switch_tailcall": "pie_rel", "computed_goto": "datarelro",
          "vtable_shapes": "vtable", "vtable_codec": "vtable"}


def load_hex(p):
    return {int(l.strip(), 16) for l in open(p) if l.strip()}


def decoy_from(stem):
    for l in open(stem + ".regions"):
        if "junk_decoy" in l:
            return int(l.split()[0], 16)
    raise SystemExit("no junk_decoy in " + stem)


def run(stem, seed, dfrom, data_only):
    args = [stem + ".elf", stem + ".gt", "--fuse", "--resolve", "--resolve-elf", seed,
            "--func-gt", stem + ".func.gt", "--decoy-from", hex(dfrom), "--dump-resolved",
            "--thresholds", "0.9"]
    if data_only:
        args.append("--resolve-data-only")
    o = subprocess.run([BENCH] + args, capture_output=True, text=True)
    d = {"edges": [], "kinds": {}}
    for ln in o.stdout.splitlines():
        p = ln.split(",")
        if p[0] == "resolved":
            d["edges"].append((int(p[1], 16), int(p[2], 16), p[4]))
            d["kinds"][p[4]] = d["kinds"].get(p[4], 0) + 1
        elif p[0] == "func_calib" and p[1] == "F":
            d["fF_auroc"] = p[3]
        elif p[0] == "fuse_calib" and p[1] == "phat":
            d["phat_ece"], d["phat_auroc"] = float(p[2]), p[3]
        elif p[0] == "func_recall":
            d["confirmed_func"] = int(p[2]); d["n_func"] = int(p[3]); d["decoy_leak_head"] = int(p[4])
    for ln in o.stderr.splitlines():
        m = re.search(r"soft_recall@R0\.5=([0-9.]+)", ln)
        if m:
            d["soft_recall"] = float(m.group(1))
    d["targets"] = {t for _, t, _ in d["edges"]}
    return d


rows = []
for elf in sorted(glob.glob(os.path.join(CID, "*__native-code-in-data.elf"))):
    stem = elf[:-4]
    name = os.path.basename(stem).split(".elf__")[0]
    dfrom = decoy_from(stem)
    seed = json.load(open(stem + ".manifest.json"))["seed"]["path"]
    igt, fgt = load_hex(stem + ".gt"), load_hex(stem + ".func.gt")
    m3, m4 = run(stem, seed, dfrom, True), run(stem, seed, dfrom, False)
    net = m4["targets"] - m3["targets"]
    rows.append({"name": name, "fam": FAMILY.get(name, "?"), "dfrom": dfrom, "igt": igt, "fgt": fgt,
                 "m3": m3, "m4": m4, "net": net, "net_insn": net & igt, "net_fn": net & fgt})


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


print("=== §1  Coverage by kind (M4) + net-new resolved targets over M3 ===")
print(f"{'spec':>15} {'family':>9} | {'reloc':>5} {'cgoto':>5} {'pie_rel':>7} {'vtable':>6} | "
      f"{'targets M3→M4':>13} | {'net-new':>7} {'(insn':>6} {'fn)':>4}")
for r in rows:
    k3, k4 = r["m3"]["kinds"], r["m4"]["kinds"]
    print(f"{r['name']:>15} {r['fam']:>9} | {k4.get('reloc',0):5d} {k4.get('computed_goto',0):5d} "
          f"{k4.get('pie_rel',0):7d} {k4.get('vtable',0):6d} | "
          f"{len(r['m3']['targets']):5d} → {len(r['m4']['targets']):<5d} | "
          f"{len(r['net']):7d} {len(r['net_insn']):6d} {len(r['net_fn']):4d}")
print("  net-new = M4 targets not resolved at M3; (insn/fn) = of those, how many are instruction-GT / function-GT addresses.")

print("\n=== §2  Flagship: real functions confirmed (F≥0.9) M3 → M4 [the number that was 0 on non-PIE C] ===")
print(f"{'spec':>15} | {'confirmed fn M3→M4':>18} | {'|func GT|':>9} | net-new real fn via M4 edge")
for r in rows:
    nf = len(r["net_fn"])
    print(f"{r['name']:>15} | {r['m3']['confirmed_func']:6d} → {r['m4']['confirmed_func']:<6d}     | "
          f"{r['m3']['n_func']:9d} | {r['m4']['confirmed_func'] - r['m3']['confirmed_func']:+d}"
          f"  (pie_rel/vtable net-new fn targets: {nf})")
tot3 = sum(r["m3"]["confirmed_func"] for r in rows)
tot4 = sum(r["m4"]["confirmed_func"] for r in rows)
print(f"{'TOTAL':>15} | {tot3:6d} → {tot4:<6d}     |           | {tot4 - tot3:+d}")

print("\n=== §3  Decoy-discipline audit: no resolved edge may target [decoy_from, end) ===")
clean = True
for r in rows:
    l3 = sum(1 for _, t, _ in r["m3"]["edges"] if t >= r["dfrom"])
    l4 = sum(1 for _, t, _ in r["m4"]["edges"] if t >= r["dfrom"])
    clean &= (l3 == 0 and l4 == 0)
    print(f"{r['name']:>15} | decoy_from=0x{r['dfrom']:x}  M3 leak={l3}  M4 leak={l4}  "
          f"func decoy_leak(F≥.9) {r['m3']['decoy_leak_head']}→{r['m4']['decoy_leak_head']}")
print(f"  → {'PASS' if clean else 'FAIL'}: exec-range gate keeps every resolved target in the real .text.")

print("\n=== §4  Both axes: function F[AUROC], soft-instruction recall, recalibrated P̂[ECE] — M3 → M4 ===")
print(f"{'spec':>15} | {'func F AUROC M3→M4':>18} | {'soft recall M3→M4':>17} | {'P̂ ECE M3→M4':>15} | {'P̂ AUROC':>8}")
for r in rows:
    m3, m4 = r["m3"], r["m4"]
    print(f"{r['name']:>15} | {m3['fF_auroc']:>7} → {m4['fF_auroc']:<7} | "
          f"{m3.get('soft_recall',0):.3f} → {m4.get('soft_recall',0):.3f} | "
          f"{m3['phat_ece']:.4f} → {m4['phat_ece']:.4f} | {m4['phat_auroc']:>8}")
print(f"{'MEAN':>15} | {mean([float(r['m3']['fF_auroc']) for r in rows]):.4f} → "
      f"{mean([float(r['m4']['fF_auroc']) for r in rows]):.4f}  | "
      f"{mean([r['m3'].get('soft_recall',0) for r in rows]):.3f} → "
      f"{mean([r['m4'].get('soft_recall',0) for r in rows]):.3f} | "
      f"{mean([r['m3']['phat_ece'] for r in rows]):.4f} → {mean([r['m4']['phat_ece'] for r in rows]):.4f}")
