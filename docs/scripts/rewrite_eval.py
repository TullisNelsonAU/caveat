#!/usr/bin/env python3
"""Behavioural evaluation of the confidence-gated rewriter (Instance 2, REWRITING_APP_SPEC).

Ground truth of a rewrite's success is BEHAVIOUR: the patched binary must reproduce the *original's*
reference I/O (stdout + exit) on every input — never a disassembler's opinion. x86_64 ELFs run in an
amd64 container; the reference is captured from the unmodified binary in the same image.

For each binary we compare, on the SAME leader set and the SAME patcher:
  * baseline  — deterministic linear-sweep rewriter, instruments every leader (commits everywhere);
  * ours(τ)   — instruments only leaders with calibrated belief bel ≥ τ, abstains below.
Headline = fraction of binaries still WORKING after rewrite, ours vs baseline, on the hard inputs.
The τ sweep gives the coverage-vs-correctness curve: calibration makes τ a meaningful safety knob.

Emits a human table to stdout and a machine CSV (rewrite_curve.csv) for the plots.
"""
import csv, os, subprocess, sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REW = os.path.join(REPO, "target/release/rewrite")
UD = os.path.join(REPO, "target/release/udstack")
BIN = os.path.join(REPO, "corpus_rw/bin")
PACKED = os.path.join(REPO, "corpus_packed")
IMG = "debian:stable-slim"
TAUS = [0.0, 0.5, 0.7, 0.9, 0.95, 0.99]
WORK = "/tmp/rw_eval"
os.makedirs(WORK, exist_ok=True)

# (name, elf, marginals, gate_col, decoy_range, [input argv strings])
#   gate_col: 3 = calibrated stack P̂ (has GT); 4 = raw probdisasm π (packed, no GT).
CORPUS = [
    ("clean_calc", f"{BIN}/clean_calc.elf", f"{BIN}/clean_calc.marg", 3, None,
     ["12 34 5", "999 1", "7", "abc 42 zzz"], "clean"),
    ("datatab", f"{BIN}/datatab.elf", f"{BIN}/datatab.marg", 3, (0x1250, 0x1450),
     ["12 34 5", "1", "7 7 7", "255 0 128"], "code-in-data (hard)"),
    ("junk_desync", f"{BIN}/junk_desync.elf", f"{BIN}/junk_desync.marg", 3, None,
     ["12 34 5", "999 1", "7", "abc 42 zzz"], "overlap-desync (hard)"),
    ("ls_packed", f"{PACKED}/ls_packed", None, 4, None,
     ["--version", "--help"], "UPX-packed (hard, uncalibrated)"),
]
SEP, EXM = "@@SEP@@", "@@EXIT@@"


def marg_for(elf, marg, gate_col):
    """Path to an instr_bel file whose phat column is the gate signal. gate_col 4 → use raw π."""
    if marg is None:  # packed: no GT — derive raw π from a dummy-GT udstack run
        import struct
        entry = struct.unpack_from("<Q", open(elf, "rb").read(0x20), 0x18)[0]  # e_entry
        gpath = os.path.join(WORK, "packed.gt")
        open(gpath, "w").write(hex(entry) + "\n")
        raw = subprocess.run([UD, elf, gpath, "--dump-instr"], capture_output=True, text=True).stdout
        marg = os.path.join(WORK, "packed.marg")
        open(marg, "w").writelines(l + "\n" for l in raw.splitlines() if l.startswith("instr_bel"))
        gate_col = 4
    if gate_col == 4:  # move raw π (col 4) into the phat slot the rewriter reads
        out = os.path.join(WORK, os.path.basename(marg) + ".pi")
        with open(marg) as f, open(out, "w") as g:
            for ln in f:
                p = ln.strip().split(",")
                if p[0] == "instr_bel":
                    g.write(f"instr_bel,{p[1]},{p[3]},{p[3]},{p[4]}\n")
        return out
    return marg


def build(elf, out, mode, marg=None, tau=0.9):
    args = [REW, elf, out, "--mode", mode, "--dump-sites"]
    if mode == "ours":
        args += ["--marginals", marg, "--tau", str(tau)]
    r = subprocess.run(args, capture_output=True, text=True)
    os.chmod(out, 0o755)
    sites, summ = [], {}
    for ln in r.stdout.splitlines():
        p = ln.split(",")
        if p[0] == "site":
            sites.append((int(p[1], 16), int(p[2])))
        elif p[0] == "rewrite_summary":
            summ = {"leaders": int(p[3]), "sites": int(p[4]), "coverage": float(p[5])}
    return sites, summ


def run_all(elf, inputs):
    """Run every input in one container; return list of (stdout, exit) for the given elf path."""
    d, b = os.path.dirname(elf), os.path.basename(elf)
    script = "".join(f'/s/{b} {i}; echo "{EXM}$?"; echo "{SEP}"; ' for i in inputs)
    r = subprocess.run(
        ["docker", "run", "--rm", "--platform", "linux/amd64", "-v", f"{d}:/s:ro", IMG, "bash", "-c", script],
        capture_output=True, text=True)
    outs = []
    for chunk in r.stdout.split(SEP)[:-1]:
        lines = chunk.split(EXM)
        outs.append((lines[0], lines[1].strip() if len(lines) > 1 else "?"))
    return outs


def decoy_sites(sites, decoy, gt, name):
    if decoy:
        return sum(1 for a, _ in sites if decoy[0] <= a < decoy[1])
    if name == "junk_desync":  # off-GT sites = desync artefacts
        return sum(1 for a, _ in sites if a not in gt)
    return 0


rows = []
print(f"{'binary':<13} {'class':<26} {'arm':<11} {'cov':>6} {'sites':>6} {'decoyS':>6} {'works':>6}")
for name, elf, marg0, gc, decoy, inputs, cls in CORPUS:
    gt = set()
    gp = elf.replace(".elf", ".gt") if elf.endswith(".elf") else None
    if gp and os.path.exists(gp):
        gt = {int(l, 16) for l in open(gp) if l.strip()}
    ref = run_all(elf, inputs)
    marg = marg_for(elf, marg0, gc)

    def report(arm, sites, summ, works):
        ds = decoy_sites(sites, decoy, gt, name)
        cov = summ.get("coverage", 0.0)
        print(f"{name:<13} {cls:<26} {arm:<11} {cov:>6.3f} {summ.get('sites',0):>6} {ds:>6} {'YES' if works else 'no':>6}")
        rows.append(dict(binary=name, cls=cls, arm=arm, coverage=cov, sites=summ.get("sites", 0),
                         decoy_sites=ds, works=int(works)))

    # baseline
    o = os.path.join(WORK, f"{name}.base.elf")
    sites, summ = build(elf, o, "baseline")
    works = run_all(o, inputs) == ref
    report("baseline", sites, summ, works)
    # ours across τ
    for tau in TAUS:
        o = os.path.join(WORK, f"{name}.ours.{tau}.elf")
        sites, summ = build(elf, o, "ours", marg, tau)
        works = run_all(o, inputs) == ref
        report(f"ours τ={tau}", sites, summ, works)

with open(os.path.join(os.path.dirname(__file__), "rewrite_curve.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["binary", "cls", "arm", "coverage", "sites", "decoy_sites", "works"])
    w.writeheader()
    w.writerows(rows)
print(f"\ncurve → {os.path.join(os.path.dirname(__file__), 'rewrite_curve.csv')}")
