#!/usr/bin/env python3
"""De-risk probe harness — DERISK_PROBE_SPEC.md. The master GO/NO-GO gate.

Does the Soft posterior (`bench --confirm-soft`) stay CALIBRATED under REAL Tigress obfuscation?
For each small header-free probe program we build the un-obfuscated baseline plus one specimen per
Tigress transform family, run the Soft engine against GT-by-construction (gen-gt/symtab), and stream
one CSV row per (program, transform). Resumable; ONE binary in memory at a time; --jobs 1 (strictly
serial); no large corpus build (~12 programs x 6 = ~72 tiny specimens).

Pipeline per specimen (SPEC sec 1):
  tigress <T>  ->  x86_64-unknown-linux-gnu-gcc --sysroot=$SR -O2 -g -no-pie  ->  gen-gt  ->  bench --confirm-soft
GT is insn_max.txt from gen-gt (the compiler's own true instruction starts; superset incl. the
neutral zone). Negatives = .text bytes gen-gt is confident are NOT starts, so a high posterior there
is a genuine confidently-wrong (the desync signature). Never a disassembler.

Alignment audit: -no-pie => ET_EXEC, fixed load base, GT vaddrs absolute (align_ok = e_type==EXEC).

Usage: derisk_probe.py [--out PATH] [--programs p01,p07] [--transforms Flatten,Virtualize] [--dry-run]
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DERISK = os.path.normpath(os.path.join(HERE, "..", "derisk"))
PROGRAMS_DIR = os.path.join(DERISK, "programs")
SCRATCH = os.path.join(DERISK, "_scratch")
BENCH = os.path.expanduser("~/lab/projects/upd-suite/target/release/bench")
GENGT = os.path.expanduser("~/lab/projects/compare/groundtruth/target/release/gen-gt")
TIGRESS_HOME = "/Applications/tigress/4.0.11"
TIGRESS = os.path.join(TIGRESS_HOME, "tigress")
XGCC = "x86_64-unknown-linux-gnu-gcc"
ENV = "x86_64:Linux:Gcc:0"

# function lists per program (Tigress needs explicit --Functions; main carries the opaque init)
FUNCS = {
    "p01_statemachine": "trans,main",
    "p02_insertsort": "isort,main",
    "p03_ackermann": "ack,main",
    "p04_fnptr": "op_add,op_mul,op_xor,op_sub,main",
    "p05_vm": "run,main",
    "p06_parser": "is_digit,eval,main",
    "p07_crc": "crc_step,main",
    "p08_binsearch": "bsearch_i,main",
    "p09_matmul": "matmul,main",
    "p10_collatz": "collatz_len,main",
    "p11_sieve": "main",
    "p12_modpow": "gcd,modpow,main",
}

# opaque-predicate infra prelude (extern protos — NO #include, so no macOS libc header inlining)
PRELUDE = ("extern void *malloc(unsigned long);\n"
           "extern void *calloc(unsigned long, unsigned long);\n"
           "extern void free(void *);\n")

# InitEntropyKinds=vars avoids timespec/time.h; InitOpaque structs need malloc (supplied by PRELUDE)
OPAQUE_PRE = ("--Transform=InitEntropy --InitEntropyKinds=vars --Functions=main "
              "--Transform=InitOpaque --InitOpaqueStructs=list,array --Functions=main")

# transform family -> (tigress arg fragment, needs_prelude). "baseline" = no tigress.
TRANSFORMS = {
    "baseline": (None, False),
    "Virtualize": ("--Transform=Virtualize --Functions={fns}", False),
    "Flatten": ("--Transform=Flatten --Functions={fns}", False),
    "EncodeArithmetic": ("--Transform=EncodeArithmetic --Functions={fns}", False),
    "AddOpaque": (OPAQUE_PRE + " --Transform=AddOpaque --Functions={fns}", True),
    "EncodeLiterals": (OPAQUE_PRE + " --Transform=EncodeLiterals --Functions={fns}", True),
}
TRANSFORM_ORDER = ["baseline", "Virtualize", "Flatten", "AddOpaque", "EncodeArithmetic", "EncodeLiterals"]

FIELDS = ["program", "transform", "gt_kind", "ok", "err", "e_type", "align_ok", "text_base",
          "text_bytes", "gt_starts", "base_rate", "ece_raw", "ece_recal", "auroc_raw", "auroc_recal",
          "reliability", "resolution", "recall0", "prec0", "f1_0",
          "prec_strict", "npred_strict", "cw_fp", "cw_flag"]


def run(cmd, timeout=900, env=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)


def tig_env():
    e = dict(os.environ)
    e["TIGRESS_HOME"] = TIGRESS_HOME
    e["PATH"] = TIGRESS_HOME + os.pathsep + e.get("PATH", "")
    return e


def sysroot():
    return run([XGCC, "-print-sysroot"]).stdout.strip()


def obfuscate(prog_c, transform, fns, out_c):
    """Produce the (possibly obfuscated) .c at out_c. baseline => copy through (+no prelude)."""
    frag, needs_prelude = TRANSFORMS[transform]
    src = prog_c
    if needs_prelude:
        src = out_c + ".src.c"
        with open(prog_c) as f:
            body = f.read()
        with open(src, "w") as f:
            f.write(PRELUDE + body)
    if frag is None:  # baseline
        shutil.copyfile(src, out_c)
        return True, ""
    args = ([TIGRESS, "--Environment=" + ENV, "--Seed=20260707"]
            + frag.format(fns=fns).split() + ["--out=" + out_c, src])
    p = run(args, env=tig_env())
    if p.returncode != 0 or not os.path.exists(out_c):
        msg = ""
        for ln in (p.stdout + p.stderr).splitlines():
            if "ERR-" in ln or "fatal error" in ln.lower():
                msg = ln.strip()[:160]
                break
        return False, "tigress:" + (msg or ("rc=%d" % p.returncode))
    return True, ""


def compile_elf(src_c, out_elf, sr):
    p = run([XGCC, "--sysroot=" + sr, "-O2", "-g", "-no-pie", src_c, "-o", out_elf])
    if p.returncode != 0 or not os.path.exists(out_elf):
        line = ""
        for ln in p.stderr.splitlines():
            if "error" in ln.lower() or "undefined ref" in ln.lower():
                line = ln.strip()[:160]
                break
        return False, "gcc:" + (line or "rc=%d" % p.returncode)
    return True, ""


def elf_audit(elf):
    """(e_type, entry) by parsing the ELF header bytes directly (no readelf on macOS).
    e_type: u16 LE @ 0x10 (2=EXEC, 3=DYN); e_entry: u64 LE @ 0x18. -no-pie => EXEC."""
    import struct as _s
    with open(elf, "rb") as f:
        hdr = f.read(64)
    if hdr[:4] != b"\x7fELF":
        return "?", "?"
    et = _s.unpack_from("<H", hdr, 0x10)[0]
    e_type = {2: "EXEC", 3: "DYN"}.get(et, str(et))
    entry = "0x%x" % _s.unpack_from("<Q", hdr, 0x18)[0]
    return e_type, entry


def gen_gt(elf, gt_dir):
    """Run gen-gt once; returns the dir holding BOTH insn_min.txt and insn_max.txt."""
    if os.path.isdir(gt_dir):
        shutil.rmtree(gt_dir)
    p = run([GENGT, elf, gt_dir])
    if not os.path.exists(os.path.join(gt_dir, "insn_max.txt")):
        return None, "gengt:" + (p.stderr.strip()[:120] or "no insn_max")
    return gt_dir, ""


CAL_RE = re.compile(r"^calibration,([\d.]+),([\d.]+),([\d.]+),([\d.]+),([\d.]+),([\d.]+)")
RECAL_RE = re.compile(r"self-recal ceiling.*AUROC ([\d.]+)")
HEAD_RE = re.compile(r"\.text (\d+) B @ (0x[0-9a-f]+), GT (\d+) starts")
BIAS_RE = re.compile(r"^(-?[\d.]+),(\d+),(\d+),([\d.]+),([\d.]+),([\d.]+)")


def bench_soft(elf, gt):
    p = run([BENCH, elf, gt, "--confirm-soft"])
    out = p.stdout                       # CSV/calibration + bias sweep on stdout
    full = p.stdout + "\n" + p.stderr    # human header + recal line on stderr
    m = CAL_RE.search(out)
    if not m:
        return None, "bench:" + (p.stderr.strip()[:120] or "no calibration line")
    base_rate, ece_raw, reliability, resolution, auroc_raw, ece_recal = (float(x) for x in m.groups())
    mr = RECAL_RE.search(full)
    auroc_recal = float(mr.group(1)) if mr else auroc_raw
    mh = HEAD_RE.search(full)
    text_bytes = int(mh.group(1)) if mh else -1
    text_base = mh.group(2) if mh else "?"
    gt_starts = int(mh.group(3)) if mh else -1
    # bias sweep rows -> dict keyed by bias
    rows = {}
    for ln in out.splitlines():
        mb = BIAS_RE.match(ln)
        if mb and not ln.startswith("bias"):
            b = float(mb.group(1))
            rows[b] = dict(n_pred=int(mb.group(2)), tp=int(mb.group(3)),
                           recall=float(mb.group(4)), precision=float(mb.group(5)), f1=float(mb.group(6)))
    r0 = rows.get(0.0, {})
    strict_b = min(rows) if rows else None
    rs = rows.get(strict_b, {}) if strict_b is not None else {}
    prec_strict = rs.get("precision", float("nan"))
    npred_strict = rs.get("n_pred", 0)
    cw_fp = round(npred_strict * (1.0 - prec_strict)) if npred_strict else 0
    return dict(
        base_rate=base_rate, ece_raw=ece_raw, ece_recal=ece_recal,
        auroc_raw=auroc_raw, auroc_recal=auroc_recal, reliability=reliability, resolution=resolution,
        text_bytes=text_bytes, text_base=text_base, gt_starts=gt_starts,
        recall0=r0.get("recall", float("nan")), prec0=r0.get("precision", float("nan")),
        f1_0=r0.get("f1", float("nan")),
        prec_strict=prec_strict, npred_strict=npred_strict, cw_fp=cw_fp,
        cw_flag=int(prec_strict == prec_strict and prec_strict < 0.90)), ""


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                done.add((r["program"], r["transform"], r.get("gt_kind", "max")))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DERISK, "derisk_probe.csv"))
    ap.add_argument("--programs", default="")
    ap.add_argument("--transforms", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    for tool, name in [(BENCH, "bench"), (GENGT, "gen-gt"), (TIGRESS, "tigress")]:
        if not os.path.exists(tool):
            print("MISSING %s at %s" % (name, tool)); return 2
    os.makedirs(SCRATCH, exist_ok=True)
    sr = sysroot()

    progs = sorted(FUNCS)
    if a.programs:
        want = set(a.programs.split(","))
        progs = [p for p in progs if p in want or p.split("_")[0] in want]
    tlist = TRANSFORM_ORDER
    if a.transforms:
        want = set(a.transforms.split(","))
        tlist = [t for t in TRANSFORM_ORDER if t in want]

    if a.dry_run:
        print("would run %d programs x %d transforms = %d specimens"
              % (len(progs), len(tlist), len(progs) * len(tlist)))
        print("programs:", ", ".join(progs))
        print("transforms:", ", ".join(tlist))
        return 0

    done = load_done(a.out)
    new_file = not os.path.exists(a.out)
    fh = open(a.out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        w.writeheader(); fh.flush()

    for prog in progs:
        prog_c = os.path.join(PROGRAMS_DIR, prog + ".c")
        fns = FUNCS[prog]
        for t in tlist:
            # both GTs (min/max) come from the SAME ELF -> build once, bench per pending kind
            pending = [k for k in ("max", "min") if (prog, t, k) not in done]
            if not pending:
                continue
            tag = "%s__%s" % (prog, t)
            out_c = os.path.join(SCRATCH, tag + ".c")
            out_elf = os.path.join(SCRATCH, tag + ".elf")
            gt_dir = os.path.join(SCRATCH, tag + ".gtdir")

            def base_row(kind):
                r = {k: "" for k in FIELDS}
                r["program"], r["transform"], r["gt_kind"], r["ok"] = prog, t, kind, 0
                return r

            rows_out = {k: base_row(k) for k in pending}
            build_err, e_type, gtd = "", "?", None
            try:
                ok, build_err = obfuscate(prog_c, t, fns, out_c)
                if ok:
                    ok, build_err = compile_elf(out_c, out_elf, sr)
                if ok:
                    e_type, _entry = elf_audit(out_elf)
                    gtd, build_err = gen_gt(out_elf, gt_dir)
                if ok and gtd is not None:
                    for kind in pending:
                        gtf = os.path.join(gtd, "insn_%s.txt" % kind)
                        res, err = bench_soft(out_elf, gtf)
                        r = rows_out[kind]
                        r["e_type"] = e_type
                        r["align_ok"] = int(e_type == "EXEC")
                        if res is None:
                            r["err"] = err
                        else:
                            r.update(res)
                            r["ok"] = 1
                else:
                    for kind in pending:
                        rows_out[kind]["err"] = build_err
                        rows_out[kind]["e_type"] = e_type
                        rows_out[kind]["align_ok"] = int(e_type == "EXEC")
            except subprocess.TimeoutExpired:
                for kind in pending:
                    rows_out[kind]["err"] = "timeout"
            except Exception as e:  # noqa
                for kind in pending:
                    rows_out[kind]["err"] = ("exc:" + str(e))[:160]
            # free artifacts (one binary in memory at a time; keep scratch small)
            for pth in (out_elf, out_c, out_c + ".src.c"):
                try:
                    os.remove(pth)
                except OSError:
                    pass
            if os.path.isdir(gt_dir):
                shutil.rmtree(gt_dir, ignore_errors=True)
            for kind in pending:
                row = rows_out[kind]
                w.writerow(row); fh.flush()
                flag = "OK" if row["ok"] else ("FAIL " + str(row["err"])[:70])
                extra = ""
                if row["ok"]:
                    extra = ("ECE %.4f→%.4f AUROC %.3f rec %.2f prec %.2f cw_fp %s%s"
                             % (row["ece_raw"], row["ece_recal"], row["auroc_raw"],
                                row["recall0"], row["prec0"], row["cw_fp"],
                                "  ⚠CONFIDENTLY-WRONG" if row["cw_flag"] else ""))
                print("[%-24s %-16s %-3s] %s %s" % (prog, t, kind, flag, extra))
    fh.close()
    print("\ndone -> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
