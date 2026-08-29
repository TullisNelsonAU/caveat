#!/usr/bin/env python3
"""Recoverability-staircase measurement harness — STAIRCASE_MEASUREMENT_SPEC.md.

Measures the achievable irreducible-uncertainty staircase U_0 >= U_2 >= ... >= U_5 per object
class x evidence rung x obfuscation, on the EXISTING small corpora only (code-in-data, desync,
decoy-heavy, plus the benign seeds as a baseline). Read-only on the engine: every rung maps to a
udstack invocation, nothing is recompiled.

HARD memory + safety rules (post-crash, non-negotiable; SPEC sec 0):
  * ONE binary in memory at a time. We shell out to `udstack` once per (binary, rung), parse its
    stdout, compute the row, free it, move on. We NEVER hold the corpus, never fan out.
  * `--jobs 1` is inherent: udstack is single-binary and does not rayon-parallelise; the harness
    loop is strictly sequential. There is no worker pool here on purpose.
  * RESUMABLE: every (binary, class, rung) row is appended to the CSV and flushed as it is computed.
    On restart we read the CSV, skip keys already present. A crash costs one row, not the run.

Evidence rungs -> engine modes (SPEC sec 1; the honesty wall means bench's decode flags leave the
raw posterior byte-identical, so the *moving calibrated posterior* is read from udstack --dump-instr):
  R0  raw pi                          udstack --dump-instr           (pi column, no anchor)
  R1  + intra-CFG cover               aggregate only (stack_instr,R)  -- per-offset R is not dumped
                                      by the read-only engine, so R1 has no per-offset entropy; its
                                      calibration is reported at aggregate level and the reachability
                                      drop is measured as E0->E2 (raw -> anchored confirmation).
  R2  + confirmation fixpoint         udstack --dump-instr           (phat column, milestone A)
  R3  + resolvers (M4/M5)             udstack --resolve-elf --dump-instr
  R4  + trace clamp                   udstack --clamp-func <half the true heads>:1.0 --dump-instr
  R5  + oracle clamp                  udstack --clamp-func <all true heads>:1.0 --dump-instr

U_k readouts (SPEC sec 1): the posterior is self-calibrated per rung with an isotonic (PAV) fit
against ground truth -- the "self-recal ceiling", i.e. the *achievable* upper bound on the true
limit. We report, over the dumped instruction domain and over the ambiguity set A_k:
  U_k(entropy)  = mean binary entropy h(q_o) of the calibrated posterior
  U_k(bayes)    = mean 0/1 Bayes risk  min(q_o, 1-q_o)
  plus base_rate / ECE / AUROC of the calibrated readout.
The lower bound h(beta_k) on the ambiguity set is emitted by staircase_ambiguity.py (kept separate so
the achiever and the bound are computed by different code paths -- an audit safeguard).

GT provenance (SPEC sec 0): instruction-start GT is the co-located `.gt` (gen-gt / symtab / DWARF);
function-head GT is `.func.gt` (benign symtab). Two axes (instruction-start, function-head) are kept
in SEPARATE csv columns and never mixed. Every row carries an `align_ok` audit flag from an
independent ELF load-base parse (the ET_DYN check the spec demands on decoy-heavy).

Usage:
  staircase_measure.py --out docs/staircase/staircase_raw.csv [--corpora cid,decoy,desync,benign]
                       [--desync-sample N] [--rungs R0,R2,R3,R4,R5] [--dry-run]
"""
import argparse, csv, glob, json, math, os, struct, subprocess, sys

ROOT = os.path.expanduser("~/lab/projects")
UDSTACK = os.path.join(ROOT, "upd-suite-stack/target/release/udstack")

# ---------------------------------------------------------------------------- corpora
# Each corpus entry: how to enumerate specimens and where the sidecars live. `struct_of` pulls the
# obfuscation sub-structure out of the filename (decoy has 5 structures; the anchored-vs-anchorless
# prediction rides on disconnected vs self-anchoring).
def cid_specs():
    d = os.path.join(ROOT, "upd-suite-stack/corpus_pie/cid")
    for elf in sorted(glob.glob(os.path.join(d, "*__native-code-in-data.elf"))):
        stem = elf[:-4]
        yield dict(elf=elf, gt=stem + ".gt", func_gt=stem + ".func.gt",
                   regions=stem + ".regions", manifest=stem + ".manifest.json",
                   name=os.path.basename(stem).split(".")[0], obf="code-in-data", struct="tiled")

def decoy_specs():
    d = os.path.join(ROOT, "upd-suite-sota/scratch/decoy-smoke")
    for elf in sorted(glob.glob(os.path.join(d, "*.elf"))):
        stem = elf[:-4]
        st = os.path.basename(stem).split("field_")[-1]
        yield dict(elf=elf, gt=stem + ".gt", func_gt=None,          # derived on demand, see func_gt_for
                   regions=stem + ".regions", manifest=stem + ".manifest.json",
                   name=os.path.basename(stem).split("__")[0] + "_" + st, obf="decoy-heavy", struct=st)

def desync_specs(sample=None):
    bins = sorted(glob.glob(os.path.join(ROOT, "probablistic/corpus/desync-pilot/stripped/*")))
    gtd = os.path.join(ROOT, "probablistic/corpus/desync-gt")
    if sample:
        bins = bins[:sample]
    for elf in bins:
        name = os.path.basename(elf)
        yield dict(elf=elf, gt=os.path.join(gtd, name + ".gt"), func_gt=None,
                   regions=None, manifest=None, name=name, obf="desync", struct="desync")

def benign_specs():
    # baseline: the pre-transform seeds. cid seeds ship .func.gt-less; use objdump for func heads.
    d = os.path.join(ROOT, "upd-suite-stack/corpus_pie/seeds")
    for elf in sorted(glob.glob(os.path.join(d, "*.elf"))):
        # benign seeds have symbols but no co-located .gt; skip if we cannot build GT cheaply.
        yield dict(elf=elf, gt=None, func_gt=None, regions=None, manifest=None,
                   name="benign_" + os.path.basename(elf).split(".")[0], obf="benign", struct="benign")

CORPORA = {"cid": cid_specs, "decoy": decoy_specs, "desync": desync_specs, "benign": benign_specs}

# ---------------------------------------------------------------------------- ELF load-base audit
def elf_loadbase(path):
    """Independent minimal ELF parse -> (e_type, e_entry, first exec PT_LOAD p_vaddr, text_hint).
    Mirrors probdisasm/src/header.rs: .text sh_addr if section headers present, else exec PT_LOAD
    p_vaddr. Used only to AUDIT that GT vaddrs fall in the loaded range (ET_DYN / rebase check)."""
    with open(path, "rb") as f:
        b = f.read()
    if b[:4] != b"\x7fELF":
        return None
    is64 = b[4] == 2
    le = "<" if b[5] == 1 else ">"
    e_type = struct.unpack_from(le + "H", b, 16)[0]
    if is64:
        e_entry = struct.unpack_from(le + "Q", b, 24)[0]
        e_phoff = struct.unpack_from(le + "Q", b, 32)[0]
        e_phentsize, e_phnum = struct.unpack_from(le + "HH", b, 54)
    else:
        e_entry = struct.unpack_from(le + "I", b, 24)[0]
        e_phoff = struct.unpack_from(le + "I", b, 28)[0]
        e_phentsize, e_phnum = struct.unpack_from(le + "HH", b, 42)
    exec_vaddr = None
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type = struct.unpack_from(le + "I", b, off)[0]
        if p_type != 1:  # PT_LOAD
            continue
        if is64:
            p_flags = struct.unpack_from(le + "I", b, off + 4)[0]
            p_vaddr = struct.unpack_from(le + "Q", b, off + 16)[0]
        else:
            p_flags = struct.unpack_from(le + "I", b, off + 24)[0]
            p_vaddr = struct.unpack_from(le + "I", b, off + 8)[0]
        if p_flags & 0x1 and exec_vaddr is None:  # PF_X
            exec_vaddr = p_vaddr
    return dict(e_type=e_type, e_entry=e_entry, exec_vaddr=exec_vaddr)

def align_audit(elf, gt_addrs):
    """True iff the GT instruction-start vaddrs sit at/above the exec load base -- i.e. they are
    absolute at the segment's p_vaddr and were not left un-rebased (0x400000 vs 0x402980 etc.)."""
    info = elf_loadbase(elf)
    if info is None or not gt_addrs:
        return None, info
    base = info["exec_vaddr"] or 0
    lo = min(gt_addrs)
    # ET_DYN (type 3) is base-0 PIE: GT == file vaddr, base 0 is fine. ET_EXEC must match the segment.
    ok = (lo >= base) if info["e_type"] != 3 else True
    return ok, info

# ---------------------------------------------------------------------------- GT / regions parsing
def read_gt(path):
    """One instruction-start (or function-head) per line; bare hex OR 0x-prefixed -- accept both."""
    out = set()
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            out.add(int(ln, 16))
    return out

def read_regions(path):
    """(start, end, label, kind) half-open spans; None if no regions sidecar."""
    if not path or not os.path.exists(path):
        return []
    spans = []
    with open(path) as f:
        for ln in f:
            if ln.startswith("#") or not ln.strip():
                continue
            p = ln.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            spans.append((int(p[0], 16), int(p[1], 16), p[2], p[3]))
    return spans

# ---------------------------------------------------------------------------- func-head GT
def func_gt_for(spec, scratch):
    """Return a path to a function-head GT for this specimen, or None if the class is unavailable.
    cid ships `.func.gt`. decoy does not -> derive from the benign SEED's .text FUNC symbols
    (objdump -t), keeping only heads inside a real_code region (the ET_DYN/rebase audit for the func
    axis). desync/benign have no func provenance here -> None (instruction-start axis only)."""
    if spec["func_gt"] and os.path.exists(spec["func_gt"]):
        return spec["func_gt"]
    if spec["obf"] != "decoy-heavy" or not spec["manifest"]:
        return None
    man = json.load(open(spec["manifest"]))
    seed = man.get("seed", {}).get("path")
    if not seed or not os.path.exists(seed):
        return None
    try:
        out = subprocess.run(["objdump", "-t", seed], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return None
    heads = set()
    for ln in out.splitlines():
        if " F .text\t" in ln:
            try:
                heads.add(int(ln.split()[0], 16))
            except ValueError:
                pass
    if not heads:
        return None
    # keep only heads that land in a real_code region of THIS binary (audits the rebase)
    real = [(s, e) for (s, e, lab, kind) in read_regions(spec["regions"]) if kind == "real_code"]
    if real:
        heads = {h for h in heads if any(s <= h < e for (s, e) in real)}
    if not heads:
        return None
    os.makedirs(scratch, exist_ok=True)
    p = os.path.join(scratch, spec["name"] + ".func.gt")
    with open(p, "w") as f:
        f.write("".join("0x%016x\n" % h for h in sorted(heads)))
    return p

# ---------------------------------------------------------------------------- isotonic calibration
def isotonic_fit(scores, labels):
    """Pool-Adjacent-Violators. Returns (xs, ys) step function: q = ys[i] for first xs[i] >= s.
    This is the self-recal ceiling -- the calibrated posterior the spec measures U_k on."""
    pts = sorted(zip(scores, labels))
    xs = [p[0] for p in pts]
    ys = [float(p[1]) for p in pts]
    w = [1.0] * len(ys)
    # PAV
    i = 0
    while i < len(ys) - 1:
        if ys[i] > ys[i + 1] + 1e-12:
            # merge i, i+1
            tw = w[i] + w[i + 1]
            ty = (ys[i] * w[i] + ys[i + 1] * w[i + 1]) / tw
            ys[i] = ty; w[i] = tw; xs[i] = xs[i + 1]
            del ys[i + 1]; del w[i + 1]; del xs[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    return xs, ys

def isotonic_apply(cal, s):
    xs, ys = cal
    # first breakpoint whose x >= s (step is right-continuous on the sorted training xs)
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] < s:
            lo = mid + 1
        else:
            hi = mid
    idx = min(lo, len(ys) - 1)
    return max(0.0, min(1.0, ys[idx]))

def h_bin(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    # rank-sum
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    rsum = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    n1 = len(pos)
    return (rsum - n1 * (n1 + 1) / 2.0) / (n1 * len(neg))

def ece(qs, labels, nbins=10):
    tot = len(qs)
    if tot == 0:
        return float("nan")
    e = 0.0
    for b in range(nbins):
        lo, hi = b / nbins, (b + 1) / nbins
        idx = [i for i in range(tot) if (qs[i] >= lo and (qs[i] < hi or (b == nbins - 1 and qs[i] <= hi)))]
        if not idx:
            continue
        conf = sum(qs[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        e += len(idx) / tot * abs(conf - acc)
    return e

# ---------------------------------------------------------------------------- udstack run + parse
def run_udstack(elf, gt, func_gt=None, resolve=False, clamp=None):
    """One process, one binary. Returns (per_offset_rows, aggregates).
    per_offset_rows: (addr, phat, pi, label01) from --dump-instr (instruction axis).
    aggregates: {layer: (ece, auroc, base_rate)} parsed from the stack_instr,* (pi/R/phat) and
    stack_func,F machine lines -- the E1 cover layer (R) and the function-head axis (F) that
    --dump-instr does not emit per-offset."""
    cmd = [UDSTACK, elf, gt, "--dump-instr"]
    if func_gt:
        cmd += ["--func-gt", func_gt]
    if resolve:
        cmd += ["--resolve-elf", elf]
    for a in (clamp or []):
        cmd += ["--clamp-func", "0x%x:1.0" % a]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    rows, agg = [], {}
    for ln in (p.stdout + "\n" + p.stderr).splitlines():
        if ln.startswith("instr_bel,"):
            f = ln.split(",")
            try:
                rows.append((int(f[1], 16), float(f[2]), float(f[3]), 1 if f[4] == "real" else 0))
            except (ValueError, IndexError):
                pass
        elif ln.startswith("stack_instr,") or ln.startswith("stack_func,"):
            f = ln.split(",")
            try:                              # stack_instr,pi,ECE,AUROC,BASE  /  stack_func,F,...
                agg[f[1]] = (float(f[2]), float(f[3]), float(f[4]))
            except (ValueError, IndexError):
                pass
    return rows, agg

# ---------------------------------------------------------------------------- rung -> score selection
def rung_score(rung, phat, pi):
    return pi if rung == "R0" else phat

def clamp_heads(rung, func_gt):
    if rung not in ("R4", "R5") or not func_gt:
        return None
    heads = sorted(read_gt(func_gt))
    if rung == "R4":                       # trace clamp: half the true heads (deterministic subset)
        return heads[::2]
    return heads                            # R5 oracle: all true heads

def needs_resolve(rung):
    return rung in ("R3", "R4", "R5")

# ---------------------------------------------------------------------------- metrics for one readout
def metrics(rows, ambig_set, rung):
    """rows: (addr, phat, pi, label01). Pick the rung's raw score (pi at R0, phat elsewhere),
    self-calibrate it against the 0/1 label, then compute U over full domain + ambiguity set."""
    scores = [rung_score(rung, r[1], r[2]) for r in rows]
    labels = [r[3] for r in rows]
    cal = isotonic_fit(scores, labels)
    qs = [isotonic_apply(cal, s) for s in scores]
    full_H = sum(h_bin(q) for q in qs) / len(qs)
    full_risk = sum(min(q, 1 - q) for q in qs) / len(qs)
    br = sum(labels) / len(labels)
    row = dict(n=len(rows), base_rate=br, U_entropy=full_H, U_bayes=full_risk,
               ece=ece(qs, labels), auroc=auroc(scores, labels))
    # ambiguity-set restricted
    if ambig_set:
        idx = [i for i, r in enumerate(rows) if r[0] in ambig_set]
        if idx:
            row["n_ambig"] = len(idx)
            row["U_entropy_ambig"] = sum(h_bin(qs[i]) for i in idx) / len(idx)
            row["U_bayes_ambig"] = sum(min(qs[i], 1 - qs[i]) for i in idx) / len(idx)
            row["beta_ambig"] = sum(labels[i] for i in idx) / len(idx)
    return row

# ---------------------------------------------------------------------------- ambiguity set A_0
def ambiguity_set(spec, gt_addrs):
    """A_0 = offsets with >=2 realizable interpretations at E0 (SPEC sec 1). By construction:
    real starts UNION decoy candidate starts. decoy-heavy: exact manifest decoy_entries. cid: the
    junk_decoy region (tiled real code) -> its offsets are candidate starts. Returns a set of vaddrs
    (real + decoy). Provenance-tagged so the tight-cell audit knows how it was built."""
    if spec.get("manifest") and os.path.exists(spec["manifest"]):
        man = json.load(open(spec["manifest"]))
        de = man.get("params", {}).get("decoy_entries")
        if isinstance(de, list) and de:
            return set(gt_addrs) | set(de), "manifest.decoy_entries"
    # cid: use the junk_decoy region extent as the decoy candidate domain
    regions = read_regions(spec.get("regions"))
    decoy_spans = [(s, e) for (s, e, lab, kind) in regions if kind == "junk_decoy"]
    if decoy_spans:
        # candidate decoy starts = every offset in the decoy span (tiled real code; a superset of
        # true decoy starts -- the tight-cell script refines this; here we bound the set).
        decoy = set()
        for s, e in decoy_spans:
            decoy |= set(range(s, e))
        return set(gt_addrs) | decoy, "junk_decoy_region"
    return set(), "none"

# ---------------------------------------------------------------------------- driver
FIELDS = ["binary", "obf", "struct", "obj_class", "rung", "n", "n_ambig", "base_rate",
          "U_entropy", "U_bayes", "U_entropy_ambig", "U_bayes_ambig", "beta_ambig",
          "ece", "auroc", "align_ok", "e_type", "exec_vaddr", "gt_n", "ambig_provenance",
          "resolve", "n_clamp", "status"]

def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                done.add((r["binary"], r["obj_class"], r["rung"]))
    return done

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/staircase/staircase_raw.csv")
    ap.add_argument("--corpora", default="cid,decoy,desync")
    ap.add_argument("--rungs", default="R0,R2,R3,R4,R5")
    ap.add_argument("--desync-sample", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    scratch = os.path.join(os.path.dirname(os.path.abspath(args.out)), "_scratch")
    rungs = args.rungs.split(",")
    done = load_done(args.out)
    new_file = not os.path.exists(args.out)
    fout = None if args.dry_run else open(args.out, "a", newline="")
    writer = None
    if fout:
        writer = csv.DictWriter(fout, fieldnames=FIELDS)
        if new_file:
            writer.writeheader(); fout.flush()

    specs = []
    for c in args.corpora.split(","):
        gen = CORPORA[c]
        specs += list(gen(args.desync_sample) if c == "desync" else gen())

    print("# %d specimens, rungs=%s, %d already done" % (len(specs), rungs, len(done)))
    for spec in specs:
        if not spec["gt"] or not os.path.exists(spec["gt"]) or not os.path.exists(spec["elf"]):
            print("  SKIP %s (no gt/elf)" % spec["name"]); continue
        gt_addrs = read_gt(spec["gt"])
        align_ok, info = align_audit(spec["elf"], gt_addrs)
        ambig, ambig_prov = ambiguity_set(spec, gt_addrs)
        func_gt = func_gt_for(spec, scratch)

        def base_row(obj_class, rung, heads, status):
            return dict(binary=spec["name"], obf=spec["obf"], struct=spec["struct"],
                        obj_class=obj_class, rung=rung, align_ok=align_ok,
                        e_type=(info or {}).get("e_type"), exec_vaddr=(info or {}).get("exec_vaddr"),
                        gt_n=len(gt_addrs), ambig_provenance=ambig_prov,
                        resolve=needs_resolve(rung), n_clamp=len(heads or []), status=status)

        def emit(row):
            writer.writerow({k: row.get(k) for k in FIELDS}); fout.flush()
            print("  %-30s %-16s %-3s U_H=%s ambig=%s %s" % (
                row["binary"], row["obj_class"], row["rung"],
                ("%.4f" % row["U_entropy"]) if row.get("U_entropy") is not None else "-",
                ("%.4f" % row["U_entropy_ambig"]) if row.get("U_entropy_ambig") is not None else "-",
                row["status"]))

        for rung in rungs:
            heads = clamp_heads(rung, func_gt)
            if rung in ("R4", "R5") and not heads:
                continue  # no oracle to clamp -> rung n/a for this specimen (documented in results)
            # what still needs doing at this rung? (instruction-start always; function-head iff func_gt;
            # R1 cover aggregate is emitted alongside R0)
            want = []
            if (spec["name"], "instruction-start", rung) not in done:
                want.append("instruction-start")
            if func_gt and (spec["name"], "function-head", rung) not in done:
                want.append("function-head")
            if rung == "R0" and (spec["name"], "instruction-start", "R1") not in done:
                want.append("R1")
            if not want:
                continue
            if args.dry_run:
                print("  would run", (spec["name"], rung), "-> classes", want,
                      "resolve=%s clamp=%s" % (needs_resolve(rung), len(heads or [])))
                continue
            # ONE udstack process for this rung (one binary in memory); serves both axes.
            try:
                rows, agg = run_udstack(spec["elf"], spec["gt"], func_gt=func_gt,
                                        resolve=needs_resolve(rung), clamp=heads)
                if not rows:
                    raise RuntimeError("no instr_bel rows")
                status = "ok"
            except Exception as ex:
                rows, agg, status = [], {}, "ERR:%s" % str(ex)[:40]

            if "instruction-start" in want:
                r = base_row("instruction-start", rung, heads, status)
                if status == "ok":
                    r.update(metrics(rows, ambig, rung))
                emit(r)
            if "R1" in want:                      # cover layer: aggregate-only (per-offset R not dumped)
                r = base_row("instruction-start", "R1", None, status if agg.get("R") else "no_R_line")
                if agg.get("R"):
                    r.update(dict(ece=agg["R"][0], auroc=agg["R"][1], base_rate=agg["R"][2],
                                  n=len(rows)))
                emit(r)
            if "function-head" in want:           # function axis: aggregate stack_func,F (U NA)
                r = base_row("function-head", rung, heads, status if agg.get("F") else "no_F_line")
                if agg.get("F"):
                    r.update(dict(ece=agg["F"][0], auroc=agg["F"][1], base_rate=agg["F"][2]))
                emit(r)
    if fout:
        fout.close()
    print("# done ->", args.out)

if __name__ == "__main__":
    main()
