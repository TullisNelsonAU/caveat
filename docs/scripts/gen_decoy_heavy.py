#!/usr/bin/env python3
"""Generate the **decoy-heavy** code-in-data corpus for the interactive/active eval (Instance 3, Arm A).

The stock `native-code-in-data` corpus (`/tmp/cid`, `region_bytes=8000`) appends ONE small block of
tiled real code past the entry. On that corpus the low-`F_h` uncertain band the active loop queries over
is *real-dominated* (the decoys sit cleanly at `F<0.05`, below the candidate floor, and the low-`F` tail
is systematically real Theorem-2 indirect functions) — so a naive "confirm-least-sure" ordering ties
EIG. That is the corpus quirk `UDSTACK_RESULTS.md` §active flagged.

This drives the SAME validated `gauntlet native-code-in-data` generator (no generator edit) with a large
`--region-bytes` so the decoy mass exceeds the real `.text`. With hundreds of tiled decoy heads, enough
pick up spurious intra-decoy call edges to land IN the candidate band `[0.05, 0.95]` — the band becomes
decoy-*dominated*. That is the regime where EIG's `F_h·ΔH` objective (which deprioritises low-`F`
likely-decoys) must beat "confirm-least-sure". GT stays by-construction: the real half's starts are the
gen-gt validated starts; the decoy half is provably-unreachable data, so no decoy byte is an instruction
start and no decoy head is a FUNC symbol.

`func.gt` is derived exactly as `gen_func_gt.py` does — `.text` FUNC symbols from the benign seed's
symtab (real symbol-table entries, not a disassembler); decoy heads get no symbol ⇒ labelled negative by
construction, which is what makes the query oracle able to *deny* a decoy.

Usage:  gen_decoy_heavy.py [--region-bytes N] [--src /tmp/cid] [--out /tmp/cid_heavy]
"""
import argparse, glob, json, os, subprocess, sys

ROOT = "/home/anon/lab/projects/probablistic"
GAUNTLET = os.path.expanduser("~/lab/projects/upd-suite-stack/target/release/gauntlet")


def func_gt_from_seed(seed_path):
    """`.text` FUNC-symbol addresses from the benign seed's symtab (objdump -t), sorted hex."""
    out = subprocess.run(["objdump", "-t", seed_path], capture_output=True, text=True).stdout
    return sorted(int(l.split()[0], 16) for l in out.splitlines() if " F .text\t" in l)


def text_size(seed_path):
    """Size of the seed's `.text` section in bytes (objdump -h), so the decoy mass can be scaled to it."""
    out = subprocess.run(["objdump", "-h", seed_path], capture_output=True, text=True).stdout
    for l in out.splitlines():
        f = l.split()
        if len(f) >= 3 and f[1] == ".text":
            return int(f[2], 16)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=1.5,
                    help="decoy region size as a multiple of each seed's .text (default 1.5× ⇒ decoy-dominated band)")
    ap.add_argument("--region-bytes", type=int, default=0,
                    help="fixed decoy region size in bytes; overrides --ratio when >0")
    ap.add_argument("--src", default="/tmp/cid", help="stock cid corpus (for the seed list)")
    ap.add_argument("--out", default="/tmp/cid_heavy")
    a = ap.parse_args()

    seeds = sorted(glob.glob(os.path.join(a.src, "*__native-code-in-data.manifest.json")))
    if not seeds:
        sys.exit(f"no cid manifests in {a.src}")
    os.makedirs(a.out, exist_ok=True)

    for man_path in seeds:
        man = json.load(open(man_path))
        seed = man["seed"]["path"]
        seed_abs = seed if os.path.isabs(seed) else os.path.join(ROOT, seed)
        seed_gt = os.path.join(ROOT, "corpus", "gt", man["seed"]["name"] + ".gt")
        if not os.path.exists(seed_gt):
            print(f"  SKIP {man['seed']['name']}: no seed gt at {seed_gt}"); continue
        region = a.region_bytes if a.region_bytes > 0 else max(8000, int(a.ratio * text_size(seed_abs)))
        r = subprocess.run(
            [GAUNTLET, "generate", "--seed-elf", seed_abs, "--seed-gt", seed_gt,
             "--out", a.out, "--gen", "native-code-in-data", "--region-bytes", str(region)],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAIL {man['seed']['name']}: {r.stderr.strip()}"); continue
        stem = os.path.join(a.out, man["seed"]["name"] + "__native-code-in-data")
        addrs = func_gt_from_seed(seed_abs)
        open(stem + ".func.gt", "w").write("\n".join(f"0x{x:x}" for x in addrs) + "\n")
        # region sizes for the log
        regs = {p[3]: (int(p[0], 16), int(p[1], 16))
                for p in (l.split() for l in open(stem + ".regions") if not l.startswith("#"))}
        real = regs["real_code"][1] - regs["real_code"][0]
        decoy = regs["junk_decoy"][1] - regs["junk_decoy"][0]
        print(f"  {man['seed']['name']}: real .text {real}B, decoy {decoy}B ({decoy/real:.2f}×), "
              f"{len(addrs)} FUNC heads")
    print(f"decoy-heavy corpus written to {a.out}")


if __name__ == "__main__":
    main()
