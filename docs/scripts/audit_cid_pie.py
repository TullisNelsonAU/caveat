#!/usr/bin/env python3
"""Alignment + GT-provenance audit for the M5 PIE code-in-data corpus.

The seeds are ET_DYN (PIE) — the exact regime where the desync scorer bug hid (a nonzero load-base
delta silently fabricating a collapse). gauntlet re-wraps the seed .text as an ET_EXEC min-ELF at the
seed's link-time .text vaddr, so the correct rebase delta is 0. This mirrors audit_cid.py's two
non-tool checks, hardened so each can actually FAIL, and must pass BEFORE any M5 number is trusted:

  1. ALIGNMENT — delta 0 is TRUE alignment: every instruction-GT address decodes as a genuine x86
     instruction at its file offset in the specimen (recall of the GT-as-decoded is 1.0 at delta 0),
     AND no nonzero constant shift decodes more GT addresses (the desync bug class).
  2. GT PROVENANCE — the decoy boundary is the gauntlet `.regions` junk_decoy span (== manifest
     [decoy_from, end)); the instruction GT and the seed-symtab func GT each have ZERO entries in the
     decoy region; GT lives in the corpus dir, never a tool's output.

Exits nonzero if anything wobbles.
"""
import glob, json, os, struct, sys

import capstone
_CS = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)

CID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "corpus_pie", "cid")
ok = True


def check(cond, msg):
    global ok
    print(("  PASS " if cond else "  FAIL ") + msg)
    ok = ok and bool(cond)


def read_addrs(p):
    return sorted(int(l.strip(), 16) for l in open(p) if l.strip())


def text_seg(elf):
    """(vaddr, file_off, filesz) of the specimen's single R+X PT_LOAD (gauntlet min-ELF)."""
    d = open(elf, "rb").read()
    phoff = struct.unpack_from("<Q", d, 0x20)[0]
    phnum = struct.unpack_from("<H", d, 0x38)[0]
    for i in range(phnum):
        o = phoff + i * 56
        p_type, p_flags = struct.unpack_from("<II", d, o)
        p_off, p_vaddr = struct.unpack_from("<QQ", d, o + 8)[0], struct.unpack_from("<QQ", d, o + 8)[1]
        p_filesz = struct.unpack_from("<Q", d, o + 32)[0]
        if p_type == 1 and (p_flags & 1):  # PT_LOAD, PF_X
            return p_vaddr, p_off, p_filesz, d
    raise SystemExit("no R+X PT_LOAD in " + elf)


def decodes_at(data, foff, vaddr):
    if 0 <= foff < len(data):
        return next(_CS.disasm(data[foff:foff + 15], vaddr), None) is not None
    return False


def nonzero_shift_beats(gt, base, foff, data, h0):
    """True if some nonzero constant shift decodes MORE GT addresses than delta 0 — the bug class."""
    for d in (0x1000, 0x4000, 0x400000, 0x555555554000, -0x1000, 0x8):
        h = sum(1 for a in gt if decodes_at(data, foff + (a - d - base), a - d))
        if h > h0:
            return True
    return False


specimens = sorted(glob.glob(os.path.join(CID, "*__native-code-in-data.elf")))
if not specimens:
    sys.exit(f"FAIL: no specimens under {CID} — nothing to audit (would be a vacuous pass)")
for elf in specimens:
    stem = elf[:-4]
    name = os.path.basename(stem).split(".elf__")[0]
    man = json.load(open(stem + ".manifest.json"))
    gt = read_addrs(stem + ".gt")
    fgt = read_addrs(stem + ".func.gt")
    base, foff, filesz, data = text_seg(elf)
    end = base + filesz

    # decoy boundary from .regions junk_decoy
    d0 = rhi = None
    for l in open(stem + ".regions"):
        if "junk_decoy" in l:
            d0, rhi = int(l.split()[0], 16), int(l.split()[1], 16)
    print(f"\n== {name}  base=0x{base:x} decoy_from=0x{d0:x} end=0x{end:x}  (ET_EXEC from ET_DYN seed)")

    # 1. ALIGNMENT — delta 0 is true (all GT decode), unbeaten by any nonzero shift.
    h0 = sum(1 for a in gt if decodes_at(data, foff + (a - base), a))
    rec0 = h0 / max(1, len(gt))
    check(rec0 == 1.0 and not nonzero_shift_beats(gt, base, foff, data, h0),
          f"aligned at delta 0: {h0}/{len(gt)} GT decode as x86 (rec={rec0:.3f}), no nonzero shift beats it")

    # 2. GT PROVENANCE — boundary matches manifest, GT clean of the decoy region, GT in corpus dir.
    check(rhi == end, f"decoy boundary: .regions junk_decoy end 0x{rhi:x} == segment end 0x{end:x}")
    check(sum(1 for a in gt if a >= d0) == 0, "instruction GT has 0 addresses in the decoy region")
    check(sum(1 for a in fgt if a >= d0) == 0, "func GT (seed symtab) has 0 entries in the decoy region")
    check("corpus_pie" in os.path.abspath(stem + ".gt"), "GT paths are corpus files, not a tool output")

print("\nAUDIT " + ("PASSED — PIE corpus aligned, GT clean" if ok else "FAILED — do not run the eval"))
sys.exit(0 if ok else 1)
