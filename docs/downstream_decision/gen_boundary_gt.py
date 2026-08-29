#!/usr/bin/env python3
"""Function-boundary ground truth for the downstream recovery probe — symtab only, never a disassembler.

The probe grades a *recovered function-boundary set* against truth, so the truth has to be real
function heads, not something a tool guessed. Two sources are legitimate here and nothing else is:
`gen-gt`'s `fn_min`/`fn_max`, and the ELF `.symtab` `STT_FUNC` entries of the unstripped original.
We take the second, because `gen-gt` has never been run over these corpora (there is not a single
`.func.gt` anywhere under `$CORP`) and the unstripped siblings are sitting right there next to every
`stripped/` directory. Same rule `docs/scripts/gen_func_gt.py` already uses for the code-in-data
specimens: `objdump -t`, `F .text` rows, take the address column.

Why the unstripped sibling is the *same* binary. The desync corpora are built as build → keep one
unstripped copy → strip a second. `objdump -h` reports byte-identical `.text` vma and size across the
pair, and every FUNC address lands inside the instruction-start GT that was generated independently
for the stripped binary (140/140 on `d1_med/desync_coreutils_64_O0_base32`). So the symbol addresses
describe the binary the engine actually sees. `reproduce.sh` re-runs that check rather than trusting
this paragraph.

What is deliberately missing. The packed regime gets no file. UPX's `b_info` chain proves a window is
*compressed data* — it yields negatives and never positives, and the original function heads do not
exist at those addresses once the section is compressed. There is no honest way to write a packed
`.func.gt`, so we do not write one and the probe reports packed F1 as undefined rather than filling
the cell.

Usage: python3 gen_boundary_gt.py [OUT_DIR]      (default: ./func_gt)
Layout: OUT_DIR/<sublabel>/<name>.func.gt — the sublabel/name pair the Rust probe keys holdout jobs by.
"""
import os
import subprocess
import sys

CORP = os.path.expanduser("~/lab/projects/probablistic/corpus")

# (sublabel, unstripped dir, the stripped dir the engine actually reads).
# The stripped dir is the roster: `unstripped/` carries binaries that never made the `_small` subsets,
# and emitting GT for those would quietly inflate the file count without adding a gradeable binary.
ARMS = [
    ("clean",    f"{CORP}/x86_64-binaries/elf/coreutils",      f"{CORP}/x86_64-binaries/elf/coreutils"),
    ("pilot",    f"{CORP}/desync-pilot/unstripped",            f"{CORP}/desync-pilot/stripped"),
    ("d1_med",   f"{CORP}/desync-dense/d1_med/unstripped",     f"{CORP}/desync-dense/d1_med/stripped_small"),
    ("d2_heavy", f"{CORP}/desync-dense/d2_heavy/unstripped",   f"{CORP}/desync-dense/d2_heavy/stripped_small"),
    ("d3_max",   f"{CORP}/desync-dense/d3_max/unstripped",     f"{CORP}/desync-dense/d3_max/stripped_small"),
    # The legit-VM false-positive gate. These ship unstripped, so — like the clean corpus — the
    # binary is its own symbol source and the layout gate below compares it against itself.
    ("vmlegit",  f"{CORP}/vm-legit/bins",                      f"{CORP}/vm-legit/bins"),
    # Tigress semantic-obfuscation arm (the declared blind spot). build_tigress_graded.sh emits the
    # obfuscated binaries unstripped (-g), so — like clean/vmlegit — each binary is its own symbol
    # source and the layout gate compares it against itself. Dirs are absent until the corpus is
    # rebuilt; the isdir() gate below then skips them cleanly on a run where Tigress isn't present.
    ("tigL",     "/tmp/tig_graded/tigL/bins",                  "/tmp/tig_graded/tigL/bins"),
    ("tigM",     "/tmp/tig_graded/tigM/bins",                  "/tmp/tig_graded/tigM/bins"),
    ("tigH",     "/tmp/tig_graded/tigH/bins",                  "/tmp/tig_graded/tigH/bins"),
]


def text_range(elf):
    """`.text` (vma, size) from the section header, or None if there is no `.text`."""
    out = subprocess.run(["objdump", "-h", elf], capture_output=True, text=True).stdout
    for ln in out.splitlines():
        f = ln.split()
        if len(f) >= 4 and f[1] == ".text":
            return int(f[3], 16), int(f[2], 16)
    return None


def func_entries(elf):
    """`.text` FUNC-symbol addresses from `objdump -t` — symbol table rows, not a disassembly."""
    out = subprocess.run(["objdump", "-t", elf], capture_output=True, text=True).stdout
    return sorted({int(ln.split()[0], 16) for ln in out.splitlines() if " F .text\t" in ln})


def main():
    out_root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "func_gt")
    total = 0
    for sublabel, unstripped, stripped in ARMS:
        if not os.path.isdir(unstripped) or not os.path.isdir(stripped):
            print(f"  SKIP {sublabel}: missing {unstripped if not os.path.isdir(unstripped) else stripped}")
            continue
        dst = os.path.join(out_root, sublabel)
        os.makedirs(dst, exist_ok=True)
        n_bins, n_addrs, n_skip = 0, 0, 0
        for name in sorted(os.listdir(stripped)):
            src = os.path.join(unstripped, name)
            if not os.path.isfile(src):
                n_skip += 1
                continue
            # The layout check is a hard gate, not a warning. If the pair's `.text` disagrees the
            # symbol addresses describe a different binary than the one the engine reads, and the
            # resulting GT would be silently wrong rather than loudly absent.
            rs, ru = text_range(os.path.join(stripped, name)), text_range(src)
            if rs is None or ru is None or rs != ru:
                print(f"  SKIP {sublabel}/{name}: .text mismatch stripped={rs} unstripped={ru}")
                n_skip += 1
                continue
            lo, size = ru
            addrs = [a for a in func_entries(src) if lo <= a < lo + size]
            with open(os.path.join(dst, name + ".func.gt"), "w") as f:
                f.write("".join(f"0x{a:016x}\n" for a in addrs))
            n_bins += 1
            n_addrs += len(addrs)
        print(f"  {sublabel:>9}: {n_bins} binaries, {n_addrs} FUNC entries"
              f"{f', {n_skip} skipped' if n_skip else ''} -> {dst}")
        total += n_bins
    print(f"wrote function-boundary GT for {total} binaries under {out_root}")
    print("packed: intentionally absent — UPX b_info gives provable data (negatives), never code heads")


if __name__ == "__main__":
    main()
