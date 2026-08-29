#!/usr/bin/env python3
"""Function-entry ground truth from the benign originals' ELF FUNC symbols (LAYER2_M2_SPEC §7 GT rule).

For each code-in-data specimen we read its manifest to find the benign seed binary, pull the `.text`
FUNC symbols out of its `.symtab` with objdump (NOT a disassembler — real symbol-table entries), and
write `<stem>.func.gt` (one 0x-hex function-entry address per line). These are the `Z_h` labels for
calibrating `prior_h` / `F_h`. Decoy-region heads get no symbol here (the benign symtab only covers
the real .text), so they are correctly labelled negative by construction.
"""
import glob, json, os, subprocess, sys

CID = "/tmp/cid"


def func_entries(elf):
    """`.text` FUNC-symbol addresses from objdump -t (` … F .text\\t<size> <name>`)."""
    out = subprocess.run(["objdump", "-t", elf], capture_output=True, text=True).stdout
    addrs = set()
    for ln in out.splitlines():
        if " F .text\t" in ln:
            addrs.add(int(ln.split()[0], 16))
    return sorted(addrs)


def main():
    specs = sorted(glob.glob(f"{CID}/*__native-code-in-data.elf"))
    if not specs:
        sys.exit("no specimens in " + CID)
    for elf in specs:
        stem = elf[:-4]
        seed = json.load(open(stem + ".manifest.json"))["seed"]["path"]
        if not os.path.exists(seed):
            print(f"  SKIP {os.path.basename(stem)}: seed missing {seed}")
            continue
        addrs = func_entries(seed)
        with open(stem + ".func.gt", "w") as f:
            f.write("".join(f"0x{a:016x}\n" for a in addrs))
        print(f"  {os.path.basename(stem):>48}: {len(addrs)} FUNC entries -> {stem}.func.gt")


if __name__ == "__main__":
    main()
