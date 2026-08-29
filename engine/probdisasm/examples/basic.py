import probdisasm

with open("tests/bins/gcc_coreutils_64_O0_make-prime-list.stripped", "rb") as f:
    for addr, insn, p in probdisasm.disassemble(f.read()):
        if p >= 0.01:
            print(f"0x{addr:010x}  {insn:<40}  {p:.6f}")
