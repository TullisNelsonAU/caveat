#!/usr/bin/env python3
"""Carve format-exact ground truth for a UPX-packed ELF from its own b_info chain.

Port of `evalkit::parse_upx_layout`. The compressed payload of a UPX binary is a chain of blocks,
each prefixed by a 12-byte `b_info` header (sz_unc, sz_cpr, method+flags). The chain starts right
after the `UPX!` l_info magic + p_info. Everything from the first block's data to the last block's
end is provable *data* (the compressed original) — never real instruction starts. That window is the
NEGATIVE region we score the packed detector against. Provenance is UPX's own format, not a
disassembler, so it is anti-circular by construction.

Emits a `.upxgt` whose `compressed NEGATIVE <vstart> <vend> ...` row is what
`consistency`'s `packed_data_window` reads.

Usage: make_upxgt.py <packed.elf> <out.upxgt>
"""
import struct, sys

def carve(path):
    b = open(path, "rb").read()
    if b[:4] != b"\x7fELF":
        raise SystemExit(f"{path}: not an ELF")
    is64 = b[4] == 2
    if not is64:
        raise SystemExit("only ELF64 handled")
    e_phoff = struct.unpack_from("<Q", b, 0x20)[0]
    e_phentsize = struct.unpack_from("<H", b, 0x36)[0]
    e_phnum = struct.unpack_from("<H", b, 0x38)[0]
    e_entry = struct.unpack_from("<Q", b, 0x18)[0]
    PT_LOAD, PF_X = 1, 1
    seg = None
    for i in range(e_phnum):
        o = e_phoff + i * e_phentsize
        p_type = struct.unpack_from("<I", b, o)[0]
        p_flags = struct.unpack_from("<I", b, o + 4)[0]
        p_offset = struct.unpack_from("<Q", b, o + 8)[0]
        p_vaddr = struct.unpack_from("<Q", b, o + 16)[0]
        p_filesz = struct.unpack_from("<Q", b, o + 32)[0]
        if p_type == PT_LOAD and (p_flags & PF_X):
            seg = (p_offset, p_vaddr, p_filesz)
            break
    if seg is None:
        raise SystemExit("no executable PT_LOAD segment")
    fstart, vaddr, filesz = seg
    fend = min(fstart + filesz, len(b))
    off2v = lambda o: vaddr + (o - fstart)

    # UPX! l_info magic in the first 0x400 of the segment.
    scan_end = min(fstart + 0x400, fend)
    magic = b.find(b"UPX!", fstart, scan_end)
    if magic < 0:
        raise SystemExit(f"{path}: no UPX! magic — not a UPX image")
    p_info = magic + 8               # magic(4)+l_lsize(2)+l_version(1)+l_format(1)
    p_filesize = struct.unpack_from("<I", b, p_info + 4)[0]

    off = p_info + 12                # walk b_info chain
    blocks = []
    while off + 12 <= fend:
        sz_unc = struct.unpack_from("<I", b, off)[0]
        sz_cpr = struct.unpack_from("<I", b, off + 4)[0]
        data = off + 12
        data_end = data + sz_cpr
        sane = sz_unc != 0 and sz_cpr != 0 and data_end <= fend and sz_cpr <= sz_unc + 4096
        if not sane:
            break
        blocks.append((data, data_end, sz_unc, sz_cpr))
        off = data_end
        if len(blocks) > 256:
            break
    if not blocks:
        raise SystemExit(f"{path}: no valid b_info blocks")
    comp_lo = blocks[0][0]
    comp_hi = blocks[-1][1]
    return dict(fstart=fstart, fend=fend, vaddr=vaddr, entry=e_entry,
                comp_lo=comp_lo, comp_hi=comp_hi, off2v=off2v,
                p_filesize=p_filesize, nblocks=len(blocks))

def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    L = carve(src)
    v = L["off2v"]
    lines = [
        f"# Format-exact GT for {src.split('/')[-1]} — derived from UPX's own b_info chain (NOT a disassembler).",
        f"# {L['nblocks']} b_info blocks; p_filesize={L['p_filesize']}.",
        "# field       label     vaddr_start   vaddr_end     file_start  file_end",
        f"exec_segment  -         0x{L['vaddr']:x}      0x{v(L['fend']):x}      0x{L['fstart']:x}       0x{L['fend']:x}",
        f"compressed    NEGATIVE  0x{v(L['comp_lo']):x}      0x{v(L['comp_hi']):x}      0x{L['comp_lo']:x}       0x{L['comp_hi']:x}  provable data ({L['comp_hi']-L['comp_lo']} B)",
    ]
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"{src.split('/')[-1]}: compressed window vaddr 0x{v(L['comp_lo']):x}..0x{v(L['comp_hi']):x} "
          f"({L['comp_hi']-L['comp_lo']} B), {L['nblocks']} blocks -> {out}")

if __name__ == "__main__":
    main()
