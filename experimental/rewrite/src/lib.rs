//! `rewrite` — the confidence-gated binary rewriter (Instance 2, `REWRITING_APP_SPEC.md`).
//!
//! Paper 2 recovers a *calibrated* CFG. This crate is the "so what": a rewriter that only touches
//! code it is confident about. It applies one concrete, behaviourally-checkable transform —
//! **basic-block instrumentation** (a call counter at confirmed block leaders) — and it applies it
//! *only where the stack's calibrated belief `bel_a ≥ τ`*, abstaining below. A deterministic-CFG
//! rewriter (the baseline) commits at every leader its linear sweep reports; on stripped / desynced /
//! packed code those leaders include mid-instruction addresses, and patching one corrupts the binary.
//! Calibrated abstention is what buys a *working* rewrite there.
//!
//! The honesty wall runs one way: this crate *consumes* `bel`; it never feeds a decision back into the
//! calibration. Ground truth of a rewrite's success is **behaviour** (the patched binary still passes
//! its reference I/O), never a disassembler's opinion — that check lives in the eval harness, not here.
//!
//! Mechanism. Both `ours` and `baseline` run the *same* patcher over the *same* candidate leaders (one
//! linear sweep). The only difference is the site filter: `ours` keeps leaders with `bel ≥ τ`,
//! `baseline` keeps them all. So any working-rate gap is attributable to the calibrated gate alone.
//!
//! Patch shape (a classic detour, kept deliberately minimal so a *correct* leader is always safe and a
//! *wrong* leader always breaks):
//!   * At a leader `S`, steal a whole number of instructions totalling `≥5` bytes, all
//!     position-independent (no RIP-relative operand, no relative branch/call/ret) and not crossing the
//!     next leader. Overwrite `[S, S+5)` with `jmp rel32 → trampoline`, NOP-pad `[S+5, S+L)`.
//!   * The trampoline (in an injected RWX segment) does `pushfq; inc qword[counter]; popfq`, replays the
//!     stolen bytes verbatim (valid *because* they are position-independent), then `jmp S+L` back.
//! A true block leader is a real boundary and no code jumps into `[S+1, S+L)`, so this is transparent.
//! A wrong leader sits mid-instruction: the real instruction overlapping `S` is clobbered and the
//! return lands mid-stream — behaviour breaks. That asymmetry is the whole experiment.

use anyhow::{anyhow, bail, Context, Result};
use capstone::prelude::*;
use capstone::InsnGroupType;
use std::collections::{BTreeSet, HashMap};

// ── Minimal ELF64 view (hand-parsed: we mutate raw phdr bytes, so we own the layout) ──────────────

const PT_LOAD: u32 = 1;
const PT_NOTE: u32 = 4;

#[derive(Clone, Copy, Debug)]
pub struct Phdr {
    pub off: usize, // byte offset of this phdr entry in the file
    pub p_type: u32,
    pub p_flags: u32,
    pub p_offset: u64,
    pub p_vaddr: u64,
    pub p_filesz: u64,
    pub p_memsz: u64,
}

pub struct Elf {
    pub bytes: Vec<u8>,
    pub e_type: u16,
    pub phdrs: Vec<Phdr>,
    /// (vaddr, file_off, size) of `.text` if section headers are present, else the R+X PT_LOAD.
    pub text: (u64, usize, usize),
}

fn u16le(d: &[u8], o: usize) -> u16 {
    u16::from_le_bytes(d[o..o + 2].try_into().unwrap())
}
fn u32le(d: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(d[o..o + 4].try_into().unwrap())
}
fn u64le(d: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(d[o..o + 8].try_into().unwrap())
}

impl Elf {
    pub fn parse(bytes: Vec<u8>) -> Result<Self> {
        if bytes.len() < 64 || &bytes[..4] != b"\x7fELF" || bytes[4] != 2 {
            bail!("not a 64-bit LE ELF");
        }
        let e_type = u16le(&bytes, 0x10);
        let e_phoff = u64le(&bytes, 0x20) as usize;
        let e_phnum = u16le(&bytes, 0x38) as usize;
        let mut phdrs = Vec::with_capacity(e_phnum);
        for i in 0..e_phnum {
            let o = e_phoff + i * 56;
            phdrs.push(Phdr {
                off: o,
                p_type: u32le(&bytes, o),
                p_flags: u32le(&bytes, o + 4),
                p_offset: u64le(&bytes, o + 8),
                p_vaddr: u64le(&bytes, o + 0x10),
                p_filesz: u64le(&bytes, o + 0x20),
                p_memsz: u64le(&bytes, o + 0x28),
            });
        }
        let text = find_text(&bytes, &phdrs).context("locating .text / R+X segment")?;
        Ok(Elf { bytes, e_type, phdrs, text })
    }

    /// File offset backing virtual address `v`, if it falls in a file-backed PT_LOAD.
    pub fn v2off(&self, v: u64) -> Option<usize> {
        self.phdrs
            .iter()
            .filter(|p| p.p_type == PT_LOAD && p.p_filesz > 0)
            .find(|p| v >= p.p_vaddr && v < p.p_vaddr + p.p_filesz)
            .map(|p| (v - p.p_vaddr + p.p_offset) as usize)
    }

    pub fn max_vaddr(&self) -> u64 {
        self.phdrs.iter().filter(|p| p.p_type == PT_LOAD).map(|p| p.p_vaddr + p.p_memsz).max().unwrap_or(0)
    }
}

/// `.text` from the section headers; fall back to the first R+X PT_LOAD (past the ELF header) when the
/// sections are gone. Returns `(vaddr, file_off, size)`.
fn find_text(d: &[u8], phdrs: &[Phdr]) -> Result<(u64, usize, usize)> {
    let e_shoff = u64le(d, 0x28) as usize;
    let e_shentsize = u16le(d, 0x3a) as usize;
    let e_shnum = u16le(d, 0x3c) as usize;
    let e_shstrndx = u16le(d, 0x3e) as usize;
    if e_shoff != 0 && e_shnum != 0 && e_shstrndx < e_shnum {
        let stroff = u64le(d, e_shoff + e_shstrndx * e_shentsize + 0x18) as usize;
        for i in 0..e_shnum {
            let sh = e_shoff + i * e_shentsize;
            let name_off = stroff + u32le(d, sh) as usize;
            let name = d[name_off..].iter().take_while(|&&c| c != 0).copied().collect::<Vec<_>>();
            if name == b".text" {
                let addr = u64le(d, sh + 0x10);
                let off = u64le(d, sh + 0x18) as usize;
                let size = u64le(d, sh + 0x20) as usize;
                return Ok((addr, off, size));
            }
        }
    }
    // Section-stripped: the R+X PT_LOAD. Skip a leading page if it maps the ELF header at file off 0.
    let seg = phdrs
        .iter()
        .find(|p| p.p_type == PT_LOAD && p.p_flags & 1 != 0)
        .ok_or_else(|| anyhow!("no R+X PT_LOAD and no .text section"))?;
    let (mut v, mut o, mut sz) = (seg.p_vaddr, seg.p_offset as usize, seg.p_filesz as usize);
    if o == 0 {
        // the header page is not code; start one page in
        let page = 0x1000usize.min(sz);
        v += page as u64;
        o += page;
        sz -= page;
    }
    Ok((v, o, sz))
}

// ── Instruction decoding + reloc-safety ───────────────────────────────────────────────────────────

fn capstone() -> Result<Capstone> {
    Capstone::new()
        .x86()
        .mode(arch::x86::ArchMode::Mode64)
        .detail(true)
        .build()
        .map_err(|e| anyhow!("capstone init: {e}"))
}

/// Is a single decoded instruction safe to *relocate by verbatim copy* — i.e. position-independent?
/// Unsafe: any relative control transfer (its target is IP-relative), and any RIP-relative memory
/// operand (its effective address is IP-relative). Everything else executes identically at the
/// trampoline address, so a verbatim copy is faithful.
fn reloc_safe(cs: &Capstone, insn: &capstone::Insn) -> bool {
    let Ok(detail) = cs.insn_detail(insn) else { return false };
    for g in detail.groups() {
        match g.0 as u32 {
            InsnGroupType::CS_GRP_JUMP
            | InsnGroupType::CS_GRP_CALL
            | InsnGroupType::CS_GRP_RET
            | InsnGroupType::CS_GRP_INT
            | InsnGroupType::CS_GRP_IRET
            | InsnGroupType::CS_GRP_PRIVILEGE
            | InsnGroupType::CS_GRP_BRANCH_RELATIVE => return false,
            _ => {}
        }
    }
    if let arch::ArchDetail::X86Detail(x86) = detail.arch_detail() {
        let rip = RegId(arch::x86::X86Reg::X86_REG_RIP as RegIdInt);
        for op in x86.operands() {
            if let arch::x86::X86OperandType::Mem(m) = op.op_type {
                if m.base() == rip || m.index() == rip {
                    return false;
                }
            }
        }
    }
    true
}

// ── Leaders: one linear sweep, textbook block-leader set ──────────────────────────────────────────

/// Block leaders from a linear sweep of `.text`: the section start, every direct-branch target that
/// lands inside `.text`, and the instruction after every terminator (branch / call / ret). This is the
/// deterministic disassembler's notion of "where a basic block begins". `ours` and `baseline` share it;
/// they differ only in the confidence gate applied afterwards.
pub fn linear_leaders(cs: &Capstone, text: &[u8], text_lo: u64) -> Result<BTreeSet<u64>> {
    let text_hi = text_lo + text.len() as u64;
    let mut leaders = BTreeSet::new();
    leaders.insert(text_lo);
    // A real linear-sweep disassembler resynchronises after an undecodable byte by skipping one byte
    // and continuing — so it "commits everywhere", planting leaders throughout junk / compressed /
    // desynced spans. `disasm_all` stops at the first bad byte, so we drive it in a resync loop.
    let mut cursor = 0usize;
    while cursor < text.len() {
        let insns = cs
            .disasm_all(&text[cursor..], text_lo + cursor as u64)
            .map_err(|e| anyhow!("linear disasm: {e}"))?;
        let mut advanced = 0usize;
        for insn in insns.iter() {
            advanced = (insn.address() + insn.bytes().len() as u64 - (text_lo + cursor as u64)) as usize;
        let Ok(detail) = cs.insn_detail(&insn) else { continue };
        let mut is_term = false;
        let mut has_target = false; // direct branch OR call: its immediate operand is a block leader
        for g in detail.groups() {
            match g.0 as u32 {
                InsnGroupType::CS_GRP_JUMP | InsnGroupType::CS_GRP_BRANCH_RELATIVE | InsnGroupType::CS_GRP_CALL => {
                    is_term = true;
                    has_target = true;
                }
                InsnGroupType::CS_GRP_RET | InsnGroupType::CS_GRP_IRET => {
                    is_term = true;
                }
                _ => {}
            }
        }
        let next = insn.address() + insn.bytes().len() as u64;
        if is_term && next < text_hi {
            leaders.insert(next);
        }
        // direct branch/call target (single immediate operand) landing in .text — a call target is a
        // function entry, itself a block leader, and (crucially) it caps a preceding leader's steal so
        // no detour ever runs across a function boundary.
        if has_target {
            if let arch::ArchDetail::X86Detail(x86) = detail.arch_detail() {
                let ops = x86.operands().collect::<Vec<_>>();
                if ops.len() == 1 {
                    if let arch::x86::X86OperandType::Imm(t) = ops[0].op_type {
                        let t = t as u64;
                        if t >= text_lo && t < text_hi {
                            leaders.insert(t);
                        }
                    }
                }
            }
        } // end if has_target
        } // end for insn
        // Resync: continue past everything that decoded; if nothing did, skip one byte. Each fresh
        // start is a candidate leader (a naive sweep begins a block wherever it (re)synchronises).
        cursor += advanced.max(1);
        if cursor < text.len() {
            leaders.insert(text_lo + cursor as u64);
        }
    }
    Ok(leaders)
}

// ── Site construction ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Site {
    pub vaddr: u64,
    pub stolen_len: usize,
    pub stolen: Vec<u8>,
    pub bel: f64, // the calibrated belief that gated this site (1.0 for baseline / ungated)
}

/// Reason a leader could not be turned into a valid site (mechanical limits, shared by both arms —
/// these are *not* confidence decisions, so we log them to keep the coverage accounting honest).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reject {
    DecodeFail,     // bytes at the leader don't decode
    NoSafeWindow,   // couldn't reach 5 relocatable bytes before the next leader
    LowBelief,      // bel < τ  (the confidence gate — ours only)
    NotACandidate,  // no marginal for this address (ours only): treat as bel 0
}

/// Steal a whole number of position-independent instructions from `S`, totalling `≥5` bytes and not
/// crossing `next_leader`. `Some((len, bytes))` if instrumentable, else the mechanical reason.
fn steal(cs: &Capstone, elf: &Elf, s: u64, next_leader: u64) -> std::result::Result<(usize, Vec<u8>), Reject> {
    let off = elf.v2off(s).ok_or(Reject::DecodeFail)?;
    let cap = ((next_leader - s) as usize).min(16);
    if cap < 5 {
        return Err(Reject::NoSafeWindow);
    }
    let window = &elf.bytes[off..(off + cap).min(elf.bytes.len())];
    let insns = cs.disasm_all(window, s).map_err(|_| Reject::DecodeFail)?;
    let mut len = 0usize;
    for insn in insns.iter() {
        // Never instrument inter-function alignment padding: a leader that begins with a NOP is dead
        // filler after a `ret`, and its bytes run straight into the next (possibly address-taken, hence
        // un-leadered) function entry. Skipping it keeps every detour inside one function.
        if len == 0 && insn.mnemonic() == Some("nop") {
            return Err(Reject::NoSafeWindow);
        }
        if !reloc_safe(cs, &insn) {
            break;
        }
        len += insn.bytes().len();
        if len >= 5 {
            let bytes = elf.bytes[off..off + len].to_vec();
            return Ok((len, bytes));
        }
    }
    Err(Reject::NoSafeWindow)
}

/// Turn leaders into instrumentation sites. `marginals = Some(map)` gates by `bel ≥ τ` (ours);
/// `None` accepts every instrumentable leader (baseline, commits everywhere). Returns the sites plus a
/// per-reason rejection tally for the audit.
pub fn plan_sites(
    elf: &Elf,
    leaders: &BTreeSet<u64>,
    marginals: Option<&HashMap<u64, f64>>,
    tau: f64,
) -> Result<(Vec<Site>, HashMap<&'static str, usize>)> {
    let cs = capstone()?;
    let sorted: Vec<u64> = leaders.iter().copied().collect();
    let text_hi = elf.text.0 + elf.text.2 as u64;
    let mut sites = Vec::new();
    let mut rej: HashMap<&'static str, usize> = HashMap::new();
    for (i, &s) in sorted.iter().enumerate() {
        let next = sorted.get(i + 1).copied().unwrap_or(text_hi).min(text_hi);
        if next <= s {
            continue;
        }
        // Confidence gate (ours only). A leader with no marginal is treated as bel 0 → abstain.
        let bel = match marginals {
            Some(m) => match m.get(&s) {
                Some(&b) => b,
                None => {
                    *rej.entry("not_a_candidate").or_default() += 1;
                    continue;
                }
            },
            None => 1.0,
        };
        if bel < tau {
            *rej.entry("low_belief").or_default() += 1;
            continue;
        }
        match steal(&cs, elf, s, next) {
            Ok((len, bytes)) => sites.push(Site { vaddr: s, stolen_len: len, stolen: bytes, bel }),
            Err(Reject::DecodeFail) => *rej.entry("decode_fail").or_default() += 1,
            Err(Reject::NoSafeWindow) => *rej.entry("no_safe_window").or_default() += 1,
            Err(_) => {}
        }
    }
    Ok((sites, rej))
}

// ── Patched-ELF emission: PT_NOTE → PT_LOAD injection + detours ───────────────────────────────────

fn align_up(x: usize, a: usize) -> usize {
    (x + a - 1) & !(a - 1)
}

/// Build the instrumented ELF: inject an RWX segment (counter + one trampoline per site) by
/// repurposing a spare `PT_NOTE` program header into a `PT_LOAD`, then write each detour + NOP pad.
/// Returns `(patched_bytes, counter_vaddr)`.
pub fn build_patched(elf: &Elf, sites: &[Site]) -> Result<(Vec<u8>, u64)> {
    let mut out = elf.bytes.clone();

    // Injected segment placed page-aligned at EOF; vaddr chosen page-congruent to its file offset.
    let seg_off = align_up(out.len(), 0x1000);
    out.resize(seg_off, 0);
    let seg_vaddr = align_up(elf.max_vaddr() as usize, 0x1000) as u64 + 0x100000;
    debug_assert_eq!(seg_vaddr as usize % 0x1000, seg_off % 0x1000);

    let counter_vaddr = seg_vaddr; // 8-byte counter at the segment's front
    let mut seg = vec![0u8; 16]; // [0..8) counter, pad to 16

    // Pass 1: assign each trampoline a vaddr (layout only needs the stolen length).
    let mut tramp_vaddr = Vec::with_capacity(sites.len());
    let mut cursor = seg.len();
    for st in sites {
        tramp_vaddr.push(seg_vaddr + cursor as u64);
        cursor += trampoline_len(st.stolen_len);
    }

    // Pass 2: emit trampolines with concrete displacements, and the site detours.
    for (st, &tv) in sites.iter().zip(&tramp_vaddr) {
        seg.extend_from_slice(&trampoline(st, tv, counter_vaddr));

        // Detour at the site: jmp rel32 → trampoline, then NOP-fill the rest of the stolen span.
        let site_off = elf.v2off(st.vaddr).ok_or_else(|| anyhow!("site 0x{:x} not file-backed", st.vaddr))?;
        let rel = (tv as i64) - (st.vaddr as i64 + 5);
        let rel = i32::try_from(rel).context("detour out of rel32 range")?;
        out[site_off] = 0xe9;
        out[site_off + 1..site_off + 5].copy_from_slice(&rel.to_le_bytes());
        for b in &mut out[site_off + 5..site_off + st.stolen_len] {
            *b = 0x90;
        }
    }

    // Append the segment and repurpose a PT_NOTE header into an RWX PT_LOAD covering it.
    out.extend_from_slice(&seg);
    let note = elf
        .phdrs
        .iter()
        .find(|p| p.p_type == PT_NOTE)
        .ok_or_else(|| anyhow!("no spare PT_NOTE to convert into a load segment"))?;
    let o = note.off;
    out[o..o + 4].copy_from_slice(&PT_LOAD.to_le_bytes()); // p_type
    out[o + 4..o + 8].copy_from_slice(&7u32.to_le_bytes()); // p_flags = R|W|X
    out[o + 8..o + 16].copy_from_slice(&(seg_off as u64).to_le_bytes()); // p_offset
    out[o + 16..o + 24].copy_from_slice(&seg_vaddr.to_le_bytes()); // p_vaddr
    out[o + 24..o + 32].copy_from_slice(&seg_vaddr.to_le_bytes()); // p_paddr
    out[o + 32..o + 40].copy_from_slice(&(seg.len() as u64).to_le_bytes()); // p_filesz
    out[o + 40..o + 48].copy_from_slice(&(seg.len() as u64).to_le_bytes()); // p_memsz
    out[o + 48..o + 56].copy_from_slice(&0x1000u64.to_le_bytes()); // p_align

    Ok((out, counter_vaddr))
}

// Fixed prologue/epilogue around the counter bump. The flag save MUST clear the 128-byte red zone
// first: many leaf functions keep live locals *below* RSP (no `sub rsp`), so a naive `pushfq` at
// `[rsp-8]` would land on a local and corrupt it. `lea rsp,[rsp±0x80]` skips the red zone and leaves
// the flags untouched (LEA sets no flags), so the counter bump is invisible and the stolen bytes then
// execute at the *exact* original RSP — position-independent, hence a faithful verbatim replay.
const LEA_DOWN: [u8; 5] = [0x48, 0x8d, 0x64, 0x24, 0x80]; // lea rsp,[rsp-0x80]  (disp8 = -128)
const LEA_UP: [u8; 8] = [0x48, 0x8d, 0xa4, 0x24, 0x80, 0x00, 0x00, 0x00]; // lea rsp,[rsp+0x80]
const PROLOGUE: usize = LEA_DOWN.len() + 1 + 7 + 1 + LEA_UP.len(); // lea; pushfq; inc; popfq; lea

fn trampoline_len(stolen_len: usize) -> usize {
    PROLOGUE + stolen_len + 5 // + stolen + jmp rel32
}

/// `lea rsp,[rsp-0x80]; pushfq; inc qword[rip+counter]; popfq; lea rsp,[rsp+0x80]; <stolen>; jmp S+L`.
fn trampoline(st: &Site, tramp_vaddr: u64, counter_vaddr: u64) -> Vec<u8> {
    let mut t = Vec::with_capacity(trampoline_len(st.stolen_len));
    t.extend_from_slice(&LEA_DOWN);
    t.push(0x9c); // pushfq
    // inc qword ptr [rip+disp32] = 48 ff 05 <disp32>, rip taken after the 7-byte instruction
    let inc_end = tramp_vaddr + (LEA_DOWN.len() + 1 + 7) as u64;
    let disp = (counter_vaddr as i64) - (inc_end as i64);
    t.extend_from_slice(&[0x48, 0xff, 0x05]);
    t.extend_from_slice(&(disp as i32).to_le_bytes());
    t.push(0x9d); // popfq
    t.extend_from_slice(&LEA_UP);
    t.extend_from_slice(&st.stolen); // replay stolen bytes verbatim, at the original RSP
    // jmp rel32 back to S+L
    let jmp_vaddr = tramp_vaddr + (PROLOGUE + st.stolen_len) as u64;
    let back = (st.vaddr as i64 + st.stolen_len as i64) - (jmp_vaddr as i64 + 5);
    t.push(0xe9);
    t.extend_from_slice(&(back as i32).to_le_bytes());
    t
}

/// A Capstone tuned for linear leader discovery (x86-64, detail on). Shared by the `rewrite` CLI
/// and by `gated_rewrite` so the one-shot analysis→rewrite hook and the standalone tool agree
/// byte-for-byte on the leader set.
pub fn capstone_for_leaders() -> Result<Capstone> {
    Capstone::new()
        .x86()
        .mode(arch::x86::ArchMode::Mode64)
        .detail(true)
        .build()
        .map_err(|e| anyhow!("capstone: {e}"))
}

/// Summary of one gated rewrite -- mirrors the CLI's `rewrite_summary` fields.
pub struct RewriteStats {
    pub leaders: usize,
    pub sites: usize,
    pub coverage: f64,
    pub counter_vaddr: u64,
}

/// One-shot confidence-gated rewrite: parse ELF bytes, discover linear-sweep leaders, gate them by
/// `marginals` (instrument a leader iff `bel ≥ τ`; pass `None` for commit-all), and return the
/// patched PIE bytes. This is the library seam the stack's analysis tool (`udstack --rewrite`)
/// calls with the calibrated marginals it already computed -- analysis result → gated rewrite, no
/// `--dump-instr` file round-trip. The standalone `rewrite` CLI keeps its own flow; both share
/// `plan_sites`/`build_patched`, so the emitted binary is identical for the same (leaders, τ).
pub fn gated_rewrite(
    elf_bytes: Vec<u8>,
    marginals: Option<&HashMap<u64, f64>>,
    tau: f64,
) -> Result<(Vec<u8>, RewriteStats)> {
    let elf = Elf::parse(elf_bytes)?;
    let cs = capstone_for_leaders()?;
    let (tv, toff, tsz) = elf.text;
    let text = &elf.bytes[toff..toff + tsz];
    let leaders = linear_leaders(&cs, text, tv)?;
    let (sites, _rej) = plan_sites(&elf, &leaders, marginals, tau)?;
    let (patched, counter) = build_patched(&elf, &sites)?;
    let coverage = if leaders.is_empty() { 0.0 } else { sites.len() as f64 / leaders.len() as f64 };
    let stats = RewriteStats { leaders: leaders.len(), sites: sites.len(), coverage, counter_vaddr: counter };
    Ok((patched, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reloc_safety_flags_ip_relative_and_control_flow() {
        let cs = capstone().unwrap();
        // safe, position-independent: `add rax, rbx`, `mov rax, [rbp-8]`, `xor ecx, ecx`
        for bytes in [&[0x48, 0x01, 0xd8][..], &[0x48, 0x8b, 0x45, 0xf8][..], &[0x31, 0xc9][..]] {
            let i = cs.disasm_all(bytes, 0x1000).unwrap();
            assert!(reloc_safe(&cs, &i.iter().next().unwrap()), "should be safe: {bytes:x?}");
        }
        // unsafe: `call rel32`, `jmp rel8`, `ret`, `lea rax,[rip+0]` (RIP-relative)
        for bytes in [&[0xe8, 0, 0, 0, 0][..], &[0xeb, 0x10][..], &[0xc3][..], &[0x48, 0x8d, 0x05, 0, 0, 0, 0][..]] {
            let i = cs.disasm_all(bytes, 0x1000).unwrap();
            assert!(!reloc_safe(&cs, &i.iter().next().unwrap()), "should be unsafe: {bytes:x?}");
        }
    }

    #[test]
    fn trampoline_encodes_counter_and_return_relative() {
        // A trampoline for a 5-byte stolen run: its `inc [rip+d]` must resolve to the counter, and its
        // final `jmp` must target S + stolen_len — both RIP-relative so they survive ASLR.
        let st = Site { vaddr: 0x1200, stolen_len: 5, stolen: vec![0x90; 5], bel: 1.0 };
        let (tv, cv) = (0x105010u64, 0x105000u64);
        let t = trampoline(&st, tv, cv);
        assert_eq!(t.len(), trampoline_len(5));
        // inc disp32 lives right after `lea; pushfq; 48 ff 05`
        let off = LEA_DOWN.len() + 1 + 3;
        let disp = i32::from_le_bytes(t[off..off + 4].try_into().unwrap()) as i64;
        assert_eq!(tv as i64 + (LEA_DOWN.len() + 1 + 7) as i64 + disp, cv as i64, "inc targets counter");
        // trailing jmp rel32 targets S + stolen_len
        let jrel = i32::from_le_bytes(t[t.len() - 4..].try_into().unwrap()) as i64;
        let jmp_vaddr = tv + (PROLOGUE + st.stolen_len) as u64;
        assert_eq!(jmp_vaddr as i64 + 5 + jrel, (st.vaddr + st.stolen_len as u64) as i64, "jmp returns to S+L");
    }

    #[test]
    fn align_up_rounds_to_page() {
        assert_eq!(align_up(0x1234, 0x1000), 0x2000);
        assert_eq!(align_up(0x2000, 0x1000), 0x2000);
    }
}
