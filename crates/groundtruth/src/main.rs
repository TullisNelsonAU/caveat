//! Min/max disassembly ground-truth generator — Rust implementation of
//! GROUNDTRUTH_FORMALISM.md (with the reachability-closure floor).
//!
//!   insn_min.txt  G_min : programmer-reachable real instructions.
//!                 Seed = DWARF .debug_line rows ∪ DW_TAG_subprogram entries, then
//!                 close under: fall-through (stop after ret / unconditional jmp) and
//!                 DIRECT branch/call targets landing in .text. CRT/linker stays OUT.
//!   insn_max.txt  G_max : the emitted instruction stream (capstone linear sweep).
//!   fn_min.txt    function-start G_min : DW_TAG_subprogram entries (programmer funcs).
//!   fn_max.txt    function-start G_max : fn_min ∪ STT_FUNC ∪ PLT stub entries.
//!
//! Usage:
//!   gen-gt <binary.dbg.elf> <out_dir>          # one binary -> 4 truth files
//!   gen-gt --batch <corpus_root> <out_root>    # every binary.dbg.elf under corpus_root
//!   gen-gt --audit <binary.dbg.elf>            # annotated MIN/NEU listing to confirm by eye

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use capstone::prelude::*;
use capstone::InsnGroupType;
use gimli::{Dwarf, EndianSlice, RunTimeEndian, SectionId};
use goblin::elf::Elf;

const SHF_EXECINSTR: u64 = 0x4;
const STT_FUNC: u8 = 2;
const PLT_STRIDE: u64 = 16;

struct InsnInfo {
    size: u64,
    is_term: bool,
    target: Option<u64>,
}

/// Everything computed for one binary — the sets plus a sorted disasm listing so the
/// audit view can show what each address actually decodes to.
struct Computed {
    insn_min: BTreeSet<u64>,
    insn_max: BTreeSet<u64>,
    fn_min: BTreeSet<u64>,
    fn_max: BTreeSet<u64>,
    listing: Vec<(u64, String, String)>, // (addr, mnemonic, op_str), address order
    leaks: usize,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.as_slice() {
        [_, flag, root, out] if flag == "--batch" => batch(Path::new(root), Path::new(out)),
        [_, flag, bin] if flag == "--audit" => audit(Path::new(bin)),
        [_, bin, out] => {
            let c = compute(Path::new(bin))?;
            write_truth(&c, Path::new(out))?;
            print_one(bin, &c);
            Ok(())
        }
        _ => bail!("usage: gen-gt <dbg.elf> <out_dir> | --batch <corpus_root> <out_root> | --audit <dbg.elf>"),
    }
}

fn print_one(name: &str, c: &Computed) {
    let neutral = c.insn_max.len().saturating_sub(c.insn_min.len());
    println!("{name}");
    println!("  insn:  min={:6}  max={:6}  neutral~{:6} ({:.0}% of max)",
        c.insn_min.len(), c.insn_max.len(), neutral,
        100.0 * neutral as f64 / c.insn_max.len().max(1) as f64);
    println!("  fn:    min={:6}  max={:6}", c.fn_min.len(), c.fn_max.len());
    println!("  containment G_min subset of G_max: {}", if c.leaks == 0 { "OK" } else { "VIOLATED" });
}

/// Annotated listing for manual confirmation: every emitted instruction, tagged
/// MIN (programmer-reachable) or NEU (neutral: CRT/PLT/padding), function entries marked.
fn audit(bin: &Path) -> Result<()> {
    let c = compute(bin)?;
    println!("=== {} ===", bin.display());
    println!("MIN = programmer-reachable (G_min)   NEU = neutral (G_max\\G_min: CRT/PLT/padding)");
    println!("<FUNC> = programmer function entry   <crt/plt> = linker/PLT entry");
    println!("{:>10}  {:<3}  {:<9}  {}", "addr", "tag", "mnemonic", "operands");
    println!("{}", "-".repeat(64));
    for (addr, mnem, op) in &c.listing {
        let tag = if c.insn_min.contains(addr) { "MIN" } else { "NEU" };
        let fmark = if c.fn_min.contains(addr) {
            "  <FUNC>"
        } else if c.fn_max.contains(addr) {
            "  <crt/plt>"
        } else {
            ""
        };
        println!("{addr:>10x}  {tag}  {mnem:<9}  {op}{fmark}");
    }
    let neutral = c.insn_max.len().saturating_sub(c.insn_min.len());
    println!("{}", "-".repeat(64));
    println!("min={} max={} neutral={} ({:.0}%)  containment: {}",
        c.insn_min.len(), c.insn_max.len(), neutral,
        100.0 * neutral as f64 / c.insn_max.len().max(1) as f64,
        if c.leaks == 0 { "OK" } else { "VIOLATED" });
    Ok(())
}

fn batch(corpus_root: &Path, out_root: &Path) -> Result<()> {
    let mut bins: Vec<PathBuf> = Vec::new();
    collect_dbg_elfs(corpus_root, &mut bins)?;
    bins.sort();
    let n = bins.len();
    if n == 0 {
        bail!("no binary.dbg.elf found under {}", corpus_root.display());
    }
    eprintln!("batch: {n} debug ELFs under {}", corpus_root.display());

    let (mut ok, mut failed, mut violated) = (0usize, 0usize, 0usize);
    let (mut sum_min, mut sum_max) = (0usize, 0usize);
    for (k, b) in bins.iter().enumerate() {
        let rel = b.parent().and_then(|p| p.strip_prefix(corpus_root).ok())
            .map(|p| p.to_path_buf()).unwrap_or_default();
        let od = out_root.join(rel);
        match compute(b) {
            // Containment is a write gate: a binary only earns a truth file when G_min subset G_max
            // actually holds. Leaks (so far: mislabeled ARM/Thumb) get reported and skipped so the
            // corpus stays trustworthy instead of silently shipping truth I can't stand behind.
            Ok(c) if c.leaks == 0 => {
                write_truth(&c, &od)?;
                ok += 1; sum_min += c.insn_min.len(); sum_max += c.insn_max.len();
                eprintln!("[{:>4}/{}] {} insn_min={} insn_max={}",
                    k + 1, n, b.display(), c.insn_min.len(), c.insn_max.len());
            }
            Ok(c) => {
                violated += 1;
                eprintln!("[{:>4}/{}] {} insn_min={} insn_max={}  !! CONTAINMENT VIOLATED -- skipped, not written",
                    k + 1, n, b.display(), c.insn_min.len(), c.insn_max.len());
            }
            Err(e) => { failed += 1; eprintln!("[{:>4}/{}] {} FAIL: {e:#}", k + 1, n, b.display()); }
        }
    }
    eprintln!("\nbatch done: {ok} ok, {failed} failed, {violated} containment violations");
    eprintln!("corpus totals: insn_min={sum_min} insn_max={sum_max} (neutral {:.0}% of max)",
        100.0 * (sum_max.saturating_sub(sum_min)) as f64 / sum_max.max(1) as f64);
    Ok(())
}

fn collect_dbg_elfs(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let p = entry?.path();
        if p.is_dir() {
            collect_dbg_elfs(&p, out)?;
        } else if p.file_name().map(|n| n == "binary.dbg.elf").unwrap_or(false) {
            out.push(p);
        }
    }
    Ok(())
}

/// Compute all sets + the disasm listing for one binary. No I/O side effects.
fn compute(bin_path: &Path) -> Result<Computed> {
    let data = std::fs::read(bin_path)?;
    let elf = Elf::parse(&data)?;
    let endian = if elf.little_endian { RunTimeEndian::Little } else { RunTimeEndian::Big };

    let mut ranges: Vec<(u64, u64)> = Vec::new();
    let mut exec_secs: Vec<(u64, &[u8])> = Vec::new();
    let mut plt_secs: Vec<(u64, u64)> = Vec::new();
    let mut text_range: Option<(u64, u64)> = None;
    for sh in &elf.section_headers {
        if sh.sh_flags & SHF_EXECINSTR == 0 {
            continue;
        }
        ranges.push((sh.sh_addr, sh.sh_addr + sh.sh_size));
        if let Some(r) = sh.file_range() {
            if r.end <= data.len() {
                exec_secs.push((sh.sh_addr, &data[r]));
            }
        }
        match elf.shdr_strtab.get_at(sh.sh_name) {
            Some(".text") => text_range = Some((sh.sh_addr, sh.sh_addr + sh.sh_size)),
            Some(".plt") | Some(".plt.sec") | Some(".plt.got") => plt_secs.push((sh.sh_addr, sh.sh_size)),
            _ => {}
        }
    }
    let in_exec = |a: u64| ranges.iter().any(|&(lo, hi)| lo <= a && a < hi);
    let in_text = |a: u64| text_range.map_or_else(|| in_exec(a), |(lo, hi)| lo <= a && a < hi);

    let mut sym_funcs: BTreeSet<u64> = BTreeSet::new();
    for sym in elf.syms.iter() {
        if sym.st_type() == STT_FUNC && sym.st_value != 0 && in_exec(sym.st_value) {
            sym_funcs.insert(sym.st_value);
        }
    }

    let mut e_line: BTreeSet<u64> = BTreeSet::new();
    let mut dwarf_funcs: BTreeSet<u64> = BTreeSet::new();
    let load = |id: SectionId| -> std::result::Result<EndianSlice<RunTimeEndian>, gimli::Error> {
        let name = id.name();
        for sh in &elf.section_headers {
            if elf.shdr_strtab.get_at(sh.sh_name) == Some(name) {
                if let Some(r) = sh.file_range() {
                    if r.end <= data.len() {
                        return Ok(EndianSlice::new(&data[r], endian));
                    }
                }
            }
        }
        Ok(EndianSlice::new(&[][..], endian))
    };
    let dwarf = Dwarf::load(load)?;
    let mut units = dwarf.units();
    while let Some(header) = units.next()? {
        let unit = dwarf.unit(header)?;
        if let Some(program) = unit.line_program.clone() {
            let mut rows = program.rows();
            while let Some((_, row)) = rows.next_row()? {
                if !row.end_sequence() && in_exec(row.address()) {
                    e_line.insert(row.address());
                }
            }
        }
        let mut entries = unit.entries();
        while let Some((_, entry)) = entries.next_dfs()? {
            if entry.tag() == gimli::DW_TAG_subprogram {
                if let Ok(Some(gimli::AttributeValue::Addr(a))) = entry.attr_value(gimli::DW_AT_low_pc) {
                    if in_exec(a) {
                        dwarf_funcs.insert(a);
                    }
                }
            }
        }
    }

    // ARM/AArch64 mapping symbols split code from data inside executable sections:
    // $x (A64), $a (A32), $t (Thumb) start code; $d starts data (literal pools, jump
    // tables). x86 emits none, so this list stays empty and nothing changes there.
    // Carving the $d spans out of the sweep keeps embedded data out of G_max, so a tool
    // that decodes a literal as an instruction is correctly charged a false positive.
    let mut mapsyms: Vec<(u64, bool)> = Vec::new(); // (addr, is_data)
    for sym in elf.syms.iter() {
        let name = match elf.strtab.get_at(sym.st_name) { Some(n) => n, None => continue };
        let is_data = match name {
            "$d" => true,
            "$x" | "$a" | "$t" => false,
            _ if name.starts_with("$d.") => true,
            _ if name.starts_with("$x.") || name.starts_with("$a.") || name.starts_with("$t.") => false,
            _ => continue,
        };
        if in_exec(sym.st_value) {
            mapsyms.push((sym.st_value, is_data));
        }
    }
    mapsyms.sort_by_key(|&(a, _)| a);
    // If addr sits inside a $d data span, return that span's end (the next mapping symbol,
    // or u64::MAX meaning "to section end"). None => it's code, so go ahead and decode it.
    let data_region_end = |addr: u64| -> Option<u64> {
        let i = mapsyms.partition_point(|&(a, _)| a <= addr); // first entry past addr
        if i == 0 { return None; }
        if mapsyms[i - 1].1 {
            Some(mapsyms.get(i).map(|&(a, _)| a).unwrap_or(u64::MAX))
        } else {
            None
        }
    };

    // Linear sweep → decode map + G_max + the disasm listing.
    // capstone's disasm_all STOPS at the first undecodable byte; AArch64 inline literal
    // pools (older gcc) then truncate G_max. So we skip-and-resume: decode a run, jump to
    // its end, and on a stall step past the bad bytes (4 on fixed-width ISAs, 1 on x86).
    let cs = build_capstone(elf.header.e_machine)?;
    let step: usize = if elf.header.e_machine == goblin::elf::header::EM_X86_64 { 1 } else { 4 };
    let mut map: HashMap<u64, InsnInfo> = HashMap::new();
    let mut listing: Vec<(u64, String, String)> = Vec::new();
    for (vaddr, bytes) in &exec_secs {
        let mut off: usize = 0;
        while off < bytes.len() {
            let addr = vaddr + off as u64;
            // Jump over $d data spans so literal pools / jump tables never reach G_max.
            if let Some(region_end) = data_region_end(addr) {
                let jump = region_end.saturating_sub(*vaddr) as usize;
                off = jump.min(bytes.len()).max(off + step);
                continue;
            }
            match cs.disasm_all(&bytes[off..], addr) {
                Ok(insns) if !insns.is_empty() => {
                    let mut new_off = off;
                    for i in insns.iter() {
                        let a = i.address();
                        if in_exec(a) {
                            let (is_term, target) = classify(&cs, i)?;
                            map.insert(a, InsnInfo { size: i.bytes().len() as u64, is_term, target });
                            listing.push((a, i.mnemonic().unwrap_or("").to_string(), i.op_str().unwrap_or("").to_string()));
                        }
                        new_off = (a - vaddr) as usize + i.bytes().len();
                    }
                    off = new_off.max(off + step); // always make progress
                }
                _ => off += step, // undecodable here (e.g. a literal-pool word) — skip it
            }
        }
    }
    listing.sort_by_key(|(a, _, _)| *a);
    let insn_max: BTreeSet<u64> = map.keys().copied().collect();

    // G_min = reachability closure from programmer anchors
    let mut insn_min: BTreeSet<u64> = BTreeSet::new();
    let mut work: Vec<u64> = e_line.iter().chain(dwarf_funcs.iter()).copied().collect();
    while let Some(a) = work.pop() {
        if !insn_min.insert(a) {
            continue;
        }
        if let Some(info) = map.get(&a) {
            if !info.is_term {
                let ft = a + info.size;
                if map.contains_key(&ft) {
                    work.push(ft);
                }
            }
            if let Some(t) = info.target {
                if in_text(t) && map.contains_key(&t) {
                    work.push(t);
                }
            }
        }
    }

    let fn_min: BTreeSet<u64> = dwarf_funcs.clone();
    let mut fn_max: BTreeSet<u64> = fn_min.iter().chain(sym_funcs.iter()).copied().collect();
    for &(lo, size) in &plt_secs {
        let mut off = 0;
        while off < size {
            fn_max.insert(lo + off);
            off += PLT_STRIDE;
        }
    }

    let leaks = insn_min.difference(&insn_max).count();
    Ok(Computed { insn_min, insn_max, fn_min, fn_max, listing, leaks })
}

fn write_truth(c: &Computed, out_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    write_set(&out_dir.join("insn_min.txt"), &c.insn_min)?;
    write_set(&out_dir.join("insn_max.txt"), &c.insn_max)?;
    write_set(&out_dir.join("fn_min.txt"), &c.fn_min)?;
    write_set(&out_dir.join("fn_max.txt"), &c.fn_max)?;
    Ok(())
}

fn classify(cs: &Capstone, insn: &capstone::Insn) -> Result<(bool, Option<u64>)> {
    let det = cs.insn_detail(insn).map_err(|e| anyhow::anyhow!("insn_detail: {e}"))?;
    let has = |g: u8| det.groups().iter().any(|x| x.0 == g);
    let is_ret = has(InsnGroupType::CS_GRP_RET as u8);
    let is_jump = has(InsnGroupType::CS_GRP_JUMP as u8);
    let is_call = has(InsnGroupType::CS_GRP_CALL as u8);
    let mnem = insn.mnemonic().unwrap_or("");
    let uncond_jmp = is_jump && matches!(mnem, "jmp" | "b" | "br");
    let is_term = is_ret || uncond_jmp;
    let target = if is_jump || is_call { parse_target(insn.op_str().unwrap_or("")) } else { None };
    Ok((is_term, target))
}

/// Pull the direct branch/call target out of an operand string, handling both
/// syntaxes: x86 prints "0x401020"; AArch64 prints "#0x400420" (and "x0, #0x4004c4"
/// for cbz/tbz). A memory operand ("[rip + 0x...]") is indirect — no static target —
/// so we bail on any '[' to avoid mistaking a displacement for a destination.
fn parse_target(op_str: &str) -> Option<u64> {
    if op_str.contains('[') {
        return None;
    }
    for tok in op_str.split(|c: char| c == ',' || c.is_whitespace()) {
        let t = tok.trim_start_matches('#');
        if let Some(h) = t.strip_prefix("0x") {
            if !h.is_empty() && h.chars().all(|c| c.is_ascii_hexdigit()) {
                return u64::from_str_radix(h, 16).ok();
            }
        }
    }
    None
}

fn build_capstone(machine: u16) -> Result<Capstone> {
    let cs = match machine {
        goblin::elf::header::EM_X86_64 => Capstone::new()
            .x86().mode(arch::x86::ArchMode::Mode64).detail(true).build(),
        goblin::elf::header::EM_AARCH64 => Capstone::new()
            .arm64().mode(arch::arm64::ArchMode::Arm).detail(true).build(),
        goblin::elf::header::EM_ARM => Capstone::new()
            .arm().mode(arch::arm::ArchMode::Arm).detail(true).build(),
        m => bail!("unsupported e_machine {m} (only x86-64, AArch64, ARM)"),
    };
    cs.map_err(|e| anyhow::anyhow!("capstone init: {e}"))
}

fn write_set(path: &Path, s: &BTreeSet<u64>) -> Result<()> {
    let mut out = String::with_capacity(s.len() * 7);
    for a in s {
        out.push_str(&format!("{a:x}\n"));
    }
    std::fs::write(path, out)?;
    Ok(())
}
