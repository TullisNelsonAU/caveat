//! `cfgprobe` — a GT-free CFG-topology channel for control-flow obfuscation.
//!
//! The consistency detector (Papers 2's switching payoff) sees *decode-level* drift — junk, overlap,
//! packer structure — but is blind to *semantic* obfuscation that preserves clean decoding (Tigress
//! virtualization / flattening). This probe asks whether a second, orthogonal channel — the *shape*
//! of the CFG we already recover — routes that blind spot to abstention. It reads only topology; it
//! runs no new engine (Soft posteriors for the recovered instruction set, `probcfg::resolve_indirect`
//! for jump-table edges — both already in the stack).
//!
//! Per recovered function it computes four GT-free topology statistics:
//!   1. **dispatcher dominance** — the busiest block's (in+out)-degree as a fraction of all edges.
//!   2. **dominator-tree collapse** — the largest fraction of blocks dominated by a single non-entry
//!      block (a flattening dispatcher dominates every case; this is its signature).
//!   3. **indirect-dispatch concentration** — the (in+out)-degree share of the busiest block that
//!      terminates in an indirect jump (the switch/computed-goto a VM or flattener routes through).
//!   4. **real-structure absence** — 1 − the fraction of edges that are straight-line fall-throughs
//!      (compiled code is fall-through-heavy; a dispatcher replaces linear flow with jumps).
//!
//! The critical honesty gate: a `legit_interp` group of hand-written / real interpreters (a 40-opcode
//! bytecode VM, a state machine, a recursive-descent parser) — legitimate switch-heavy code that a
//! naive flattening detector fires on. We report the false-positive rate on it **explicitly**.
//!
//! ```text
//! cfgprobe --group obf_flatten DIR --group obf_virt DIR \
//!          --group benign_normal DIR --group legit_interp DIR \
//!          --out results.csv
//! ```
//! Group-label semantics: a label containing "flatten"/"virt" is an obfuscated positive; "legit" is
//! the FP-audit set; anything else is a benign negative. AUROC is flatten-vs-benign and virt-vs-benign
//! on each statistic and on the mean; the FP audit thresholds at the benign-normal max.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use evalkit::run_soft;
use goblin::Object;
use probcfg::{resolve_indirect, ResolveConfig};
use probdisasm::{extract_text_section as extract_text, Superset};

/// A recovered instruction is "code" when its Soft posterior clears this bar — the recovered-CFG
/// membership test. The engine's calibrated posterior, thresholded; no ground truth.
const PI_CODE: f64 = 0.5;
/// Only functions with at least this many blocks drive a binary's per-binary statistic — tiny stubs
/// (PLT thunks, `_start` fragments) have degenerate topology and would add noise, not signal.
const MIN_BLOCKS: usize = 8;

fn main() -> Result<()> {
    let args = Args::parse(std::env::args().skip(1))?;

    let mut rows: Vec<Row> = Vec::new();
    for (label, dir) in &args.groups {
        let mut bins: Vec<PathBuf> = fs::read_dir(dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file())
            .collect();
        bins.sort();
        for bin in bins {
            match analyze_binary(&bin) {
                Ok(m) => {
                    let name = bin.file_name().and_then(|s| s.to_str()).unwrap_or("?").to_string();
                    eprintln!(
                        "  {:<14} {:<22} funcs={:<3} mb={:<3} disp={:.3} domcol={:.3} indir={:.3} noStruct={:.3}",
                        label, name, m.n_funcs, m.max_blocks, m.dispatcher, m.dom_collapse, m.indirect, m.struct_absence
                    );
                    rows.push(Row { name, label: label.clone(), m });
                }
                Err(e) => eprintln!("  !! {} skipped: {e:#}", bin.display()),
            }
        }
    }

    write_csv(&args.out, &rows)?;
    report(&rows);
    Ok(())
}

// ── Per-binary topology statistics ─────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default)]
struct Metrics {
    n_funcs: usize,
    max_blocks: usize,
    dispatcher: f64,
    dom_collapse: f64,
    indirect: f64,
    struct_absence: f64,
}

impl Metrics {
    /// The combined flattening score — the plain mean of the four (each already in [0,1]).
    fn combined(&self) -> f64 {
        (self.dispatcher + self.dom_collapse + self.indirect + self.struct_absence) / 4.0
    }
}

struct Row {
    name: String,
    label: String,
    m: Metrics,
}

/// Recover the CFG of one ELF (GT-free) and reduce it to the four topology statistics — the
/// per-binary value of each is the max over qualifying functions (obfuscation hides in one function).
fn analyze_binary(path: &Path) -> Result<Metrics> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let (base, code) = extract_text(&bytes)?;
    let e_entry = elf_entry(&bytes)?;

    // The recovered instruction set: Soft posteriors thresholded at PI_CODE. This is "the CFG we
    // already recover" — no new engine.
    let sup = Superset::new(base, code).map_err(|e| anyhow::anyhow!("superset: {e}"))?;
    let post = run_soft(base, code, 0.0, false)?;
    let recovered: HashSet<u64> = post.iter().filter(|&&(_, p)| p >= PI_CODE).map(|&(a, _)| a).collect();
    if recovered.is_empty() {
        bail!("no recovered instructions");
    }

    // Jump-table / pointer targets (the switch cases a dispatcher routes to). resolve_indirect
    // attributes them to the entry root; we re-attach them to the indirect-jump block by address span.
    let resolved = resolve_indirect(&sup, &bytes, e_entry, &ResolveConfig::default());
    let jt_targets: Vec<u64> = resolved.iter().map(|e| e.t).collect();
    if std::env::var("CFGPROBE_JT").is_ok() {
        // Count indirect jumps in the recovered set — do the flattened binaries even HAVE jump-table
        // dispatchers, or did gcc lower the switch to a comparison chain?
        let n_indjmp = recovered
            .iter()
            .filter(|&&a| sup.at(a).map_or(false, |i| i.is_jump() && i.branch_target.is_none()))
            .count();
        eprintln!("    [jt] {} resolved-targets, {} indirect-jumps in recovered set",
            jt_targets.len(), n_indjmp);
    }

    // GT-free function entries. `main` (and its flattened dispatcher) is the problem: in a -no-pie
    // dynamic ELF `_start` passes `main` to `__libc_start_main` as a *pointer*, not a direct call, so
    // call-target discovery alone misses it. Instead: entries = recovered instructions with NO
    // intra-procedural predecessor (a function head is reached only by call/pointer, which we don't
    // model as an intra edge), plus the ELF entry and any direct call target. Jump-table case blocks
    // have the dispatcher as a predecessor, so they are correctly excluded.
    let entries = discover_entries(&sup, &recovered, &jt_targets, e_entry);

    // Contiguous-span function assignment: function i owns [entry_i, entry_{i+1}). Holds for these
    // non-stripped -O2 -no-pie builds; Tigress keeps functions contiguous.
    //
    // Per-binary statistic = the metrics of the LARGEST function (most blocks). Obfuscation inflates
    // `main` into the biggest function in these programs — flattening/virtualization multiply its
    // block count — so the largest function IS where the obfuscation lives. Selecting by size (not by
    // score) avoids keying on libc/crt boilerplate (`register_tm_clones` etc.), whose fixed small
    // shape is identical across every binary and would otherwise swamp the signal.
    // Per-binary statistic = the max of each topology metric over all functions with ≥ MIN_BLOCKS
    // blocks. Obfuscation lives in one function (the inflated `main`); with the corrected fan-out
    // dominator metric, ordinary libc/crt boilerplate scores low on every axis, so the max surfaces
    // the obfuscated function without needing to name it. Each metric is maxed independently — the
    // flattened dispatcher maximizes them together anyway, and a legit switch/VM will too (the FP
    // surface we audit).
    let text_hi = base + code.len() as u64;
    let mut n_funcs = 0usize;
    let mut best = Metrics::default();
    let mut max_blocks = 0usize;
    for i in 0..entries.len() {
        let lo = entries[i];
        let hi = *entries.get(i + 1).unwrap_or(&text_hi);
        let Some(cfg) = build_function_cfg(&sup, &recovered, lo, hi, &jt_targets) else { continue };
        n_funcs += 1;
        max_blocks = max_blocks.max(cfg.blocks.len());
        if cfg.blocks.len() < MIN_BLOCKS {
            continue;
        }
        let m = cfg_metrics(&cfg);
        if std::env::var("CFGPROBE_DUMP").ok().as_deref() == path.file_name().and_then(|s| s.to_str()) {
            eprintln!(
                "    [dump] fn@{:#x} blocks={} edges={} disp={:.3} dom={:.3} indir={:.3} noStruct={:.3}",
                lo, cfg.blocks.len(), cfg.total_edges, m.dispatcher, m.dom_collapse, m.indirect, m.struct_absence
            );
        }
        best.dispatcher = best.dispatcher.max(m.dispatcher);
        best.dom_collapse = best.dom_collapse.max(m.dom_collapse);
        best.indirect = best.indirect.max(m.indirect);
        best.struct_absence = best.struct_absence.max(m.struct_absence);
    }
    best.n_funcs = n_funcs;
    best.max_blocks = max_blocks;
    Ok(best)
}

/// Discover function entries GT-free: recovered instructions with no intra-procedural predecessor
/// (function heads are reached only by call / code-pointer, which we don't model as intra edges),
/// unioned with the ELF entry and direct call targets. Jump-table case blocks have the dispatcher as
/// a predecessor, so they are not mistaken for entries.
fn discover_entries(
    sup: &Superset,
    recovered: &HashSet<u64>,
    jt_targets: &[u64],
    e_entry: u64,
) -> Vec<u64> {
    let jt: HashSet<u64> = jt_targets.iter().copied().collect();
    // Global intra-procedural successors (calls contribute only their fall-through).
    let intra_succ = |a: u64| -> Vec<u64> {
        let Some(insn) = sup.at(a) else { return Vec::new() };
        let ft = a + insn.size as u64;
        if insn.is_ret() {
            return Vec::new();
        }
        if insn.is_call() {
            return if recovered.contains(&ft) { vec![ft] } else { Vec::new() };
        }
        if insn.is_jump() {
            match insn.branch_target {
                Some(t) => {
                    let mut v = Vec::new();
                    if recovered.contains(&t) {
                        v.push(t);
                    }
                    if insn.mnemonic != "jmp" && recovered.contains(&ft) {
                        v.push(ft);
                    }
                    v
                }
                None => jt_targets.iter().copied().filter(|&t| recovered.contains(&t)).collect(),
            }
        } else if recovered.contains(&ft) {
            vec![ft]
        } else {
            Vec::new()
        }
    };
    let mut has_pred: HashSet<u64> = HashSet::new();
    for &a in recovered {
        for t in intra_succ(a) {
            has_pred.insert(t);
        }
    }
    let mut entries: Vec<u64> = recovered
        .iter()
        .copied()
        .filter(|a| !has_pred.contains(a) && !jt.contains(a))
        .collect();
    if recovered.contains(&e_entry) {
        entries.push(e_entry);
    }
    for &a in recovered {
        if let Some(insn) = sup.at(a) {
            if insn.is_call() {
                if let Some(t) = insn.branch_target {
                    if recovered.contains(&t) && !jt.contains(&t) {
                        entries.push(t);
                    }
                }
            }
        }
    }
    entries.sort_unstable();
    entries.dedup();
    if entries.is_empty() {
        entries.push(*recovered.iter().min().unwrap());
    }
    entries
}

// ── Recovered per-function CFG (basic blocks + edges) ──────────────────────────

struct FunctionCfg {
    /// Block entry address → block index.
    index: HashMap<u64, usize>,
    blocks: Vec<u64>, // block leader addresses, index-aligned
    /// Directed block edges (unique), by block index.
    succ: Vec<HashSet<usize>>,
    pred: Vec<HashSet<usize>>,
    /// Block index of the function entry.
    entry_block: usize,
    /// Blocks whose terminator is an indirect jump (no resolved static target on the branch itself).
    indirect_blocks: HashSet<usize>,
    /// Count of straight-line fall-through block edges (target is the source block's fall-through).
    fallthrough_edges: usize,
    /// Total unique block edges.
    total_edges: usize,
}

/// Build the intra-procedural recovered CFG for the function spanning `[lo, hi)`. Instructions are the
/// recovered set in span; edges are direct successors (calls contribute only their fall-through) plus,
/// for a block ending in an indirect jump, edges to every resolved jump-table target inside the span.
fn build_function_cfg(
    sup: &Superset,
    recovered: &HashSet<u64>,
    lo: u64,
    hi: u64,
    jt_targets: &[u64],
) -> Option<FunctionCfg> {
    // Recovered instructions in this function's span, address-sorted.
    let mut insns: Vec<u64> = recovered.iter().copied().filter(|&a| a >= lo && a < hi).collect();
    insns.sort_unstable();
    if insns.len() < 2 {
        return None;
    }
    let in_span = |a: u64| a >= lo && a < hi && recovered.contains(&a);
    let jt_in_span: Vec<u64> = jt_targets.iter().copied().filter(|&a| a >= lo && a < hi).collect();

    // Intra-procedural instruction successors.
    let insn_succ = |a: u64| -> (Vec<u64>, bool) {
        // returns (targets, is_indirect_jump)
        let Some(insn) = sup.at(a) else { return (Vec::new(), false) };
        let ft = a + insn.size as u64;
        if insn.is_ret() {
            return (Vec::new(), false);
        }
        if insn.is_call() {
            // Intra: do not descend into the callee; flow returns to the fall-through.
            return (if in_span(ft) { vec![ft] } else { Vec::new() }, false);
        }
        if insn.is_jump() {
            match insn.branch_target {
                Some(t) => {
                    let mut v = Vec::new();
                    if in_span(t) {
                        v.push(t);
                    }
                    // Conditional jump also falls through.
                    if insn.mnemonic != "jmp" && in_span(ft) {
                        v.push(ft);
                    }
                    (v, false)
                }
                None => {
                    // Indirect jump: attach resolved jump-table targets in span (the dispatcher edges).
                    let v: Vec<u64> = jt_in_span.iter().copied().filter(|&t| t != a).collect();
                    (v, true)
                }
            }
        } else {
            (if in_span(ft) { vec![ft] } else { Vec::new() }, false)
        }
    };

    // Leaders: function entry, any branch/indirect target, and the instruction after a terminator.
    let insn_set: HashSet<u64> = insns.iter().copied().collect();
    let mut leaders: HashSet<u64> = HashSet::new();
    leaders.insert(insns[0]);
    for &a in &insns {
        let (succs, _) = insn_succ(a);
        let Some(insn) = sup.at(a) else { continue };
        let ft = a + insn.size as u64;
        let terminator = insn.is_ret() || insn.is_jump();
        for t in &succs {
            leaders.insert(*t); // branch/jump-table targets start blocks
        }
        if terminator && insn_set.contains(&ft) {
            leaders.insert(ft); // instruction after a jump/ret starts a block
        }
    }
    // Restrict leaders to recovered in-span instructions.
    let mut block_addrs: Vec<u64> = leaders.into_iter().filter(|&a| insn_set.contains(&a)).collect();
    block_addrs.sort_unstable();
    let mut index: HashMap<u64, usize> = HashMap::new();
    for (i, &a) in block_addrs.iter().enumerate() {
        index.insert(a, i);
    }
    let n = block_addrs.len();
    if n < 2 {
        return None;
    }

    // For each block, walk instructions until the next leader / terminator, collect out-edges.
    let mut succ: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    let mut pred: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    let mut indirect_blocks: HashSet<usize> = HashSet::new();
    let mut fallthrough_edges = 0usize;
    // Map an instruction address to the next recovered in-span instruction address (linear order).
    let next_of: BTreeMap<u64, u64> = insns
        .windows(2)
        .map(|w| (w[0], w[1]))
        .collect();

    for (bi, &b) in block_addrs.iter().enumerate() {
        let mut a = b;
        loop {
            let Some(insn) = sup.at(a) else { break };
            let (succs, is_indirect) = insn_succ(a);
            let is_term = insn.is_ret() || insn.is_jump();
            let next_leader = next_of.get(&a).copied().filter(|t| index.contains_key(t));
            // A block ends at a terminator, or right before the next leader.
            let ends_here = is_term || next_leader.map_or(true, |t| index.contains_key(&t) && block_addrs.binary_search(&t).is_ok() && is_leader_boundary(&index, t, a, insn.size));
            if is_indirect {
                indirect_blocks.insert(bi);
            }
            if is_term {
                let ft = a + insn.size as u64;
                for t in &succs {
                    if let Some(&ti) = index.get(t) {
                        succ[bi].insert(ti);
                        if *t == ft {
                            fallthrough_edges += 1;
                        }
                    }
                }
                break;
            }
            // Non-terminator: continue to next instruction unless it starts a new block.
            let ft = a + insn.size as u64;
            match index.get(&ft) {
                Some(&ti) => {
                    // fall-through into a new block leader → single fall-through edge.
                    succ[bi].insert(ti);
                    fallthrough_edges += 1;
                    break;
                }
                None => {
                    if insn_set.contains(&ft) {
                        a = ft; // extend the block
                        let _ = ends_here;
                        continue;
                    } else {
                        break; // fell off recovered set
                    }
                }
            }
        }
    }
    for (bi, s) in succ.iter().enumerate() {
        for &ti in s {
            pred[ti].insert(bi);
        }
    }
    let total_edges: usize = succ.iter().map(|s| s.len()).sum();
    let entry_block = *index.get(&block_addrs[0]).unwrap();

    Some(FunctionCfg {
        index,
        blocks: block_addrs,
        succ,
        pred,
        entry_block,
        indirect_blocks,
        fallthrough_edges,
        total_edges,
    })
}

/// Whether address `t` is a genuine block boundary relative to the current instruction (helper kept
/// simple — a leader always is; retained for readability of the block-walk).
fn is_leader_boundary(index: &HashMap<u64, usize>, t: u64, _a: u64, _sz: u8) -> bool {
    index.contains_key(&t)
}

// ── The four topology statistics from one function CFG ─────────────────────────

fn cfg_metrics(cfg: &FunctionCfg) -> Metrics {
    let n = cfg.blocks.len();
    let te = cfg.total_edges.max(1) as f64;

    // 1. dispatcher dominance: busiest block's (in+out) degree over all edges.
    let mut dispatcher = 0.0f64;
    for i in 0..n {
        let deg = (cfg.succ[i].len() + cfg.pred[i].len()) as f64;
        dispatcher = dispatcher.max(deg / te);
    }

    // 3. indirect-dispatch concentration: busiest indirect-jump block's (in+out) degree share.
    let mut indirect = 0.0f64;
    for &i in &cfg.indirect_blocks {
        let deg = (cfg.succ[i].len() + cfg.pred[i].len()) as f64;
        indirect = indirect.max(deg / te);
    }

    // 2. dominator-tree collapse: largest fraction of blocks dominated by a single non-entry block.
    let dom_collapse = dominator_collapse(cfg);

    // 4. real-structure absence: 1 − fraction of edges that are straight-line fall-throughs.
    let struct_absence = 1.0 - (cfg.fallthrough_edges as f64 / te);

    Metrics {
        n_funcs: 0,
        max_blocks: n,
        dispatcher,
        dom_collapse,
        indirect,
        struct_absence: struct_absence.clamp(0.0, 1.0),
    }
}

/// Dominator-tree fan-out: the largest fraction of blocks that share a single immediate dominator.
/// This is the flattening signature — the dispatcher is the immediate dominator of *every* case, so
/// one node has enormous dominator-tree fan-out. Ordinary sequential/structured code has a path-like
/// dominator tree (fan-out ~1–2), so it scores low; a *legitimate* switch/VM dispatcher scores high
/// too (the honest FP surface we audit). Iterative dominators (Cooper–Harvey–Kennedy) over blocks
/// reachable from the entry; islands still count in the denominator.
fn dominator_collapse(cfg: &FunctionCfg) -> f64 {
    let n = cfg.blocks.len();
    if n < 3 {
        return 0.0;
    }
    // Reverse-postorder from the entry.
    let mut rpo: Vec<usize> = Vec::new();
    let mut seen = vec![false; n];
    dfs_post(cfg.entry_block, &cfg.succ, &mut seen, &mut rpo);
    rpo.reverse();
    let mut order = vec![usize::MAX; n]; // node → rpo position
    for (pos, &node) in rpo.iter().enumerate() {
        order[node] = pos;
    }
    let reachable: Vec<usize> = rpo.clone();
    if reachable.len() < 3 {
        return 0.0;
    }

    // idom via CHK.
    let mut idom = vec![usize::MAX; n];
    idom[cfg.entry_block] = cfg.entry_block;
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &rpo {
            if b == cfg.entry_block {
                continue;
            }
            let mut new_idom = usize::MAX;
            for &p in &cfg.pred[b] {
                if idom[p] == usize::MAX {
                    continue; // p not yet processed / unreachable
                }
                new_idom = if new_idom == usize::MAX {
                    p
                } else {
                    intersect(p, new_idom, &idom, &order)
                };
            }
            if new_idom != usize::MAX && idom[b] != new_idom {
                idom[b] = new_idom;
                changed = true;
            }
        }
    }

    // Dominator-tree fan-out: for each block, tally its immediate dominator; the max tally is the
    // number of case-blocks a single dispatcher directly dominates.
    let mut children = vec![0usize; n];
    for &b in &reachable {
        if b == cfg.entry_block {
            continue;
        }
        let d = idom[b];
        if d != usize::MAX {
            children[d] += 1;
        }
    }
    let max_children = children.iter().copied().max().unwrap_or(0);
    max_children as f64 / n as f64
}

fn dfs_post(u: usize, succ: &[HashSet<usize>], seen: &mut [bool], out: &mut Vec<usize>) {
    seen[u] = true;
    for &v in &succ[u] {
        if !seen[v] {
            dfs_post(v, succ, seen, out);
        }
    }
    out.push(u);
}

/// CHK intersect on rpo positions (smaller position = closer to entry).
fn intersect(mut a: usize, mut b: usize, idom: &[usize], order: &[usize]) -> usize {
    while a != b {
        while order[a] > order[b] {
            a = idom[a];
            if a == usize::MAX {
                return b;
            }
        }
        while order[b] > order[a] {
            b = idom[b];
            if b == usize::MAX {
                return a;
            }
        }
    }
    a
}

// ── ELF entry ──────────────────────────────────────────────────────────────────

fn elf_entry(bytes: &[u8]) -> Result<u64> {
    match Object::parse(bytes)? {
        Object::Elf(elf) => Ok(elf.header.e_entry),
        _ => bail!("not an ELF"),
    }
}

// ── Reporting: AUROC + the explicit FP audit ───────────────────────────────────

fn report(rows: &[Row]) {
    // Group semantics by label substring:
    //   flatten / virt   — obfuscated positives
    //   ctrl             — controlled negative: the SAME programs, clean-compiled (isolates the
    //                      obfuscation effect from program identity)
    //   switch           — real-world switch-heavy benign (coreutils) — the honest FP surface
    //   legit            — legitimate interpreters/VMs — the interpreter FP gate
    let has = |sub: &'static str| move |l: &str| l.contains(sub);
    let is_flat = has("flatten");
    let is_virt = has("virt");
    let is_ctrl = has("ctrl");
    let is_switch = has("switch");
    let is_legit = has("legit");

    let pick = |f: &dyn Fn(&str) -> bool, g: &dyn Fn(&Metrics) -> f64| -> Vec<f64> {
        rows.iter().filter(|r| f(&r.label)).map(|r| g(&r.m)).collect()
    };

    let stats: [(&str, fn(&Metrics) -> f64); 5] = [
        ("dispatcher", |m| m.dispatcher),
        ("dom_collapse", |m| m.dom_collapse),
        ("indirect", |m| m.indirect),
        ("struct_absence", |m| m.struct_absence),
        ("COMBINED", |m| m.combined()),
    ];
    let groups: [(&str, &dyn Fn(&str) -> bool); 5] = [
        ("flatten", &is_flat),
        ("virt", &is_virt),
        ("clean_ctrl", &is_ctrl),
        ("benign_switch", &is_switch),
        ("legit_interp", &is_legit),
    ];

    println!("\n════════════════ CFG-TOPOLOGY OBFUSCATION PROBE ════════════════");
    for (name, f) in &groups {
        println!("  group {:<14} n={}", name, rows.iter().filter(|r| f(&r.label)).count());
    }

    println!("\n— Group means per statistic —");
    print!("  {:<15}", "stat");
    for (name, _) in &groups {
        print!(" {:>13}", name);
    }
    println!();
    for (sname, g) in &stats {
        print!("  {:<15}", sname);
        for (_, f) in &groups {
            print!(" {:>13.3}", mean(&pick(f, g)));
        }
        println!();
    }

    // The AUROC matrix: each obfuscation positive vs each negative group. Separating from clean_ctrl
    // = "the topology channel sees the obfuscation of the SAME code". Failing to separate from
    // benign_switch / legit_interp = "the signature is not obfuscation-specific — legit switch/VM code
    // shares it". Both numbers are the honest story.
    println!("\n— AUROC matrix (positive vs each negative; 0.5=chance) —");
    let negs: [(&str, &dyn Fn(&str) -> bool); 3] =
        [("clean_ctrl", &is_ctrl), ("benign_switch", &is_switch), ("legit_interp", &is_legit)];
    for (pname, pf) in [("flatten", &is_flat as &dyn Fn(&str) -> bool), ("virt", &is_virt)] {
        println!("  {} vs:", pname);
        println!("    {:<15} {:>12} {:>13} {:>13}", "stat", negs[0].0, negs[1].0, negs[2].0);
        for (sname, g) in &stats {
            let pos = pick(&pf, g);
            print!("    {:<15}", sname);
            for (_, nf) in &negs {
                print!(" {:>12}", fmt_auc(auroc(&pos, &pick(nf, g))));
            }
            println!();
        }
    }

    // FP audit: threshold each statistic at the clean_ctrl MAX (the operating point that would flag
    // any obfuscation of the small controls), then report how many benign_switch / legit_interp
    // binaries the same threshold FALSELY fires on — the "worthless if it fires on every interpreter"
    // gate, made concrete.
    println!("\n— FALSE-POSITIVE AUDIT (threshold = clean_ctrl max; fire = strictly above) —");
    for (sname, g) in &stats {
        let thr = pick(&is_ctrl, g).iter().cloned().fold(f64::MIN, f64::max);
        let fp_switch = pick(&is_switch, g).iter().filter(|&&v| v > thr).count();
        let n_switch = pick(&is_switch, g).len();
        let fp_legit = pick(&is_legit, g).iter().filter(|&&v| v > thr).count();
        let n_legit = pick(&is_legit, g).len();
        // Also: how many flatten/virt clear this same threshold (true-positive sensitivity)?
        let tp_f = pick(&is_flat, g).iter().filter(|&&v| v > thr).count();
        let tp_v = pick(&is_virt, g).iter().filter(|&&v| v > thr).count();
        println!(
            "  {:<15} thr={:.3}  TP flatten={}/{} virt={}/{}   FP switch={}/{} legit={}/{}",
            sname, thr, tp_f, pick(&is_flat, g).len(), tp_v, pick(&is_virt, g).len(),
            fp_switch, n_switch, fp_legit, n_legit
        );
    }
    println!("\n— legit_interp per-binary (the interpreter FP gate, explicit) —");
    for r in rows.iter().filter(|r| is_legit(&r.label)) {
        println!(
            "  {:<20} disp={:.3} dom={:.3} indir={:.3} noStruct={:.3}",
            r.name, r.m.dispatcher, r.m.dom_collapse, r.m.indirect, r.m.struct_absence
        );
    }
    println!("═══════════════════════════════════════════════════════════════");
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

/// AUROC of `pos` vs `neg` (Mann–Whitney). None-safe formatting via `fmt_auc`.
fn auroc(pos: &[f64], neg: &[f64]) -> Option<f64> {
    if pos.is_empty() || neg.is_empty() {
        return None;
    }
    let mut gt = 0.0;
    let mut n = 0.0;
    for &p in pos {
        for &q in neg {
            n += 1.0;
            if p > q {
                gt += 1.0;
            } else if (p - q).abs() < 1e-12 {
                gt += 0.5;
            }
        }
    }
    Some(gt / n)
}

fn fmt_auc(a: Option<f64>) -> String {
    a.map(|v| format!("{v:.3}")).unwrap_or_else(|| "n/a".into())
}

// ── CSV ────────────────────────────────────────────────────────────────────────

fn write_csv(path: &Path, rows: &[Row]) -> Result<()> {
    let mut s = String::from("name,label,n_funcs,max_blocks,dispatcher,dom_collapse,indirect,struct_absence,combined\n");
    use std::fmt::Write as _;
    for r in rows {
        writeln!(
            s,
            "{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
            r.name, r.label, r.m.n_funcs, r.m.max_blocks,
            r.m.dispatcher, r.m.dom_collapse, r.m.indirect, r.m.struct_absence, r.m.combined()
        )
        .ok();
    }
    fs::write(path, s).with_context(|| format!("writing {}", path.display()))?;
    eprintln!("wrote {}", path.display());
    Ok(())
}

// ── CLI ────────────────────────────────────────────────────────────────────────

struct Args {
    groups: Vec<(String, PathBuf)>,
    out: PathBuf,
}

impl Args {
    fn parse(mut it: impl Iterator<Item = String>) -> Result<Self> {
        const USAGE: &str = "usage: cfgprobe --group LABEL DIR [--group LABEL DIR ...] --out results.csv";
        let mut groups = Vec::new();
        let mut out = None;
        while let Some(a) = it.next() {
            let mut next = |flag: &str| it.next().with_context(|| format!("{flag} needs a value"));
            match a.as_str() {
                "--group" => {
                    let label = next("--group label")?;
                    let dir = PathBuf::from(next("--group dir")?);
                    groups.push((label, dir));
                }
                "--out" => out = Some(PathBuf::from(next("--out")?)),
                "-h" | "--help" => {
                    eprintln!("{USAGE}");
                    std::process::exit(0);
                }
                other => bail!("unexpected argument: {other}\n{USAGE}"),
            }
        }
        if groups.is_empty() {
            bail!("need at least one --group\n{USAGE}");
        }
        Ok(Args { groups, out: out.context(USAGE)? })
    }
}
