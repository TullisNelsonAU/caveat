//! `probcfg` — Layer-2 probabilistic reachability over the superset CFG.
//!
//! Layer 1 (`probdisasm` Soft) answers "does this byte *decode* as a real instruction start?".
//! This crate adds the orthogonal Layer-2 question: is an instruction actually *reached* from the
//! program entry via the control-flow graph? The motivating hypothesis (see
//! `probablistic/LAYER2_DESIGN.md`) is that reachability suppresses *code-shaped-but-unreachable*
//! bytes — an appended decoy / code-in-data region — that no per-byte posterior can separate,
//! because that decoy genuinely *is* code.
//!
//! This is a **probe to measure the leak**, not a finished method. A decoy is real code, so once
//! reachability enters it via one boundary fall-through edge, the decoy's own internal jumps keep it
//! lit. The decoy is cleanly unreachable only when the boundary instruction is `ret`/`jmp` (no
//! fall-through edge at all). How often that holds — and how much the decoy leaks otherwise — is the
//! empirical question this crate exists to answer.
//!
//! **Honesty wall (non-negotiable):** reachability is a *decode weight* only. It never overwrites the
//! reported per-byte posterior. Callers (e.g. `bench --reach`) fold it into the cover's selection
//! weight and leave the calibration axis untouched.

use std::collections::{HashMap, HashSet, VecDeque};

use probdisasm::Superset;

/// Tunables for [`reachability`].
pub struct ReachConfig {
    /// Per-hop decay applied on fall-through edges (e.g. `0.9`). `1.0` = no decay. Jump/call-target
    /// edges never decay: a taken branch fully reaches its target. Decaying only fall-through is what
    /// lets a region reached *only* by a long fall-through chain (a decoy) fade toward 0 while real
    /// functions — refreshed at each jump/call entry — stay high.
    pub fall_decay: f64,
    /// `true`: anchor reachability on the entry + direct CALL targets (genuine function heads).
    /// `false`: also anchor on jump targets. The `false` case is an ablation expected to leak *more* —
    /// a decoy's own internal jump targets self-anchor to `r = 1` (trap #2 from the design doc).
    pub anchors_calls_only: bool,
    /// Safety cap on worklist pops. Monotone max-propagation converges well within this on real
    /// `.text`; the cap only guards against pathological inputs.
    pub max_iters: usize,
}

impl Default for ReachConfig {
    fn default() -> Self {
        Self { fall_decay: 0.9, anchors_calls_only: true, max_iters: 5_000_000 }
    }
}

/// Reachability score in `[0, 1]` per valid-instruction vaddr, propagated from `entry` plus the
/// anchors selected by `cfg` over the superset CFG.
///
/// Absent key ⇒ `0.0` (unreached). The algorithm is a worklist fixpoint over monotone
/// max-propagation: a node's score is the max, over all paths, of the product of edge factors
/// (`fall_decay` per fall-through hop, `1.0` per jump/call-target hop). Because every update strictly
/// raises a bounded score, it converges.
pub fn reachability(sup: &Superset, entry: u64, cfg: &ReachConfig) -> HashMap<u64, f64> {
    let mut r: HashMap<u64, f64> = HashMap::new();
    let mut queue: VecDeque<u64> = VecDeque::new();

    let is_valid = |addr: u64| sup.at(addr).is_some();

    // ── Step 1: anchors (r = 1.0). ───────────────────────────────────────────────────────────────
    let anchor = |addr: u64, r: &mut HashMap<u64, f64>, q: &mut VecDeque<u64>| {
        if is_valid(addr) && r.insert(addr, 1.0).is_none() {
            q.push_back(addr);
        }
    };
    anchor(entry, &mut r, &mut queue);
    for insn in sup.iter_valid() {
        if let Some(t) = insn.branch_target {
            if insn.is_call() {
                anchor(t, &mut r, &mut queue); // direct CALL target = confirmed function head
            } else if !cfg.anchors_calls_only && insn.is_jump() {
                anchor(t, &mut r, &mut queue); // ablation: jump targets too (leaks more)
            }
        }
    }

    // ── Step 2: propagate. ───────────────────────────────────────────────────────────────────────
    let mut iters = 0usize;
    while let Some(a) = queue.pop_front() {
        iters += 1;
        if iters > cfg.max_iters {
            break;
        }
        let ra = r[&a];
        // `at(a)` is Some — `a` only ever enters the queue as a valid instruction.
        let size = sup.at(a).map(|i| i.size as u64).unwrap_or(0);
        let fall_through = a.wrapping_add(size);
        for s in sup.successors_of(a) {
            if !is_valid(s) {
                continue;
            }
            let contrib = if s == fall_through { ra * cfg.fall_decay } else { ra };
            let entry_r = r.entry(s).or_insert(0.0);
            if contrib > *entry_r + 1e-9 {
                *entry_r = contrib;
                queue.push_back(s);
            }
        }
    }

    r
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// Function confirmation (Layer-2 milestone).
//
// The reachability probe (above) LEAKED: an appended code-in-data decoy is real code, so it
// self-anchors through its *own* direct CALLs (`r = 1`) no matter the entry or boundary. The fix is
// not "anchor on call targets" but **transitive confirmation from the true entry**: a function is
// real only if a *confirmed* function calls it. The decoy is a disconnected component of the
// entry-rooted call graph — nothing in the real program calls it — so it is never confirmed.
//
// Honesty wall still holds: confirmation is a *decode gate* only (a hard `r ∈ {0,1}` the caller folds
// into the cover weight). It never touches the reported per-byte posterior.
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// One function body recovered by intra-procedural recursive descent from a head.
#[derive(Debug, Clone, Default)]
pub struct Function {
    /// Head (entry) address of the function.
    pub head: u64,
    /// Instruction addresses reached intra-procedurally from `head`.
    pub body: Vec<u64>,
    /// Direct CALL targets found in the body (call-graph out-edges).
    pub calls: Vec<u64>,
    /// Unconditional `jmp` targets that leave the function (tail calls).
    pub tail_jumps: Vec<u64>,
}

/// Decode one function body by intra-procedural recursive descent from `head`.
///
/// Edge semantics:
/// - normal insn → follow fall-through;
/// - `call` → record the (direct) target in `calls`, continue at the return address (fall-through);
/// - conditional jump → follow BOTH the intra-function target and the fall-through;
/// - unconditional `jmp` → if the target is another candidate `head` OR outside `[head, head+max_span)`,
///   record it in `tail_jumps` and STOP that path; otherwise follow it (intra-function);
/// - `ret` (or an indirect branch with no static target) → stop that path.
///
/// Exploration is bounded to `[head, head + max_span)` and to valid instructions.
pub fn extract_function(sup: &Superset, head: u64, heads: &HashSet<u64>, max_span: usize) -> Function {
    let mut f = Function { head, ..Default::default() };
    let hi = head.wrapping_add(max_span as u64);
    let within = |a: u64| a >= head && a < hi;

    let mut seen: HashSet<u64> = HashSet::new();
    let mut stack: Vec<u64> = vec![head];
    while let Some(a) = stack.pop() {
        if !within(a) || !seen.insert(a) {
            continue;
        }
        // Another function's entry is a hard boundary on EVERY path (fall-through, conditional target,
        // intra-`jmp`): a function does not bleed into the next one. Without this a function that ends
        // without a `ret` would absorb its successor — and, at the real/decoy boundary, walk straight
        // into the contiguously-valid decoy, spuriously lighting it. Reached heads are still confirmed
        // as call/jump *targets* elsewhere; we just don't fold their bodies into this one.
        if a != head && heads.contains(&a) {
            continue;
        }
        let Some(insn) = sup.at(a) else {
            continue;
        };
        f.body.push(a);
        let fall = a.wrapping_add(insn.size as u64);

        if insn.is_ret() {
            continue; // path ends
        }
        if insn.is_call() {
            if let Some(t) = insn.branch_target {
                f.calls.push(t); // direct call — a call-graph out-edge
            }
            stack.push(fall); // returns to the fall-through
            continue;
        }
        if insn.is_jump() {
            let target = insn.branch_target;
            if insn.mnemonic == "jmp" {
                // unconditional: either a tail call (leaves the function) or intra-function flow.
                match target {
                    Some(t) if heads.contains(&t) || !within(t) => f.tail_jumps.push(t),
                    Some(t) => stack.push(t),
                    None => {} // indirect jmp: no static target, stop this path
                }
            } else {
                // conditional: follow the intra-function target (if any) AND the fall-through.
                if let Some(t) = target {
                    if within(t) && !heads.contains(&t) {
                        stack.push(t);
                    }
                }
                stack.push(fall);
            }
            continue;
        }
        // normal instruction
        stack.push(fall);
    }
    f
}

/// Tunables for [`confirm_from_entry`].
pub struct ConfirmConfig {
    /// Cap on a single function's byte span during intra-procedural descent (e.g. `65536`).
    pub max_fn_span: usize,
    /// If `true`, an unconditional-`jmp` tail call also confirms its target (tail-called functions).
    pub confirm_via_tail_jumps: bool,
}

impl Default for ConfirmConfig {
    fn default() -> Self {
        Self { max_fn_span: 65536, confirm_via_tail_jumps: false }
    }
}

/// Result of transitive function confirmation from the entry.
pub struct Confirmation {
    /// Candidate heads = `{ entry } ∪ { direct-CALL targets over all valid instructions }`. This
    /// deliberately INCLUDES a decoy's own internal call targets — they just never get confirmed.
    pub all_heads: HashSet<u64>,
    /// Heads transitively confirmed from `entry` via the call graph.
    pub confirmed_heads: HashSet<u64>,
    /// Union of confirmed functions' bodies — the decode gate.
    pub confirmed_insns: HashSet<u64>,
}

/// Confirm functions transitively reachable from `entry` over the direct-call graph.
///
/// Candidate heads are the entry plus every direct-CALL target. A BFS from `entry` confirms a head's
/// `calls` (and optionally its `tail_jumps`); `confirmed_insns` is the union of the confirmed heads'
/// intra-procedural bodies. A decoy has no direct caller reachable from the entry, so it is never
/// confirmed — that is the whole point.
pub fn confirm_from_entry(sup: &Superset, entry: u64, cfg: &ConfirmConfig) -> Confirmation {
    // Candidate heads: entry + every direct-CALL target that decodes.
    let mut all_heads: HashSet<u64> = HashSet::new();
    if sup.at(entry).is_some() {
        all_heads.insert(entry);
    }
    for insn in sup.iter_valid() {
        if insn.is_call() {
            if let Some(t) = insn.branch_target {
                if sup.at(t).is_some() {
                    all_heads.insert(t);
                }
            }
        }
    }

    // Transitive confirmation: BFS the call graph from the entry.
    let mut confirmed_heads: HashSet<u64> = HashSet::new();
    let mut confirmed_insns: HashSet<u64> = HashSet::new();
    let mut queue: VecDeque<u64> = VecDeque::new();
    if sup.at(entry).is_some() && confirmed_heads.insert(entry) {
        queue.push_back(entry);
    }
    while let Some(h) = queue.pop_front() {
        let f = extract_function(sup, h, &all_heads, cfg.max_fn_span);
        confirmed_insns.extend(f.body.iter().copied());
        let mut out: Vec<u64> = f.calls;
        if cfg.confirm_via_tail_jumps {
            out.extend(f.tail_jumps);
        }
        for t in out {
            if all_heads.contains(&t) && confirmed_heads.insert(t) {
                queue.push_back(t);
            }
        }
    }

    Confirmation { all_heads, confirmed_heads, confirmed_insns }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// Milestone 2 — CALIBRATED function confirmation (eqs 1–4 of LAYER2_M2_SPEC).
//
// M1's hard gate suppresses the decoy (F = 0) but excludes the ~36% indirect-only real tail with it.
// M2 replaces the Boolean gate with a probabilistic reachability FIXPOINT so the tail degrades to
// honest uncertainty (≈ the mixing base rate β₀, Theorem 2) instead of silent loss:
//
//   edge_evidence  (eq 1): C_{g→h} = 1 − ∏_c (1 − π_c)          noisy-OR over call sites
//   local_prior    (eq 2): prior_h = σ(w·φ_h)                    calibrated local shape prior
//   confirm_fixpoint(eq 3): F_h = 1 − (1−prior_h)·∏_g(1 − F_g·C_{g→h})   least fixpoint (Tarski)
//   reachedness    (eq 4): R_a = 1 − ∏_{g∋a}(1 − F_g·ρ(g,a))     noisy-OR over containing fns
//
// The honesty wall still governs the DECODE (mode a): R is a weight only. The FUSED posterior
// (mode b) is a *deliberately recalibrated* Layer-2 confidence, isotonic-mapped (Theorem 1), never a
// silent overwrite of π. Both live in `bench`.
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// Eq (1) — noisy-OR edge evidence: the probability that (real) function `g` really contains a call
/// to `h`, given the per-call-site instruction posteriors. `C = 0` for no sites (empty product ⇒ 1).
pub fn edge_evidence(site_pis: &[f64]) -> f64 {
    1.0 - site_pis.iter().map(|&p| 1.0 - p.clamp(0.0, 1.0)).product::<f64>()
}

/// Local features φ_h for the head prior (eq 2). Deliberately *shape-only*: whether `h` decodes into
/// a function prologue, and whether it is the ELF entry. `#incoming call sites` is intentionally
/// EXCLUDED from the prior — that is call-graph evidence already carried by `C_{g→h}` in the
/// fixpoint, and folding it in here would double-count it (and let a decoy's own internal call
/// structure inflate its prior). Keeping φ shape-only is what makes Theorem 2 exact: decoy heads and
/// indirect-only real heads are tiled real code, so they share the φ-law and get the same prior.
#[derive(Debug, Clone, Copy, Default)]
pub struct HeadFeatures {
    /// `h` is the ELF entry point.
    pub is_entry: bool,
    /// `h` decodes into a recognizable function prologue (`endbr64` / `push` reg / `sub rsp,imm`).
    pub prologue: bool,
    /// Incoming direct call sites (reported for diagnostics / β₀; NOT used by [`local_prior`]).
    pub n_callsites: usize,
}

/// Eq (2) — a calibrated local prior `P(Z_h = 1 | φ_h)`. Hand-set logistic on {prologue, is_entry};
/// isotonic-recalibrated against FUNC-symbol GT downstream (§5). The entry is pinned to 1.
pub fn local_prior(ft: &HeadFeatures) -> f64 {
    if ft.is_entry {
        return 1.0;
    }
    // σ(−2.2 + 2.2·prologue): prologue ⇒ 0.5, no-prologue ⇒ ~0.10. Shape-only, weak by design.
    let z = -2.2 + 2.2 * f64::from(ft.prologue);
    1.0 / (1.0 + (-z).exp())
}

/// Does the decode at `head` look like a function prologue? Weak, shape-only signal for [`local_prior`].
fn looks_like_prologue(sup: &Superset, head: u64) -> bool {
    let Some(i0) = sup.at(head) else {
        return false;
    };
    match i0.mnemonic.as_str() {
        "endbr64" => true,                              // CET landing pad — near-certain fn head
        "push" => true,                                 // callee-saved register save
        "sub" if i0.op_str.starts_with("rsp") => true,  // stack frame allocation
        _ => false,
    }
}

/// Eq (3) — the confirmation fixpoint `F : head → P(real function)`. Jacobi iteration of the monotone
/// operator `T` from `F⁰ = prior` (`F_entry = 1`) to `‖F^{k+1} − F^k‖_∞ < eps`; by Knaster–Tarski
/// this converges to the least fixpoint. Pure in its inputs so the M1-reduction test can drive it
/// with synthetic edges.
///
/// `edges_into[h]` = the incoming call-graph edges `(g, C_{g→h})`. Absent ⇒ no caller.
pub fn confirm_fixpoint(
    entry: u64,
    heads: &[u64],
    prior: &HashMap<u64, f64>,
    edges_into: &HashMap<u64, Vec<(u64, f64)>>,
    eps: f64,
    max_iter: usize,
) -> HashMap<u64, f64> {
    let mut f: HashMap<u64, f64> =
        heads.iter().map(|&h| (h, if h == entry { 1.0 } else { prior.get(&h).copied().unwrap_or(0.0) })).collect();

    for _ in 0..max_iter {
        let mut next = f.clone();
        let mut delta = 0.0f64;
        for &h in heads {
            if h == entry {
                next.insert(h, 1.0);
                continue;
            }
            let ph = prior.get(&h).copied().unwrap_or(0.0);
            let prod: f64 = edges_into
                .get(&h)
                .map(|es| es.iter().map(|&(g, c)| 1.0 - f.get(&g).copied().unwrap_or(0.0) * c).product())
                .unwrap_or(1.0);
            let v = 1.0 - (1.0 - ph) * prod;
            delta = delta.max((v - f.get(&h).copied().unwrap_or(0.0)).abs());
            next.insert(h, v);
        }
        f = next;
        if delta < eps {
            break;
        }
    }
    f
}

/// Eq (4) — per-instruction reachedness `R_a = 1 − ∏_{g : a ∈ body(g)} (1 − F_g·ρ(g,a))`, noisy-OR
/// over the functions containing `a`. `ρ = 1` (cleanly decoded body) here; the API keeps the door
/// open to a fall-through-decayed ρ. Absent address ⇒ not in any body ⇒ `R = 0`.
pub fn reachedness(bodies: &HashMap<u64, Vec<u64>>, f: &HashMap<u64, f64>) -> HashMap<u64, f64> {
    let mut prod: HashMap<u64, f64> = HashMap::new();
    for (g, body) in bodies {
        let fg = f.get(g).copied().unwrap_or(0.0);
        for &a in body {
            *prod.entry(a).or_insert(1.0) *= 1.0 - fg;
        }
    }
    prod.into_iter().map(|(a, p)| (a, 1.0 - p)).collect()
}

/// Tunables for [`build_soft_confirm`].
pub struct SoftConfig {
    /// Cap on a single function's byte span during intra-procedural descent.
    pub max_fn_span: usize,
    /// Fixpoint convergence tolerance (‖·‖_∞).
    pub eps: f64,
    /// Fixpoint iteration cap.
    pub max_iter: usize,
}

impl Default for SoftConfig {
    fn default() -> Self {
        Self { max_fn_span: 65536, eps: 1e-6, max_iter: 10_000 }
    }
}

/// The full soft-confirmation result: candidate heads, their features/priors, the call-graph edges
/// with noisy-OR evidence, per-head bodies, the fixpoint `F`, and per-instruction reachedness `R`.
pub struct SoftConfirm {
    /// Candidate heads (sorted) = `{entry} ∪ {direct-CALL targets}`.
    pub heads: Vec<u64>,
    /// Local features per head.
    pub features: HashMap<u64, HeadFeatures>,
    /// Local prior `prior_h` per head (eq 2).
    pub prior: HashMap<u64, f64>,
    /// Incoming call-graph edges `h → [(g, C_{g→h})]` (eq 1).
    pub edges_into: HashMap<u64, Vec<(u64, f64)>>,
    /// Intra-procedural body per head.
    pub bodies: HashMap<u64, Vec<u64>>,
    /// Confirmation fixpoint `F_h` (eq 3).
    pub f: HashMap<u64, f64>,
    /// Per-instruction reachedness `R_a` (eq 4).
    pub r: HashMap<u64, f64>,
}

/// Assemble and solve the M2 soft-confirmation model from a superset, entry, and the Layer-1
/// posteriors `pmap` (used as call-site evidence `π_c` in eq 1).
pub fn build_soft_confirm(
    sup: &Superset,
    entry: u64,
    pmap: &HashMap<u64, f64>,
    cfg: &SoftConfig,
) -> SoftConfirm {
    // The M2 model is exactly the M3a model with no resolved edges (self-edges are still excluded in
    // the shared builder; mutual recursion still confirms via the fixpoint).
    build_soft_confirm_resolved(sup, entry, pmap, &[], cfg)
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// Milestone 3a — INDIRECT-TARGET RESOLUTION (recover real call-graph edges from data).
//
// Theorem 2 (M2) proved the uncalled tail U = D ⊔ R_ind is *locally* indistinguishable, so the ONLY
// way to move a specific head out of the tail with per-head confidence is NON-LOCAL information: a
// *resolved* edge into it (LAYER2_M3_SPEC §0/§1). A resolver reads the binary's DATA — relocations,
// `.init_array`/`.fini_array` code pointers, and jump/function-pointer tables in read-only data — and
// proposes edges `g →_𝓡 t` with confidence `q`. These are REAL edges (a code pointer to `t` provably
// exists in the program image), not guesses. Folded into the eq-1 edge evidence (M3a-1) and re-run
// through the *same* M2 fixpoint, a resolved-real `t` gains a confirmed caller and moves from the tail
// (F≈β₀) to the core (F≈1) with earned confidence. Nothing overwrites a posterior — resolvers add
// STRUCTURE only.
//
// Precision is structural: the resolver reads code pointers that live in the benign program's own
// data, and none of them point at an appended code-in-data decoy (the decoy is unreferenced by
// construction), so the decoy is never resolved. `q < 1` for jump tables (bounds-recovery quality);
// `q ≈ 1` for relocation / section-semantic pointers.
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// The provenance of a resolved indirect edge (drives the confidence `q` and is reported for audit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolveKind {
    /// A `.init_array` entry: a constructor run by the runtime before `main` (reached indirectly).
    InitArray,
    /// A `.fini_array` entry: a destructor run by the runtime at exit (reached indirectly).
    FiniArray,
    /// An `R_X86_64_RELATIVE` relocation whose addend lands in `.text` (a PIE code pointer).
    Relocation,
    /// A run (length ≥ 2) of consecutive aligned code pointers in read-only data — a jump/switch
    /// table or vtable. Recovered by the bounded table scanner; `q` scaled by the run's validity.
    JumpTable,
    /// An isolated aligned code pointer in data (a stored function pointer / dispatch-table slot).
    DataPointer,
    // ── M4: code-anchored kinds — the table is found by following a DISPATCH INSTRUCTION (or a
    // vtable's section structure), not by blindly scanning every aligned word of data. Strictly more
    // precise than the blind scan: only tables an actual `jmp`/`lea` references are read. ──────────
    /// A computed-goto / dense-switch jump: an indirect `jmp *disp(,idx,8)` whose absolute 8-byte
    /// entry table (at `disp`) is read directly. The 8-byte-absolute form overlaps the blind
    /// [`ResolveKind::JumpTable`] scan on non-PIE binaries; code-anchoring re-tags those with their
    /// true provenance and reads only tables a real dispatch references.
    ComputedGoto,
    /// A PIE relative jump table: `lea reg, [rip+disp]` loads the table base, and each entry is a
    /// **4-byte signed self-relative offset** (`target = table_base + i32_entry`). The blind 8-byte
    /// scan CANNOT see these (wrong width, relative not absolute); this is the resolver's genuinely
    /// new coverage on position-independent code.
    PieRelJumpTable,
    /// A C++ vtable: a run of function pointers in `.data.rel.ro`, read with vtable structure in mind.
    Vtable,
}

/// Code-anchored kinds (M4) are found by following a dispatch instruction / section structure; they
/// take precedence when they land on the same `(g,t)` a blind data scan already proposed, because the
/// code-anchored provenance is the more precise explanation of the same real edge.
fn is_code_anchored(kind: ResolveKind) -> bool {
    matches!(kind, ResolveKind::ComputedGoto | ResolveKind::PieRelJumpTable | ResolveKind::Vtable)
}

/// A recovered indirect call-graph edge `g →_𝓡 t` with resolution confidence `q ∈ (0,1]`. `g` is the
/// confirmed source that reaches `t` non-locally; for data-anchored pointers (init/fini/reloc/table)
/// the runtime is the caller, modelled as an edge from the entry *root* (`F_entry = 1`).
#[derive(Debug, Clone, Copy)]
pub struct ResolvedEdge {
    pub g: u64,
    pub t: u64,
    pub q: f64,
    pub kind: ResolveKind,
}

/// Tunables for [`resolve_indirect`].
pub struct ResolveConfig {
    /// Confidence for a relocation / init-array / fini-array / isolated-pointer edge (`≈ 1`).
    pub q_pointer: f64,
    /// Confidence for a jump-table entry (`< 1`, bounds-recovery quality).
    pub q_jump_table: f64,
    /// A run of this many or more consecutive aligned code pointers is classed as a jump table.
    pub table_run_min: usize,
    /// Safety cap on entries read from a single contiguous table run.
    pub max_table_entries: usize,
    /// Run the M4 code-anchored resolvers (computed-goto, PIE-relative jump table, vtable) in addition
    /// to the data-anchored passes. `false` reproduces M3 exactly (data-anchored only) — the eval uses
    /// this to isolate the code-anchored contribution.
    pub code_anchored: bool,
}

impl Default for ResolveConfig {
    fn default() -> Self {
        Self { q_pointer: 0.99, q_jump_table: 0.9, table_run_min: 2, max_table_entries: 4096, code_anchored: true }
    }
}

/// Resolve indirect targets from an ELF's data and return call-graph edges `g → t` with confidence
/// `q` (LAYER2_M3_SPEC §1). Every target is validated against `sup` — it must be a valid instruction
/// start in the analyzed text — so junk offsets and pointers outside the code never enter the graph.
///
/// Sources, in order of certainty:
/// - `.init_array` / `.fini_array`: absolute 8-byte code pointers (non-PIE) → constructors/destructors.
/// - `R_X86_64_RELATIVE` relocations whose addend lands in text (PIE code pointers).
/// - read-only / writable data sections: aligned 8-byte code pointers. Consecutive runs (≥
///   `table_run_min`) are classed as jump/switch tables (`q_jump_table`); isolated pointers as stored
///   function pointers (`q_pointer`).
///
/// `entry` is the root source used for data-anchored edges (the runtime is the caller). Falls back to
/// program headers (non-executable `PT_LOAD`) when a stripped image has no section headers.
pub fn resolve_indirect(sup: &Superset, elf: &[u8], entry: u64, cfg: &ResolveConfig) -> Vec<ResolvedEdge> {
    use goblin::elf::section_header::{SHF_EXECINSTR, SHT_NOBITS};
    use goblin::elf::program_header::{PF_X, PT_LOAD};
    use goblin::elf::reloc::R_X86_64_RELATIVE;
    use goblin::Object;

    // Keyed by `(g,t)` so a later code-anchored pass can UPGRADE the provenance of an edge a blind
    // data scan already proposed (same real edge, more precise explanation) without duplicating it.
    let mut edge_map: HashMap<(u64, u64), ResolvedEdge> = HashMap::new();

    let Ok(Object::Elf(obj)) = Object::parse(elf) else {
        return Vec::new();
    };

    // A code pointer's target must land in EXECUTABLE text — the resolver ELF's own `.text` section
    // (or its executable `PT_LOAD` when headerless) — not in arbitrary data. This matters at the
    // real/decoy boundary: a `.rodata`→`.rodata` data pointer (a string/table reference) has a value
    // *past* the real text end, which the analyzed superset (with the decoy tiled there) would decode
    // as a "valid instruction". Restricting the target domain to real executable text rejects those
    // data references, so the decoy is never resolved.
    let (exe_lo, exe_hi) = elf_exec_range(&obj).unwrap_or((sup.base_addr, sup.base_addr + sup.bytes.len() as u64));
    // Additionally the target must decode to a valid instruction in the analyzed superset. This single
    // gate is the decoy discipline: an appended code-in-data decoy is tiled PAST `exe_hi` (the real
    // `.text` end), so no resolved target — data- OR code-anchored — can ever land in it.
    let valid = |v: u64| v >= exe_lo && v < exe_hi && sup.at(v).is_some();
    let push = |g: u64, t: u64, q: f64, kind: ResolveKind, m: &mut HashMap<(u64, u64), ResolvedEdge>| {
        if valid(t) && t != g {
            m.entry((g, t))
                .and_modify(|e| {
                    // A code-anchored kind upgrades a data-anchored one on the same real edge.
                    if !is_code_anchored(e.kind) && is_code_anchored(kind) {
                        e.kind = kind;
                        e.q = q;
                    }
                })
                .or_insert(ResolvedEdge { g, t, q, kind });
        }
    };

    // ── Relocations: R_X86_64_RELATIVE addend = a resolved vaddr (PIE code pointer). ──────────────
    for relocs in [&obj.dynrelas, &obj.dynrels, &obj.pltrelocs] {
        for r in relocs.iter() {
            if r.r_type == R_X86_64_RELATIVE {
                if let Some(a) = r.r_addend {
                    push(entry, a as u64, cfg.q_pointer, ResolveKind::Relocation, &mut edge_map);
                }
            }
        }
    }

    // Scan an aligned-u64 window of `data` (mapped at `vbase`) for code pointers, classifying
    // maximal consecutive runs as jump tables and isolated hits as stored function pointers.
    let scan_pointers = |data: &[u8], vbase: u64, kind_override: Option<ResolveKind>, m: &mut HashMap<(u64, u64), ResolvedEdge>| {
        let n = data.len() / 8;
        let mut i = 0usize;
        while i < n {
            let v = u64::from_le_bytes(data[i * 8..i * 8 + 8].try_into().unwrap());
            if !valid(v) {
                i += 1;
                continue;
            }
            // Extend a maximal run of consecutive valid pointers.
            let start = i;
            let mut run: Vec<u64> = Vec::new();
            while i < n && run.len() < cfg.max_table_entries {
                let vv = u64::from_le_bytes(data[i * 8..i * 8 + 8].try_into().unwrap());
                if valid(vv) {
                    run.push(vv);
                    i += 1;
                } else {
                    break;
                }
            }
            let kind = kind_override.unwrap_or(if run.len() >= cfg.table_run_min {
                ResolveKind::JumpTable
            } else {
                ResolveKind::DataPointer
            });
            let q = match kind {
                ResolveKind::JumpTable => cfg.q_jump_table,
                _ => cfg.q_pointer,
            };
            let _ = (start, vbase); // vbase kept for future per-site attribution
            for t in run {
                push(entry, t, q, kind, m);
            }
        }
    };

    // Data sections that legitimately hold code pointers a compiler emits (jump/switch tables,
    // vtables, function-pointer tables, constructor/destructor arrays, GOT). Deliberately an
    // ALLOWLIST: symbol tables, relocation tables, debug info and `.eh_frame` also contain code
    // addresses, but treating those as "the program's pointers" would be non-robust (they vanish on
    // strip) and would leak addresses no runtime pointer actually uses. Relocations are handled
    // structurally above; here we read only what the loaded program dereferences.
    const DATA_SECTIONS: &[&str] = &[
        ".rodata", ".data", ".data.rel.ro", ".init_array", ".fini_array", ".ctors", ".dtors",
        ".got", ".got.plt", ".tdata",
    ];
    if obj.section_headers.iter().any(|sh| !obj.shdr_strtab.get_at(sh.sh_name).unwrap_or("").is_empty()) {
        // Section headers present: scan only the allowlisted data sections.
        for sh in &obj.section_headers {
            if sh.sh_type == SHT_NOBITS || sh.sh_flags & u64::from(SHF_EXECINSTR) != 0 {
                continue;
            }
            let name = obj.shdr_strtab.get_at(sh.sh_name).unwrap_or("");
            if !DATA_SECTIONS.contains(&name) {
                continue;
            }
            let Some(range) = sh.file_range() else { continue };
            let Some(data) = elf.get(range) else { continue };
            let kind_override = match name {
                ".init_array" | ".ctors" => Some(ResolveKind::InitArray),
                ".fini_array" | ".dtors" => Some(ResolveKind::FiniArray),
                _ => None,
            };
            scan_pointers(data, sh.sh_addr, kind_override, &mut edge_map);
        }
    } else {
        // Stripped image: scan non-executable PT_LOAD segments.
        for ph in &obj.program_headers {
            if ph.p_type != PT_LOAD || ph.p_flags & PF_X != 0 {
                continue;
            }
            let start = ph.p_offset as usize;
            let end = start.saturating_add(ph.p_filesz as usize);
            let Some(data) = elf.get(start..end) else { continue };
            scan_pointers(data, ph.p_vaddr, None, &mut edge_map);
        }
    }

    // ── M4: code-anchored passes. Each follows a DISPATCH INSTRUCTION (or vtable structure) to the
    // exact table, so it reads only tables a real jmp/lea/vtable references — strictly more precise
    // than the blind data scan. Targets go through the SAME `valid` gate ⇒ the decoy stays unresolved.
    if cfg.code_anchored {
        // Read `len` bytes at a virtual address, via section headers (mapped image) or PT_LOAD.
        let read_vaddr = |vaddr: u64, len: usize| -> Option<Vec<u8>> {
            let hi = vaddr.checked_add(len as u64)?;
            for sh in &obj.section_headers {
                if sh.sh_type == SHT_NOBITS || sh.sh_addr == 0 {
                    continue;
                }
                if vaddr >= sh.sh_addr && hi <= sh.sh_addr + sh.sh_size {
                    let foff = (sh.sh_offset + (vaddr - sh.sh_addr)) as usize;
                    return elf.get(foff..foff + len).map(<[u8]>::to_vec);
                }
            }
            for ph in &obj.program_headers {
                if ph.p_type != PT_LOAD {
                    continue;
                }
                if vaddr >= ph.p_vaddr && hi <= ph.p_vaddr + ph.p_filesz {
                    let foff = (ph.p_offset + (vaddr - ph.p_vaddr)) as usize;
                    return elf.get(foff..foff + len).map(<[u8]>::to_vec);
                }
            }
            None
        };

        for insn in sup.iter_valid() {
            let off = (insn.address - sup.base_addr) as usize;
            let Some(ibytes) = sup.bytes.get(off..off + insn.size as usize) else { continue };

            // (1) Computed goto / dense switch: `jmp *disp(,idx,8)` → read the absolute 8-byte table.
            if insn.is_jump() && insn.mnemonic == "jmp" && insn.branch_target.is_none() {
                if let Some(table) = decode_indexed_jmp_table(ibytes) {
                    for k in 0..cfg.max_table_entries {
                        let Some(b) = read_vaddr(table + (k as u64) * 8, 8) else { break };
                        let v = u64::from_le_bytes(b.try_into().unwrap());
                        if !valid(v) {
                            break; // a switch table ends at its first non-code word
                        }
                        push(entry, v, cfg.q_jump_table, ResolveKind::ComputedGoto, &mut edge_map);
                    }
                }
            }

            // (2) PIE relative jump table: `lea reg,[rip+disp]` sets the table base; entries are 4-byte
            // signed self-relative offsets (`target = table_base + i32`). The blind 8-byte scan cannot
            // see these. A `lea` is common (address-of-anything), so we only emit when a RUN of
            // ≥ table_run_min consecutive entries all resolve to valid code — random data won't.
            if let Some(disp) = decode_lea_rip(ibytes) {
                let table = (insn.address + insn.size as u64).wrapping_add(disp as u64);
                if !valid(table) {
                    // table base is data, not code — the expected case for a real jump table.
                    let mut run: Vec<u64> = Vec::new();
                    for k in 0..cfg.max_table_entries {
                        let Some(b) = read_vaddr(table + (k as u64) * 4, 4) else { break };
                        let e = i32::from_le_bytes(b.try_into().unwrap());
                        let t = table.wrapping_add(e as i64 as u64);
                        if !valid(t) {
                            break;
                        }
                        run.push(t);
                    }
                    if run.len() >= cfg.table_run_min {
                        for t in run {
                            push(entry, t, cfg.q_jump_table, ResolveKind::PieRelJumpTable, &mut edge_map);
                        }
                    }
                }
            }
        }

        // (3) Vtables: runs of function pointers in `.data.rel.ro` (a C++ vtable's virtual-fn array).
        // Structurally the same words the blind scan reads there, but tagged with vtable provenance;
        // on PIE images the addresses arrive via `R_X86_64_RELATIVE` (handled above) and here we read
        // any surviving absolute pointers. Requires a run ≥ table_run_min (a lone slot is not a vtable).
        for sh in &obj.section_headers {
            if obj.shdr_strtab.get_at(sh.sh_name) != Some(".data.rel.ro") {
                continue;
            }
            let Some(range) = sh.file_range() else { continue };
            let Some(data) = elf.get(range) else { continue };
            let n = data.len() / 8;
            let mut i = 0usize;
            while i < n {
                let mut run: Vec<u64> = Vec::new();
                while i < n && run.len() < cfg.max_table_entries {
                    let v = u64::from_le_bytes(data[i * 8..i * 8 + 8].try_into().unwrap());
                    if valid(v) {
                        run.push(v);
                        i += 1;
                    } else {
                        break;
                    }
                }
                if run.len() >= cfg.table_run_min {
                    for t in run {
                        push(entry, t, cfg.q_pointer, ResolveKind::Vtable, &mut edge_map);
                    }
                } else if run.is_empty() {
                    i += 1;
                }
            }
        }
    }

    let mut edges: Vec<ResolvedEdge> = edge_map.into_values().collect();
    edges.sort_by_key(|e| (e.g, e.t));
    edges
}

/// Decode an indirect `jmp *disp32(,index,8)` (computed goto / dense switch dispatch) to its absolute
/// 8-byte-entry table base. Returns `None` unless the operand is exactly the no-base, scale-8 indexed
/// form a compiler emits for a jump table. Byte-level (not string) so it is robust to operand syntax.
fn decode_indexed_jmp_table(bytes: &[u8]) -> Option<u64> {
    let mut i = skip_prefixes(bytes);
    if *bytes.get(i)? != 0xff {
        return None; // group-5 opcode (jmp/call r/m)
    }
    i += 1;
    let modrm = *bytes.get(i)?;
    i += 1;
    let (md, reg, rm) = (modrm >> 6, (modrm >> 3) & 7, modrm & 7);
    if reg != 4 || rm != 4 {
        return None; // reg=/4 ⇒ jmp r/m; rm=100 ⇒ a SIB byte follows
    }
    let sib = *bytes.get(i)?;
    i += 1;
    let (scale, index, base) = (sib >> 6, (sib >> 3) & 7, sib & 7);
    if scale != 3 || index == 4 || md != 0 || base != 5 {
        return None; // scale=8, has an index, and [index*8 + disp32] with no base register
    }
    let disp = i32::from_le_bytes(bytes.get(i..i + 4)?.try_into().ok()?);
    Some((disp as i64) as u64)
}

/// Decode a `lea reg, [rip+disp32]` to its rip-relative displacement (added to the address of the NEXT
/// instruction by the caller). Returns `None` for any other operand form.
fn decode_lea_rip(bytes: &[u8]) -> Option<i64> {
    let mut i = skip_prefixes(bytes); // skips the REX.W lea in 64-bit code carries
    if *bytes.get(i)? != 0x8d {
        return None; // lea opcode
    }
    i += 1;
    let modrm = *bytes.get(i)?;
    i += 1;
    if modrm >> 6 != 0 || modrm & 7 != 5 {
        return None; // mod=00, rm=101 ⇒ [rip+disp32]
    }
    let disp = i32::from_le_bytes(bytes.get(i..i + 4)?.try_into().ok()?);
    Some(disp as i64)
}

/// Skip x86 legacy prefixes AND a REX prefix, returning the index of the opcode byte.
fn skip_prefixes(bytes: &[u8]) -> usize {
    let mut i = 0;
    while let Some(&b) = bytes.get(i) {
        // legacy prefix groups (segment/operand/address/lock/rep)
        if matches!(b, 0x66 | 0x67 | 0xf0 | 0xf2 | 0xf3 | 0x2e | 0x36 | 0x3e | 0x26 | 0x64 | 0x65) {
            i += 1;
        } else {
            break;
        }
    }
    if let Some(&b) = bytes.get(i) {
        if b & 0xf0 == 0x40 {
            i += 1; // REX
        }
    }
    i
}

/// The executable-code vaddr range of an ELF: the `.text` section when section headers survive, else
/// the first executable `PT_LOAD` segment. A resolved code pointer must target this range.
fn elf_exec_range(obj: &goblin::elf::Elf) -> Option<(u64, u64)> {
    use goblin::elf::program_header::{PF_X, PT_LOAD};
    for sh in &obj.section_headers {
        if obj.shdr_strtab.get_at(sh.sh_name) == Some(".text") {
            return Some((sh.sh_addr, sh.sh_addr + sh.sh_size));
        }
    }
    obj.program_headers
        .iter()
        .find(|p| p.p_type == PT_LOAD && p.p_flags & PF_X != 0)
        .map(|p| (p.p_vaddr, p.p_vaddr + p.p_memsz))
}

/// Fold resolved edges into an existing `edges_into` map via M3a-1: `C_{g→t} ← 1 − (1−C)(1−q)`
/// (noisy-OR merge on the same `(g,t)`, or a new incoming edge). Returns the set of targets that were
/// *newly introduced* as heads (had no prior incoming edge) so the caller can add them as candidates.
fn fold_resolved(
    edges_into: &mut HashMap<u64, Vec<(u64, f64)>>,
    resolved: &[ResolvedEdge],
) {
    for e in resolved {
        let ins = edges_into.entry(e.t).or_default();
        if let Some(slot) = ins.iter_mut().find(|(g, _)| *g == e.g) {
            slot.1 = 1.0 - (1.0 - slot.1) * (1.0 - e.q); // noisy-OR on the same caller
        } else {
            ins.push((e.g, e.q));
        }
    }
}

/// [`build_soft_confirm`] extended with M3a resolved edges. With `resolved = &[]` it is byte-identical
/// to `build_soft_confirm` (same heads, edges, fixpoint). With resolved edges it (i) adds each resolved
/// target as a candidate head (with a shape-only prior + body + reachedness), (ii) folds the edges into
/// the eq-1 evidence (M3a-1), and (iii) re-runs the *same* eq-3 fixpoint. Resolved-real targets thereby
/// gain a confirmed caller and rise from the tail to the core.
pub fn build_soft_confirm_resolved(
    sup: &Superset,
    entry: u64,
    pmap: &HashMap<u64, f64>,
    resolved: &[ResolvedEdge],
    cfg: &SoftConfig,
) -> SoftConfirm {
    // Candidate heads = entry ∪ direct-CALL targets ∪ resolved targets; tally incoming direct calls.
    let mut head_set: HashSet<u64> = HashSet::new();
    if sup.at(entry).is_some() {
        head_set.insert(entry);
    }
    let mut incoming: HashMap<u64, usize> = HashMap::new();
    for insn in sup.iter_valid() {
        if insn.is_call() {
            if let Some(t) = insn.branch_target {
                if sup.at(t).is_some() {
                    head_set.insert(t);
                    *incoming.entry(t).or_insert(0) += 1;
                }
            }
        }
    }
    for e in resolved {
        if sup.at(e.t).is_some() {
            head_set.insert(e.t);
        }
    }
    let mut heads: Vec<u64> = head_set.iter().copied().collect();
    heads.sort_unstable();

    // Bodies + call-site posteriors per (g → h) edge (self-edges excluded — see build_soft_confirm).
    let mut bodies: HashMap<u64, Vec<u64>> = HashMap::new();
    let mut site_pis: HashMap<(u64, u64), Vec<f64>> = HashMap::new();
    for &g in &heads {
        let func = extract_function(sup, g, &head_set, cfg.max_fn_span);
        for &a in &func.body {
            if let Some(insn) = sup.at(a) {
                if insn.is_call() {
                    if let Some(h) = insn.branch_target {
                        if h != g && head_set.contains(&h) {
                            site_pis.entry((g, h)).or_default().push(pmap.get(&a).copied().unwrap_or(0.0));
                        }
                    }
                }
            }
        }
        bodies.insert(g, func.body);
    }
    let mut edges_into: HashMap<u64, Vec<(u64, f64)>> = HashMap::new();
    for ((g, h), pis) in &site_pis {
        edges_into.entry(*h).or_default().push((*g, edge_evidence(pis)));
    }
    // M3a-1: fold the resolved edges into the evidence.
    fold_resolved(&mut edges_into, resolved);

    // Features + prior.
    let mut features: HashMap<u64, HeadFeatures> = HashMap::new();
    let mut prior: HashMap<u64, f64> = HashMap::new();
    for &h in &heads {
        let ft = HeadFeatures {
            is_entry: h == entry,
            prologue: looks_like_prologue(sup, h),
            n_callsites: incoming.get(&h).copied().unwrap_or(0),
        };
        prior.insert(h, local_prior(&ft));
        features.insert(h, ft);
    }

    let f = confirm_fixpoint(entry, &heads, &prior, &edges_into, cfg.eps, cfg.max_iter);
    let r = reachedness(&bodies, &f);
    SoftConfirm { heads, features, prior, edges_into, bodies, f, r }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// Milestone 3b — per-binary calibrated base rate β̂₀(b) for the residual exchangeable tail.
//
// After M3a, the residual tail U'(b) still holds indirect-only real code we could NOT resolve, mixed
// with decoy. Theorem 2 ⇒ those heads are exchangeable (same local law), so per-head confidence finer
// than the group base rate β₀(b) is false precision. We estimate β₀(b) from BINARY-LEVEL features
// ψ(b) with a calibrated group regressor (logistic + isotonic), and assign every residual-tail head
// F_h = β̂₀(b). Theorem 3: if β̂₀ is a calibrated estimate of the group base rate, the residual-tail
// confidences are aggregate-calibrated (LAYER2_M3_SPEC §2).
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// Binary-level features ψ(b) predictive of the residual-tail real fraction β₀(b). All are observable
/// at inference (no labels): they measure how much genuine, unresolved indirect control flow the
/// binary has — more of it ⇒ more real functions hiding in the exchangeable tail.
#[derive(Debug, Clone, Copy, Default)]
pub struct Beta0Features {
    /// |residual tail| / |candidate heads| — how large the exchangeable remainder is.
    pub tail_frac: f64,
    /// Unresolved indirect call/jump sites per confirmed-core instruction (genuine indirect flow we
    /// could not statically resolve — a proxy for hidden real callees).
    pub unresolved_indirect_density: f64,
    /// Resolved code-pointer targets per candidate head (how pointer-heavy the binary's data is).
    pub data_pointer_density: f64,
    /// Mean local prior over the residual tail (shape evidence the tail carries).
    pub mean_tail_prior: f64,
}

impl Beta0Features {
    /// Feature vector in a fixed order, for the regressor.
    pub fn to_vec(&self) -> Vec<f64> {
        vec![self.tail_frac, self.unresolved_indirect_density, self.data_pointer_density, self.mean_tail_prior]
    }
}

/// Compute ψ(b) from a solved (post-M3a) soft-confirmation model, the resolved-edge count, and the
/// number of unresolved indirect call/jump sites in the confirmed core. A head is in the residual
/// tail when no *confirmed* caller reaches it (max_g F_g·C < 0.5).
pub fn beta0_features(
    sc: &SoftConfirm,
    entry: u64,
    n_resolved: usize,
    unresolved_indirect_sites: usize,
) -> Beta0Features {
    let mut tail = 0usize;
    let mut core_insns = 0usize;
    let mut tail_prior_sum = 0.0;
    for &h in &sc.heads {
        if h == entry {
            core_insns += sc.bodies.get(&h).map(|b| b.len()).unwrap_or(0);
            continue;
        }
        let called = sc
            .edges_into
            .get(&h)
            .map(|es| es.iter().any(|&(g, c)| sc.f.get(&g).copied().unwrap_or(0.0) * c >= 0.5))
            .unwrap_or(false);
        if called {
            core_insns += sc.bodies.get(&h).map(|b| b.len()).unwrap_or(0);
        } else {
            tail += 1;
            tail_prior_sum += sc.prior.get(&h).copied().unwrap_or(0.0);
        }
    }
    let n_heads = sc.heads.len().max(1) as f64;
    Beta0Features {
        tail_frac: tail as f64 / n_heads,
        unresolved_indirect_density: unresolved_indirect_sites as f64 / core_insns.max(1) as f64,
        data_pointer_density: n_resolved as f64 / n_heads,
        mean_tail_prior: if tail > 0 { tail_prior_sum / tail as f64 } else { 0.0 },
    }
}

/// A calibrated group-level regressor `β̂₀(b) = h(ψ(b))` — a standardized logistic fit by
/// L2-regularized gradient descent, then isotonic-recalibrated against the observed β₀ (Theorem 3's
/// calibration hypothesis). Trained on `(ψ(b), β₀(b))` rows across binaries.
pub struct Beta0Model {
    mean: Vec<f64>,
    std: Vec<f64>,
    w: Vec<f64>,
    b: f64,
    iso: evalkit::IsotonicMap,
}

impl Beta0Model {
    /// Fit on `(ψ, β₀)` rows. `l2` regularizes the logistic (n is tiny — a few binaries), and the
    /// isotonic map recalibrates the logistic output to the observed group base rate.
    pub fn fit(rows: &[(Vec<f64>, f64)], l2: f64, iters: usize) -> Self {
        let d = rows.first().map(|r| r.0.len()).unwrap_or(0);
        let n = rows.len().max(1) as f64;
        let mean: Vec<f64> = (0..d).map(|j| rows.iter().map(|r| r.0[j]).sum::<f64>() / n).collect();
        let std: Vec<f64> = (0..d)
            .map(|j| (rows.iter().map(|r| (r.0[j] - mean[j]).powi(2)).sum::<f64>() / n).sqrt().max(1e-9))
            .collect();
        let z = |x: &[f64]| -> Vec<f64> { (0..d).map(|j| (x[j] - mean[j]) / std[j]).collect() };

        let (mut w, mut b) = (vec![0.0f64; d], 0.0f64);
        let lr = 0.3;
        for _ in 0..iters {
            let mut gw = vec![0.0f64; d];
            let mut gb = 0.0f64;
            for (x, y) in rows {
                let zx = z(x);
                let p = 1.0 / (1.0 + (-(zx.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>() + b)).exp());
                let e = p - y;
                for j in 0..d {
                    gw[j] += e * zx[j];
                }
                gb += e;
            }
            for j in 0..d {
                w[j] -= lr * (gw[j] / n + l2 * w[j]);
            }
            b -= lr * gb / n;
        }
        // Isotonic recalibration of the logistic output vs the observed β₀ (calibration hypothesis).
        let mut model = Beta0Model { mean, std, w, b, iso: evalkit::IsotonicMap::fit(&[]) };
        let samples: Vec<(f64, f64)> = rows.iter().map(|(x, y)| (model.raw(x), *y)).collect();
        model.iso = evalkit::IsotonicMap::fit(&samples);
        model
    }

    /// The raw (pre-isotonic) logistic output.
    fn raw(&self, x: &[f64]) -> f64 {
        let d = self.w.len();
        let z: f64 = (0..d).map(|j| (x[j] - self.mean[j]) / self.std[j] * self.w[j]).sum();
        1.0 / (1.0 + (-(z + self.b)).exp())
    }

    /// The calibrated β̂₀(b) for a feature vector ψ(b).
    pub fn apply(&self, x: &[f64]) -> f64 {
        self.iso.apply(self.raw(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hand-assembled x86-64 so the superset has a known CFG. Layout (base 0x1000):
    //   0x1000 entry:  jne +? (2 bytes)  75 06      → cond jump to C (0x1008), fall-through to A
    //   0x1002 A:      nop              90          → fall-through to B
    //   0x1003 B:      jmp +3           eb 03       → unconditional jump to a valid target, no F-T
    //   0x1005 (pad)   nop x3           90 90 90    → jmp lands at 0x1008 = C
    //   0x1008 C:      ret              c3
    // We anchor the entry manually; C is only reachable via the jne/jmp target edges.
    //
    // The reachability contract we assert here:
    //   r[entry] = 1 (anchor); r[A] = fall_decay (one F-T hop from entry);
    //   r[B] = fall_decay^2 (two F-T hops); C reached by a *jump target* so r[C] = 1 up to the
    //   non-decaying branch edge from entry's taken side (jne target = 0x1008 = C directly → r=1).
    fn build(bytes: &[u8]) -> Superset {
        Superset::new(0x1000, bytes).expect("superset build")
    }

    #[test]
    fn fall_through_decays_jump_target_refreshes() {
        // entry jne C ; A nop ; B jmp +? ; ret at C. Craft so jne target == C == 0x1008.
        // 0x1000: 75 06        jne 0x1008
        // 0x1002: 90           nop            (A)
        // 0x1003: 90           nop            (B)
        // 0x1004: 90           nop
        // 0x1005: 90           nop
        // 0x1006: 90           nop
        // 0x1007: 90           nop
        // 0x1008: c3           ret            (C)
        let bytes = [0x75, 0x06, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0xc3];
        let sup = build(&bytes);
        let cfg = ReachConfig { fall_decay: 0.9, anchors_calls_only: true, max_iters: 10_000 };
        let r = reachability(&sup, 0x1000, &cfg);

        // entry anchored.
        assert_eq!(r[&0x1000], 1.0);
        // A = 0x1002 reached by one fall-through hop from entry.
        assert!((r[&0x1002] - 0.9).abs() < 1e-9, "A: {:?}", r.get(&0x1002));
        // B = 0x1003 reached by two fall-through hops.
        assert!((r[&0x1003] - 0.81).abs() < 1e-9, "B: {:?}", r.get(&0x1003));
        // C = 0x1008 reached by the jne *target* edge (non-decaying) → full 1.0.
        assert!((r[&0x1008] - 1.0).abs() < 1e-9, "C: {:?}", r.get(&0x1008));
    }

    #[test]
    fn isolated_instruction_is_unreached() {
        // A single reachable chain plus an isolated instruction the entry can't reach.
        // 0x1000: c3           ret            (entry, no successors)
        // 0x1001: 90           nop            (isolated D — nothing reaches it)
        let bytes = [0xc3, 0x90];
        let sup = build(&bytes);
        let cfg = ReachConfig::default();
        let r = reachability(&sup, 0x1000, &cfg);
        assert_eq!(r[&0x1000], 1.0);
        // D absent (or 0) — the entry `ret` has no fall-through, so D is never reached.
        assert_eq!(r.get(&0x1001).copied().unwrap_or(0.0), 0.0);
    }

    #[test]
    fn ret_has_no_fall_through_successor() {
        // entry ret at 0x1000; the byte at 0x1001 must NOT inherit reachability, because
        // `Superset::successors_of` returns no fall-through edge for `ret`.
        // 0x1000: c3           ret
        // 0x1001: 90           nop      (would-be fall-through of the ret)
        let bytes = [0xc3, 0x90];
        let sup = build(&bytes);
        let r = reachability(&sup, 0x1000, &ReachConfig::default());
        assert!(
            r.get(&0x1001).copied().unwrap_or(0.0) < 1e-9,
            "ret fall-through leaked: {:?}",
            r.get(&0x1001)
        );
    }

    #[test]
    fn call_target_is_anchored_as_function_head() {
        // A direct call anchors its target to r=1 even with the entry elsewhere.
        // 0x1000: e8 03 00 00 00   call 0x1008   (5 bytes) → target 0x1008
        // 0x1005: 90               nop           (fall-through/return point)
        // 0x1006: 90               nop
        // 0x1007: 90               nop
        // 0x1008: c3               ret           (call target = function head)
        let bytes = [0xe8, 0x03, 0x00, 0x00, 0x00, 0x90, 0x90, 0x90, 0xc3];
        let sup = build(&bytes);
        // Anchor from a bogus entry so the ONLY way 0x1008 becomes 1.0 is the call-target anchor.
        let r = reachability(&sup, 0x1000, &ReachConfig::default());
        assert_eq!(r[&0x1008], 1.0, "call target not anchored: {:?}", r.get(&0x1008));
    }

    #[test]
    fn anchors_all_lights_jump_targets() {
        // With anchors_calls_only=false, a jump target self-anchors to 1.0 even if the jumping
        // instruction is itself unreachable — the ablation that leaks more.
        // 0x1000: c3            ret               (entry, dead-ends immediately)
        // 0x1001: eb 03         jmp 0x1006        (unreachable from entry, but its target anchors)
        // 0x1003: 90 90 90      nop x3
        // 0x1006: c3            ret               (jump target)
        let bytes = [0xc3, 0xeb, 0x03, 0x90, 0x90, 0x90, 0xc3];
        let sup = build(&bytes);
        let calls_only = reachability(&sup, 0x1000, &ReachConfig::default());
        assert_eq!(
            calls_only.get(&0x1006).copied().unwrap_or(0.0),
            0.0,
            "calls-only should not anchor a jump target"
        );
        let all = reachability(
            &sup,
            0x1000,
            &ReachConfig { anchors_calls_only: false, ..ReachConfig::default() },
        );
        assert_eq!(all[&0x1006], 1.0, "anchors-all should anchor the jump target");
    }

    // ── Function confirmation. ────────────────────────────────────────────────────────────────────
    //
    // Hand-assembled superset (base 0x1000) that encodes the three behaviors the milestone hinges on:
    // a direct-call chain (confirmed), an indirect-only callee (disconnected → excluded), and a
    // self-calling island decoy (disconnected → excluded).
    //
    //   0x1000 E: e8 03 00 00 00   call 0x1008 (F)    ← direct call, confirms F
    //   0x1005    ff d0            call rax           ← indirect call to "G": no static target
    //   0x1007    c3               ret                ← E path ends
    //   0x1008 F: c3               ret                ← F is a direct-call target (a real head)
    //   0x1009 G: 90               nop                ← reached only indirectly → not a candidate head
    //   0x100a    c3               ret
    //   0x100b D: e8 fb ff ff ff   call 0x100b (self) ← island decoy: a head, but nothing confirmed
    //   0x1010    c3               ret                   calls it → never confirmed
    const CONFIRM_BYTES: [u8; 17] = [
        0xe8, 0x03, 0x00, 0x00, 0x00, // 0x1000 call 0x1008 (F)
        0xff, 0xd0, // 0x1005 call rax (indirect → G)
        0xc3, // 0x1007 ret
        0xc3, // 0x1008 F: ret
        0x90, // 0x1009 G: nop
        0xc3, // 0x100a ret
        0xe8, 0xfb, 0xff, 0xff, 0xff, // 0x100b D: call 0x100b (self)
        0xc3, // 0x1010 ret
    ];

    #[test]
    fn extract_function_stops_at_ret_and_records_direct_call() {
        let sup = build(&CONFIRM_BYTES);
        let heads: HashSet<u64> = [0x1000, 0x1008, 0x100b].into_iter().collect();
        let f = extract_function(&sup, 0x1000, &heads, 0x10000);
        // E's body is exactly its three instructions; it stops at the ret (never reaches F/G/D).
        let body: HashSet<u64> = f.body.iter().copied().collect();
        assert_eq!(body, [0x1000, 0x1005, 0x1007].into_iter().collect());
        // The direct call to F is recorded; the indirect `call rax` (no static target) is not.
        assert_eq!(f.calls, vec![0x1008]);
    }

    #[test]
    fn confirm_excludes_indirect_only_and_self_call_island() {
        let sup = build(&CONFIRM_BYTES);
        let conf = confirm_from_entry(&sup, 0x1000, &ConfirmConfig::default());

        // The island decoy D is a *candidate* head (a direct-call target) but never confirmed.
        assert!(conf.all_heads.contains(&0x100b), "D should be a candidate head");
        assert!(!conf.confirmed_heads.contains(&0x100b), "D must NOT be confirmed");

        // Only E and F are confirmed: G is indirect-only, D is disconnected.
        assert_eq!(conf.confirmed_heads, [0x1000, 0x1008].into_iter().collect());

        // confirmed_insns covers E- and F-bodies …
        for a in [0x1000u64, 0x1005, 0x1007, 0x1008] {
            assert!(conf.confirmed_insns.contains(&a), "confirmed_insns missing {a:#x}");
        }
        // … and excludes the indirect-only G and the island D.
        assert!(!conf.confirmed_insns.contains(&0x1009), "G (indirect-only) leaked into confirmed");
        assert!(!conf.confirmed_insns.contains(&0x100b), "D (island) leaked into confirmed");
        assert!(!conf.confirmed_insns.contains(&0x1010), "D body leaked into confirmed");
    }

    // ── M2: calibrated confirmation (eqs 1–4). ─────────────────────────────────────────────────────

    #[test]
    fn edge_evidence_is_noisy_or() {
        assert!((edge_evidence(&[]) - 0.0).abs() < 1e-12, "no sites ⇒ 0");
        assert!((edge_evidence(&[1.0]) - 1.0).abs() < 1e-12, "certain site ⇒ 1");
        // two independent 0.5 sites: 1 − 0.5·0.5 = 0.75.
        assert!((edge_evidence(&[0.5, 0.5]) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn fixpoint_reduces_to_m1_boolean_result() {
        // Eq (3) with prior ≡ 0 (h≠entry) and C thresholded to {0,1} MUST equal M1's Boolean
        // transitive confirmation from the entry. Build both from the same superset and compare.
        let sup = build(&CONFIRM_BYTES);
        let entry = 0x1000;
        let pmap: HashMap<u64, f64> = HashMap::new(); // π unused in the reduction
        let sc = build_soft_confirm(&sup, entry, &pmap, &SoftConfig::default());

        // Boolean reduction: prior = 0 for all non-entry heads; every real edge → C = 1.
        let prior0: HashMap<u64, f64> = sc.heads.iter().map(|&h| (h, 0.0)).collect();
        let edges_bool: HashMap<u64, Vec<(u64, f64)>> = sc
            .edges_into
            .iter()
            .map(|(&h, es)| (h, es.iter().map(|&(g, _)| (g, 1.0)).collect()))
            .collect();
        let fb = confirm_fixpoint(entry, &sc.heads, &prior0, &edges_bool, 1e-9, 1000);

        let m1 = confirm_from_entry(&sup, entry, &ConfirmConfig::default());
        let boolean_confirmed: HashSet<u64> =
            fb.iter().filter(|&(_, &v)| v > 0.5).map(|(&h, _)| h).collect();
        assert_eq!(
            boolean_confirmed, m1.confirmed_heads,
            "eq-3 fixpoint (prior=0, C∈{{0,1}}) must reduce to M1 Boolean confirmation"
        );
    }

    #[test]
    fn soft_confirm_lights_core_leaves_island_low() {
        // With a real prior + high call-site posteriors, the entry-rooted core (E→F) confirms near 1,
        // while the disconnected self-call island D stays at its bare local prior (no confirmed caller).
        let sup = build(&CONFIRM_BYTES);
        // High π everywhere ⇒ strong edge evidence.
        let pmap: HashMap<u64, f64> = (0x1000u64..=0x1010).map(|a| (a, 0.99)).collect();
        let sc = build_soft_confirm(&sup, 0x1000, &pmap, &SoftConfig::default());

        assert!(sc.f[&0x1000] > 0.99, "entry pinned");
        assert!(sc.f[&0x1008] > 0.9, "F (direct-called) confirmed high: {}", sc.f[&0x1008]);
        // D is a candidate head (self-call target) but no CONFIRMED function calls it → F ≈ prior,
        // which is low (its self-edge multiplies by its own low F, so it can't bootstrap).
        assert!(sc.f[&0x100b] < 0.6, "island D must stay low-confidence: {}", sc.f[&0x100b]);
        // Reachedness of a confirmed-body insn is high; of the island body, low.
        assert!(sc.r[&0x1008] > 0.9, "confirmed insn reachedness high");
        assert!(sc.r.get(&0x1010).copied().unwrap_or(0.0) < 0.6, "island insn reachedness low");
    }

    // ── M3a: indirect-target resolution. ─────────────────────────────────────────────────────────

    #[test]
    fn resolved_build_with_no_edges_is_identical_to_m2() {
        // build_soft_confirm_resolved(&[]) MUST reproduce build_soft_confirm exactly (heads, F, R).
        let sup = build(&CONFIRM_BYTES);
        let pmap: HashMap<u64, f64> = (0x1000u64..=0x1010).map(|a| (a, 0.9)).collect();
        let m2 = build_soft_confirm(&sup, 0x1000, &pmap, &SoftConfig::default());
        let m3 = build_soft_confirm_resolved(&sup, 0x1000, &pmap, &[], &SoftConfig::default());
        assert_eq!(m2.heads, m3.heads);
        for &h in &m2.heads {
            assert!((m2.f[&h] - m3.f[&h]).abs() < 1e-12, "F mismatch at {h:#x}");
        }
    }

    #[test]
    fn synthetic_resolved_edge_lifts_the_island_from_the_tail() {
        // The self-call island D (0x100b) sits at its bare prior in M2 (no confirmed caller). A
        // resolved edge from the entry root (a recovered code pointer to D) MUST lift it to the core.
        let sup = build(&CONFIRM_BYTES);
        let pmap: HashMap<u64, f64> = (0x1000u64..=0x1010).map(|a| (a, 0.9)).collect();
        let base = build_soft_confirm(&sup, 0x1000, &pmap, &SoftConfig::default());
        assert!(base.f[&0x100b] < 0.6, "precondition: D low in M2");

        let resolved = [ResolvedEdge { g: 0x1000, t: 0x100b, q: 0.99, kind: ResolveKind::InitArray }];
        let m3 = build_soft_confirm_resolved(&sup, 0x1000, &pmap, &resolved, &SoftConfig::default());
        // Entry F = 1, so the folded edge C = 0.99 confirms D near 0.99.
        assert!(m3.f[&0x100b] > 0.9, "resolved edge must lift D to the core: {}", m3.f[&0x100b]);
        // Other heads unchanged (the edge is strictly additive structure).
        assert!(m3.f[&0x1008] >= base.f[&0x1008] - 1e-9, "F must be monotone under added edges");
    }

    #[test]
    fn resolve_indirect_on_headerless_code_only_elf_finds_nothing() {
        // A code-only minimal ELF (no data segment) has no code pointers to resolve — resolve_indirect
        // must parse it and return an empty edge set (not panic).
        let elf = evalkit::build_min_elf(0x1000, &CONFIRM_BYTES);
        let sup = build(&CONFIRM_BYTES);
        let edges = resolve_indirect(&sup, &elf, 0x1000, &ResolveConfig::default());
        assert!(edges.is_empty(), "code-only ELF should yield no resolved edges, got {}", edges.len());
    }

    // ── M4: code-anchored resolvers. ───────────────────────────────────────────────────────────────
    //
    // Each resolver is proven on a CRAFTED specimen that exercises exactly its idiom. A headerless
    // two-`PT_LOAD` ELF (executable text + read-only data) suffices for the instruction-anchored
    // resolvers; the vtable resolver additionally needs a named `.data.rel.ro` section.

    /// A headerless ELF with one R|X text segment (`text` at `tvaddr`) and one R data segment (`data`
    /// at `dvaddr`) — enough for `resolve_indirect`'s program-header path.
    fn two_seg_elf(tvaddr: u64, text: &[u8], dvaddr: u64, data: &[u8]) -> Vec<u8> {
        const EHDR: u64 = 64;
        const PHDR: u64 = 56;
        let t_off = EHDR + 2 * PHDR;
        let d_off = t_off + text.len() as u64;
        let mut e = Vec::new();
        e.extend_from_slice(&[0x7f, b'E', b'L', b'F', 2, 1, 1, 0]);
        e.extend_from_slice(&[0u8; 8]);
        e.extend_from_slice(&2u16.to_le_bytes()); // ET_EXEC
        e.extend_from_slice(&62u16.to_le_bytes()); // EM_X86_64
        e.extend_from_slice(&1u32.to_le_bytes());
        e.extend_from_slice(&tvaddr.to_le_bytes()); // e_entry
        e.extend_from_slice(&EHDR.to_le_bytes()); // e_phoff
        e.extend_from_slice(&0u64.to_le_bytes()); // e_shoff = none
        e.extend_from_slice(&0u32.to_le_bytes());
        e.extend_from_slice(&(EHDR as u16).to_le_bytes());
        e.extend_from_slice(&(PHDR as u16).to_le_bytes());
        e.extend_from_slice(&2u16.to_le_bytes()); // e_phnum = 2
        e.extend_from_slice(&0u16.to_le_bytes());
        e.extend_from_slice(&0u16.to_le_bytes());
        e.extend_from_slice(&0u16.to_le_bytes());
        let phdr = |flags: u32, off: u64, vaddr: u64, sz: u64, e: &mut Vec<u8>| {
            e.extend_from_slice(&1u32.to_le_bytes()); // PT_LOAD
            e.extend_from_slice(&flags.to_le_bytes());
            e.extend_from_slice(&off.to_le_bytes());
            e.extend_from_slice(&vaddr.to_le_bytes());
            e.extend_from_slice(&vaddr.to_le_bytes());
            e.extend_from_slice(&sz.to_le_bytes());
            e.extend_from_slice(&sz.to_le_bytes());
            e.extend_from_slice(&0x1000u64.to_le_bytes());
        };
        phdr(5, t_off, tvaddr, text.len() as u64, &mut e); // R|X
        phdr(4, d_off, dvaddr, data.len() as u64, &mut e); // R
        e.extend_from_slice(text);
        e.extend_from_slice(data);
        e
    }

    #[test]
    fn computed_goto_resolves_absolute_switch_table() {
        // A dense-switch dispatch `jmp *0x2000(,%rax,8)` and a 2-entry absolute table at 0x2000 that
        // points at two `ret`s in text. The code-anchored pass must recover BOTH targets, tagged
        // ComputedGoto (it re-tags the blind scan's JumpTable on the same edge).
        //   0x1000: ff 24 c5 00 20 00 00   jmp *0x2000(,%rax,8)
        //   0x1007: c3                     ret   (case A)
        //   0x1008: c3                     ret   (case B)
        let text = [0xff, 0x24, 0xc5, 0x00, 0x20, 0x00, 0x00, 0xc3, 0xc3];
        let mut data = Vec::new();
        data.extend_from_slice(&0x1007u64.to_le_bytes());
        data.extend_from_slice(&0x1008u64.to_le_bytes());
        let elf = two_seg_elf(0x1000, &text, 0x2000, &data);
        let sup = build(&text);
        let edges = resolve_indirect(&sup, &elf, 0x1000, &ResolveConfig::default());

        let cg: HashSet<u64> =
            edges.iter().filter(|e| e.kind == ResolveKind::ComputedGoto).map(|e| e.t).collect();
        assert_eq!(cg, [0x1007, 0x1008].into_iter().collect(), "computed-goto targets: {edges:?}");

        // With code-anchoring OFF (M3), the same table is still found by the blind 8-byte scan, but
        // NOT tagged ComputedGoto — that provenance is the M4 resolver's contribution.
        let m3 = resolve_indirect(
            &sup,
            &elf,
            0x1000,
            &ResolveConfig { code_anchored: false, ..ResolveConfig::default() },
        );
        assert!(
            m3.iter().all(|e| e.kind != ResolveKind::ComputedGoto),
            "data-only mode must not emit ComputedGoto"
        );
        assert!(
            m3.iter().any(|e| e.t == 0x1007),
            "the absolute table is still visible to the blind scan in M3"
        );
    }

    /// An ELF with section headers: `.text` (exec) at `tvaddr` and one named data section (`sec_name`,
    /// ALLOC|WRITE) at `svaddr`. Enough for `elf_exec_range` (`.text`), the blind allowlist scan, and
    /// the vtable pass (which keys on the `.data.rel.ro` section name).
    fn sectioned_elf(tvaddr: u64, text: &[u8], sec_name: &str, svaddr: u64, sec: &[u8]) -> Vec<u8> {
        let text_off = 64u64;
        let sec_off = text_off + text.len() as u64;
        // .shstrtab: "\0.text\0<sec_name>\0.shstrtab\0"
        let mut shstr = vec![0u8];
        let name_text = shstr.len() as u32;
        shstr.extend_from_slice(b".text\0");
        let name_sec = shstr.len() as u32;
        shstr.extend_from_slice(sec_name.as_bytes());
        shstr.push(0);
        let name_shstr = shstr.len() as u32;
        shstr.extend_from_slice(b".shstrtab\0");
        let shstr_off = sec_off + sec.len() as u64;
        let shoff = shstr_off + shstr.len() as u64;

        let mut e = Vec::new();
        e.extend_from_slice(&[0x7f, b'E', b'L', b'F', 2, 1, 1, 0]);
        e.extend_from_slice(&[0u8; 8]);
        e.extend_from_slice(&2u16.to_le_bytes()); // ET_EXEC
        e.extend_from_slice(&62u16.to_le_bytes()); // EM_X86_64
        e.extend_from_slice(&1u32.to_le_bytes());
        e.extend_from_slice(&tvaddr.to_le_bytes()); // e_entry
        e.extend_from_slice(&0u64.to_le_bytes()); // e_phoff = none
        e.extend_from_slice(&shoff.to_le_bytes()); // e_shoff
        e.extend_from_slice(&0u32.to_le_bytes());
        e.extend_from_slice(&64u16.to_le_bytes()); // e_ehsize
        e.extend_from_slice(&0u16.to_le_bytes()); // e_phentsize
        e.extend_from_slice(&0u16.to_le_bytes()); // e_phnum
        e.extend_from_slice(&64u16.to_le_bytes()); // e_shentsize
        e.extend_from_slice(&4u16.to_le_bytes()); // e_shnum = null,.text,sec,.shstrtab
        e.extend_from_slice(&3u16.to_le_bytes()); // e_shstrndx
        debug_assert_eq!(e.len(), 64);
        e.extend_from_slice(text);
        e.extend_from_slice(sec);
        e.extend_from_slice(&shstr);

        let shdr = |name: u32, typ: u32, flags: u64, addr: u64, off: u64, size: u64, e: &mut Vec<u8>| {
            e.extend_from_slice(&name.to_le_bytes());
            e.extend_from_slice(&typ.to_le_bytes());
            e.extend_from_slice(&flags.to_le_bytes());
            e.extend_from_slice(&addr.to_le_bytes());
            e.extend_from_slice(&off.to_le_bytes());
            e.extend_from_slice(&size.to_le_bytes());
            e.extend_from_slice(&0u32.to_le_bytes()); // sh_link
            e.extend_from_slice(&0u32.to_le_bytes()); // sh_info
            e.extend_from_slice(&1u64.to_le_bytes()); // sh_addralign
            e.extend_from_slice(&0u64.to_le_bytes()); // sh_entsize
        };
        // SHT_PROGBITS=1, SHT_STRTAB=3; SHF_ALLOC=2, SHF_WRITE=1, SHF_EXECINSTR=4.
        shdr(0, 0, 0, 0, 0, 0, &mut e); // null
        shdr(name_text, 1, 2 | 4, tvaddr, text_off, text.len() as u64, &mut e); // .text
        shdr(name_sec, 1, 2 | 1, svaddr, sec_off, sec.len() as u64, &mut e); // data section
        shdr(name_shstr, 3, 0, 0, shstr_off, shstr.len() as u64, &mut e); // .shstrtab
        e
    }

    #[test]
    fn vtable_resolves_data_rel_ro_function_pointer_run() {
        // A C++ vtable: a run of absolute function pointers in `.data.rel.ro` pointing at real text.
        //   text 0x1000: c3 c3 c3 c3   four `ret`s (0x1000 is the entry root; 0x1001-3 are fn bodies)
        //   .data.rel.ro 0x3000: [0x1001, 0x1002, 0x1003]  (the vtable's virtual-fn array)
        let text = [0xc3, 0xc3, 0xc3, 0xc3];
        let mut dr = Vec::new();
        for &t in &[0x1001u64, 0x1002, 0x1003] {
            dr.extend_from_slice(&t.to_le_bytes());
        }
        let elf = sectioned_elf(0x1000, &text, ".data.rel.ro", 0x3000, &dr);
        let sup = build(&text);
        let edges = resolve_indirect(&sup, &elf, 0x1000, &ResolveConfig::default());

        let vt: HashSet<u64> =
            edges.iter().filter(|e| e.kind == ResolveKind::Vtable).map(|e| e.t).collect();
        assert_eq!(vt, [0x1001, 0x1002, 0x1003].into_iter().collect(), "vtable targets: {edges:?}");

        // Data-only (M3) still reads the run (it is in the allowlist) but tags it a generic JumpTable,
        // never Vtable — the structural provenance is the M4 resolver's contribution.
        let m3 = resolve_indirect(
            &sup,
            &elf,
            0x1000,
            &ResolveConfig { code_anchored: false, ..ResolveConfig::default() },
        );
        assert!(m3.iter().all(|e| e.kind != ResolveKind::Vtable), "data-only must not emit Vtable");
        assert!(m3.iter().any(|e| e.t == 0x1003), "the run is still visible to the blind scan in M3");
    }

    #[test]
    fn pie_relative_jump_table_resolves_self_relative_entries() {
        // The PIE idiom: `lea rax,[rip+disp]` loads the table base; each entry is a 4-byte SIGNED
        // self-relative offset (target = table_base + i32). The blind 8-byte-absolute scan cannot read
        // these — this is the resolver's genuinely new coverage.
        //   0x1000: 48 8d 05 f9 0f 00 00   lea rax,[rip+0xff9]   ; base = 0x1007 + 0xff9 = 0x2000
        //   0x1007: ff e0                  jmp rax               ; indirect (no static target)
        //   0x1009: c3                     ret                   ; case A
        //   0x100a: c3                     ret                   ; case B
        let text = [0x48, 0x8d, 0x05, 0xf9, 0x0f, 0x00, 0x00, 0xff, 0xe0, 0xc3, 0xc3];
        let mut data = Vec::new();
        data.extend_from_slice(&((0x1009i64 - 0x2000) as i32).to_le_bytes()); // case A, relative to base
        data.extend_from_slice(&((0x100ai64 - 0x2000) as i32).to_le_bytes()); // case B
        let elf = two_seg_elf(0x1000, &text, 0x2000, &data);
        let sup = build(&text);
        let edges = resolve_indirect(&sup, &elf, 0x1000, &ResolveConfig::default());

        let pie: HashSet<u64> =
            edges.iter().filter(|e| e.kind == ResolveKind::PieRelJumpTable).map(|e| e.t).collect();
        assert_eq!(pie, [0x1009, 0x100a].into_iter().collect(), "pie-rel targets: {edges:?}");

        // The 4-byte-relative table is invisible to the data-anchored scan: M3 finds nothing here.
        let m3 = resolve_indirect(
            &sup,
            &elf,
            0x1000,
            &ResolveConfig { code_anchored: false, ..ResolveConfig::default() },
        );
        assert!(m3.is_empty(), "blind 8-byte scan must miss a 4-byte-relative table, got {m3:?}");
    }

    // ── M3b: per-binary β̂₀ regressor. ────────────────────────────────────────────────────────────

    #[test]
    fn beta0_model_recovers_a_monotone_relationship() {
        // Synthetic: β₀ rises with the first feature. The isotonic-recalibrated regressor should
        // return a higher β̂₀ for a high-feature binary than a low-feature one, and stay in [0,1].
        let rows: Vec<(Vec<f64>, f64)> = (0..8)
            .map(|i| {
                let x = i as f64 / 7.0;
                (vec![x, 0.1 * x, 0.2, 0.3], 0.1 + 0.8 * x)
            })
            .collect();
        let model = Beta0Model::fit(&rows, 1e-3, 3000);
        let lo = model.apply(&[0.0, 0.0, 0.2, 0.3]);
        let hi = model.apply(&[1.0, 0.1, 0.2, 0.3]);
        assert!((0.0..=1.0).contains(&lo) && (0.0..=1.0).contains(&hi));
        assert!(hi > lo, "β̂₀ should increase with the driving feature: lo={lo} hi={hi}");
    }
}
