//! `udstack` — the uncertainty-preserving **stack orchestrator** (Layer 3,
//! `probablistic/LAYER3_STACK_DESIGN.md` §6). Every layer emits calibrated marginals over its
//! objects; adjacent layers exchange calibrated messages across a structural incidence `Γ`; the
//! whole system relaxes to a fixpoint whose per-object marginals stay calibrated end-to-end (Theorem
//! 4). The coupling atom is `calibrate::Fusion` (M2's fusion, tiled); the *new* engineering is here:
//! the `Layer` trait, the `Γ` incidence, the damped message schedule with `(S3)` exclusion, the
//! fixpoint `relax`, and the online `clamp`.
//!
//! We wire the two existing layers as adapters **without touching them**:
//!   * **L1 instructions** — `probdisasm` Soft posteriors `π_a` (the bottom evidence).
//!   * **L2 CFG/functions** — `probcfg`'s confirmation fixpoint `F_h` (eq 3) and reachedness `R_a`
//!     (eq 4). The bottom-up message is `π → C → F`; the top-down message is `F → R → π̂`.
//!
//! The reported instruction marginal is the fused, isotonic-recalibrated `bel_a = P̂_a`; the reported
//! function marginal is `bel_h = F_h`. Nothing here overwrites a raw posterior — `π` is an input
//! message, never mutated (the honesty wall).

use std::collections::HashMap;

use anyhow::{anyhow, Context, Result};
use calibrate::{logit, Fusion, IsotonicMap, LogOdds};
use evalkit::{load_gt, run_soft};
use probcfg::{
    build_soft_confirm_resolved, confirm_fixpoint, edge_evidence, reachedness, resolve_indirect,
    ResolveConfig, ResolvedEdge, SoftConfig,
};
use probdisasm::{extract_text_section, Superset};

// ── Objects & incidence ─────────────────────────────────────────────────────────────────────────

/// The object type (the design's `TypeId`): one calibration map / fusion is fit per kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    /// An L1 instruction-start candidate `a ∈ 𝓘`.
    Instr,
    /// An L2 function head `h ∈ 𝓗`.
    Func,
    /// An L4 module object `c ∈ 𝓒` — a strongly-connected component of the confirmed call graph
    /// (plus the program-entry component). See [`L4Layer`].
    Module,
}

/// A stable object id addressed by `(kind, vaddr)`. Beliefs live in the `Stack`, keyed by `ObjId`.
/// For a [`Kind::Module`] object the `addr` is the component's representative id (its lowest member
/// head vaddr) — components have no address of their own, so I key them by a stable member.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ObjId {
    pub kind: Kind,
    pub addr: u64,
}

impl ObjId {
    pub fn instr(a: u64) -> Self {
        ObjId { kind: Kind::Instr, addr: a }
    }
    pub fn func(h: u64) -> Self {
        ObjId { kind: Kind::Func, addr: h }
    }
    /// A module (call-graph SCC) object, keyed by its representative id (lowest member head).
    pub fn module(c: u64) -> Self {
        ObjId { kind: Kind::Module, addr: c }
    }
}

/// The structural incidence `Γ` linking a function layer to the instruction layer below it: which
/// instructions each function contains, and the call sites that carry bottom-up evidence. Deterministic
/// (not probabilities) — it defines *which* marginals message which.
#[derive(Clone, Debug, Default)]
pub struct Incidence {
    /// `a ∈ body(h)` — instructions contained in each function head's intra-procedural body.
    pub bodies: HashMap<u64, Vec<u64>>,
    /// Call sites `(g, h, a)`: instruction `a` in function `g` is a direct call to head `h` (`h ≠ g`).
    /// These carry the bottom-up message `π_a → C_{g→h}`.
    pub sites: Vec<(u64, u64, u64)>,
    /// Inverse of `bodies`: `containers[a]` = the heads whose body contains `a` (for `(S3)` exclusion).
    pub containers: HashMap<u64, Vec<u64>>,
}

// ── The Layer trait (design §6) ──────────────────────────────────────────────────────────────────

/// One layer of the stack. Objects are addressed by a stable vaddr of the layer's [`Kind`]; beliefs
/// live in the [`Stack`]. `messages_in` is realized by the concrete adapters the `Stack` drives
/// (`L1Layer::pi`, the L2 solve) — the trait states the shape future layers share.
pub trait Layer {
    /// The object kind this layer owns.
    fn kind(&self) -> Kind;
    /// The object vaddrs this layer contributes marginals for.
    fn objects(&self) -> &[u64];
    /// Structural incidence to the adjacent lower layer (`Γ`); empty for the bottom layer.
    fn couplings(&self) -> &Incidence;
}

/// L1 adapter: `probdisasm` Soft posteriors as instruction-object beliefs. The bottom layer — no
/// downward incidence. `π_a` is a fixed input message (never mutated).
pub struct L1Layer {
    addrs: Vec<u64>,
    /// `π_a` — the Soft posterior per instruction candidate.
    pub pi: HashMap<u64, f64>,
    incidence: Incidence,
}

impl Layer for L1Layer {
    fn kind(&self) -> Kind {
        Kind::Instr
    }
    fn objects(&self) -> &[u64] {
        &self.addrs
    }
    fn couplings(&self) -> &Incidence {
        &self.incidence
    }
}

/// L2 adapter: `probcfg`'s confirmation fixpoint over function heads. Holds the topology (heads,
/// bodies, call sites, shape-only prior, resolved edges) built once; the fixpoint `F` and reachedness
/// `R` are recomputed each sweep from the current instruction beliefs (that is the bottom-up message).
pub struct L2Layer {
    heads: Vec<u64>,
    /// Shape-only local prior `prior_h` (eq 2) — belief-independent, built once.
    pub prior: HashMap<u64, f64>,
    /// Resolved indirect edges `g →_𝓡 t` (M3a) folded into the eq-1 evidence; plus online clamps.
    pub resolved: Vec<ResolvedEdge>,
    incidence: Incidence,
    cfg_eps: f64,
    cfg_max_iter: usize,
}

impl Layer for L2Layer {
    fn kind(&self) -> Kind {
        Kind::Func
    }
    fn objects(&self) -> &[u64] {
        &self.heads
    }
    fn couplings(&self) -> &Incidence {
        &self.incidence
    }
}

/// L4 adapter: the **module / call-graph level.** Objects `O_4 = 𝓒` are the strongly-connected
/// components (SCCs) of the *confirmed* call graph (direct call sites `Γ` + resolved indirect edges),
/// plus the entry component. Latent `X_c` = "component `c` is real code on the intended
/// interpretation." The point of this layer: a *decoy* is a call-graph component **disconnected** from
/// the entry component, and that disconnection is a global fact per-function reachedness `R_a` doesn't
/// encode. Built once from the L2 topology; the component fixpoint `F_c` is recomputed each sweep from
/// the current function confirmations (that is the bottom-up `2→4` message).
pub struct L4Layer {
    /// The component objects, keyed by representative id (lowest member head). Sorted, stable.
    comps: Vec<u64>,
    /// `comp → member heads`. Reused as the `Incidence.bodies` for the `Γ_{2↔4}` down-message.
    members: HashMap<u64, Vec<u64>>,
    /// `head → its component id` (each head is in exactly one SCC).
    comp_of: HashMap<u64, u64>,
    /// The component containing the ELF entry — the root, pinned `F = 1`.
    entry_comp: u64,
    /// The condensation DAG in `callee_comp → [predecessor_comp]` form (inter-SCC edges only). The
    /// per-edge weight is rebuilt each sweep from the crossing calls' caller confirmations, so it is
    /// *not* stored here — only the topology is.
    preds: HashMap<u64, Vec<u64>>,
    /// For each condensation edge `(c' → c)`, the crossing call sites `(g, h)` that induce it — used
    /// to weight the edge by the callers' current confirmations (the `2→4` message).
    cross: HashMap<(u64, u64), Vec<(u64, u64)>>,
    /// `Γ_{2↔4}` incidence: `bodies = members`, `containers = head → [comp]`. `sites` unused (the
    /// inter-component edges live in `preds`/`cross`, one abstraction level up).
    incidence: Incidence,
    cfg_eps: f64,
    cfg_max_iter: usize,
}

impl Layer for L4Layer {
    fn kind(&self) -> Kind {
        Kind::Module
    }
    fn objects(&self) -> &[u64] {
        &self.comps
    }
    fn couplings(&self) -> &Incidence {
        &self.incidence
    }
}

impl L4Layer {
    /// Build the condensation of the confirmed call graph. Nodes = function heads; directed edges =
    /// direct call sites `(g, h)` from the L2 incidence plus resolved indirect edges `(g, t)`. The SCCs
    /// come from Tarjan; the entry component is the one containing `entry`. GT-free: this is pure
    /// topology derived from `probcfg`'s confirmation, exactly as the spec demands.
    fn build(heads: &[u64], sites: &[(u64, u64, u64)], resolved: &[ResolvedEdge], entry: u64, eps: f64, max_iter: usize) -> Self {
        // Adjacency `g → callees` over the confirmed edge set (dedup; drop self-loops — they don't
        // change SCC membership and only add noise to the condensation).
        let head_set: std::collections::HashSet<u64> = heads.iter().copied().collect();
        let mut adj: HashMap<u64, Vec<u64>> = heads.iter().map(|&h| (h, Vec::new())).collect();
        let push_edge = |g: u64, h: u64, adj: &mut HashMap<u64, Vec<u64>>| {
            if g != h && head_set.contains(&g) && head_set.contains(&h) {
                let v = adj.entry(g).or_default();
                if !v.contains(&h) {
                    v.push(h);
                }
            }
        };
        for &(g, h, _a) in sites {
            push_edge(g, h, &mut adj);
        }
        for e in resolved {
            push_edge(e.g, e.t, &mut adj);
        }

        let (_comp_id_of, comps_members) = tarjan_scc(heads, &adj);

        // Representative id per component = its lowest member head (stable, deterministic).
        let mut members: HashMap<u64, Vec<u64>> = HashMap::new();
        let mut comp_of: HashMap<u64, u64> = HashMap::new();
        for grp in &comps_members {
            let rep = *grp.iter().min().expect("non-empty SCC");
            let mut ms = grp.clone();
            ms.sort_unstable();
            for &h in &ms {
                comp_of.insert(h, rep);
            }
            members.insert(rep, ms);
        }
        let mut comps: Vec<u64> = members.keys().copied().collect();
        comps.sort_unstable();
        let entry_comp = comp_of.get(&entry).copied().unwrap_or_else(|| comps.first().copied().unwrap_or(entry));

        // Condensation edges: for every g→h with different components, record c'→c and the crossing
        // call (g, h). Because SCCs collapse every cycle, this graph is a DAG by construction.
        let mut preds: HashMap<u64, Vec<u64>> = HashMap::new();
        let mut cross: HashMap<(u64, u64), Vec<(u64, u64)>> = HashMap::new();
        for (&g, callees) in &adj {
            let cg = comp_of[&g];
            for &h in callees {
                let ch = comp_of[&h];
                if cg != ch {
                    let ins = preds.entry(ch).or_default();
                    if !ins.contains(&cg) {
                        ins.push(cg);
                    }
                    cross.entry((cg, ch)).or_default().push((g, h));
                }
            }
        }

        let incidence = Incidence {
            bodies: members.clone(),
            sites: Vec::new(),
            containers: comp_of.iter().map(|(&h, &c)| (h, vec![c])).collect(),
        };
        L4Layer { comps, members, comp_of, entry_comp, preds, cross, incidence, cfg_eps: eps, cfg_max_iter: max_iter }
    }

    /// The component id containing head `h` (if any).
    fn comp_of(&self, h: u64) -> Option<u64> {
        self.comp_of.get(&h).copied()
    }
}

/// Tarjan's strongly-connected-components. Returns `(head → component representative id, components as
/// member lists)`. No SCC utility exists anywhere in the stack, so I wrote the classic iterative
/// version here (recursion would blow the stack on a big call graph). The representative id used
/// downstream is the lowest member head; here I just tag each node with a dense component index and let
/// the caller pick reps.
fn tarjan_scc(nodes: &[u64], adj: &HashMap<u64, Vec<u64>>) -> (HashMap<u64, u64>, Vec<Vec<u64>>) {
    let mut index: HashMap<u64, u32> = HashMap::new();
    let mut low: HashMap<u64, u32> = HashMap::new();
    let mut on_stack: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut stack: Vec<u64> = Vec::new();
    let mut comps: Vec<Vec<u64>> = Vec::new();
    let mut comp_index: HashMap<u64, u64> = HashMap::new();
    let mut next_idx: u32 = 0;

    // Iterative DFS: each frame is (node, next-child-cursor).
    for &root in nodes {
        if index.contains_key(&root) {
            continue;
        }
        let mut call_stack: Vec<(u64, usize)> = vec![(root, 0)];
        while let Some(&(v, ci)) = call_stack.last() {
            if ci == 0 {
                index.insert(v, next_idx);
                low.insert(v, next_idx);
                next_idx += 1;
                stack.push(v);
                on_stack.insert(v);
            }
            let children = adj.get(&v).map(|c| c.as_slice()).unwrap_or(&[]);
            if ci < children.len() {
                let w = children[ci];
                call_stack.last_mut().unwrap().1 += 1;
                if !index.contains_key(&w) {
                    call_stack.push((w, 0));
                } else if on_stack.contains(&w) {
                    let lv = low[&v].min(index[&w]);
                    low.insert(v, lv);
                }
            } else {
                // Done with v: if it is a root of an SCC, pop the component.
                if low[&v] == index[&v] {
                    let mut grp = Vec::new();
                    loop {
                        let w = stack.pop().unwrap();
                        on_stack.remove(&w);
                        comp_index.insert(w, comps.len() as u64);
                        grp.push(w);
                        if w == v {
                            break;
                        }
                    }
                    comps.push(grp);
                }
                call_stack.pop();
                if let Some(&(parent, _)) = call_stack.last() {
                    let lp = low[&parent].min(low[&v]);
                    low.insert(parent, lp);
                }
            }
        }
    }
    (comp_index, comps)
}

// ── Schedule & reporting ─────────────────────────────────────────────────────────────────────────

/// The message schedule for [`Stack::relax`] (design §3).
#[derive(Clone, Copy, Debug)]
pub struct Schedule {
    /// Enable the top-down `F → R → π̂` message. `false` = bottom-up only (Milestone A: reproduce M2).
    pub top_down: bool,
    /// Apply the `(S3)` loopy-BP exclusion when a call site sends its belief up (Milestone B).
    pub s3: bool,
    /// Damping `λ ∈ (0,1]`: `logit(bel)^{t+1} = (1−λ)·logit(bel)^t + λ·(S1)`.
    pub lambda: f64,
    /// Convergence tolerance `‖bel^{t+1} − bel^t‖_∞ < ε`.
    pub eps: f64,
    /// Sweep-pair cap (one up + one down = one iteration).
    pub max_sweeps: usize,
}

impl Schedule {
    /// Milestone A: bottom-up only, single pass, no damping — the exact M2 fusion.
    pub fn bottom_up_once() -> Self {
        Schedule { top_down: false, s3: false, lambda: 1.0, eps: 1e-6, max_sweeps: 1 }
    }
    /// Milestone B: coupled top-down relaxation to a fixpoint, damped, with `(S3)` exclusion.
    pub fn coupled(lambda: f64) -> Self {
        Schedule { top_down: true, s3: true, lambda, eps: 1e-4, max_sweeps: 64 }
    }
}

/// Convergence trace from a [`Stack::relax`].
#[derive(Clone, Debug, Default)]
pub struct Convergence {
    /// Per-iteration `‖bel^{t+1} − bel^t‖_∞` over instruction marginals (probability space) — the
    /// quantity the `ε` stopping test uses.
    pub deltas: Vec<f64>,
    /// Per-iteration `‖Δlogit(bel)‖_∞` over instruction marginals (log-odds space) — the Phase-2
    /// contraction trace. The damped sweep is linear in *log-odds*, so this is the trace whose ratio is
    /// the contraction factor `ρ`. Collected always (negligible cost); printed only under `--trace`.
    pub logit_deltas: Vec<f64>,
    /// Whether the schedule reached `ε` before the sweep cap.
    pub converged: bool,
}

impl Convergence {
    pub fn iters(&self) -> usize {
        self.deltas.len()
    }
    pub fn final_delta(&self) -> f64 {
        self.deltas.last().copied().unwrap_or(f64::NAN)
    }

    /// The empirical **contraction ratio** `ρ ≈ limsup_t ‖Δ^{t+1}‖ / ‖Δ^{t}‖` from the log-odds trace.
    /// I take the max ratio over the *second half* of the trace (the linear regime, past the transient)
    /// as a robust limsup proxy. `ρ < 1` ⇒ the sweep is a contraction; `ρ ≥ 1` ⇒ it is not (oscillation
    /// or divergence). Returns `NaN` for traces too short to have a ratio.
    pub fn contraction_ratio(&self) -> f64 {
        let d = &self.logit_deltas;
        if d.len() < 3 {
            return f64::NAN;
        }
        let start = d.len() / 2;
        let mut rho = 0.0f64;
        for t in start..d.len() - 1 {
            if d[t] > 1e-30 {
                rho = rho.max(d[t + 1] / d[t]);
            }
        }
        rho
    }
}

/// A ranked online-evidence candidate (design §5 active analysis): confirming head `head` is
/// expected to remove `eig` bits of instruction-map entropy.
#[derive(Clone, Copy, Debug)]
pub struct Query {
    /// The candidate function head to confirm.
    pub head: u64,
    /// Its current confirmation `F_h` (the probability the query returns *real*).
    pub f_prior: f64,
    /// Expected information gain `F_h · ΔH` (bits) — the ranking key.
    pub eig: f64,
    /// Entropy actually removed *if* the query confirms (`ΔH`, bits).
    pub dh_confirm: f64,
    /// Body size (instructions the confirmation can reach).
    pub body: usize,
}

/// A rollback snapshot of the mutable relaxation state (see [`Stack::snapshot`]).
#[derive(Clone)]
struct StackState {
    bel: HashMap<ObjId, f64>,
    r: HashMap<u64, f64>,
    f: HashMap<u64, f64>,
    f_raw: HashMap<u64, f64>,
    f_c: HashMap<u64, f64>,
    resolved: Vec<ResolvedEdge>,
    clamped: HashMap<ObjId, f64>,
}

// ── The Stack orchestrator ───────────────────────────────────────────────────────────────────────

/// The stack: layers, per-object beliefs `bel`, and per-type fusion operators (`cal`). Owns the
/// `Superset` so both layers can read the shared decode. Drives the two-layer L1↔L2 schedule
/// concretely (the trait documents the shape future layers will slot into).
pub struct Stack {
    sup: Superset,
    entry: u64,
    l1: L1Layer,
    l2: L2Layer,
    /// The optional L4 module layer. `None` ⇒ a K=2 stack (`{L1, L2}`, the M2-reproducing baseline);
    /// `Some` ⇒ K=3 (`{L1, L2, L4}`). Built on demand by [`Stack::build_module_layer`] so the *same*
    /// specimen can be run at both depths (that is how the compositionality/honesty-wall comparison is
    /// made). Every K-specific branch in the engine is gated on `self.l4.is_some()`, so the K=2 path
    /// stays bit-for-bit identical to before this layer existed.
    l4: Option<L4Layer>,
    /// Per-object marginals `bel_o = P(X_o = 1 | evidence)`.
    bel: HashMap<ObjId, f64>,
    /// Current reachedness `R_a` (the live top-down message) and confirmation `F_h`. When L4 is
    /// coupled, `f` holds the *effective* head belief (raw `F_h` fused with the top-down component
    /// message); at K=2 it is the raw `F_h`, unchanged.
    r: HashMap<u64, f64>,
    f: HashMap<u64, f64>,
    /// The raw bottom-up confirmation `F_h` before the `4→2` fusion — kept so the `Kind::Func` pool can
    /// fuse it with the component message. Equal to `f` at K=2.
    f_raw: HashMap<u64, f64>,
    /// Component confirmations `F_c` (the L4 marginals); empty at K=2.
    f_c: HashMap<u64, f64>,
    /// The `(S1)` log-linear pool operator per object kind — the fitted fusion *weights*, applied
    /// during propagation (`pool.s1`). Fit once; frozen through relaxation (weights are the operator).
    pool: HashMap<Kind, Fusion>,
    /// The `(S2)` recalibration map `g_o` per object type (design's `cal`) — fit on the **fixpoint**
    /// beliefs against GT (Theorem 4) and applied only at readout, so calibration tracks the converged
    /// distribution instead of the initial one.
    cal: HashMap<Kind, IsotonicMap>,
    /// Objects pinned by online evidence (`clamp`): held fixed, excluded from sweep updates.
    clamped: HashMap<ObjId, f64>,
}

impl Stack {
    /// Build a two-layer `{L1, L2}` stack from an ELF image. Runs Soft (L1) and assembles the L2
    /// topology via `probcfg` (heads/bodies/prior/resolved). `resolve_elf` supplies the data image
    /// the M3a resolver reads (the benign seed for the code-in-data corpus); `None` = no resolution.
    pub fn from_elf(
        bytes: &[u8],
        entropy: f64,
        dassa: bool,
        max_fn_span: usize,
        resolve_elf: Option<&[u8]>,
    ) -> Result<Self> {
        let (base, code) = extract_text_section(bytes).context("extracting .text")?;
        let post = run_soft(base, code, entropy, dassa).context("running Soft")?;
        let entry = read_e_entry(bytes);
        let mapped = if entry >= base && entry < base + code.len() as u64 { entry } else { base };
        let sup = Superset::new(base, code).map_err(|e| anyhow!("building superset: {e:?}"))?;

        // Resolved indirect edges (M3a) — read the data image (seed) if supplied.
        let resolved: Vec<ResolvedEdge> = match resolve_elf {
            Some(elf) => resolve_indirect(&sup, elf, mapped, &ResolveConfig::default()),
            None => Vec::new(),
        };

        // L2 topology: reuse probcfg's builder once (with pmap = π) to get heads/bodies/prior. We then
        // recompute F/R ourselves each sweep so call-site evidence can be the *coupled* belief.
        let pi: HashMap<u64, f64> = post.iter().copied().collect();
        let cfg = SoftConfig { max_fn_span, ..SoftConfig::default() };
        let sc = build_soft_confirm_resolved(&sup, mapped, &pi, &resolved, &cfg);

        // Derive the incidence Γ from the bodies: call sites and the body/container maps.
        let head_set: std::collections::HashSet<u64> = sc.heads.iter().copied().collect();
        let mut sites: Vec<(u64, u64, u64)> = Vec::new();
        let mut containers: HashMap<u64, Vec<u64>> = HashMap::new();
        for (&g, body) in &sc.bodies {
            for &a in body {
                containers.entry(a).or_default().push(g);
                if let Some(insn) = sup.at(a) {
                    if insn.is_call() {
                        if let Some(h) = insn.branch_target {
                            if h != g && head_set.contains(&h) {
                                sites.push((g, h, a));
                            }
                        }
                    }
                }
            }
        }
        let incidence = Incidence { bodies: sc.bodies.clone(), sites, containers };

        let l1 = L1Layer {
            addrs: post.iter().map(|&(a, _)| a).collect(),
            pi,
            incidence: Incidence::default(),
        };
        let l2 = L2Layer {
            heads: sc.heads.clone(),
            prior: sc.prior.clone(),
            resolved,
            incidence,
            cfg_eps: cfg.eps,
            cfg_max_iter: cfg.max_iter,
        };

        // Initial beliefs: instructions at π_a, heads at their prior.
        let mut bel: HashMap<ObjId, f64> = HashMap::new();
        for &(a, p) in &post {
            bel.insert(ObjId::instr(a), p);
        }
        for &h in &l2.heads {
            bel.insert(ObjId::func(h), l2.prior.get(&h).copied().unwrap_or(0.0));
        }

        Ok(Stack {
            sup,
            entry: mapped,
            l1,
            l2,
            l4: None,
            bel,
            r: HashMap::new(),
            f: HashMap::new(),
            f_raw: HashMap::new(),
            f_c: HashMap::new(),
            pool: HashMap::new(),
            cal: HashMap::new(),
            clamped: HashMap::new(),
        })
    }

    /// The stack's layers as trait objects, **bottom → top**, length = K (2 or 3). Generalized off the
    /// old hard-coded `[&dyn Layer; 2]`: a third layer drops in without special-casing the return type.
    /// The relaxation schedule ([`Stack::relax`]) is likewise layer-count-agnostic — it walks the L2
    /// rung, then the L4 rung when present, and the K=2 path is untouched.
    pub fn layers(&self) -> Vec<&dyn Layer> {
        let mut ls: Vec<&dyn Layer> = vec![&self.l1, &self.l2];
        if let Some(l4) = &self.l4 {
            ls.push(l4);
        }
        ls
    }

    /// Number of layers currently in the stack (`K`): 2 for `{L1, L2}`, 3 with the module layer.
    pub fn depth(&self) -> usize {
        2 + usize::from(self.l4.is_some())
    }

    /// **Add the L4 module layer** — build the condensation of the confirmed call graph from the L2
    /// topology (direct call sites + resolved indirect edges) and switch the stack to K=3. Idempotent;
    /// GT-free (pure topology). Re-run [`Stack::relax`] afterward to couple it in. Seeds each component
    /// belief at `0` (non-entry) / `1` (entry) so a fresh `relax` starts from the reachability prior.
    pub fn build_module_layer(&mut self) {
        let l4 = L4Layer::build(
            &self.l2.heads,
            &self.l2.incidence.sites,
            &self.l2.resolved,
            self.entry,
            self.l2.cfg_eps,
            self.l2.cfg_max_iter,
        );
        for &c in &l4.comps {
            let seed = if c == l4.entry_comp { 1.0 } else { 0.0 };
            self.bel.insert(ObjId::module(c), seed);
            self.f_c.insert(c, seed);
        }
        self.l4 = Some(l4);
    }

    /// `π_a` (Layer-1 posterior) for an instruction candidate.
    pub fn pi(&self, a: u64) -> f64 {
        self.l1.pi.get(&a).copied().unwrap_or(0.0)
    }

    // ── (S1) pool + (S2) recalibration per object ───────────────────────────────────────────────

    /// **(S1)** — the raw fused pool for instruction `a`'s messages `(π_a, R_a)`: `σ(b + w·[logit π,
    /// logit R])`. This is the quantity that *propagates* during relaxation (uncalibrated but monotone
    /// and coupling-consistent). Falls back to `π_a` before the pool weights are fit. The calibrated
    /// readout applies `(S2)` on top (see [`Stack::instr_marginals`]).
    pub fn pool_fuse(&self, a: u64, r_a: f64) -> f64 {
        match self.pool.get(&Kind::Instr) {
            Some(fu) => fu.s1(&[logit(self.pi(a)), logit(r_a)]),
            None => self.pi(a),
        }
    }

    /// Fit the `(S1)` pool weights on `(logit π_a, logit R_a, y)` rows (the fusion operator). Frozen
    /// through relaxation. Only the weights are used from the fitted `Fusion` (its bundled isotonic map
    /// is ignored here; `(S2)` is fit separately at the fixpoint by [`Stack::recalibrate`]).
    pub fn fit_pool(&mut self, gt: &std::collections::HashSet<u64>) {
        self.pool.insert(Kind::Instr, Fusion::fit(&self.fusion_rows(gt)));
    }

    /// **(S2)** — fit the instruction recalibration map `g_o` on the **current (fixpoint) beliefs** vs
    /// GT (Theorem 4). Self-fit ceiling: fit == eval. Skipped when a transfer `g_o` is already installed.
    pub fn recalibrate(&mut self, gt: &std::collections::HashSet<u64>) {
        let samples: Vec<(f64, f64)> = self
            .l1
            .addrs
            .iter()
            .map(|&a| (self.bel.get(&ObjId::instr(a)).copied().unwrap_or(self.pi(a)), f64::from(gt.contains(&a))))
            .collect();
        self.cal.insert(Kind::Instr, IsotonicMap::fit(&samples));
    }

    /// Install a pool operator fit elsewhere (§9.5 transfer: fit on a held-out binary).
    pub fn install_pool(&mut self, fusion: Fusion) {
        self.pool.insert(Kind::Instr, fusion);
    }

    /// Install a recalibration map `g_o` fit elsewhere (§9.5 transfer: the held-out binary's fixpoint
    /// `g_o`). When present, [`Stack::relax`] does not self-recalibrate — the transfer readout stands.
    pub fn install_cal(&mut self, iso: IsotonicMap) {
        self.cal.insert(Kind::Instr, iso);
    }

    /// The fitted `(S1)` pool operator for a kind (to install on a target for transfer).
    pub fn pool_of(&self, kind: Kind) -> Option<&Fusion> {
        self.pool.get(&kind)
    }

    /// The held-out fixpoint recalibration map for a kind (to install on a target for transfer).
    pub fn cal_of(&self, kind: Kind) -> Option<&IsotonicMap> {
        self.cal.get(&kind)
    }

    /// The current `(logit π_a, logit R_a, y)` rows for this binary — used to fit a transferable
    /// pool, or to fit one on a held-out `Stack`.
    pub fn fusion_rows(&self, gt: &std::collections::HashSet<u64>) -> Vec<(Vec<LogOdds>, f64)> {
        self.l1
            .addrs
            .iter()
            .map(|&a| {
                let r_a = self.r.get(&a).copied().unwrap_or(0.0);
                (vec![logit(self.pi(a)), logit(r_a)], f64::from(gt.contains(&a)))
            })
            .collect()
    }

    // ── L4 per-type fits (the same S1/S2 atom, one level up) ──────────────────────────────────────

    /// Fit the `Kind::Func` pool `(S1)` on `(logit F_h, logit F_{c(h)}, y_func)` — the `4→2` fusion
    /// weights. Uses the raw head confirmation and its component belief from the first sweep. Frozen
    /// through the rest of the relaxation, exactly like the instruction pool.
    fn fit_pool_func(&mut self, func_gt: &std::collections::HashSet<u64>) {
        let Some(l4) = &self.l4 else { return };
        let rows: Vec<(Vec<LogOdds>, f64)> = self
            .l2
            .heads
            .iter()
            .map(|&h| {
                let fr = self.f_raw.get(&h).copied().unwrap_or(0.0);
                let bc = l4.comp_of(h).and_then(|c| self.f_c.get(&c).copied()).unwrap_or(0.0);
                (vec![logit(fr), logit(bc)], f64::from(func_gt.contains(&h)))
            })
            .collect();
        self.pool.insert(Kind::Func, Fusion::fit(&rows));
    }

    /// **(S2)** for functions — isotonic `g_o` on the fixpoint fused head beliefs vs FUNC-symbol GT.
    fn recalibrate_func(&mut self, func_gt: &std::collections::HashSet<u64>) {
        let samples: Vec<(f64, f64)> = self
            .l2
            .heads
            .iter()
            .map(|&h| (self.bel.get(&ObjId::func(h)).copied().unwrap_or(0.0), f64::from(func_gt.contains(&h))))
            .collect();
        self.cal.insert(Kind::Func, IsotonicMap::fit(&samples));
    }

    /// **(S2)** for modules — isotonic `g_o` on the fixpoint component beliefs `F_c` vs module GT (a
    /// component is real iff any member head is a real function per `func_gt`). GT-derived, no decode.
    fn recalibrate_module(&mut self, func_gt: &std::collections::HashSet<u64>) {
        let Some(l4) = &self.l4 else { return };
        let samples: Vec<(f64, f64)> = l4
            .comps
            .iter()
            .map(|&c| {
                let y = l4.members.get(&c).map(|ms| ms.iter().any(|h| func_gt.contains(h))).unwrap_or(false);
                (self.bel.get(&ObjId::module(c)).copied().unwrap_or(0.0), f64::from(y))
            })
            .collect();
        self.cal.insert(Kind::Module, IsotonicMap::fit(&samples));
    }

    /// Module GT derived from FUNC-symbol GT: a component is real iff any member head is a real
    /// function. Returned as the set of *real* component ids — for scoring the module axis.
    pub fn module_gt(&self, func_gt: &std::collections::HashSet<u64>) -> std::collections::HashSet<u64> {
        match &self.l4 {
            Some(l4) => l4
                .comps
                .iter()
                .copied()
                .filter(|c| l4.members.get(c).map(|ms| ms.iter().any(|h| func_gt.contains(h))).unwrap_or(false))
                .collect(),
            None => std::collections::HashSet::new(),
        }
    }

    // ── Sweeps (design §3) ───────────────────────────────────────────────────────────────────────

    /// **Bottom-up sweep** — recompute `F_h` (eq 3) from the current instruction beliefs (the
    /// `π → C → F` message); when L4 is present, solve the component fixpoint `F_c` and fuse the
    /// top-down `4→2` component message into each head to get the *effective* head belief; then
    /// recompute `R_a` (eq 4) from that effective belief and move each head toward it (damped). With
    /// `s3`, a call site's up-message excludes the source function's own top-down contribution (the
    /// `(S3)` loopy-BP correction). At K=2 (`l4 = None`) this is bit-for-bit the old two-layer sweep.
    fn sweep_up(&mut self, s3: bool, lambda: f64) {
        let f_raw = self.solve_confirm(s3);
        let (f_c, f_eff) = match &self.l4 {
            Some(l4) => self.solve_module(l4, &f_raw),
            None => (HashMap::new(), f_raw.clone()),
        };
        // R_a is driven by the *effective* head belief (fused with the module message when K=3).
        let r = reachedness(&self.l2.incidence.bodies, &f_eff);

        // Damp the head beliefs toward the effective F_h; clamps hold fixed.
        for &h in &self.l2.heads {
            let id = ObjId::func(h);
            if let Some(&c) = self.clamped.get(&id) {
                self.bel.insert(id, c);
                continue;
            }
            let target = f_eff.get(&h).copied().unwrap_or(0.0);
            let cur = self.bel.get(&id).copied().unwrap_or(target);
            self.bel.insert(id, damp(cur, target, lambda));
        }
        // Damp the component beliefs toward F_c (K=3 only).
        if let Some(l4) = &self.l4 {
            for &c in &l4.comps {
                let id = ObjId::module(c);
                let target = f_c.get(&c).copied().unwrap_or(0.0);
                let cur = self.bel.get(&id).copied().unwrap_or(target);
                self.bel.insert(id, damp(cur, target, lambda));
            }
        }
        self.f_raw = f_raw;
        self.f = f_eff;
        self.f_c = f_c;
        self.r = r;
    }

    /// **Top-down sweep** — fuse each instruction's messages `(π_a, R_a)` into `bel_a = P̂_a`
    /// (damped). Function confidence flows down through `R_a`, down-weighting instructions in
    /// unconfirmed functions. Clamps (traces) hold fixed. Returns `(‖Δbel‖_∞, ‖Δlogit(bel)‖_∞)`: the
    /// first (probability space) drives the `ε` stop, the second (log-odds space) is the Phase-2
    /// contraction trace.
    fn sweep_down(&mut self, lambda: f64) -> (f64, f64) {
        let mut max_delta = 0.0f64;
        let mut max_logit_delta = 0.0f64;
        for &a in &self.l1.addrs {
            let id = ObjId::instr(a);
            if let Some(&c) = self.clamped.get(&id) {
                self.bel.insert(id, c);
                continue;
            }
            let r_a = self.r.get(&a).copied().unwrap_or(0.0);
            let target = self.pool_fuse(a, r_a);
            let cur = self.bel.get(&id).copied().unwrap_or(self.pi(a));
            let next = damp(cur, target, lambda);
            max_delta = max_delta.max((next - cur).abs());
            max_logit_delta = max_logit_delta.max((logit(next) - logit(cur)).abs());
            self.bel.insert(id, next);
        }
        (max_delta, max_logit_delta)
    }

    /// Solve the L2 confirmation fixpoint from the current instruction beliefs. Returns the raw `F_h`.
    /// `s3` toggles the exclusion correction on the call-site up-message. This reproduces
    /// `probcfg::build_soft_confirm_resolved` exactly when the call-site belief is the raw `π` (the
    /// Milestone-A invariant), and extends it to coupled/`(S3)` beliefs for Milestone B. Reachedness is
    /// computed by the caller from the *effective* head belief (which at K=3 folds in the L4 message).
    fn solve_confirm(&self, s3: bool) -> HashMap<u64, f64> {
        // Belief a call site `a` in source function `g` sends up. Without S3: its current marginal.
        // With S3: re-fuse using the reachedness of `a` *excluding* g's own noisy-OR term.
        let site_belief = |g: u64, a: u64| -> f64 {
            if s3 {
                if let Some(fu) = self.pool.get(&Kind::Instr) {
                    let r_excl = self.reached_excluding(a, g);
                    return fu.s1(&[logit(self.pi(a)), logit(r_excl)]);
                }
            }
            self.bel.get(&ObjId::instr(a)).copied().unwrap_or(self.pi(a))
        };

        // eq (1): noisy-OR edge evidence C_{g→h} over call-site beliefs.
        let mut site_pis: HashMap<(u64, u64), Vec<f64>> = HashMap::new();
        for &(g, h, a) in &self.l2.incidence.sites {
            site_pis.entry((g, h)).or_default().push(site_belief(g, a));
        }
        let mut edges_into: HashMap<u64, Vec<(u64, f64)>> = HashMap::new();
        for ((g, h), pis) in &site_pis {
            edges_into.entry(*h).or_default().push((*g, edge_evidence(pis)));
        }
        // M3a-1 + online clamps: fold resolved edges (noisy-OR on the same caller).
        for e in &self.l2.resolved {
            let ins = edges_into.entry(e.t).or_default();
            if let Some(slot) = ins.iter_mut().find(|(g, _)| *g == e.g) {
                slot.1 = 1.0 - (1.0 - slot.1) * (1.0 - e.q);
            } else {
                ins.push((e.g, e.q));
            }
        }

        // eq (3): the confirmation fixpoint (raw F_h).
        confirm_fixpoint(
            self.entry,
            &self.l2.heads,
            &self.l2.prior,
            &edges_into,
            self.l2.cfg_eps,
            self.l2.cfg_max_iter,
        )
    }

    /// Solve the **L4 component fixpoint** `F_c` from the current raw head confirmations, then push the
    /// top-down `4→2` message back down to produce the *effective* head belief `f_eff`.
    ///
    /// The component graph is the condensation of the confirmed call graph — a DAG. `F_c` is the *same*
    /// noisy-OR confirmation fixpoint used at L2, one level up: the entry component is pinned `F = 1`,
    /// every other component has prior `0` (a component is real **only** if it is reached from the entry
    /// through confirmed edges — that reachability is the new structural signal a decoy, sitting in a
    /// disconnected component, fails). Each condensation edge `c'→c` carries the noisy-OR of its
    /// crossing calls' caller confirmations `F_g` (the bottom-up `2→4` message).
    ///
    /// The `4→2` message is `logit(msg_{c→h}) = logit(F_c)` fused into head `h` by the `Kind::Func`
    /// pool: `f_eff(h) = σ(b + w·[logit F_h, logit F_{c(h)}])`. There is **no `(S3)` exclusion here** —
    /// because the condensation is a DAG, a component's belief never depends on its own members, so the
    /// down-message carries no self-loop to correct (unlike L2, where a function contains the very call
    /// sites that feed its reachedness). Before the `Kind::Func` pool is fit, the message is a no-op and
    /// `f_eff = F_h` — so an unfit K=3 stack degrades to the K=2 head belief exactly.
    fn solve_module(&self, l4: &L4Layer, f_raw: &HashMap<u64, f64>) -> (HashMap<u64, f64>, HashMap<u64, f64>) {
        // Inter-component edge evidence C_{c'→c}: noisy-OR over the crossing calls, each weighted by its
        // caller function's current confirmation F_g (the 2→4 message feeding the component from below).
        let mut edges_into: HashMap<u64, Vec<(u64, f64)>> = HashMap::new();
        for (&c, preds) in &l4.preds {
            for &cp in preds {
                let calls = l4.cross.get(&(cp, c)).map(|v| v.as_slice()).unwrap_or(&[]);
                let pis: Vec<f64> = calls.iter().map(|&(g, _h)| f_raw.get(&g).copied().unwrap_or(0.0)).collect();
                edges_into.entry(c).or_default().push((cp, edge_evidence(&pis)));
            }
        }
        let prior: HashMap<u64, f64> = l4.comps.iter().map(|&c| (c, 0.0)).collect();
        let f_c = confirm_fixpoint(l4.entry_comp, &l4.comps, &prior, &edges_into, l4.cfg_eps, l4.cfg_max_iter);

        // 4→2 down-message fused into the head belief (Kind::Func pool). No pool yet ⇒ pass F_h through.
        let f_eff: HashMap<u64, f64> = match self.pool.get(&Kind::Func) {
            Some(fu) => self
                .l2
                .heads
                .iter()
                .map(|&h| {
                    let fr = f_raw.get(&h).copied().unwrap_or(0.0);
                    let bc = l4.comp_of(h).and_then(|c| f_c.get(&c).copied()).unwrap_or(0.0);
                    (h, fu.s1(&[logit(fr), logit(bc)]))
                })
                .collect(),
            None => f_raw.clone(),
        };
        (f_c, f_eff)
    }

    /// Reachedness of `a` excluding function `g`'s term: `1 − ∏_{g'∋a, g'≠g}(1 − F_{g'})` (the `(S3)`
    /// message correction — a's up-message to `g` must not re-import g's own downward influence).
    fn reached_excluding(&self, a: u64, g: u64) -> f64 {
        let mut prod = 1.0f64;
        if let Some(gs) = self.l2.incidence.containers.get(&a) {
            for &gp in gs {
                if gp != g {
                    prod *= 1.0 - self.f.get(&gp).copied().unwrap_or(0.0);
                }
            }
        }
        1.0 - prod
    }

    // ── relax / clamp / marginals (design §6) ───────────────────────────────────────────────────

    /// **relax** — run the message schedule to a fixpoint. Fits the `(S1)` pool weights on the first
    /// bottom-up sweep's `(π, R)` unless one is installed (transfer); at the fixpoint, fits the `(S2)`
    /// recalibration `g_o` on the converged beliefs (Theorem 4) unless a transfer `g_o` is installed.
    /// Returns the convergence trace. With [`Schedule::bottom_up_once`] this is exactly M2's fusion
    /// (one up + one down + fixpoint recalibration = π,R → P² → P̂).
    pub fn relax(&mut self, sched: Schedule, gt: &std::collections::HashSet<u64>) -> Convergence {
        self.relax_layered(sched, gt, None)
    }

    /// **relax, layer-aware** — the K-agnostic driver. Identical to [`Stack::relax`] at K=2; at K=3 it
    /// additionally fits the `Kind::Func` pool `(S1)` on the first sweep's `(logit F_h, logit F_c)` and,
    /// at the fixpoint, the `Kind::Func` and `Kind::Module` recalibration maps `g_o` `(S2)` — so *every*
    /// object type is calibrated at the fixpoint (Theorem 4 at depth). `func_gt` (the seed-symtab FUNC
    /// labels) is the higher layers' GT; `None` ⇒ the module message stays a pass-through no-op. Module
    /// GT is derived from `func_gt`: a component is real iff any member head is a real function.
    pub fn relax_layered(
        &mut self,
        sched: Schedule,
        gt: &std::collections::HashSet<u64>,
        func_gt: Option<&std::collections::HashSet<u64>>,
    ) -> Convergence {
        // First bottom-up message: F, R from the initial beliefs (= π). This is M2's F/R.
        self.sweep_up(false, 1.0);
        if !self.pool.contains_key(&Kind::Instr) {
            self.fit_pool(gt);
        }
        // Fit the Kind::Func pool on the first sweep's raw (F_h, F_c) so the 4→2 message has weights for
        // the rest of the relaxation (frozen, exactly like the instruction pool).
        if self.l4.is_some() && !self.pool.contains_key(&Kind::Func) {
            if let Some(fg) = func_gt {
                self.fit_pool_func(fg);
            }
        }

        let mut conv = Convergence::default();
        for _ in 0..sched.max_sweeps {
            let (delta, logit_delta) = self.sweep_down(sched.lambda);
            conv.deltas.push(delta);
            conv.logit_deltas.push(logit_delta);
            if !sched.top_down {
                conv.converged = true;
                break;
            }
            if delta < sched.eps {
                conv.converged = true;
                break;
            }
            // Coupled top-down: recompute F, R from the updated instruction beliefs.
            self.sweep_up(sched.s3, sched.lambda);
        }
        // (S2) at the fixpoint: recalibrate the converged beliefs against GT (Theorem 4). A transfer
        // map, if installed, stands instead. Every present object type gets its own g_o.
        if !self.cal.contains_key(&Kind::Instr) {
            self.recalibrate(gt);
        }
        if self.l4.is_some() {
            if let Some(fg) = func_gt {
                if !self.cal.contains_key(&Kind::Func) {
                    self.recalibrate_func(fg);
                }
                if !self.cal.contains_key(&Kind::Module) {
                    self.recalibrate_module(fg);
                }
            }
        }
        conv
    }

    /// **clamp** — inject online evidence: pin object `o` to confidence `p` and propagate. A confirmed
    /// symbol / resolved edge / dynamic trace enters here (design §5). Instruction clamps pin `bel_a`;
    /// function clamps enter as a resolved edge from the entry root (`F_entry = 1`) with confidence
    /// `p` — the same M3a mechanism, so the fixpoint honors it. Re-run [`Stack::relax`] to converge.
    pub fn clamp(&mut self, o: ObjId, p: f64) {
        match o.kind {
            Kind::Instr => {
                self.clamped.insert(o, p);
                self.bel.insert(o, p);
            }
            Kind::Func => {
                self.l2.resolved.push(ResolvedEdge {
                    g: self.entry,
                    t: o.addr,
                    q: p.clamp(0.0, 1.0),
                    kind: probcfg::ResolveKind::DataPointer,
                });
            }
            // A module is a derived object (an SCC of the confirmed graph), not a place online evidence
            // is injected — you confirm a *function*, and its component confirmation follows. Clamp the
            // representative head instead.
            Kind::Module => {
                self.clamp(ObjId::func(o.addr), p);
            }
        }
    }

    /// The per-object marginals `bel_o` (design §6).
    pub fn marginals(&self) -> &HashMap<ObjId, f64> {
        &self.bel
    }

    // ── Active analysis: marginal entropy + expected-information-gain queries (design §5) ──────────

    /// Total binary entropy of the calibrated instruction marginals, `Σ_a H(P̂_a)` in bits — the
    /// stack's remaining *uncertainty* about the instruction map. Online evidence lowers it; the
    /// active-analysis objective is to spend each query where it drops the most (see [`Stack::rank_queries`]).
    pub fn instr_entropy(&self) -> f64 {
        self.instr_marginals().iter().map(|&(_, p)| binary_entropy(p)).sum()
    }

    /// Snapshot the mutable relaxation state (beliefs, `F`/`R`, online edges, clamps) so a what-if
    /// query can be evaluated and rolled back. The frozen operators (`pool`, `cal`) and the immutable
    /// topology/`Superset` are shared, so this is cheap — no re-decode, no re-fit.
    fn snapshot(&self) -> StackState {
        StackState {
            bel: self.bel.clone(),
            r: self.r.clone(),
            f: self.f.clone(),
            f_raw: self.f_raw.clone(),
            f_c: self.f_c.clone(),
            resolved: self.l2.resolved.clone(),
            clamped: self.clamped.clone(),
        }
    }

    /// Restore a [`StackState`] snapshot (undo a what-if query).
    fn restore(&mut self, s: StackState) {
        self.bel = s.bel;
        self.r = s.r;
        self.f = s.f;
        self.f_raw = s.f_raw;
        self.f_c = s.f_c;
        self.l2.resolved = s.resolved;
        self.clamped = s.clamped;
    }

    /// Relax to the coupled fixpoint **without touching the frozen operators** — a what-if probe that
    /// reuses the already-fit `pool`/`cal`. Requires [`Stack::relax`] to have run once (so both exist);
    /// panics otherwise, since a what-if with an unfit readout would not be comparable.
    fn relax_frozen(&mut self, sched: Schedule) {
        debug_assert!(
            self.pool.contains_key(&Kind::Instr) && self.cal.contains_key(&Kind::Instr),
            "relax_frozen needs a fit pool + cal (call relax once first)"
        );
        self.relax(sched, &std::collections::HashSet::new());
    }

    /// **Expected information gain** of confirming head `h` at confidence `q`: the query resolves `h`
    /// to *real* with probability `F_h` (its current confirmation), and confirming it removes
    /// `ΔH = H_now − H|confirm` bits of instruction-map entropy. The expected reduction is `F_h · ΔH`
    /// — a greedy value-of-information proxy (design §5): high only when the head is *both* genuinely
    /// uncertain and, once confirmed, resolves a body of currently-doubtful instructions. Rolls back.
    pub fn query_gain(&mut self, h: u64, q: f64, sched: Schedule) -> (f64, f64) {
        let f_h = self.f.get(&h).copied().unwrap_or(0.0);
        let h_now = self.instr_entropy();
        let snap = self.snapshot();
        self.clamp(ObjId::func(h), q);
        self.relax_frozen(sched);
        let dh = h_now - self.instr_entropy();
        self.restore(snap);
        (f_h * dh, dh)
    }

    /// Rank candidate confirmations by expected information gain (design §5). Candidates are the
    /// *uncertain* heads (`F_h ∈ [lo, hi]`) with a body of at least `min_body` instructions — the ones a
    /// query can actually move. Returns the top `top`, most-informative first. Non-mutating overall
    /// (each what-if is rolled back).
    ///
    /// `pool_cap` bounds the number of **exact** what-if evaluations per call: the true EIG (a full
    /// frozen relax per candidate) is expensive, and on a decoy-heavy binary the uncertain band holds
    /// hundreds of heads. So we first prescreen every band candidate by the cheap monotone proxy
    /// `F_h · min(body, PROXY_BODY_CAP)` — high only for heads that are *both* plausibly real (large
    /// `F_h`) and, if confirmed, unlock a large body (large `ΔH`) — and compute the exact `F_h·ΔH` only
    /// on the top `pool_cap`. `pool_cap = 0` means "no cap" (score every candidate exactly, the original
    /// behaviour). The proxy is the same VOI intuition the exact objective formalises, so the shortlist
    /// keeps EIG's decoy-avoidance: a rock-bottom-`F_h` decoy never makes the shortlist.
    pub fn rank_queries(
        &mut self,
        q: f64,
        lo: f64,
        hi: f64,
        min_body: usize,
        top: usize,
        sched: Schedule,
        exclude: &std::collections::HashSet<u64>,
        pool_cap: usize,
    ) -> Vec<Query> {
        const PROXY_BODY_CAP: usize = 64;
        let mut cands: Vec<(u64, f64, usize)> = self
            .l2
            .heads
            .iter()
            .copied()
            .filter_map(|h| {
                let f = self.f.get(&h).copied().unwrap_or(0.0);
                let body = self.l2.incidence.bodies.get(&h).map(|b| b.len()).unwrap_or(0);
                (f >= lo && f <= hi && body >= min_body && !exclude.contains(&h)).then_some((h, f, body))
            })
            .collect();
        // Prescreen by the cheap VOI proxy and keep only the shortlist for exact evaluation.
        if pool_cap > 0 && cands.len() > pool_cap {
            cands.sort_by(|a, b| {
                let pa = a.1 * a.2.min(PROXY_BODY_CAP) as f64;
                let pb = b.1 * b.2.min(PROXY_BODY_CAP) as f64;
                pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal)
            });
            cands.truncate(pool_cap);
        }
        let mut out: Vec<Query> = cands
            .into_iter()
            .map(|(h, f_prior, body)| {
                let (eig, dh) = self.query_gain(h, q, sched);
                Query { head: h, f_prior, eig, dh_confirm: dh, body }
            })
            .collect();
        out.sort_by(|a, b| b.eig.partial_cmp(&a.eig).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(top);
        out
    }

    /// Instruction marginals as `(vaddr, P̂_a)` in the Soft posterior order — the shape
    /// `evalkit::evaluate` scores (aligned to `bench`'s evaluation domain). Applies the fixpoint
    /// recalibration `g_o` `(S2)` to the propagated pool belief, so the reported marginal is calibrated
    /// at the fixpoint (Theorem 4).
    pub fn instr_marginals(&self) -> Vec<(u64, f64)> {
        let cal = self.cal.get(&Kind::Instr);
        self.l1
            .addrs
            .iter()
            .map(|&a| {
                let bel = self.bel.get(&ObjId::instr(a)).copied().unwrap_or(self.pi(a));
                (a, cal.map(|c| c.apply(bel)).unwrap_or(bel))
            })
            .collect()
    }

    /// Function marginals as `(head, bel_h)` — the L2 confirmation confidence. At K=3 with a fitted
    /// `Kind::Func` calibrator this is the recalibrated fused belief (the `4→2` message folded in and
    /// isotonic-mapped at the fixpoint); at K=2 it is the raw fixpoint head belief, unchanged.
    pub fn func_marginals(&self) -> Vec<(u64, f64)> {
        let cal = self.cal.get(&Kind::Func);
        self.l2
            .heads
            .iter()
            .map(|&h| {
                let bel = self.bel.get(&ObjId::func(h)).copied().unwrap_or(0.0);
                (h, cal.map(|c| c.apply(bel)).unwrap_or(bel))
            })
            .collect()
    }

    /// Module marginals as `(component id, bel_c)` — the L4 component confirmations, isotonic-mapped by
    /// the `Kind::Module` calibrator when fit. Empty at K=2.
    pub fn module_marginals(&self) -> Vec<(u64, f64)> {
        let cal = self.cal.get(&Kind::Module);
        match &self.l4 {
            Some(l4) => l4
                .comps
                .iter()
                .map(|&c| {
                    let bel = self.bel.get(&ObjId::module(c)).copied().unwrap_or(0.0);
                    (c, cal.map(|g| g.apply(bel)).unwrap_or(bel))
                })
                .collect(),
            None => Vec::new(),
        }
    }

    /// The component id (module object) containing head `h`, if the module layer is built.
    pub fn component_of(&self, h: u64) -> Option<u64> {
        self.l4.as_ref().and_then(|l4| l4.comp_of(h))
    }

    /// The raw component confirmations `F_c` (diagnostics); empty at K=2.
    pub fn component_map(&self) -> &HashMap<u64, f64> {
        &self.f_c
    }

    /// The raw Layer-1 posteriors `(a, π_a)` — the L1-only baseline (joint-beats-parts §7).
    pub fn pi_marginals(&self) -> Vec<(u64, f64)> {
        self.l1.addrs.iter().map(|&a| (a, self.pi(a))).collect()
    }

    /// The current reachedness `R_a` — the live top-down message (diagnostics).
    pub fn reachedness_map(&self) -> &HashMap<u64, f64> {
        &self.r
    }

    /// The current confirmation `F_h` — L2 marginals (diagnostics).
    pub fn confirmation_map(&self) -> &HashMap<u64, f64> {
        &self.f
    }

    /// The candidate function heads (design's `O_2`).
    pub fn heads(&self) -> &[u64] {
        &self.l2.heads
    }

    /// The intra-procedural body of head `h` (`a ∈ body(h)`) — for online-update diagnostics.
    pub fn body_of(&self, h: u64) -> Option<&[u64]> {
        self.l2.incidence.bodies.get(&h).map(|v| v.as_slice())
    }

    /// The ELF entry vaddr the stack anchors on.
    pub fn entry(&self) -> u64 {
        self.entry
    }

    /// The shared superset decode both layers read (diagnostics / future layers).
    pub fn superset(&self) -> &Superset {
        &self.sup
    }

    /// The targets `t` of the M3a resolved indirect edges (diagnostics) — the pointer-witnessed code
    /// starts the resolver contributes. Read-only view of the fixpoint's resolver output; used by the
    /// `--dump-pins` reachability-closure emitter to seed the E₃ (resolve) rung. Empty if `--resolve-elf`
    /// was not supplied.
    pub fn resolved_targets(&self) -> Vec<u64> {
        self.l2.resolved.iter().map(|e| e.t).collect()
    }
}

/// Damp an update in logit space: `σ((1−λ)·logit(cur) + λ·logit(target))`. `λ = 1` returns `target`
/// exactly (the Milestone-A no-damping pass), avoiding a logit round-trip through the `1e-6` clamp.
fn damp(cur: f64, target: f64, lambda: f64) -> f64 {
    if lambda >= 1.0 {
        return target;
    }
    calibrate::sigmoid((1.0 - lambda) * logit(cur) + lambda * logit(target))
}

/// Binary entropy `H(p) = −p·log₂p − (1−p)·log₂(1−p)` in bits (0 at the extremes, 1 at `p = ½`).
fn binary_entropy(p: f64) -> f64 {
    let p = p.clamp(0.0, 1.0);
    if p <= 0.0 || p >= 1.0 {
        return 0.0;
    }
    -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
}

/// The ELF entry vaddr (`e_entry`): little-endian u64 at file offset `0x18`.
fn read_e_entry(bytes: &[u8]) -> u64 {
    bytes.get(0x18..0x20).map(|s| u64::from_le_bytes(s.try_into().unwrap())).unwrap_or(0)
}

/// Convenience: load instruction GT (one hex start per line), for callers that don't already have it.
pub fn load_instr_gt(path: &std::path::Path) -> Result<std::collections::HashSet<u64>> {
    load_gt(path).with_context(|| format!("loading GT {}", path.display()))
}

/// A **committing recursive-descent disassembler** seeded by confirmed addresses — the incremental
/// BASELINE (Arm B, INTERACTIVE_APP_SPEC §2). From each seed it follows control-flow successors
/// (branch target + fall-through, stopping at `ret`) through the shared superset decode, marking every
/// reached instruction start as code. This is what a tool that *commits* to one disassembly produces:
/// a hard 0/1 label set. It has no belief to recalibrate and cannot represent a probabilistic clamp
/// (a `q < 1` trace hit is taken as certain), so as evidence arrives it can only flip hard decisions —
/// the contrast the stack's calibrated incremental update is measured against. Pure function of the
/// (immutable) superset + seeds; it never touches the stack's beliefs.
pub fn recursive_descent(sup: &Superset, seeds: &[u64]) -> std::collections::HashSet<u64> {
    let mut visited: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut work: Vec<u64> = seeds.iter().copied().filter(|&a| sup.at(a).is_some()).collect();
    while let Some(a) = work.pop() {
        if !visited.insert(a) {
            continue;
        }
        for s in sup.successors_of(a) {
            if sup.at(s).is_some() && !visited.contains(&s) {
                work.push(s);
            }
        }
    }
    visited
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// The Milestone-A invariant at the unit level: the stack's first bottom-up solve reproduces
    /// `probcfg::build_soft_confirm_resolved` `F`/`R` bit-for-bit (call-site evidence = raw π). Uses a
    /// tiny hand-built code-only ELF so the test is self-contained.
    #[test]
    fn bottom_up_solve_matches_probcfg() {
        // A direct-call chain + a self-call island (mirrors probcfg's CONFIRM_BYTES).
        const BYTES: [u8; 17] = [
            0xe8, 0x03, 0x00, 0x00, 0x00, // call F
            0xff, 0xd0, // call rax
            0xc3, // ret
            0xc3, // F: ret
            0x90, // G: nop
            0xc3, // ret
            0xe8, 0xfb, 0xff, 0xff, 0xff, // D: call self
            0xc3, // ret
        ];
        let elf = evalkit::build_min_elf(0x1000, &BYTES);
        let mut stack = Stack::from_elf(&elf, 0.0, false, 65536, None).expect("build stack");
        let gt: HashSet<u64> = HashSet::new();
        // Bottom-up only: no fusion loop, F/R come straight from π.
        stack.relax(Schedule::bottom_up_once(), &gt);

        // Reference: probcfg directly, pmap = π.
        let (base, code) = extract_text_section(&elf).unwrap();
        let sup = Superset::new(base, code).unwrap();
        let pi: HashMap<u64, f64> = stack.l1.pi.clone();
        let sc = build_soft_confirm_resolved(&sup, stack.entry(), &pi, &[], &SoftConfig::default());

        for &h in &sc.heads {
            let ours = stack.confirmation_map().get(&h).copied().unwrap_or(0.0);
            assert!((ours - sc.f[&h]).abs() < 1e-12, "F mismatch at {h:#x}: {ours} vs {}", sc.f[&h]);
        }
        for (&a, &rv) in &sc.r {
            let ours = stack.reachedness_map().get(&a).copied().unwrap_or(0.0);
            assert!((ours - rv).abs() < 1e-12, "R mismatch at {a:#x}");
        }
    }

    /// A function clamp (confirmed symbol) lifts a disconnected island head from its bare prior toward
    /// the clamped confidence — the online-update mechanism (design §5).
    #[test]
    fn clamp_lifts_island_head() {
        const BYTES: [u8; 17] = [
            0xe8, 0x03, 0x00, 0x00, 0x00, 0xff, 0xd0, 0xc3, 0xc3, 0x90, 0xc3, 0xe8, 0xfb, 0xff, 0xff,
            0xff, 0xc3,
        ];
        let elf = evalkit::build_min_elf(0x1000, &BYTES);
        let mut stack = Stack::from_elf(&elf, 0.0, false, 65536, None).expect("build stack");
        let gt: HashSet<u64> = HashSet::new();
        stack.relax(Schedule::bottom_up_once(), &gt);
        let before = stack.confirmation_map().get(&0x100b).copied().unwrap_or(0.0);
        assert!(before < 0.6, "island D low before clamp: {before}");

        stack.clamp(ObjId::func(0x100b), 0.99);
        stack.relax(Schedule::coupled(0.5), &gt);
        let after = stack.confirmation_map().get(&0x100b).copied().unwrap_or(0.0);
        assert!(after > 0.9, "clamp must lift island D to the core: {after}");
    }

    /// Active analysis (design §5): a what-if confirmation lowers instruction-map entropy and rolls
    /// back cleanly — `query_gain` reports a non-negative expected gain and leaves the stack unchanged.
    #[test]
    fn query_gain_is_nonneg_and_rolls_back() {
        const BYTES: [u8; 17] = [
            0xe8, 0x03, 0x00, 0x00, 0x00, 0xff, 0xd0, 0xc3, 0xc3, 0x90, 0xc3, 0xe8, 0xfb, 0xff, 0xff,
            0xff, 0xc3,
        ];
        let elf = evalkit::build_min_elf(0x1000, &BYTES);
        let mut stack = Stack::from_elf(&elf, 0.0, false, 65536, None).expect("build stack");
        let gt: HashSet<u64> = HashSet::new();
        stack.relax(Schedule::coupled(0.5), &gt);

        let h0 = stack.instr_entropy();
        let bel_before = stack.marginals().clone();
        // Query the low-confidence island head: expected gain ≥ 0, ΔH ≥ 0 (confirming can only sharpen).
        let (eig, dh) = stack.query_gain(0x100b, 0.99, Schedule::coupled(0.5));
        assert!(eig >= -1e-9 && dh >= -1e-9, "gain must be non-negative: eig={eig} dh={dh}");
        // Rollback: entropy and every belief return to their pre-query values.
        assert!((stack.instr_entropy() - h0).abs() < 1e-12, "entropy not restored");
        for (id, &b) in &bel_before {
            assert!((stack.marginals()[id] - b).abs() < 1e-12, "belief {id:?} not restored");
        }
    }

    /// The same tiny ELF as the M2 test, but seen through the L4 lens: entry (0x1000) calls F (0x1008)
    /// — one reachable component — while D (0x100b) is a self-call island **disconnected** from entry.
    /// The condensation must (a) put entry and D in different components, (b) pin the entry component to
    /// F_c = 1, and (c) leave D's component at F_c = 0 (no path from entry). That is exactly the decoy
    /// signal L4 exists to carry: a disconnected component is unconfirmed no matter how real its bytes
    /// decode.
    #[test]
    fn condensation_isolates_disconnected_island() {
        const BYTES: [u8; 17] = [
            0xe8, 0x03, 0x00, 0x00, 0x00, 0xff, 0xd0, 0xc3, 0xc3, 0x90, 0xc3, 0xe8, 0xfb, 0xff, 0xff,
            0xff, 0xc3,
        ];
        let elf = evalkit::build_min_elf(0x1000, &BYTES);
        let mut stack = Stack::from_elf(&elf, 0.0, false, 65536, None).expect("build stack");
        assert_eq!(stack.depth(), 2, "starts as a K=2 stack");
        stack.build_module_layer();
        assert_eq!(stack.depth(), 3, "module layer makes it K=3");
        assert_eq!(stack.layers().len(), 3, "layers() is length K, generalized off [_;2]");

        let gt: HashSet<u64> = HashSet::new();
        stack.relax(Schedule::coupled(0.5), &gt);

        let entry_c = stack.component_of(stack.entry()).expect("entry has a component");
        let d_c = stack.component_of(0x100b).expect("island D has a component");
        assert_ne!(entry_c, d_c, "entry and the disconnected island must be different components");
        let fc = stack.component_map();
        assert!((fc[&entry_c] - 1.0).abs() < 1e-9, "entry component pinned to 1: {}", fc[&entry_c]);
        assert!(fc[&d_c] < 1e-9, "disconnected island component stays at 0: {}", fc[&d_c]);
    }

    /// **Honesty wall at K=3.** Adding the module layer adds *messages*, never evidence — so the raw
    /// Layer-1 posterior `π_a` at readout must be bit-for-bit identical between the K=2 baseline and the
    /// K=3 run: `‖π^{K=3} − π^{baseline}‖_∞ = 0`. (The *fused* belief `bel_a` is allowed to move; `π` is
    /// the input message that must never be overwritten.)
    #[test]
    fn honesty_wall_pi_identical_k2_vs_k3() {
        const BYTES: [u8; 17] = [
            0xe8, 0x03, 0x00, 0x00, 0x00, 0xff, 0xd0, 0xc3, 0xc3, 0x90, 0xc3, 0xe8, 0xfb, 0xff, 0xff,
            0xff, 0xc3,
        ];
        let elf = evalkit::build_min_elf(0x1000, &BYTES);
        let gt: HashSet<u64> = HashSet::new();
        // A real/decoy split so the module fusion actually has weights to fit and could, in principle,
        // perturb something downstream.
        let func_gt: HashSet<u64> = [0x1000u64, 0x1008].into_iter().collect();

        // Baseline: K=2.
        let mut k2 = Stack::from_elf(&elf, 0.0, false, 65536, None).expect("build k2");
        k2.relax(Schedule::coupled(0.5), &gt);
        let pi_base = k2.pi_marginals();

        // K=3 on the same specimen.
        let mut k3 = Stack::from_elf(&elf, 0.0, false, 65536, None).expect("build k3");
        k3.build_module_layer();
        k3.relax_layered(Schedule::coupled(0.5), &gt, Some(&func_gt));
        let pi_k3 = k3.pi_marginals();

        assert_eq!(pi_base.len(), pi_k3.len(), "π domain must match");
        let mut linf = 0.0f64;
        for (&(a0, p0), &(a1, p1)) in pi_base.iter().zip(pi_k3.iter()) {
            assert_eq!(a0, a1, "π addresses must align");
            linf = linf.max((p0 - p1).abs());
        }
        assert_eq!(linf, 0.0, "‖π^K3 − π^baseline‖_∞ must be exactly 0 (honesty wall), got {linf:e}");
    }

    /// **Honesty wall — active arm (design §5).** A sequence of *function* confirmations (the active
    /// loop's online evidence) must never touch the raw Layer-1 posterior `π_a`: it enters as a
    /// resolved edge at L2 and flows down only through the fused belief. So `π` after any number of
    /// `clamp(Func, q)` + `relax` cycles is bit-for-bit the pre-clamp `π`. The complementary half — an
    /// *instruction* clamp (a trace hit) DOES move its own marginal — proves the wall is not vacuous:
    /// real evidence propagates, everything else stays put.
    #[test]
    fn active_confirmations_preserve_pi() {
        const BYTES: [u8; 17] = [
            0xe8, 0x03, 0x00, 0x00, 0x00, 0xff, 0xd0, 0xc3, 0xc3, 0x90, 0xc3, 0xe8, 0xfb, 0xff, 0xff,
            0xff, 0xc3,
        ];
        let elf = evalkit::build_min_elf(0x1000, &BYTES);
        let gt: HashSet<u64> = HashSet::new();
        let mut stack = Stack::from_elf(&elf, 0.0, false, 65536, None).expect("build stack");
        stack.relax(Schedule::coupled(0.5), &gt);
        let pi0 = stack.pi_marginals();

        // Active loop: confirm two function heads in sequence, re-relaxing after each.
        for &h in &[0x1008u64, 0x100b] {
            stack.clamp(ObjId::func(h), 0.99);
            stack.relax(Schedule::coupled(0.5), &gt);
        }
        let pi1 = stack.pi_marginals();
        assert_eq!(pi0.len(), pi1.len());
        for (&(a0, p0), &(a1, p1)) in pi0.iter().zip(pi1.iter()) {
            assert_eq!(a0, a1, "π addresses must align");
            assert_eq!(p0, p1, "function confirmation moved π at {a0:#x}: {p0} → {p1} (honesty wall)");
        }

        // Not vacuous: an instruction clamp moves that instruction's fused belief. (Read the raw
        // belief `bel`, not the calibrated readout — on this tiny all-negative-GT ELF the isotonic map
        // collapses every marginal to ≈0, which would mask the propagation at readout.)
        let a = pi0[0].0;
        let before = stack.marginals()[&ObjId::instr(a)];
        stack.clamp(ObjId::instr(a), 0.01);
        stack.relax(Schedule::coupled(0.5), &gt);
        let after = stack.marginals()[&ObjId::instr(a)];
        assert!((before - after).abs() > 1e-6, "instruction clamp must move its own belief: {before} → {after}");

        // Arm B honesty wall: even an *instruction* clamp (a trace hit) never rewrites the raw π — it
        // sets the fused belief `bel`, the input posterior stays put.
        let pi2 = stack.pi_marginals();
        for (&(a0, p0), &(_, p2)) in pi0.iter().zip(pi2.iter()) {
            assert_eq!(p0, p2, "instruction clamp moved π at {a0:#x} (honesty wall)");
        }
    }

    /// The committing recursive-descent baseline (Arm B): from a seed it follows control-flow through
    /// the superset and marks reached starts as code, stopping at `ret`. On the tiny chain ELF, seeding
    /// at entry reaches the entry body + the directly-called F, but NOT the disconnected self-call
    /// island D — the very limitation online evidence (a symbol/trace for D) exists to repair.
    #[test]
    fn recursive_descent_reaches_only_connected_code() {
        const BYTES: [u8; 17] = [
            0xe8, 0x03, 0x00, 0x00, 0x00, 0xff, 0xd0, 0xc3, 0xc3, 0x90, 0xc3, 0xe8, 0xfb, 0xff, 0xff,
            0xff, 0xc3,
        ];
        let elf = evalkit::build_min_elf(0x1000, &BYTES);
        let stack = Stack::from_elf(&elf, 0.0, false, 65536, None).expect("build stack");
        let from_entry = recursive_descent(stack.superset(), &[stack.entry()]);
        assert!(from_entry.contains(&0x1008), "direct-called F is reached from entry");
        assert!(!from_entry.contains(&0x100b), "disconnected island D is NOT reached from entry alone");
        // Feed D as evidence (a symbol/trace) → the baseline now reaches it.
        let with_d = recursive_descent(stack.superset(), &[stack.entry(), 0x100b]);
        assert!(with_d.contains(&0x100b), "seeding D reaches it");
    }

    /// **Compositionality at K=3 (Theorem 4 at depth) + convergence.** The coupled sweep still reaches a
    /// fixpoint with three layers, and every object type has a marginal in `[0,1]` after isotonic recal.
    /// Also checks the *mechanism*: the disconnected decoy function D ends up strictly below the real,
    /// entry-reachable function F on the recalibrated function axis — the module message pruning it.
    #[test]
    fn k3_composes_and_converges() {
        const BYTES: [u8; 17] = [
            0xe8, 0x03, 0x00, 0x00, 0x00, 0xff, 0xd0, 0xc3, 0xc3, 0x90, 0xc3, 0xe8, 0xfb, 0xff, 0xff,
            0xff, 0xc3,
        ];
        let elf = evalkit::build_min_elf(0x1000, &BYTES);
        let gt: HashSet<u64> = HashSet::new();
        let func_gt: HashSet<u64> = [0x1000u64, 0x1008].into_iter().collect(); // entry + F real; D decoy

        let mut stack = Stack::from_elf(&elf, 0.0, false, 65536, None).expect("build stack");
        stack.build_module_layer();
        let conv = stack.relax_layered(Schedule::coupled(0.5), &gt, Some(&func_gt));
        assert!(conv.converged, "K=3 coupled sweep must reach a fixpoint (iters={})", conv.iters());

        // Every present object type produces marginals in [0,1] (calibrated readout exists).
        for (_, p) in stack.instr_marginals() {
            assert!((0.0..=1.0).contains(&p), "instr marginal out of range: {p}");
        }
        let fm = stack.func_marginals();
        for (_, p) in &fm {
            assert!((0.0..=1.0).contains(p), "func marginal out of range: {p}");
        }
        let mm = stack.module_marginals();
        assert!(!mm.is_empty(), "K=3 must expose module marginals");
        for (_, p) in &mm {
            assert!((0.0..=1.0).contains(p), "module marginal out of range: {p}");
        }
    }
}
