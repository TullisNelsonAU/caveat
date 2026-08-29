//! Probabilistic disassembly inference.
//!
//! Hard mode: Miller Algorithm 1 as published (unchanged).
//! Soft mode: explicit-factor loopy sum-product BP on a factor graph with:
//!   - Unary factors φ_a: local hint coincidence evidence at each address
//!   - Overlap factors ψ_ovl: mutual exclusion for overlapping instruction candidates
//!   - Fall-through factors ψ_ft: soft coupling from instruction to its known successor(s)

use std::collections::{HashMap, HashSet};

use crate::hints::{HintKey, HintPair};
use crate::superset::{Instruction, Superset};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Sentinel for log-probability zero. Used instead of NEG_INFINITY to avoid
/// propagating -inf through log-sum-exp. Must dominate any finite log-potential.
const LOG_ZERO: f64 = -1e30;

const BP_MAX_ITER: usize = 400;
const BP_CONVERGE: f64 = 1e-6;
const BP_MARG_WINDOW: usize = 10;
const BP_MARG_TOL: f64 = 1e-4;

// ── Public types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnalysisMode {
    /// Miller Algorithm 1 as published: hard-kills occluded instructions in the
    /// iteration loop; backward propagation at full strength (σ = 1.0).
    Hard,
    /// Loopy sum-product BP on an explicit factor graph. Replaces the heuristic
    /// Luce normalization and max-product backward pass with proper marginal inference.
    Soft,
}

#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    pub mode: AnalysisMode,
    /// Message mixing coefficient λ ∈ [0, 1) for numerical stability.
    /// Applied as μ_new = (1-λ)·μ_raw + λ·μ_old in linear space (log-sum-exp in log space).
    /// Soft mode only. Default 0.5.
    pub msg_damp: f64,
    /// Fall-through factor softness ε ∈ (0, 1].
    /// log ψ_ft(x_src=1, x_tgt=0) = ln(ε). Smaller ε → stronger coupling.
    /// This is the sweep target parameter. Default 0.5 (honest starting point; learn from corpus).
    pub ft_eps: f64,
    /// Global evidence scale ∈ (0, ∞]. Applied to unary (CtrlWeak phi) and HintCoupling
    /// log_weight only. Does NOT scale overlap (-1e30) or fall-through (log ft_eps).
    /// Default 1.0 (unscaled). Values < 1.0 weaken evidence; useful for calibration sweep.
    pub evidence_scale: f64,
    /// Log-weight for Transfer pairwise factor (1,1) corner.
    /// Applied as transfer_log_weight * evidence_scale. Default 4.0.
    pub transfer_log_weight: f64,
    /// Prior P(code) for valid-decode addresses with no firing single-address hints.
    /// Sets phi[a] = [log(1-p), log(p)] instead of [0, 0]. Default 0.2 (corpus base rate).
    pub unhinted_code_prob: f64,
    /// Reaching-hints transport — the §3.4 evidence-transport mechanism, and the
    /// dominant design choice (turning it off drops obfuscated recall ~0.97 → ~0.78).
    /// When > 0, forward-propagate hints to a fixpoint and build each address's unary
    /// from its reaching-hint log-product, scaled by this factor: phi[a] = [s·L(a), −s·L(a)]
    /// with L(a) = Σ_{h∈R(a)} ln p_h ≤ 0. This OVERRIDES the local-hint unary of Eq. (3)
    /// for any address it reaches (R(a) ⊇ local hints). Default 1.0 = the reported config;
    /// 0.0 disables transport and falls back to the local-hint unary.
    pub reaching_scale: f64,
    /// Strength of the entropy-aware data prior (DASSA-style). 0.0 = off (the default and the
    /// behavior before this knob existed). When > 0, every valid-decode address gets a log-odds
    /// push toward *data* of `strength · max(0, local_entropy_bits − entropy_floor_bits)`, so
    /// high-entropy regions (packed / encrypted payloads) are pulled away from "code". The
    /// statistical-properties-of-data idea from DASSA, folded into the unary potential.
    pub entropy_prior_strength: f64,
    /// Local-entropy floor, in bits, below which the entropy prior contributes nothing. Set to
    /// code-typical entropy (~6 bits) so ordinary code is left untouched; only the compressed
    /// tail above it is penalized. Default 6.0.
    pub entropy_floor_bits: f64,
    /// Strength of the forward decode-chain-consistency data prior. 0.0 = off (the default, and the
    /// behavior before this knob existed — with it 0, output is bit-for-bit the pre-chainfwd engine).
    /// When > 0, every valid-decode address gets a log-odds push *toward code* of `strength · c`,
    /// where `c ∈ [0,1]` is the normalized forward consistent fall-through chain length from that
    /// address (see [`chain_fwd`]). This is the mirror image of the entropy prior: entropy pushes
    /// high-entropy bytes toward *data*, chainfwd pushes chain-consistent bytes toward *code*.
    ///
    /// The blob go/no-go study found this is the one structural per-byte feature that generalizes
    /// out-of-distribution — the walk is ported verbatim from `blobscan/src/features.rs` (`chainfwd`)
    /// so the integrated prior is the same signal the study validated. Like the entropy prior it is a
    /// prior on the *posterior*: it reweights the unary evidence and then flows through BP and the
    /// downstream isotonic recal, it never overwrites π. Opt-in on purpose — the shared engine's
    /// default has to stay untouched for the papers building on it. Default 0.0.
    pub chainfwd_strength: f64,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            mode: AnalysisMode::Hard,
            msg_damp: 0.5,
            ft_eps: 0.5,
            evidence_scale: 1.0,
            transfer_log_weight: 4.0,
            unhinted_code_prob: 0.2,
            reaching_scale: 1.0,
            entropy_prior_strength: 0.0,
            entropy_floor_bits: 6.0,
            chainfwd_strength: 0.0,
        }
    }
}

// ── Hard-mode data probability (unchanged) ────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataProb {
    DefinitelyData,
    Estimated(f64),
    Unknown,
}

// ── BP factor graph types ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
enum BpFactorKind {
    /// log ψ(x_a=1, x_b=1) = LOG_ZERO; all others = 0.
    Overlap,
    /// log ψ(x_src=1, x_tgt=0) = ln(ft_eps); all others = 0.
    /// vars[0] is the source, vars[1] is the target.
    FallThrough,
    /// Relational hint coupling: log ψ(1,1) = log_weight; all others = 0.
    /// Symmetric. log_weight = -log(p_h) > 0 (upweights joint-code config).
    HintCoupling(f64),
    /// Direct-branch transfer: log ψ(1,1) = log_weight; all others = 0.
    /// Identical potential to HintCoupling; semantic difference is conceptual only
    /// (direct branch src→tgt, not a hint pair).
    Transfer(f64),
}

#[derive(Clone, Debug)]
struct BpFactor {
    kind: BpFactorKind,
    vars: [usize; 2],
}

struct BpState {
    n: usize,
    /// phi[i] = [log φ(x=0), log φ(x=1)]
    phi: Vec<[f64; 2]>,
    factors: Vec<BpFactor>,
    /// factor-to-variable messages: cur_f2v[f][side][x_val]
    cur_f2v: Vec<[[f64; 2]; 2]>,
    /// variable-to-factor messages: cur_v2f[f][side][x_val]
    cur_v2f: Vec<[[f64; 2]; 2]>,
}

// ── Factor graph construction ─────────────────────────────────────────────────

/// Half-width, in bytes, of the window used to measure local entropy for the entropy prior.
/// It has to be wide enough to reach the 8-bit ceiling — a W-byte window holds at most W
/// distinct values, so its entropy caps at log2(W); 128 each side (256 total) spans 0..8 bits.
const ENTROPY_HALF_WINDOW: usize = 128;

/// Shannon entropy, in bits, of the byte distribution in the window centered on `offset`.
/// Returns 0 for a constant window and approaches 8 as the distribution flattens toward
/// uniform — which is what compressed and encrypted data look like, and what plain code does
/// not. Used only by the entropy prior; cheap enough to call per candidate.
fn local_entropy(bytes: &[u8], offset: usize, half_window: usize) -> f64 {
    let lo = offset.saturating_sub(half_window);
    let hi = (offset + half_window).min(bytes.len());
    let window = &bytes[lo..hi];
    if window.is_empty() {
        return 0.0;
    }
    let mut counts = [0u32; 256];
    for &b in window {
        counts[b as usize] += 1;
    }
    let len = window.len() as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / len;
            -p * p.log2()
        })
        .sum()
}

/// Cap on the forward fall-through chain walk, matching the reference `chainfwd` in
/// `blobscan/src/features.rs` (chain length is normalized by this cap). Max x86 instruction is 15
/// bytes, so 16 steps reach well past a typical basic block.
const CHAINFWD_CAP: usize = 16;

/// Length of the forward consistent fall-through chain starting at `offset`, capped at
/// [`CHAINFWD_CAP`]. Ported verbatim from `chain_fwd` in `blobscan/src/features.rs`: advance
/// `o += size[o]` while each landing offset decodes and stays in bounds, counting steps. A real
/// instruction start begins a long self-consistent chain; a mid-instruction or junk offset desyncs
/// fast. This is the exact walk the go/no-go study validated as the lone OOD-robust structural prior
/// — do not silently redefine it; if the reference changes, change it here to match.
fn chain_fwd(superset: &Superset, offset: usize, cap: usize) -> usize {
    let n = superset.instructions.len();
    let mut o = offset;
    let mut steps = 0;
    while steps < cap && o < n {
        // Landing offset must decode with a positive size, else the chain is broken.
        let Some(ins) = superset.instructions[o].as_ref() else {
            break;
        };
        let sz = ins.size as usize;
        if sz == 0 {
            break;
        }
        steps += 1;
        o += sz;
    }
    steps
}

fn build_factor_graph(
    superset: &Superset,
    hint_priors: &HashMap<HintKey, f64>,
    hint_pairs: &[HintPair],
    _config: &AnalysisConfig,
    reaching: Option<&HashMap<u64, f64>>,
) -> BpState {
    let n = superset.instructions.len();
    let base = superset.base_addr;

    // Unary log-potentials from hint likelihood ratios.
    //
    // Each hint h has prior p_h = P(hint fires | random byte) — the false-positive rate.
    // Very small priors (1/255 … 1/2^32) mean the hint rarely fires by coincidence.
    //
    // Likelihood ratio:
    //   phi[0] = log P(hints | data) = Σ log p_h  (small → very negative for strong hints)
    //   phi[1] = log P(hints | code) = 0           (hints reliably fire at code starts)
    //
    // At addresses with no hints: phi = [0, 0] (uninformative uniform prior).
    // Invalid decodes (None): phi[1] = LOG_ZERO (code structurally impossible).
    let hints_by_src = group_hints_by_source(hint_priors);
    // Base-rate prior for unhinted valid addresses: P(code) = unhinted_code_prob.
    // Hinted addresses override this; invalid-decode addresses get LOG_ZERO on phi[1].
    let p = _config
        .unhinted_code_prob
        .clamp(f64::MIN_POSITIVE, 1.0 - f64::MIN_POSITIVE);
    let base_phi = [(1.0 - p).ln(), p.ln()];
    let mut phi = vec![base_phi; n];
    for offset in 0..n {
        let addr = base + offset as u64;
        if let Some(local) = hints_by_src.get(&addr) {
            let log_data: f64 = local
                .iter()
                .filter_map(|h| hint_priors.get(h))
                .map(|&p| p.ln())
                .sum::<f64>()
                * _config.evidence_scale;
            phi[offset] = [log_data, -log_data];
        }
        // else: unhinted valid address keeps base_phi = [log(1-p), log(p)]
        // E4: reaching-hints transport overrides the local unary when present.
        // R(a) includes a's own hints, so this subsumes the local-hint case above.
        if let Some(r) = reaching {
            if let Some(&log_data) = r.get(&addr) {
                let l = log_data * _config.reaching_scale; // log_data ≤ 0
                phi[offset] = [l, -l];
            }
        }
        // Entropy-aware data prior (opt-in). Applied after the hint/reaching unary so it adjusts
        // whatever evidence we already have: a valid decode sitting in a high-entropy region gets
        // pushed toward data in proportion to how far its local entropy exceeds the floor. Invalid
        // decodes are handled below and stay impossible regardless.
        if _config.entropy_prior_strength > 0.0 && superset.instructions[offset].is_some() {
            let bits = local_entropy(&superset.bytes, offset, ENTROPY_HALF_WINDOW);
            let penalty = _config.entropy_prior_strength * (bits - _config.entropy_floor_bits).max(0.0);
            phi[offset][0] += penalty; // toward data
            phi[offset][1] -= penalty; // away from code
        }
        // Forward decode-chain-consistency prior (opt-in). Mirror image of the entropy prior above:
        // a valid decode that begins a long self-consistent fall-through chain gets pushed *toward*
        // code, in proportion to how much of the cap-16 chain it sustains. Applied to the same unary
        // the hint/reaching/entropy evidence already shaped, so it flows through BP and isotonic recal
        // like every other factor — never a post-hoc π overwrite. strength 0 skips this block, which
        // is what keeps the default engine byte-for-byte unchanged.
        if _config.chainfwd_strength > 0.0 && superset.instructions[offset].is_some() {
            let chain = chain_fwd(superset, offset, CHAINFWD_CAP) as f64 / CHAINFWD_CAP as f64;
            let push = _config.chainfwd_strength * chain; // chain ∈ [0,1]
            phi[offset][1] += push; // toward code
            phi[offset][0] -= push; // away from data
        }
        if superset.instructions[offset].is_none() {
            phi[offset][1] = LOG_ZERO;
        }
    }

    let mut factors: Vec<BpFactor> = Vec::new();

    // Overlap factors (mutual exclusion).
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    for offset in 0..n {
        if superset.instructions[offset].is_none() {
            continue;
        }
        let addr = base + offset as u64;
        for peer in occluding_addrs_of(superset, addr) {
            let Some(po) = peer.checked_sub(base).map(|o| o as usize) else {
                continue;
            };
            if po >= n {
                continue;
            }
            let key = if offset < po {
                (offset, po)
            } else {
                (po, offset)
            };
            if seen.insert(key) {
                factors.push(BpFactor {
                    kind: BpFactorKind::Overlap,
                    vars: [key.0, key.1],
                });
            }
        }
    }

    // Fall-through factors.
    for offset in 0..n {
        let Some(ref insn) = superset.instructions[offset] else {
            continue;
        };
        let addr = base + offset as u64;
        for tgt in ft_factor_targets(insn, addr) {
            let Some(to) = tgt.checked_sub(base).map(|o| o as usize) else {
                continue;
            };
            if to >= n {
                continue;
            }
            // Skip FT to invalid-decode targets: phi[to][code]=LOG_ZERO would otherwise
            // leak extreme data-certainty signal back to valid source through the factor,
            // bypassing the soft-evidence intent of the factor graph.
            if superset.instructions[to].is_none() {
                continue;
            }
            factors.push(BpFactor {
                kind: BpFactorKind::FallThrough,
                vars: [offset, to],
            });
        }
    }

    // HintCoupling factors from pairwise relational hints (CtrlConv, CtrlCross, RegDefUse).
    for pair in hint_pairs {
        let Some(oa) = pair.addr_a.checked_sub(base).map(|o| o as usize) else {
            continue;
        };
        let Some(ob) = pair.addr_b.checked_sub(base).map(|o| o as usize) else {
            continue;
        };
        if oa >= n || ob >= n {
            continue;
        }
        if superset.instructions[oa].is_none() || superset.instructions[ob].is_none() {
            continue;
        }
        factors.push(BpFactor {
            kind: BpFactorKind::HintCoupling(pair.log_weight * _config.evidence_scale),
            vars: [oa, ob],
        });
    }

    // Transfer factors: direct branches (jmp, jcc, call) with resolvable target in superset.
    // Same (1,1)-upweight potential as HintCoupling; src=branch instruction, tgt=branch target.
    // Call targets are included (unlike FallThrough which skips call targets per procedure-boundary rule).
    let transfer_lw = _config.transfer_log_weight * _config.evidence_scale;
    for offset in 0..n {
        let Some(ref insn) = superset.instructions[offset] else {
            continue;
        };
        if !insn.is_jump() && !insn.is_call() {
            continue;
        }
        let Some(tgt) = insn.branch_target else {
            continue;
        };
        let Some(to) = tgt.checked_sub(base).map(|o| o as usize) else {
            continue;
        };
        if to >= n {
            continue;
        }
        if superset.instructions[to].is_none() {
            continue;
        }
        factors.push(BpFactor {
            kind: BpFactorKind::Transfer(transfer_lw),
            vars: [offset, to],
        });
    }

    let nf = factors.len();
    BpState {
        n,
        phi,
        factors,
        cur_f2v: vec![[[0.0; 2]; 2]; nf],
        cur_v2f: vec![[[0.0; 2]; 2]; nf],
    }
}

/// Addresses whose byte ranges overlap that of the instruction at `addr`.
fn occluding_addrs_of(superset: &Superset, addr: u64) -> Vec<u64> {
    let Some(insn) = superset.at(addr) else {
        return Vec::new();
    };
    let i_end = addr + insn.size as u64;
    let scan_start = addr.saturating_sub(14);
    let scan_end = i_end + 14;
    let mut out = Vec::new();
    let mut a = scan_start;
    while a < scan_end {
        if a != addr {
            if let Some(j) = superset.at(a) {
                let j_end = a + j.size as u64;
                if a < i_end && addr < j_end {
                    out.push(a);
                }
            }
        }
        a += 1;
    }
    out
}

/// Addresses that receive a fall-through factor FROM the instruction at `addr`.
///
/// Rules:
///   Sequential / conditional-jmp fall-through  → ft factor to fallthrough address
///   Conditional jmp target                      → ft factor to branch target
///   Unconditional direct jmp                    → ft factor to target only (no fallthrough)
///   Call fall-through (return site)             → ft factor to fallthrough only
///   Call target                                 → NO factor (procedure boundary)
///   Indirect branch (no known target)           → no factor
///   Ret                                         → no factor
fn ft_factor_targets(insn: &Instruction, addr: u64) -> Vec<u64> {
    let fallthrough = addr + insn.size as u64;
    if insn.is_ret() {
        return vec![];
    }
    if insn.is_call() {
        return vec![fallthrough];
    } // return site only; not call target
    if insn.is_jump() {
        return match insn.branch_target {
            Some(tgt) if insn.mnemonic == "jmp" => vec![tgt], // unconditional
            Some(tgt) => vec![tgt, fallthrough],              // conditional
            None => vec![],                                   // indirect
        };
    }
    vec![fallthrough]
}

// ── Sum-product BP ────────────────────────────────────────────────────────────

#[inline]
fn log_potential(kind: BpFactorKind, xa: usize, xb: usize, log_ft_eps: f64) -> f64 {
    match kind {
        BpFactorKind::Overlap => {
            if xa == 1 && xb == 1 {
                LOG_ZERO
            } else {
                0.0
            }
        }
        // Asymmetric, matching the AnalysisConfig spec: code falls through to code
        // (penalize src=1,tgt=0); data constrains nothing (0,1) and (0,0) free.
        // E1 experiment 2026-06-04: was symmetric `xa != xb`, which let wrong-phase
        // data runs drag true instructions down as hard as code runs pulled them up.
        BpFactorKind::FallThrough => {
            if xa == 1 && xb == 0 {
                log_ft_eps
            } else {
                0.0
            }
        }
        BpFactorKind::HintCoupling(lw) => {
            if xa == 1 && xb == 1 {
                lw
            } else {
                0.0
            }
        }
        BpFactorKind::Transfer(lw) => {
            if xa == 1 && xb == 1 {
                lw
            } else {
                0.0
            }
        }
    }
}

#[inline]
fn log_sum_exp2(a: f64, b: f64) -> f64 {
    let max = a.max(b);
    if !max.is_finite() {
        return max;
    }
    max + ((a - max).exp() + (b - max).exp()).ln()
}

/// Compute P(xi=1) for every variable from the current message state.
fn marginals(state: &BpState) -> Vec<f64> {
    let mut total_in = vec![[0.0f64; 2]; state.n];
    for fi in 0..state.factors.len() {
        for s in 0..2 {
            let vi = state.factors[fi].vars[s];
            total_in[vi][0] += state.cur_f2v[fi][s][0];
            total_in[vi][1] += state.cur_f2v[fi][s][1];
        }
    }
    (0..state.n)
        .map(|i| {
            let b0 = state.phi[i][0] + total_in[i][0];
            let b1 = state.phi[i][1] + total_in[i][1];
            (b1 - log_sum_exp2(b0, b1)).exp()
        })
        .collect()
}

/// Synchronous (flooding) loopy sum-product BP with message damping.
///
/// Two convergence criteria (either triggers early exit):
///   1. Message convergence: max |Δf2v| < BP_CONVERGE (1e-6).
///   2. Marginal stability: max |marginal(t) − marginal(t − BP_MARG_WINDOW)| < BP_MARG_TOL (1e-4),
///      checked every BP_MARG_WINDOW iterations. Handles oscillating messages whose marginals
///      have stabilised (common in loopy graphs with strong damping).
///
/// Returns iteration count at first convergence; returns BP_MAX_ITER if neither fires.
/// Returns `(iters_ran, final_max_marg_delta)`.
/// Runs for up to BP_MAX_ITER iterations. Exits early if message convergence fires
/// (max |Δf2v| < BP_CONVERGE) or marginal stability fires (max |Δmarginal| < BP_MARG_TOL
/// over BP_MARG_WINDOW iterations). On the iteration cap the marginals are stable for this
/// graph topology even without formal convergence; callers should use the posterior state
/// regardless.
fn run_bp(state: &mut BpState, config: &AnalysisConfig) -> (usize, f64) {
    let nf = state.factors.len();
    if nf == 0 {
        return (0, 0.0);
    }

    let log_ft_eps = config.ft_eps.ln();
    let log_lambda = config.msg_damp.ln();
    let log_1m_lambda = (1.0 - config.msg_damp).ln();

    let mut nxt_f2v: Vec<[[f64; 2]; 2]> = vec![[[0.0; 2]; 2]; nf];
    let mut nxt_v2f: Vec<[[f64; 2]; 2]> = vec![[[0.0; 2]; 2]; nf];
    let mut prev_marg: Option<Vec<f64>> = None;
    let mut last_marg_delta = f64::NAN;

    for iter in 0..BP_MAX_ITER {
        let mut total_in: Vec<[f64; 2]> = vec![[0.0, 0.0]; state.n];
        for fi in 0..nf {
            for s in 0..2 {
                let vi = state.factors[fi].vars[s];
                total_in[vi][0] += state.cur_f2v[fi][s][0];
                total_in[vi][1] += state.cur_f2v[fi][s][1];
            }
        }

        for fi in 0..nf {
            for s in 0..2 {
                let vi = state.factors[fi].vars[s];
                for x in 0..2 {
                    nxt_v2f[fi][s][x] =
                        state.phi[vi][x] + total_in[vi][x] - state.cur_f2v[fi][s][x];
                }
            }
        }

        // f→v: marginalize, normalize (subtract lse to bound messages to (-∞, 0]).
        for fi in 0..nf {
            let kind = state.factors[fi].kind;
            for s in 0..2usize {
                let os = 1 - s;
                for xi in 0..2usize {
                    let t0 = {
                        let (xa, xb) = if s == 0 { (xi, 0) } else { (0, xi) };
                        log_potential(kind, xa, xb, log_ft_eps) + state.cur_v2f[fi][os][0]
                    };
                    let t1 = {
                        let (xa, xb) = if s == 0 { (xi, 1) } else { (1, xi) };
                        log_potential(kind, xa, xb, log_ft_eps) + state.cur_v2f[fi][os][1]
                    };
                    nxt_f2v[fi][s][xi] = log_sum_exp2(t0, t1);
                }
                let log_z = log_sum_exp2(nxt_f2v[fi][s][0], nxt_f2v[fi][s][1]);
                nxt_f2v[fi][s][0] -= log_z;
                nxt_f2v[fi][s][1] -= log_z;
            }
        }

        let mut max_delta = 0.0f64;
        for fi in 0..nf {
            for s in 0..2 {
                for x in 0..2 {
                    let raw = nxt_f2v[fi][s][x];
                    let old = state.cur_f2v[fi][s][x];
                    let dmp = log_sum_exp2(log_1m_lambda + raw, log_lambda + old);
                    max_delta = max_delta.max((dmp - old).abs());
                    state.cur_f2v[fi][s][x] = dmp;
                    state.cur_v2f[fi][s][x] = nxt_v2f[fi][s][x];
                }
            }
        }

        if max_delta < BP_CONVERGE {
            return (iter + 1, 0.0);
        }

        if (iter + 1) % BP_MARG_WINDOW == 0 {
            let cur = marginals(state);
            if let Some(prev) = prev_marg.take() {
                let mmd = cur
                    .iter()
                    .zip(prev.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f64, f64::max);
                last_marg_delta = mmd;
                if mmd < BP_MARG_TOL {
                    return (iter + 1, mmd);
                }
            }
            prev_marg = Some(cur);
        }
    }
    (BP_MAX_ITER, last_marg_delta)
}

fn extract_posteriors(state: &BpState, superset: &Superset, out: &mut HashMap<u64, f64>) {
    let base = superset.base_addr;
    let mut total_in: Vec<[f64; 2]> = vec![[0.0, 0.0]; state.n];
    for fi in 0..state.factors.len() {
        for s in 0..2 {
            let vi = state.factors[fi].vars[s];
            total_in[vi][0] += state.cur_f2v[fi][s][0];
            total_in[vi][1] += state.cur_f2v[fi][s][1];
        }
    }
    for offset in 0..state.n {
        if superset.instructions[offset].is_none() {
            continue;
        }
        let addr = base + offset as u64;
        let b0 = state.phi[offset][0] + total_in[offset][0];
        let b1 = state.phi[offset][1] + total_in[offset][1];
        let log_z = log_sum_exp2(b0, b1);
        out.insert(addr, (b1 - log_z).exp().clamp(0.0, 1.0));
    }
}

// ── Cavity / consistency hook (read-only post-hoc pass over the converged graph) ─
//
// This is the object Paper 2's consistency detector needs. For each candidate
// instruction-start `a`, the *cavity belief* is what the rest of the graph says about
// `a` on its own — the incoming structural messages BEFORE `a`'s local decode factor
// φ_a is folded in:
//
//     q_a = normalize( Π incoming f→v messages into a )   [no φ_a]
//
// The full posterior is π_a ∝ φ_a · Π(incoming); the cavity drops φ_a. Because BP has
// converged, both live in `state` already — computing this is a pure read, no
// re-inference, and it does not touch `posterior`/π. (See `cavity_leaves_pi_unchanged`.)
//
// The per-address *surprise* s_a = −log P(e_a | q_a): how startled the rest of the
// graph is by the local decode evidence. Two forms are exposed (§1 of the spec):
//   - soft (primary): s_a = −log( q0·m0 + q1·m1 ), the negative log of the local
//     likelihood averaged under the cavity — a proper Bayesian surprise on the
//     normalized local measurement m_a = softmax(φ_a). Uninformative φ (m=½) floors it
//     at ln2; the empirical clean null absorbs that constant, so only *excess* reads as
//     drift.
//   - NIS/hard: the standardized residual (m1 − q1)/sqrt(q1(1−q1)) and its ½·residual²,
//     the Bernoulli innovation-squared. Signed residual feeds the spatial (run-length)
//     statistic downstream.
#[derive(Debug, Clone, Copy)]
pub struct CavityStat {
    /// q_a = P(x_a = code | rest of graph), local factor φ_a excluded.
    pub cavity_code_prob: f64,
    /// m_a = softmax(φ_a)[code]: the local decode evidence as a probability.
    pub local_code_prob: f64,
    /// Soft surprise s_a = −ln( q0·m0 + q1·m1 ). Primary statistic.
    pub surprise: f64,
    /// Signed standardized Bernoulli residual (m1 − q1)/sqrt(q1(1−q1)). Feeds spatial clustering.
    pub residual: f64,
    /// NIS analogue ½·residual² (hard-indicator surprise).
    pub nis: f64,
    /// Local decode log-likelihood-ratio φ_a[1] − φ_a[0] (a's raw evidence strength).
    pub llr_local: f64,
}

/// Read-only pass: extract the cavity belief and per-address surprise from the converged
/// graph. Mirrors [`extract_posteriors`]'s `total_in` accumulation but stops one step short —
/// it never adds φ (that is the whole point of the cavity) and never writes π.
fn extract_cavity(state: &BpState, superset: &Superset, out: &mut HashMap<u64, CavityStat>) {
    // Guard the standardizing denominator away from 0/0 at the Bernoulli variance boundary.
    const Q_EPS: f64 = 1e-4;
    let base = superset.base_addr;
    let mut total_in: Vec<[f64; 2]> = vec![[0.0, 0.0]; state.n];
    for fi in 0..state.factors.len() {
        for s in 0..2 {
            let vi = state.factors[fi].vars[s];
            total_in[vi][0] += state.cur_f2v[fi][s][0];
            total_in[vi][1] += state.cur_f2v[fi][s][1];
        }
    }
    for offset in 0..state.n {
        if superset.instructions[offset].is_none() {
            continue;
        }
        let addr = base + offset as u64;
        // Cavity logit = structural messages only (no φ). Local logit = φ only.
        let cav_logit = total_in[offset][1] - total_in[offset][0];
        let loc_logit = state.phi[offset][1] - state.phi[offset][0];
        let q1 = sigmoid(cav_logit);
        let m1 = sigmoid(loc_logit);
        let (q0, m0) = (1.0 - q1, 1.0 - m1);
        // Soft surprise: negative log of the local likelihood averaged under the cavity.
        let z = (q0 * m0 + q1 * m1).max(f64::MIN_POSITIVE);
        let surprise = -z.ln();
        // NIS/hard: standardized Bernoulli innovation of the local measurement vs the cavity.
        let qc = q1.clamp(Q_EPS, 1.0 - Q_EPS);
        let residual = (m1 - q1) / (qc * (1.0 - qc)).sqrt();
        out.insert(
            addr,
            CavityStat {
                cavity_code_prob: q1,
                local_code_prob: m1,
                surprise,
                residual,
                nis: 0.5 * residual * residual,
                llr_local: loc_logit,
            },
        );
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// ── Analysis struct ───────────────────────────────────────────────────────────

pub struct Analysis<'a> {
    superset: &'a Superset,
    data_byte: Vec<DataProb>,
    reaching_hints: HashMap<u64, HashSet<HintKey>>,
    posterior: HashMap<u64, f64>,
    /// Read-only cavity/surprise stats, populated in Soft mode alongside `posterior`.
    /// Empty in Hard mode and until a Soft run completes. Never feeds back into π.
    cavity: HashMap<u64, CavityStat>,
}

impl<'a> Analysis<'a> {
    pub fn new(superset: &'a Superset) -> Self {
        let data_byte = superset
            .instructions
            .iter()
            .map(|i| {
                if i.is_some() {
                    DataProb::Unknown
                } else {
                    DataProb::DefinitelyData
                }
            })
            .collect();
        Self {
            superset,
            data_byte,
            reaching_hints: HashMap::new(),
            posterior: HashMap::new(),
            cavity: HashMap::new(),
        }
    }

    pub fn run(&mut self, hint_priors: &HashMap<HintKey, f64>) {
        self.run_with_config(hint_priors, &[], &AnalysisConfig::default());
    }

    pub fn run_with_config(
        &mut self,
        hint_priors: &HashMap<HintKey, f64>,
        hint_pairs: &[HintPair],
        config: &AnalysisConfig,
    ) {
        match config.mode {
            AnalysisMode::Hard => self.run_hard(hint_priors),
            AnalysisMode::Soft => {
                // E4: transport-only forward pass — Miller's hint propagation to
                // fixpoint, with neither occlusion kill nor backward invalidation.
                let reaching: Option<HashMap<u64, f64>> = if config.reaching_scale > 0.0 {
                    const MAX_FWD_ITER: usize = 100;
                    for _ in 0..MAX_FWD_ITER {
                        if !self.propagate_hints_forward(hint_priors) {
                            break;
                        }
                    }
                    Some(
                        self.reaching_hints
                            .iter()
                            .map(|(&addr, hs)| (addr, log_product(hs, hint_priors)))
                            .collect(),
                    )
                } else {
                    None
                };
                let mut state = build_factor_graph(
                    self.superset,
                    hint_priors,
                    hint_pairs,
                    config,
                    reaching.as_ref(),
                );
                run_bp(&mut state, config);
                extract_posteriors(&state, self.superset, &mut self.posterior);
                // Read-only post-hoc pass: cavity belief + surprise. Runs after π is
                // finalized, reads the same converged `state`, and writes only `cavity` —
                // π above is untouched (asserted by `cavity_leaves_pi_unchanged`).
                extract_cavity(&state, self.superset, &mut self.cavity);
            }
        }
    }

    pub fn sorted_posteriors(&self) -> Vec<(u64, f64)> {
        let mut out: Vec<(u64, f64)> = self.posterior.iter().map(|(&a, &p)| (a, p)).collect();
        out.sort_by_key(|(addr, _)| *addr);
        out
    }

    /// Per-address cavity/surprise stats from the converged graph, sorted by address.
    /// Empty unless a Soft run has completed. This is a post-hoc read of the same fixed
    /// point that produced the posteriors — it never altered them.
    pub fn sorted_cavity(&self) -> Vec<(u64, CavityStat)> {
        let mut out: Vec<(u64, CavityStat)> = self.cavity.iter().map(|(&a, &c)| (a, c)).collect();
        out.sort_by_key(|(addr, _)| *addr);
        out
    }

    // ── Hard mode (Miller Algorithm 1, untouched) ─────────────────────────────

    fn run_hard(&mut self, hint_priors: &HashMap<HintKey, f64>) {
        const MAX_ITER: usize = 100;
        let predecessors = self.build_predecessor_map();
        for _ in 0..MAX_ITER {
            let fwd = self.propagate_hints_forward(hint_priors);
            let occ = self.propagate_to_occlusion_space();
            let bwd = self.propagate_invalidity_backward(&predecessors, 1.0);
            if !fwd && !occ && !bwd {
                break;
            }
        }
        self.normalize();
    }

    fn build_predecessor_map(&self) -> HashMap<u64, Vec<u64>> {
        let mut map: HashMap<u64, Vec<u64>> = HashMap::new();
        for offset in 0..self.superset.instructions.len() {
            let addr = self.superset.base_addr + offset as u64;
            for succ in self.superset.successors_of(addr) {
                map.entry(succ).or_default().push(addr);
            }
        }
        map
    }

    fn propagate_hints_forward(&mut self, hint_priors: &HashMap<HintKey, f64>) -> bool {
        let hints_by_source = group_hints_by_source(hint_priors);
        let mut changed = false;
        for offset in 0..self.superset.instructions.len() {
            if self.data_byte[offset] == DataProb::DefinitelyData {
                continue;
            }
            let addr = self.superset.base_addr + offset as u64;
            if let Some(local) = hints_by_source.get(&addr) {
                if self.merge_reaching_hints(addr, offset, local.iter().copied(), hint_priors) {
                    changed = true;
                }
            }
            let reaching: HashSet<HintKey> =
                self.reaching_hints.get(&addr).cloned().unwrap_or_default();
            if reaching.is_empty() {
                continue;
            }
            for succ in self.superset.successors_of(addr) {
                let Some(so) = succ
                    .checked_sub(self.superset.base_addr)
                    .map(|o| o as usize)
                else {
                    continue;
                };
                if so >= self.data_byte.len() {
                    continue;
                }
                if self.data_byte[so] == DataProb::DefinitelyData {
                    continue;
                }
                if self.merge_reaching_hints(succ, so, reaching.iter().copied(), hint_priors) {
                    changed = true;
                }
            }
        }
        changed
    }

    fn propagate_to_occlusion_space(&mut self) -> bool {
        let mut changed = false;
        for offset in 0..self.superset.instructions.len() {
            if self.data_byte[offset] != DataProb::Unknown {
                continue;
            }
            let addr = self.superset.base_addr + offset as u64;
            let mut min_lp: Option<f64> = None;
            for peer in occluding_addrs_of(self.superset, addr) {
                let Some(po) = peer
                    .checked_sub(self.superset.base_addr)
                    .map(|o| o as usize)
                else {
                    continue;
                };
                let lp = match self.data_byte[po] {
                    DataProb::Estimated(lp) => lp,
                    DataProb::DefinitelyData => 0.0,
                    DataProb::Unknown => continue,
                };
                min_lp = Some(min_lp.map_or(lp, |m: f64| m.min(lp)));
            }
            if let Some(mlp) = min_lp {
                let new_lp = (1.0 - mlp.exp()).max(f64::MIN_POSITIVE).ln();
                self.data_byte[offset] = DataProb::Estimated(new_lp);
                changed = true;
            }
        }
        changed
    }

    fn propagate_invalidity_backward(
        &mut self,
        predecessors: &HashMap<u64, Vec<u64>>,
        damping: f64,
    ) -> bool {
        let mut changed = false;
        let empty: Vec<u64> = Vec::new();
        for offset in (0..self.superset.instructions.len()).rev() {
            let addr = self.superset.base_addr + offset as u64;
            let d_i = match self.data_byte[offset] {
                DataProb::Estimated(lp) => lp,
                DataProb::DefinitelyData => 0.0,
                DataProb::Unknown => continue,
            };
            let propagated = d_i + damping.ln();
            for &p in predecessors.get(&addr).unwrap_or(&empty) {
                let Some(po) = p.checked_sub(self.superset.base_addr).map(|o| o as usize) else {
                    continue;
                };
                if po >= self.data_byte.len() {
                    continue;
                }
                let update = match self.data_byte[po] {
                    DataProb::Unknown => true,
                    DataProb::Estimated(lp) => lp < propagated,
                    DataProb::DefinitelyData => false,
                };
                if update {
                    self.data_byte[po] = if propagated >= 0.0 {
                        DataProb::DefinitelyData
                    } else {
                        DataProb::Estimated(propagated)
                    };
                    changed = true;
                }
            }
        }
        changed
    }

    fn normalize(&mut self) {
        for offset in 0..self.superset.instructions.len() {
            let addr = self.superset.base_addr + offset as u64;
            let log_dp = match self.data_byte[offset] {
                DataProb::DefinitelyData => {
                    self.posterior.insert(addr, 0.0);
                    continue;
                }
                DataProb::Estimated(lp) => lp,
                DataProb::Unknown => continue,
            };
            let mut neg_logs = vec![-log_dp];
            for peer in occluding_addrs_of(self.superset, addr) {
                let Some(po) = peer
                    .checked_sub(self.superset.base_addr)
                    .map(|o| o as usize)
                else {
                    continue;
                };
                match self.data_byte[po] {
                    DataProb::Estimated(lp) => neg_logs.push(-lp),
                    DataProb::DefinitelyData => neg_logs.push(0.0),
                    DataProb::Unknown => {}
                }
            }
            let max_t = neg_logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if !max_t.is_finite() {
                continue;
            }
            let log_z = max_t + neg_logs.iter().map(|x| (x - max_t).exp()).sum::<f64>().ln();
            self.posterior
                .insert(addr, (-log_dp - log_z).exp().clamp(0.0, 1.0));
        }
    }

    fn merge_reaching_hints(
        &mut self,
        addr: u64,
        offset: usize,
        hints: impl IntoIterator<Item = HintKey>,
        priors: &HashMap<HintKey, f64>,
    ) -> bool {
        let reaching = self.reaching_hints.entry(addr).or_default();
        let before = reaching.len();
        reaching.extend(hints);
        if reaching.len() == before {
            return false;
        }
        self.data_byte[offset] = DataProb::Estimated(log_product(reaching, priors));
        true
    }
}

// ── Shared helpers ────────────────────────────────────────────────────────────

fn log_product(rh: &HashSet<HintKey>, priors: &HashMap<HintKey, f64>) -> f64 {
    rh.iter()
        .filter_map(|h| priors.get(h))
        .map(|p| p.ln())
        .sum()
}

fn group_hints_by_source(priors: &HashMap<HintKey, f64>) -> HashMap<u64, Vec<HintKey>> {
    let mut m: HashMap<u64, Vec<HintKey>> = HashMap::new();
    for &k in priors.keys() {
        m.entry(k.source_addr).or_default().push(k);
    }
    m
}


#[cfg(test)]
mod tests {
    use super::*;

    /// Enumerate all 2^n configs; return normalised per-variable marginals.
    /// phi_lin[i] = [phi(x=0), phi(x=1)] in LINEAR space.
    fn brute_force(
        phi_lin: &[[f64; 2]],
        factors: &[(BpFactorKind, usize, usize)],
        ft_eps: f64,
    ) -> Vec<[f64; 2]> {
        let n = phi_lin.len();
        let log_ft_eps = ft_eps.ln();
        let mut marg = vec![[0.0f64; 2]; n];
        for cfg in 0u32..(1 << n) {
            let xs: Vec<usize> = (0..n).map(|i| ((cfg >> i) & 1) as usize).collect();
            // Skip forbidden configs (overlap constraints).
            let forbidden = factors
                .iter()
                .any(|&(kind, a, b)| kind == BpFactorKind::Overlap && xs[a] == 1 && xs[b] == 1);
            if forbidden {
                continue;
            }
            let mut log_w: f64 = xs
                .iter()
                .enumerate()
                .map(|(i, &x)| phi_lin[i][x].ln())
                .sum();
            for &(kind, a, b) in factors {
                log_w += log_potential(kind, xs[a], xs[b], log_ft_eps);
            }
            let w = log_w.exp();
            for i in 0..n {
                marg[i][xs[i]] += w;
            }
        }
        for m in &mut marg {
            let z = m[0] + m[1];
            if z > 0.0 {
                m[0] /= z;
                m[1] /= z;
            }
        }
        marg
    }

    fn make_state(
        phi_lin: &[[f64; 2]],
        factors: &[(BpFactorKind, usize, usize)],
        ft_eps: f64,
        msg_damp: f64,
    ) -> (BpState, AnalysisConfig) {
        let n = phi_lin.len();
        let phi: Vec<[f64; 2]> = phi_lin.iter().map(|p| [p[0].ln(), p[1].ln()]).collect();
        let bp_factors: Vec<BpFactor> = factors
            .iter()
            .map(|&(kind, a, b)| BpFactor { kind, vars: [a, b] })
            .collect();
        let nf = bp_factors.len();
        let state = BpState {
            n,
            phi,
            factors: bp_factors,
            cur_f2v: vec![[[0.0; 2]; 2]; nf],
            cur_v2f: vec![[[0.0; 2]; 2]; nf],
        };
        let config = AnalysisConfig {
            mode: AnalysisMode::Soft,
            msg_damp,
            ft_eps,
            evidence_scale: 1.0,
            transfer_log_weight: 4.0,
            unhinted_code_prob: 0.2,
            reaching_scale: 0.0,
            entropy_prior_strength: 0.0,
            entropy_floor_bits: 6.0,
            chainfwd_strength: 0.0,
        };
        (state, config)
    }

    fn beliefs(state: &BpState) -> Vec<f64> {
        let mut total_in = vec![[0.0f64; 2]; state.n];
        for fi in 0..state.factors.len() {
            for s in 0..2 {
                let vi = state.factors[fi].vars[s];
                total_in[vi][0] += state.cur_f2v[fi][s][0];
                total_in[vi][1] += state.cur_f2v[fi][s][1];
            }
        }
        (0..state.n)
            .map(|i| {
                let b0 = state.phi[i][0] + total_in[i][0];
                let b1 = state.phi[i][1] + total_in[i][1];
                (b1 - log_sum_exp2(b0, b1)).exp()
            })
            .collect()
    }

    /// Tree: A overlaps B, A falls-through to C.
    /// Acyclic graph → BP must recover exact marginals (to 1e-6).
    #[test]
    fn toy_tree_exact_marginals() {
        let ft_eps = 0.5f64;
        let phi_lin = [[0.05, 0.95], [0.50, 0.50], [0.55, 0.45]];
        let factors = [
            (BpFactorKind::Overlap, 0, 1),
            (BpFactorKind::FallThrough, 0, 2),
        ];
        let expected = brute_force(&phi_lin, &factors, ft_eps);
        let (mut state, config) = make_state(&phi_lin, &factors, ft_eps, 0.01);
        run_bp(&mut state, &config);
        let got = beliefs(&state);
        for i in 0..3 {
            println!(
                "toy_tree var {i}: BP={:.9}  BF={:.9}  diff={:.2e}",
                got[i],
                expected[i][1],
                (got[i] - expected[i][1]).abs()
            );
            assert!(
                (got[i] - expected[i][1]).abs() < 1e-6,
                "var {i}: BP={:.9} brute-force={:.9} diff={:.2e}",
                got[i],
                expected[i][1],
                (got[i] - expected[i][1]).abs()
            );
        }
    }

    /// Loopy: Overlap triangle — 3 variables with pairwise mutual exclusion.
    /// Factor graph has a length-6 cycle (x0-f01-x1-f12-x2-f02-x0).
    /// Exact marginals: P(xi=1) = 1/4 for all i (4 valid configs: all-0, or exactly one 1).
    /// Flooding BP converges (Overlap hard constraints settle quickly) but gives P(xi=1) ≠ 1/4
    /// due to double-counting in the cycle.
    #[test]
    fn toy_loopy_converges() {
        let ft_eps = 0.5f64; // not used by Overlap factors; required by make_state
        let phi_lin = [[0.5_f64, 0.5], [0.5, 0.5], [0.5, 0.5]];
        let factors = [
            (BpFactorKind::Overlap, 0, 1),
            (BpFactorKind::Overlap, 1, 2),
            (BpFactorKind::Overlap, 0, 2),
        ];
        let (mut state, config) = make_state(&phi_lin, &factors, ft_eps, 0.5);
        let (n_iters, _) = run_bp(&mut state, &config);
        assert!(
            n_iters < BP_MAX_ITER,
            "loopy BP failed to converge in {BP_MAX_ITER} iters"
        );

        let got = beliefs(&state);
        for (i, &p) in got.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&p),
                "var {i} posterior {p} out of range"
            );
        }

        // Brute-force exact: P(xi=1) = 1/4 by symmetry.
        let bf = brute_force(&phi_lin, &factors, ft_eps);
        assert!(
            (bf[0][1] - 0.25).abs() < 1e-9,
            "brute-force sanity: {}",
            bf[0][1]
        );
        // Loopy BP differs from exact due to double-counting.
        assert!(
            (got[0] - bf[0][1]).abs() > 1e-4,
            "BP and brute-force too similar on loopy graph: BP={:.6}, BF={:.6}",
            got[0],
            bf[0][1]
        );
    }

    /// Hinted code anchor + overlapping unhinted + ft chain mimicking a real binary region.
    /// A (strongly hinted code) overlaps B (unhinted, should be data).
    /// A ft→C (A's sequential successor, should be code).
    /// B ft→C (B is 1 byte shorter, so B+size(B) = C = A+size(A)).
    /// Also: C ft→D (D = C's successor). Tests that B gets pushed to data even
    /// with a competing ft signal from C (which is code-likely from A's ft).
    #[test]
    fn hinted_overlap_pushes_to_data() {
        // Simulate: instruction at A(0) size 5, overlapping decode at B(1) size 4.
        // Both have ft to C(5). C(5) has ft to D(8).
        // phi_lin[A] = [1e-11, 1.0] (very strongly hinted code, one CtrlConvLong hint).
        // phi_lin[B] = [1.0, 1.0]   (unhinted, uniform).
        // phi_lin[C] = [1.0, 1.0]   (unhinted).
        // phi_lin[D] = [1.0, 1.0]   (unhinted).
        let ft_eps = 0.9_f64;
        let p_hint = 1.0_f64 / u32::MAX as f64; // CtrlConvLong prior = 1/2^32
        let phi_lin = [[p_hint, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
        // A(0)↔B(1): overlap. A(0)→C(2): ft. B(1)→C(2): ft. C(2)→D(3): ft.
        let factors = [
            (BpFactorKind::Overlap, 0, 1),
            (BpFactorKind::FallThrough, 0, 2),
            (BpFactorKind::FallThrough, 1, 2),
            (BpFactorKind::FallThrough, 2, 3),
        ];
        let expected = brute_force(&phi_lin, &factors, ft_eps);
        let (mut state, config) = make_state(&phi_lin, &factors, ft_eps, 0.5);
        run_bp(&mut state, &config);
        let got = beliefs(&state);

        println!(
            "hinted_overlap: A={:.6} B={:.6} C={:.6} D={:.6}",
            got[0], got[1], got[2], got[3]
        );
        println!(
            "brute-force:    A={:.6} B={:.6} C={:.6} D={:.6}",
            expected[0][1], expected[1][1], expected[2][1], expected[3][1]
        );

        // B (overlapping with code-certain A) must be pushed to data.
        assert!(
            got[1] < 0.1,
            "B should be data (overlaps code-certain A) but got P(code)={:.6}",
            got[1]
        );
        // A must be code.
        assert!(
            got[0] > 0.9,
            "A should be code (strongly hinted) but got P(code)={:.6}",
            got[0]
        );
        // C should be code (ft from code-certain A).
        assert!(
            got[2] > 0.5,
            "C should be code-likely (ft from A) but got P(code)={:.6}",
            got[2]
        );
    }

    /// Two-variable fall-through chain: x_a has strong code hint, x_b is uninformative.
    /// FT factor a→b. No overlap factors. On an acyclic graph, BP is exact.
    /// FT pushes b toward code when a is code: P(b=1) must follow a's posterior.
    #[test]
    fn toy_fallthrough_reinforces() {
        let ft_eps = 0.3f64;
        // x_a: log-odds +3 for code ≈ P(code)≈0.95. phi_lin = [exp(-3/2), exp(+3/2)] centered.
        // Use [e^{-1.5}, e^{+1.5}] so phi_lin[1]/phi_lin[0] = e^3.
        let phi_lin = [
            [(-1.5f64).exp(), (1.5f64).exp()], // x_a: strongly code
            [1.0f64, 1.0f64],                  // x_b: uninformative
        ];
        let factors = [(BpFactorKind::FallThrough, 0, 1)];
        let expected = brute_force(&phi_lin, &factors, ft_eps);
        let (mut state, config) = make_state(&phi_lin, &factors, ft_eps, 0.01);
        run_bp(&mut state, &config);
        let got = beliefs(&state);

        println!(
            "toy_ft_reinforces: a={:.6} b={:.6}  BF: a={:.6} b={:.6}",
            got[0], got[1], expected[0][1], expected[1][1]
        );

        // Dump per-factor messages after convergence.
        {
            let mut total_in = vec![[0.0f64; 2]; state.n];
            for fi in 0..state.factors.len() {
                for s in 0..2 {
                    let vi = state.factors[fi].vars[s];
                    total_in[vi][0] += state.cur_f2v[fi][s][0];
                    total_in[vi][1] += state.cur_f2v[fi][s][1];
                }
            }
            for fi in 0..state.factors.len() {
                println!(
                    "  factor {fi} ({:?} vars={:?}):",
                    state.factors[fi].kind, state.factors[fi].vars
                );
                println!(
                    "    f2v[side=0 → var {}]: data={:.6}  code={:.6}  diff(code-data)={:.6}",
                    state.factors[fi].vars[0],
                    state.cur_f2v[fi][0][0],
                    state.cur_f2v[fi][0][1],
                    state.cur_f2v[fi][0][1] - state.cur_f2v[fi][0][0]
                );
                println!(
                    "    f2v[side=1 → var {}]: data={:.6}  code={:.6}  diff(code-data)={:.6}",
                    state.factors[fi].vars[1],
                    state.cur_f2v[fi][1][0],
                    state.cur_f2v[fi][1][1],
                    state.cur_f2v[fi][1][1] - state.cur_f2v[fi][1][0]
                );
                println!(
                    "    v2f[side=0 ← var {}]: data={:.6}  code={:.6}",
                    state.factors[fi].vars[0], state.cur_v2f[fi][0][0], state.cur_v2f[fi][0][1]
                );
                println!(
                    "    v2f[side=1 ← var {}]: data={:.6}  code={:.6}",
                    state.factors[fi].vars[1], state.cur_v2f[fi][1][0], state.cur_v2f[fi][1][1]
                );
            }
            println!("  phi[a]={:?}  phi[b]={:?}", state.phi[0], state.phi[1]);
            println!(
                "  total_in[a]: data={:.6}  code={:.6}",
                total_in[0][0], total_in[0][1]
            );
            println!(
                "  total_in[b]: data={:.6}  code={:.6}",
                total_in[1][0], total_in[1][1]
            );
        }

        // BP must match brute force to 1e-6 (acyclic graph → exact).
        for i in 0..2 {
            assert!(
                (got[i] - expected[i][1]).abs() < 1e-6,
                "var {i}: BP={:.9} BF={:.9} diff={:.2e}",
                got[i],
                expected[i][1],
                (got[i] - expected[i][1]).abs()
            );
        }
        // Symmetric FT pushes b toward a's state AND penalizes a=data→b=code.
        // With a strongly code, b is pushed code-ward but now also held back by the
        // symmetric penalty. BF gives P(b)≈0.744; threshold updated from 0.7 (still holds).
        assert!(
            got[1] > 0.7,
            "FT factor should push b toward code when a is code-certain. P(b=1)={:.6}",
            got[1]
        );
        // Symmetric FT widens the gap vs asymmetric (old: ~0.18, new: ~0.21). Threshold ≤ 0.25.
        assert!(
            (got[0] - got[1]).abs() < 0.25,
            "a and b posteriors should be close via FT coupling. a={:.6} b={:.6}",
            got[0],
            got[1]
        );
    }

    /// Loopy graph: Overlap(a,b) + FT(a→c) + FT(b→c).
    /// Creates cycle: a — Overlap — b — FT — c — FT — a (length-6 in factor graph).
    /// Strong positive hint on a, weak negative hint on b, no hint on c.
    /// Expected (from brute force with ft_eps=0.3, log-odds: a=+3, b=-1):
    ///   P(a=1)≈0.91, P(b=1)≈0.02, P(c=1)≈0.75.
    /// If BP is broken: c stalls near 0.5 (conflicting FT signals not reinforcing).
    #[test]
    fn toy_loopy_overlap_plus_fallthrough() {
        let ft_eps = 0.3f64;
        // a: log-odds +3 → phi_lin = [e^{-1.5}, e^{+1.5}]
        // b: log-odds -1 → phi_lin = [e^{+0.5}, e^{-0.5}]
        // c: uninformative
        let phi_lin = [
            [(-1.5f64).exp(), (1.5f64).exp()],
            [(0.5f64).exp(), (-0.5f64).exp()],
            [1.0f64, 1.0f64],
        ];
        let factors = [
            (BpFactorKind::Overlap, 0, 1),
            (BpFactorKind::FallThrough, 0, 2),
            (BpFactorKind::FallThrough, 1, 2),
        ];
        let expected = brute_force(&phi_lin, &factors, ft_eps);
        let (mut state, config) = make_state(&phi_lin, &factors, ft_eps, 0.5);
        let (n_iters, _) = run_bp(&mut state, &config);
        let got = beliefs(&state);

        let converged = n_iters < BP_MAX_ITER;
        // Report convergence status — includes marginal-stability criterion.
        println!(
            "toy_loopy_ovlp_ft [damp=0.5]: iters={n_iters}/{BP_MAX_ITER}  converged={converged}"
        );
        println!("  BP:  a={:.6}  b={:.6}  c={:.6}", got[0], got[1], got[2]);
        println!(
            "  BF:  a={:.6}  b={:.6}  c={:.6}",
            expected[0][1], expected[1][1], expected[2][1]
        );
        for i in 0..3 {
            println!(
                "  var {i}: |BP-BF| = {:.2e}",
                (got[i] - expected[i][1]).abs()
            );
        }

        // Marginal accuracy: loopy BP is approximate but should track brute force closely
        // on this small graph. Tolerance 0.05 allows for loopy-BP approximation error.
        for i in 0..3 {
            assert!(
                (got[i] - expected[i][1]).abs() < 0.05,
                "var {i}: BP={:.6} BF={:.6} diff={:.2e} — too far from brute force",
                got[i],
                expected[i][1],
                (got[i] - expected[i][1]).abs()
            );
        }
        assert!(
            got[0] > 0.85,
            "P(a=1) should be high (strong code hint). got={:.6}",
            got[0]
        );
        assert!(
            got[1] < 0.10,
            "P(b=1) should be low (overlap + weak data hint). got={:.6}",
            got[1]
        );
        // Symmetric FT: c gets pull toward code (from a) AND pull toward data (from b).
        // Old asymmetric gave c≈0.75. New symmetric: competing FT signals → c≈0.47 (uncertain).
        // BF confirms ~0.466. Assert c stays within [0.35, 0.65] — genuinely uncertain.
        assert!(
            got[2] > 0.35 && got[2] < 0.65,
            "P(c=1) should be uncertain with competing FT signals. got={:.6}",
            got[2]
        );

        // msg_damp=0.7: heavier damping; test if marginal stability fires sooner.
        {
            let (mut state7, config7) = make_state(&phi_lin, &factors, ft_eps, 0.7);
            let (n7, _) = run_bp(&mut state7, &config7);
            let got7 = beliefs(&state7);
            let conv7 = n7 < BP_MAX_ITER;
            println!("toy_loopy_ovlp_ft [damp=0.7]: iters={n7}/{BP_MAX_ITER}  converged={conv7}");
            println!(
                "  BP:  a={:.6}  b={:.6}  c={:.6}",
                got7[0], got7[1], got7[2]
            );
        }
    }

    /// Two variables, one HintCoupling(log_weight=5.0), uniform unary priors.
    /// ψ(1,1)=exp(5); all other configs weight 1. Symmetric → P(x0=1)=P(x1=1).
    /// Brute force: Z = 3 + exp(5); P(xi=1) = (1+exp(5))/Z ≈ 0.987.
    #[test]
    fn toy_hint_coupling_couples() {
        let ft_eps = 0.5f64; // unused by HintCoupling; required by make_state
        let log_w = 5.0f64;
        let phi_lin = [[1.0f64, 1.0], [1.0f64, 1.0]];
        let factors = [(BpFactorKind::HintCoupling(log_w), 0usize, 1usize)];
        let expected = brute_force(&phi_lin, &factors, ft_eps);
        let (mut state, config) = make_state(&phi_lin, &factors, ft_eps, 0.01);
        run_bp(&mut state, &config);
        let got = beliefs(&state);

        println!(
            "toy_hint_coupling: BP x0={:.9} x1={:.9}  BF x0={:.9} x1={:.9}",
            got[0], got[1], expected[0][1], expected[1][1]
        );

        // Symmetric: both variables must have identical posteriors.
        assert!(
            (got[0] - got[1]).abs() < 1e-9,
            "symmetry broken: P(x0=1)={:.9} P(x1=1)={:.9}",
            got[0],
            got[1]
        );

        // Must be above 0.5 (coupling rewards joint-code).
        assert!(got[0] > 0.5, "P(x0=1) should be > 0.5, got={:.9}", got[0]);

        // Must match brute force to 1e-9 (acyclic → exact BP).
        for i in 0..2 {
            assert!(
                (got[i] - expected[i][1]).abs() < 1e-9,
                "var {i}: BP={:.9} BF={:.9} diff={:.2e}",
                got[i],
                expected[i][1],
                (got[i] - expected[i][1]).abs()
            );
        }
    }

    /// Transfer factor with log_weight=5.0, uniform unary priors.
    /// Potential identical to HintCoupling: ψ(1,1)=exp(5); all other configs weight 1.
    /// Brute force: Z = 3 + exp(5); P(xi=1) = (1+exp(5))/Z ≈ 0.987. Symmetric by construction.
    #[test]
    fn toy_transfer_couples() {
        let ft_eps = 0.5f64; // unused by Transfer; required by make_state
        let log_w = 5.0f64;
        let phi_lin = [[1.0f64, 1.0], [1.0f64, 1.0]];
        let factors = [(BpFactorKind::Transfer(log_w), 0usize, 1usize)];
        let expected = brute_force(&phi_lin, &factors, ft_eps);
        let (mut state, config) = make_state(&phi_lin, &factors, ft_eps, 0.01);
        run_bp(&mut state, &config);
        let got = beliefs(&state);

        println!(
            "toy_transfer: BP x0={:.9} x1={:.9}  BF x0={:.9} x1={:.9}",
            got[0], got[1], expected[0][1], expected[1][1]
        );

        // Symmetric: identical potential shape → both variables must have identical posteriors.
        assert!(
            (got[0] - got[1]).abs() < 1e-9,
            "symmetry broken: P(x0=1)={:.9} P(x1=1)={:.9}",
            got[0],
            got[1]
        );

        // Transfer rewards (1,1) → must be above 0.5.
        assert!(got[0] > 0.5, "P(x0=1) should be > 0.5, got={:.9}", got[0]);

        // Acyclic graph (2 nodes, 1 factor) → BP is exact; match brute force to 1e-9.
        for i in 0..2 {
            assert!(
                (got[i] - expected[i][1]).abs() < 1e-9,
                "var {i}: BP={:.9} BF={:.9} diff={:.2e}",
                got[i],
                expected[i][1],
                (got[i] - expected[i][1]).abs()
            );
        }
    }

    /// Convergence-coupling proof:
    ///   Step 2: HintCoupling(22.2) between b1 and b2; t uninformative; no unary.
    ///           Both b1 and b2 should be code-likely (coupling rewards joint-code).
    ///   Step 3: Add phi[b1]=[+2,-2] (data-push on b1). P(b2=1) must DROP — coupling
    ///           drags b2 down with b1. This is the structural difference vs. old
    ///           independent-unary approach, which would leave b2 unchanged.
    #[test]
    fn toy_convergence_couples() {
        let ft_eps = 0.5f64; // not used by HintCoupling; required by make_state
        let log_w = 22.2f64;
        // b1=0, b2=1, t=2. Only b1 and b2 coupled; t has no factors → stays at 0.5.
        let factors = [(BpFactorKind::HintCoupling(log_w), 0usize, 1usize)];

        // ── Step 2: coupling only, uniform priors ─────────────────────────────
        let phi2 = [[1.0f64, 1.0], [1.0f64, 1.0], [1.0f64, 1.0]];
        let bf2 = brute_force(&phi2, &factors, ft_eps);
        let (mut s2, c2) = make_state(&phi2, &factors, ft_eps, 0.01);
        run_bp(&mut s2, &c2);
        let bp2 = beliefs(&s2);

        println!("step 2 (coupling only, log_w={log_w}):");
        println!(
            "  BP  b1={:.12}  b2={:.12}  t={:.6}",
            bp2[0], bp2[1], bp2[2]
        );
        println!(
            "  BF  b1={:.12}  b2={:.12}  t={:.6}",
            bf2[0][1], bf2[1][1], bf2[2][1]
        );

        // Acyclic → exact.
        for i in 0..3 {
            assert!(
                (bp2[i] - bf2[i][1]).abs() < 1e-9,
                "step2 var {i}: BP={:.12} BF={:.12}",
                bp2[i],
                bf2[i][1]
            );
        }
        assert!(
            bp2[0] > 0.9,
            "P(b1=1) step2 should be > 0.9, got={:.12}",
            bp2[0]
        );
        assert!(
            bp2[1] > 0.9,
            "P(b2=1) step2 should be > 0.9, got={:.12}",
            bp2[1]
        );
        assert!(
            (bp2[0] - bp2[1]).abs() < 1e-6,
            "b1 and b2 must be symmetric, diff={:.2e}",
            (bp2[0] - bp2[1]).abs()
        );

        // ── Step 3: data-push on b1, coupling should drag b2 down ─────────────
        // phi[b1] = [+2.0, -2.0] in log space → phi_lin[b1] = [exp(2), exp(-2)]
        let phi3 = [
            [(2.0f64).exp(), (-2.0f64).exp()],
            [1.0f64, 1.0],
            [1.0f64, 1.0],
        ];
        let bf3 = brute_force(&phi3, &factors, ft_eps);
        let (mut s3, c3) = make_state(&phi3, &factors, ft_eps, 0.01);
        run_bp(&mut s3, &c3);
        let bp3 = beliefs(&s3);

        println!("step 3 (b1 data-push phi=[+2,-2]):");
        println!(
            "  BP  b1={:.12}  b2={:.12}  t={:.6}",
            bp3[0], bp3[1], bp3[2]
        );
        println!(
            "  BF  b1={:.12}  b2={:.12}  t={:.6}",
            bf3[0][1], bf3[1][1], bf3[2][1]
        );
        println!("  ΔP(b2=1) step3-step2 = {:.4e}", bp3[1] - bp2[1]);

        // Acyclic → exact.
        for i in 0..3 {
            assert!(
                (bp3[i] - bf3[i][1]).abs() < 1e-9,
                "step3 var {i}: BP={:.12} BF={:.12}",
                bp3[i],
                bf3[i][1]
            );
        }
        // Coupling: data-push on b1 must reduce P(b2=1).
        assert!(
            bp3[1] < bp2[1],
            "coupling should drag b2 down when b1 pushed to data. step2={:.12} step3={:.12}",
            bp2[1],
            bp3[1]
        );
    }

    /// Sanity: a single isolated variable with phi=[0,0] and no factors gives P=0.5.
    #[test]
    fn toy_single_no_factors() {
        let state = BpState {
            n: 1,
            phi: vec![[0.0f64, 0.0]],
            factors: vec![],
            cur_f2v: vec![],
            cur_v2f: vec![],
        };
        // extract_posteriors equivalent:
        let total_in = vec![[0.0f64; 2]; 1];
        let b0 = state.phi[0][0] + total_in[0][0];
        let b1 = state.phi[0][1] + total_in[0][1];
        let p = (b1 - log_sum_exp2(b0, b1)).exp();
        println!("single isolated variable: P(code)={:.6}", p);
        assert!(
            (p - 0.5).abs() < 1e-9,
            "isolated uniform should be 0.5, got {p}"
        );
    }

    /// 6-node chain: overlap pairs (0,1),(2,3),(4,5), no FT, uniform phi.
    /// BF: each pair resolves to one or the other, producing P≈0.333 per node.
    /// Scale test: does 6-node saturate where 3-node doesn't?
    #[test]
    fn toy_6node_overlap_chain() {
        let ft_eps = 1.0f64; // trivial FT = no FT
        let phi_lin = [[1.0f64; 2]; 6];
        let factors = [
            (BpFactorKind::Overlap, 0, 1),
            (BpFactorKind::Overlap, 2, 3),
            (BpFactorKind::Overlap, 4, 5),
        ];
        let expected = brute_force(&phi_lin, &factors, ft_eps);
        let (mut state, config) = make_state(&phi_lin, &factors, ft_eps, 0.5);
        run_bp(&mut state, &config);
        let got = beliefs(&state);

        println!("6node_overlap_chain:");
        println!(
            "  BP:  {:?}",
            got.iter().map(|p| format!("{:.3}", p)).collect::<Vec<_>>()
        );
        println!(
            "  BF:  {:?}",
            expected
                .iter()
                .map(|e| format!("{:.3}", e[1]))
                .collect::<Vec<_>>()
        );

        // BF: each pair is a 2-node overlap with uniform phi.
        // Brute force: P(x=1) = 1/3 ≈ 0.333 for all nodes in a pair (by symmetry).
        for i in 0..6 {
            assert!(
                (expected[i][1] - 1.0 / 3.0).abs() < 1e-9,
                "BF sanity for var {i}: got {:.6}",
                expected[i][1]
            );
        }
        // BP must not saturate.
        for i in 0..6 {
            assert!(
                got[i] < 0.99,
                "var {i} saturated to 1.0 on 6-node with no hints. P={:.6}",
                got[i]
            );
        }
    }

    /// 10-node overlap PATH: 0-1, 1-2, ..., 8-9. Each interior node has 2 overlap partners.
    /// Mirrors the real binary's dense overlap structure. Does BP saturate here?
    #[test]
    fn toy_10node_overlap_path() {
        let ft_eps = 1.0f64; // no FT
        let phi_lin = [[1.0f64; 2]; 10];
        let factors: Vec<(BpFactorKind, usize, usize)> =
            (0..9).map(|i| (BpFactorKind::Overlap, i, i + 1)).collect();
        let expected = brute_force(&phi_lin, &factors, ft_eps);
        let (mut state, config) = make_state(&phi_lin, &factors, ft_eps, 0.5);
        run_bp(&mut state, &config);
        let got = beliefs(&state);

        println!("10node_overlap_path:");
        println!(
            "  BP: {:?}",
            got.iter().map(|p| format!("{:.3}", p)).collect::<Vec<_>>()
        );
        println!(
            "  BF: {:?}",
            expected
                .iter()
                .map(|e| format!("{:.3}", e[1]))
                .collect::<Vec<_>>()
        );
        println!(
            "  max: BP={:.4} BF={:.4}",
            got.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            expected
                .iter()
                .map(|e| e[1])
                .fold(f64::NEG_INFINITY, f64::max)
        );

        let saturated = got.iter().filter(|&&p| p > 0.99).count();
        println!("  saturated(>0.99): {}", saturated);
        // If ANY node saturates to code with no hints, we've found the scale where it breaks.
        for i in 0..10 {
            assert!(
                got[i] < 0.99,
                "var {i} saturated on 10-node path with no hints! P={:.6}",
                got[i]
            );
        }
    }

    /// Test 1: 3-node graph, uniform phi, Overlap(a,b) + FT(a→c) + FT(b→c), no hints.
    /// BF gives P(a)=P(b)≈0.308, P(c)≈0.385 (data-biased from overlap).
    /// If BP saturates to 1.0, the saturation source is the inference loop itself.
    #[test]
    fn toy_no_hints_ft_overlap() {
        let ft_eps = 0.5f64;
        let phi_lin = [[1.0f64, 1.0], [1.0f64, 1.0], [1.0f64, 1.0]];
        let factors = [
            (BpFactorKind::Overlap, 0, 1),
            (BpFactorKind::FallThrough, 0, 2),
            (BpFactorKind::FallThrough, 1, 2),
        ];
        let expected = brute_force(&phi_lin, &factors, ft_eps);
        let (mut state, config) = make_state(&phi_lin, &factors, ft_eps, 0.5);
        run_bp(&mut state, &config);
        let got = beliefs(&state);

        println!("toy_no_hints_ft_overlap:");
        println!("  BP: a={:.6}  b={:.6}  c={:.6}", got[0], got[1], got[2]);
        println!(
            "  BF: a={:.6}  b={:.6}  c={:.6}",
            expected[0][1], expected[1][1], expected[2][1]
        );

        // BF: a≈0.308, b≈0.308, c≈0.385 — data-biased, NOT near 1.0.
        for i in 0..3 {
            assert!(
                expected[i][1] < 0.5,
                "BF sanity: var {i} should be data-biased without code hints, got BF={:.6}",
                expected[i][1]
            );
        }
        // BP must match BF (loopy-tolerant).
        for i in 0..3 {
            assert!(
                (got[i] - expected[i][1]).abs() < 0.05,
                "var {i}: BP={:.6} BF={:.6} — too far from brute force",
                got[i],
                expected[i][1]
            );
        }
        // None should saturate to code.
        for i in 0..3 {
            assert!(
                got[i] < 0.99,
                "var {i}: saturated to 1.0 with no code hints — saturation bug. got={:.6}",
                got[i]
            );
        }
    }

    /// Test 2: Same graph but b has phi_code = LOG_ZERO (invalid-decode simulation).
    /// Construct BpState directly to set phi[b][1] = LOG_ZERO exactly.
    /// BF: P(a)≈0.444, P(b)≈0, P(c)≈0.333 — b is definitively data, a and c moderately data-biased.
    /// If BP drives a and c to 1.0, LOG_ZERO leaks through FT(b→c) into the valid graph.
    #[test]
    fn toy_log_zero_leak_via_ft() {
        let ft_eps = 0.5f64;
        let log_ft_eps = ft_eps.ln();
        let n = 3usize;
        let mut phi = vec![[0.0f64; 2]; n];
        phi[0] = [0.0, 0.0]; // a: uniform
        phi[1] = [0.0, LOG_ZERO]; // b: invalid-decode (definitively data)
        phi[2] = [0.0, 0.0]; // c: uniform

        let bp_factors = vec![
            BpFactor {
                kind: BpFactorKind::Overlap,
                vars: [0, 1],
            },
            BpFactor {
                kind: BpFactorKind::FallThrough,
                vars: [0, 2],
            },
            BpFactor {
                kind: BpFactorKind::FallThrough,
                vars: [1, 2],
            },
        ];
        let nf = bp_factors.len();
        let mut state = BpState {
            n,
            phi: phi.clone(),
            factors: bp_factors,
            cur_f2v: vec![[[0.0; 2]; 2]; nf],
            cur_v2f: vec![[[0.0; 2]; 2]; nf],
        };
        let config = AnalysisConfig {
            mode: AnalysisMode::Soft,
            msg_damp: 0.5,
            ft_eps,
            evidence_scale: 1.0,
            transfer_log_weight: 4.0,
            unhinted_code_prob: 0.2,
            reaching_scale: 0.0,
            entropy_prior_strength: 0.0,
            entropy_floor_bits: 6.0,
            chainfwd_strength: 0.0,
        };
        run_bp(&mut state, &config);
        let got = beliefs(&state);

        // Brute force (manual, since brute_force() uses phi_lin.ln()):
        // Enumerate 8 configs; LOG_ZERO makes all (b=code) configs weight ≈ 0.
        // Valid configs with b=0 (b=data only):
        // (0,0,0): e^0*e^0*e^0 * ovl(0,0)=0 * ft(0,0)=0 * ft(0,0)=0 = 1
        // (0,0,1): 1*1*1 * ft(0,1)=log_ft * ft(0,1)=log_ft = ft_eps^2 = 0.25
        // (1,0,0): 1*1*1 * ft(1,0)=log_ft * ft(0,0)=0 = 0.5
        // (1,0,1): 1*1*1 * ft(1,1)=0 * ft(0,1)=log_ft = 0.5
        // (b=1 configs): weight ≈ 0 due to phi[b][code]=LOG_ZERO
        let z = 1.0 + 0.25 + 0.5 + 0.5;
        let bf_a = (0.5 + 0.5) / z; // configs (1,0,0) + (1,0,1)
        let bf_b = 0.0_f64; // all b=code configs → 0
        let bf_c = (0.25 + 0.5) / z; // configs (0,0,1) + (1,0,1)

        println!("toy_log_zero_leak_via_ft:");
        println!("  BP: a={:.6}  b={:.6}  c={:.6}", got[0], got[1], got[2]);
        println!("  BF: a={:.6}  b={:.6}  c={:.6}", bf_a, bf_b, bf_c);
        println!(
            "  Diagnose: is a or c saturated to 1.0? a_sat={} c_sat={}",
            got[0] > 0.99,
            got[2] > 0.99
        );

        // b must be near 0 (definitively data from LOG_ZERO phi).
        assert!(
            got[1] < 0.01,
            "P(b=1) should be ≈0 (invalid decode). got={:.6}",
            got[1]
        );

        // a and c: BF gives 0.444 and 0.333. If saturated to 1.0, LOG_ZERO is leaking.
        // We assert they should NOT be saturated — this is the test for the bug.
        assert!(
            got[0] < 0.99,
            "LOG_ZERO LEAK: a saturated to 1.0 via FT(b→c)→c→... chain. got={:.6}",
            got[0]
        );
        assert!(
            got[2] < 0.99,
            "LOG_ZERO LEAK: c saturated to 1.0 via FT(b→c). got={:.6}",
            got[2]
        );
        // BF match (loopy-tolerant).
        assert!(
            (got[0] - bf_a).abs() < 0.05,
            "a: BP={:.6} BF={:.6}",
            got[0],
            bf_a
        );
        assert!(
            (got[2] - bf_c).abs() < 0.05,
            "c: BP={:.6} BF={:.6}",
            got[2],
            bf_c
        );
    }
}
