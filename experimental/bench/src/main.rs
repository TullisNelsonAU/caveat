//! `bench` — instruction-recovery precision/recall of the Soft model against ground truth.
//!
//! The accuracy head-to-head. A deterministic disassembler gives one (recall, precision) *point*;
//! Soft is probabilistic, so we read out a *curve*. Two readouts:
//!
//!   * `--threshold` (default): predicted starts = `{ addr : P ≥ τ }`, swept over τ. Crude — it
//!     counts overlapping in-instruction offsets, so precision is understated.
//!   * `--cover`: a max-weight non-overlapping instruction set (weighted interval scheduling over
//!     the superset, weight = log-odds(P) + bias). This enforces "you can't pick an instruction and
//!     a byte inside it" and skips low-confidence junk, which is what recovers precision. Swept over
//!     the bias to trace the curve.
//!   * `--reach`: the same cover, but the decode weight is nudged by a Layer-2 reachability score
//!     `r ∈ [0,1]` from `probcfg`: `weight = log-odds(P) + bias + gamma·(2r − 1)`. Reachable
//!     instructions get a boost, unreached ones a penalty — the mechanism that can suppress an
//!     appended code-in-data decoy the per-byte posterior cannot. **Honesty wall:** `--reach` only
//!     changes the decode; the reported per-byte posterior (and the ECE/AUROC line) is untouched and
//!     identical to `--cover`.
//!   * `--confirm`: same cover, but the Layer-2 score is a **hard confirmation gate** — `r = 1` iff
//!     the instruction lies in a function *transitively confirmed* from the true entry over the
//!     direct-call graph (`probcfg::confirm_from_entry`), else `r = 0`. Unlike `--reach` (which
//!     leaks: the decoy self-anchors through its own CALLs), the appended code-in-data decoy has no
//!     confirmed caller, so it scores 0 and takes a `−gamma` penalty. Same honesty wall.
//!   * `--confirm-soft`: mode (a) of Milestone 2 — the same decode gate, but `r` is the **soft**
//!     reachedness `R_a ∈ [0,1]` from the M2 confirmation fixpoint (`probcfg::build_soft_confirm`,
//!     eqs 1–4) instead of a hard 0/1. Still decode-only; the posterior line is untouched.
//!   * `--fuse`: mode (b) — the **real Layer-2 posterior**. Fuses `π` and `R` as a product-of-experts
//!     (eq 5), isotonic-recalibrates (Theorem 1), and reports the recalibrated `P̂` alongside `π` on
//!     both axes plus a `P̂` risk–coverage sweep. `P̂` is a *distinct, deliberately recalibrated*
//!     confidence — it never silently overwrites `π`. `--func-gt` adds function-level (`F_h` vs FUNC
//!     symbols) calibration + the β₀ estimate; `--fit-elf/--fit-gt` fit the fusion on a held-out
//!     binary (§9.5 transfer).
//!
//! Where the curve passes above/right of a competitor's point, Soft dominates.
//!
//! ```text
//! bench <binary> <gt> [--entropy S] [--dassa] [--cover|--reach [--gamma F] [--fall-decay F]
//!       [--anchors-all]|--confirm [--gamma F] [--max-fn-span N] [--confirm-tail-jumps]]
//!       [--decoy-from ADDR] [--thresholds ..|--biases ..]
//! ```
//! Reference points (desync-cc O0, measured separately): linear sweep ≈ 0.939 / 0.899;
//! superset ceiling ≈ 1.000 / 0.288.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use evalkit::{evaluate, load_gt, run_soft, IsotonicMap, Metrics};
use probcfg::{
    beta0_features, build_soft_confirm_resolved, confirm_from_entry, reachability, resolve_indirect,
    ConfirmConfig, ReachConfig, ResolveConfig, ResolveKind, ResolvedEdge, SoftConfig, SoftConfirm,
};
use probdisasm::{extract_text_section, Superset};

fn main() -> Result<()> {
    let args = Args::parse(std::env::args().skip(1))?;

    let bytes = fs::read(&args.binary).with_context(|| format!("reading {}", args.binary.display()))?;
    let (base, code) = extract_text_section(&bytes).context("extracting .text")?;
    let gt: HashSet<u64> = load_gt(&args.gt)?;
    if gt.is_empty() {
        bail!("ground truth {} is empty", args.gt.display());
    }
    let post = run_soft(base, code, args.entropy, args.dassa).context("running Soft")?;

    eprintln!(
        "{} : .text {} B @ 0x{base:x}, GT {} starts, entropy={}, dassa={}, mode={}",
        args.binary.display(),
        code.len(),
        gt.len(),
        args.entropy,
        args.dassa,
        if args.fuse { "fuse" } else if args.confirm_soft { "confirm-soft" }
        else if args.confirm { "confirm" } else if args.reach { "reach" }
        else if args.cover { "cover" } else { "threshold" }
    );

    // Honesty axis, from the RAW (untouched) posteriors: does a stated 0.5 land at a 50% real
    // start-rate over all superset offsets? Reported next to the accuracy readout so we can see
    // both — and confirm the decode never touched the confidence.
    let cal = evaluate(&post, &gt);
    let auroc = cal.auroc.map(|a| format!("{a:.4}")).unwrap_or_else(|| "NA".into());
    eprintln!(
        "  HONESTY (raw posterior): base_rate {:.3}  ECE {:.4}  reliability {:.4}  resolution {:.4}  AUROC {auroc}",
        cal.base_rate, cal.ece, cal.reliability, cal.resolution
    );
    // Recalibration headroom: fit an isotonic map on this binary's own posteriors+GT and re-score.
    // Self-fit is an *optimistic ceiling* (fit and eval on the same data), but it proves the map
    // machinery and shows how much ECE a calibration map can recover; AUROC ≈ unchanged confirms the
    // monotone remap preserved ranking (it never touched discrimination).
    let recal = evaluate(&IsotonicMap::fit_from_gt(&post, &gt).apply_all(&post), &gt);
    let recal_auroc = recal.auroc.map(|a| format!("{a:.4}")).unwrap_or_else(|| "NA".into());
    eprintln!(
        "  HONESTY (self-recal ceiling): ECE {:.4} (raw {:.4})  AUROC {recal_auroc} (ranking preserved)",
        recal.ece, cal.ece
    );

    // Also emit a machine-readable calibration line to stdout so a corpus sweep can aggregate it.
    println!(
        "calibration,{:.4},{:.4},{:.4},{:.4},{auroc},{:.4}",
        cal.base_rate, cal.ece, cal.reliability, cal.resolution, recal.ece
    );

    // Optional function-entry GT (FUNC symbols from the benign original) for Layer-2 function-level
    // calibration reporting (§9.1). One hex address per line, same loader as instruction GT.
    let func_gt = args.func_gt.as_ref().map(|p| load_gt(p)).transpose()?;

    if args.fuse {
        // Mode (b): the real Layer-2 posterior. Build the soft model, fit the product-of-experts
        // fusion (eq 5), isotonic-recalibrate (Theorem 1), and report P̂ next to π on both axes.
        run_fuse(&args, &bytes, base, code, &post, &gt, func_gt.as_ref())?;
    } else if args.cover || args.reach || args.confirm || args.confirm_soft {
        // Layer-2 decode weight. The ELF entry point (e_entry, LE u64 at file offset 0x18) is mapped
        // into the analyzed .text; if it lands outside, fall back to `base`. Each mode produces a
        // per-address score `r ∈ [0,1]` folded into the cover weight as `gamma·(2r − 1)`;
        // `reach=None` ⇒ plain cover (weights untouched).
        //
        //   * --reach:        `r` = the (leaky) reachability propagation from `probcfg`.
        //   * --confirm:      `r` = HARD confirmation gate ∈ {0,1} (M1).
        //   * --confirm-soft: `r` = SOFT reachedness R_a ∈ [0,1] from the M2 fixpoint (mode a). The
        //     decode is nudged; the reported posterior is untouched (honesty wall).
        let reach = if args.reach || args.confirm || args.confirm_soft {
            let entry = read_e_entry(&bytes);
            let mapped = if entry >= base && entry < base + code.len() as u64 { entry } else { base };
            let sup = Superset::new(base, code).map_err(|e| anyhow!("building superset: {e:?}"))?;
            let r = if args.confirm_soft {
                let pmap: HashMap<u64, f64> = post.iter().copied().collect();
                let cfg = SoftConfig { max_fn_span: args.max_fn_span, ..SoftConfig::default() };
                let resolved = resolve_edges(&args, &bytes, &sup, mapped, false)?;
                let sc = build_soft_confirm_resolved(&sup, mapped, &pmap, &resolved, &cfg);
                report_soft_summary(&sc, &gt, entry, mapped, args.gamma);
                if let Some(fg) = &func_gt {
                    report_function_calibration(&sc, fg, mapped, args.decoy_from);
                }
                if args.beta0_perbin {
                    report_beta0_features(&sc, &sup, mapped, resolved.len());
                }
                sc.r
            } else if args.confirm {
                let cfg = ConfirmConfig {
                    max_fn_span: args.max_fn_span,
                    confirm_via_tail_jumps: args.confirm_tail_jumps,
                };
                let conf = confirm_from_entry(&sup, mapped, &cfg);
                let hits = conf.confirmed_insns.iter().filter(|a| gt.contains(a)).count();
                eprintln!(
                    "  CONFIRM: entry 0x{entry:x} → root 0x{mapped:x}  gamma={}  max_fn_span={}  confirm_tail_jumps={}",
                    args.gamma, args.max_fn_span, args.confirm_tail_jumps
                );
                eprintln!(
                    "  CONFIRM: candidate_heads={}  confirmed_heads={}  confirmed_insns={}  confirmed_real_recall={:.4} ({}/{})",
                    conf.all_heads.len(),
                    conf.confirmed_heads.len(),
                    conf.confirmed_insns.len(),
                    hits as f64 / gt.len() as f64,
                    hits,
                    gt.len()
                );
                // W1 ablation (read-only): M1 direct-call boolean decoy leak = confirmed heads past
                // the decoy boundary. This is the *strict* boolean-reachability baseline (direct
                // calls only, no resolved indirect edges).
                if let Some(lo) = args.decoy_from {
                    let m1_leak = conf.confirmed_heads.iter().filter(|&&h| h >= lo).count();
                    let m1_real = conf.confirmed_heads.iter().filter(|&&h| h < lo).count();
                    eprintln!("  W1-ABLATION(M1 direct-call): decoy heads confirmed past 0x{lo:x}={m1_leak}  (real heads confirmed={m1_real})");
                    println!("w1_m1,{m1_leak},{m1_real}");
                }
                // Hard gate → r ∈ {0,1}.
                conf.confirmed_insns.into_iter().map(|a| (a, 1.0)).collect()
            } else {
                eprintln!(
                    "  REACH: entry 0x{entry:x} → anchor 0x{mapped:x}  gamma={}  fall_decay={}  anchors_calls_only={}",
                    args.gamma, args.fall_decay, !args.anchors_all
                );
                let cfg = ReachConfig {
                    fall_decay: args.fall_decay,
                    anchors_calls_only: !args.anchors_all,
                    max_iters: 5_000_000,
                };
                reachability(&sup, mapped, &cfg)
            };
            Some((r, args.gamma))
        } else {
            None
        };
        cover_curve(base, code, &post, &gt, &args.biases, reach.as_ref(), args.decoy_from)?;
    } else {
        threshold_curve(&post, &gt, &args.thresholds);
    }
    Ok(())
}

/// The ELF entry vaddr (`e_entry`): little-endian u64 at file offset 0x18. `0` if the buffer is too
/// short to hold an ELF header.
fn read_e_entry(bytes: &[u8]) -> u64 {
    bytes
        .get(0x18..0x20)
        .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
        .unwrap_or(0)
}

// ── Layer-2 indirect resolution (M3a) ───────────────────────────────────────────────────────────

/// Resolve indirect call-graph edges for M3a. Reads the resolve source ELF (`--resolve-elf`, default
/// the primary binary) and runs `probcfg::resolve_indirect` against `sup`. Empty when `--resolve` is
/// off. Reports a resolution summary (edge count by source, distinct targets) to stderr.
fn resolve_edges(
    args: &Args,
    primary: &[u8],
    sup: &Superset,
    mapped: u64,
    quiet: bool,
) -> Result<Vec<ResolvedEdge>> {
    if !args.resolve {
        return Ok(Vec::new());
    }
    let owned;
    let elf_bytes: &[u8] = match &args.resolve_elf {
        Some(p) => {
            owned = fs::read(p).with_context(|| format!("reading --resolve-elf {}", p.display()))?;
            &owned
        }
        None => primary,
    };
    let cfg = ResolveConfig { code_anchored: !args.resolve_data_only, ..ResolveConfig::default() };
    let edges = resolve_indirect(sup, elf_bytes, mapped, &cfg);
    if !quiet {
        let count = |k: ResolveKind| edges.iter().filter(|e| e.kind == k).count();
        let targets: HashSet<u64> = edges.iter().map(|e| e.t).collect();
        eprintln!(
            "  RESOLVE: source={} code_anchored={}  edges={} (init_array={} fini_array={} reloc={} jump_table={} data_ptr={} | computed_goto={} pie_rel={} vtable={})  distinct_targets={}",
            args.resolve_elf.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "<primary>".into()),
            cfg.code_anchored,
            edges.len(),
            count(ResolveKind::InitArray),
            count(ResolveKind::FiniArray),
            count(ResolveKind::Relocation),
            count(ResolveKind::JumpTable),
            count(ResolveKind::DataPointer),
            count(ResolveKind::ComputedGoto),
            count(ResolveKind::PieRelJumpTable),
            count(ResolveKind::Vtable),
            targets.len(),
        );
        // --dump-resolved: one machine-readable line per resolved edge, for the decoy-discipline audit
        // (`resolved,g,t,q,kind` — the eval asserts no `t` lands in [decoy_from, end)).
        if args.dump_resolved {
            for e in &edges {
                println!("resolved,{:#x},{:#x},{:.4},{}", e.g, e.t, e.q, kind_str(e.kind));
            }
        }
    }
    Ok(edges)
}

/// Short stable label for a resolved-edge provenance (used by `--dump-resolved` and the M4 eval).
fn kind_str(kind: ResolveKind) -> &'static str {
    match kind {
        ResolveKind::InitArray => "init_array",
        ResolveKind::FiniArray => "fini_array",
        ResolveKind::Relocation => "reloc",
        ResolveKind::JumpTable => "jump_table",
        ResolveKind::DataPointer => "data_ptr",
        ResolveKind::ComputedGoto => "computed_goto",
        ResolveKind::PieRelJumpTable => "pie_rel",
        ResolveKind::Vtable => "vtable",
    }
}

/// Count unresolved indirect call/jump sites (indirect branch, no static target) — a β̂₀ feature.
fn count_indirect_sites(sup: &Superset) -> usize {
    sup.iter_valid()
        .filter(|i| (i.is_call() || i.is_jump()) && i.branch_target.is_none())
        .count()
}

// ── Layer-2 soft confirmation (M2): reporting + fusion ──────────────────────────────────────────

fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Summarize the soft-confirmation fixpoint to stderr: candidate heads, the confirmed core at a few
/// `F` thresholds, and the reachedness recall ceiling at `R ≥ 0.5` (compare M1's hard ceiling).
fn report_soft_summary(sc: &SoftConfirm, gt: &HashSet<u64>, entry: u64, mapped: u64, gamma: f64) {
    let n_core = |thr: f64| sc.f.values().filter(|&&v| v >= thr).count();
    // Recall of the soft gate at R≥0.5 — the analogue of M1's confirmed_real_recall.
    let hit = sc.r.iter().filter(|(a, &v)| v >= 0.5 && gt.contains(a)).count();
    eprintln!(
        "  CONFIRM-SOFT: entry 0x{entry:x} → root 0x{mapped:x}  gamma={gamma}  candidate_heads={}",
        sc.heads.len()
    );
    eprintln!(
        "  CONFIRM-SOFT: confirmed core |F≥0.9|={}  |F≥0.5|={}  |F≥0.1|={}  soft_recall@R0.5={:.4} ({}/{})",
        n_core(0.9),
        n_core(0.5),
        n_core(0.1),
        hit as f64 / gt.len() as f64,
        hit,
        gt.len()
    );
}

/// Function-level calibration (§9.1): is `F_h` (and the raw prior) calibrated against FUNC-symbol GT?
/// Also estimate the uncalled-tail base rate β₀ (Theorem 2) and the confirmed-core vs tail split.
/// Emits human lines to stderr and machine-readable `func_calib` / `beta0` lines to stdout.
fn report_function_calibration(sc: &SoftConfirm, func_gt: &HashSet<u64>, entry: u64, decoy_from: Option<u64>) {
    // §4.1 function RECALL and §4.2 precision-hold — robust to M3a adding intra-function jump-table
    // labels as heads (labels are real code < the decoy boundary; they are neither in `func_gt` nor
    // past `decoy_from`, so they touch neither number). `func_recall` = confirmed real function
    // entries / all FUNC symbols; `decoy_leak` = heads past the decoy boundary confirmed high.
    let confirmed_func: usize = func_gt.iter().filter(|h| sc.f.get(h).copied().unwrap_or(0.0) >= 0.9).count();
    let func_recall = if func_gt.is_empty() { 0.0 } else { confirmed_func as f64 / func_gt.len() as f64 };
    let decoy_leak = decoy_from
        .map(|lo| sc.heads.iter().filter(|&&h| h >= lo && sc.f.get(&h).copied().unwrap_or(0.0) >= 0.9).count());

    let prior_ps: Vec<(u64, f64)> = sc.heads.iter().map(|&h| (h, sc.prior[&h])).collect();
    let f_ps: Vec<(u64, f64)> = sc.heads.iter().map(|&h| (h, sc.f[&h])).collect();
    let cprior = evaluate(&prior_ps, func_gt);
    let cf = evaluate(&f_ps, func_gt);
    // Isotonic recal ceiling of F vs FUNC GT (Theorem 1 on the function axis).
    let fr = evaluate(&IsotonicMap::fit_from_gt(&f_ps, func_gt).apply_all(&f_ps), func_gt);

    // Uncalled tail U = heads (≠entry) with no *confirmed* incoming caller (max_g F_g·C < 0.5);
    // β₀ = fraction of U that are real functions (FUNC symbols). Core = the complement.
    let (mut u, mut u_real, mut core, mut core_real) = (0usize, 0usize, 0usize, 0usize);
    let mut f_core_sum = 0.0;
    let mut f_tail_sum = 0.0;
    for &h in &sc.heads {
        if h == entry {
            continue;
        }
        let called = sc
            .edges_into
            .get(&h)
            .map(|es| es.iter().any(|&(g, c)| sc.f.get(&g).copied().unwrap_or(0.0) * c >= 0.5))
            .unwrap_or(false);
        let z = usize::from(func_gt.contains(&h));
        if called {
            core += 1;
            core_real += z;
            f_core_sum += sc.f[&h];
        } else {
            u += 1;
            u_real += z;
            f_tail_sum += sc.f[&h];
        }
    }
    let beta0 = if u > 0 { u_real as f64 / u as f64 } else { 0.0 };
    let auroc = |m: &Metrics| m.auroc.map(|a| format!("{a:.4}")).unwrap_or_else(|| "NA".into());

    // THM-2 FALSIFICATION (read-only). Theorem 2 assumes local indistinguishability: conditioned on
    // being uncalled, the local features φ carry no signal for real-vs-decoy. The φ-derived score is
    // the local prior prior_h = σ(w·φ_h) — the ONLY function of φ the model forms. So we test the
    // assumption directly: over the *uncalled* heads that are either a real FUNC symbol or a decoy
    // (past the boundary), can prior_h separate them? AUROC ≈ 0.5 ⇒ φ is uninformative ⇒ the
    // assumption holds and the base-rate predictor is Bayes-optimal (Thm 2/3).
    if let Some(lo) = decoy_from {
        let mut pairs: Vec<(f64, f64)> = Vec::new();
        for &h in &sc.heads {
            if h == entry {
                continue;
            }
            let called = sc
                .edges_into
                .get(&h)
                .map(|es| es.iter().any(|&(g, c)| sc.f.get(&g).copied().unwrap_or(0.0) * c >= 0.5))
                .unwrap_or(false);
            if called {
                continue;
            }
            let is_real = func_gt.contains(&h);
            let is_decoy = h >= lo;
            if is_real || is_decoy {
                pairs.push((sc.prior[&h], if is_real { 1.0 } else { 0.0 }));
            }
        }
        let n_real = pairs.iter().filter(|&&(_, y)| y > 0.5).count();
        let a = evalkit::auroc(&pairs);
        eprintln!(
            "  THM2-φ-INDIST: AUROC(prior_h; real vs decoy | uncalled) = {}  (n_uncalled={} real={} decoy={})  [≈0.5 ⇒ φ uninformative]",
            a.map(|v| format!("{v:.4}")).unwrap_or_else(|| "NA".into()),
            pairs.len(), n_real, pairs.len() - n_real
        );
        println!("thm2_phi_auroc,{},{},{}", a.map(|v| format!("{v:.4}")).unwrap_or_else(|| "nan".into()), n_real, pairs.len() - n_real);
    }

    eprintln!(
        "  FUNC-CAL: heads={} base_rate(FUNC)={:.3}  prior[ECE {:.4} AUROC {}]  F[ECE {:.4} AUROC {}]  F-isoceil[ECE {:.4}]",
        sc.heads.len(),
        cf.base_rate,
        cprior.ece,
        auroc(&cprior),
        cf.ece,
        auroc(&cf),
        fr.ece
    );
    eprintln!(
        "  FUNC-CAL: confirmed core n={core} realFUNC={core_real} meanF={:.3}   uncalled tail n={u} realFUNC={u_real} meanF={:.3}  β₀={beta0:.4}",
        if core > 0 { f_core_sum / core as f64 } else { 0.0 },
        if u > 0 { f_tail_sum / u as f64 } else { 0.0 },
    );
    eprintln!(
        "  FUNC-RECALL: confirmed real functions {confirmed_func}/{} = {func_recall:.4} (F≥0.9)   decoy_leak(F≥0.9)={}   |U|={u}",
        func_gt.len(),
        decoy_leak.map(|d| d.to_string()).unwrap_or_else(|| "NA".into())
    );
    // W1 ablation (read-only): decoy leak at several F thresholds. F>0 ⇒ boolean-reachable from
    // entry (a path of positive-weight edges exists — what a hard M1 confirmation gate commits);
    // F≥0.9 ⇒ soft-confirmed (what the calibrated method commits). The gap is what the probabilistic
    // layer suppresses over plain reachability.
    if let Some(lo) = decoy_from {
        let leak_at = |t: f64| sc.heads.iter().filter(|&&h| h >= lo && sc.f.get(&h).copied().unwrap_or(0.0) >= t).count();
        let real_reach = sc.heads.iter().filter(|&&h| h < lo && sc.f.get(&h).copied().unwrap_or(0.0) > 1e-6).count();
        eprintln!(
            "  W1-ABLATION: decoy heads past 0x{lo:x} — boolean-reachable(F>0)={}  softF≥0.5={}  softF≥0.9={}   (real heads reachable={real_reach})",
            leak_at(1e-6), leak_at(0.5), leak_at(0.9)
        );
        println!("w1_ablation,{},{},{},{}", leak_at(1e-6), leak_at(0.5), leak_at(0.9), real_reach);

        // W1 FRONTIER (read-only): sweep the *boolean* reachability edge-confidence threshold t and
        // trace (real_recall, decoy_leak). Boolean reach at t = heads reachable from entry using only
        // call edges with confidence ≥ t. Compared against the calibrated belief F_h ≥ τ. The point:
        // no single boolean t gets both the indirect real-function tail and low decoy leak, because a
        // decoy reached through several moderate edges and a real function reached through one strong
        // edge are indistinguishable to any edge-threshold — only the multiplicative belief
        // (edge-confidence propagation) separates them.
        if !func_gt.is_empty() {
            // Forward adjacency g → [(h, C_{g→h})], built once from the incoming-edge map.
            let mut fwd: HashMap<u64, Vec<(u64, f64)>> = HashMap::new();
            for (&h, es) in &sc.edges_into {
                for &(g, c) in es {
                    fwd.entry(g).or_default().push((h, c));
                }
            }
            // Boolean reach at edge-confidence threshold t: BFS from entry over edges with C ≥ t.
            let boolean_reach = |t: f64| -> HashSet<u64> {
                let mut reached: HashSet<u64> = HashSet::new();
                let mut q: VecDeque<u64> = VecDeque::new();
                reached.insert(entry);
                q.push_back(entry);
                while let Some(g) = q.pop_front() {
                    if let Some(outs) = fwd.get(&g) {
                        for &(h, c) in outs {
                            if c >= t && reached.insert(h) {
                                q.push_back(h);
                            }
                        }
                    }
                }
                reached
            };
            let nfg = func_gt.len();
            let recall = |s: &HashSet<u64>| func_gt.iter().filter(|h| s.contains(h)).count();
            let leak = |s: &HashSet<u64>| s.iter().filter(|&&h| h >= lo).count();
            eprintln!("  W1-FRONTIER boolean(edge-conf≥t): t → recall, decoy_leak");
            for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99] {
                let s = boolean_reach(t);
                eprintln!("    bool t={t:.2}  recall={:.3} ({}/{nfg})  decoy_leak={}", recall(&s) as f64 / nfg as f64, recall(&s), leak(&s));
                println!("w1_frontier_bool,{t:.2},{},{nfg},{}", recall(&s), leak(&s));
            }
            eprintln!("  W1-FRONTIER calibrated(F≥τ): τ → recall, decoy_leak");
            for tau in [0.1, 0.3, 0.5, 0.7, 0.9] {
                let cr = func_gt.iter().filter(|h| sc.f.get(h).copied().unwrap_or(0.0) >= tau).count();
                let cl = sc.heads.iter().filter(|&&h| h >= lo && sc.f.get(&h).copied().unwrap_or(0.0) >= tau).count();
                eprintln!("    calib τ={tau:.2}  recall={:.3} ({cr}/{nfg})  decoy_leak={cl}", cr as f64 / nfg as f64);
                println!("w1_frontier_cal,{tau:.2},{cr},{nfg},{cl}");
            }
        }
    }
    println!("func_calib,prior,{:.4},{},{:.4}", cprior.ece, auroc(&cprior), cprior.base_rate);
    println!("func_calib,F,{:.4},{},{:.4}", cf.ece, auroc(&cf), cf.base_rate);
    println!("func_calib,F_isoceil,{:.4},{},{:.4}", fr.ece, auroc(&fr), fr.base_rate);
    println!(
        "func_recall,{func_recall:.4},{confirmed_func},{},{},{u}",
        func_gt.len(),
        decoy_leak.map(|d| d.to_string()).unwrap_or_else(|| "-1".into())
    );
    println!("beta0,{beta0:.4},{u_real},{u},{core_real},{core}");
}

/// M3b: emit the per-binary β̂₀ feature vector ψ(b) to stdout (`beta0_feats,`) alongside the observed
/// β₀ line, so a corpus sweep can fit the group regressor + run the Theorem-3 aggregate/transfer check.
fn report_beta0_features(sc: &SoftConfirm, sup: &Superset, entry: u64, n_resolved: usize) {
    let n_indirect = count_indirect_sites(sup);
    let psi = beta0_features(sc, entry, n_resolved, n_indirect);
    let v = psi.to_vec();
    eprintln!(
        "  BETA0-ψ: tail_frac={:.4} unresolved_indirect_density={:.5} data_ptr_density={:.4} mean_tail_prior={:.4}  (n_resolved={n_resolved} n_indirect_sites={n_indirect})",
        v[0], v[1], v[2], v[3]
    );
    println!("beta0_feats,{:.6},{:.6},{:.6},{:.6}", v[0], v[1], v[2], v[3]);
}

/// A fitted product-of-experts fusion (eq 5): `logit(P²) = α·z(logit π) + β·z(logit R) + b`, where
/// `z(·)` standardizes each feature (fit by MLE / batch gradient descent for numerical stability).
struct Fusion {
    m1: f64,
    s1: f64,
    m2: f64,
    s2: f64,
    a: f64,
    b: f64,
    c: f64,
}

impl Fusion {
    /// Fit on `(logit_pi, logit_R, y)` rows.
    fn fit(rows: &[(f64, f64, f64)]) -> Self {
        let n = rows.len().max(1) as f64;
        let (m1, m2) = (
            rows.iter().map(|r| r.0).sum::<f64>() / n,
            rows.iter().map(|r| r.1).sum::<f64>() / n,
        );
        let var = |sel: fn(&(f64, f64, f64)) -> f64, m: f64| {
            (rows.iter().map(|r| (sel(r) - m).powi(2)).sum::<f64>() / n).sqrt().max(1e-9)
        };
        let (s1, s2) = (var(|r| r.0, m1), var(|r| r.1, m2));
        let (mut a, mut b, mut c) = (1.0f64, 1.0f64, 0.0f64);
        let lr = 0.1;
        for _ in 0..4000 {
            let (mut ga, mut gb, mut gc) = (0.0, 0.0, 0.0);
            for r in rows {
                let (x1, x2) = ((r.0 - m1) / s1, (r.1 - m2) / s2);
                let e = sigmoid(a * x1 + b * x2 + c) - r.2;
                ga += e * x1;
                gb += e * x2;
                gc += e;
            }
            a -= lr * ga / n;
            b -= lr * gb / n;
            c -= lr * gc / n;
        }
        Fusion { m1, s1, m2, s2, a, b, c }
    }
    /// Raw fused posterior `P²` for a `(π, R)` pair.
    fn p2(&self, pi: f64, r: f64) -> f64 {
        sigmoid(self.a * (logit(pi) - self.m1) / self.s1 + self.b * (logit(r) - self.m2) / self.s2 + self.c)
    }
    fn coeffs(&self) -> (f64, f64, f64) {
        (self.a, self.b, self.c)
    }
}

/// Build `(logit_pi, logit_R, y)` fusion rows for an ELF: run Soft, solve the soft-confirm fixpoint,
/// and pair each posterior with its reachedness and instruction-GT label. Used both for the primary
/// binary and (for the §9.5 transfer test) an optional held-out fit binary.
fn fusion_rows(
    elf: &std::path::Path,
    gt_path: &std::path::Path,
    entropy: f64,
    dassa: bool,
    max_fn_span: usize,
    resolve: bool,
    code_anchored: bool,
) -> Result<Vec<(f64, f64, f64)>> {
    let bytes = fs::read(elf).with_context(|| format!("reading {}", elf.display()))?;
    let (base, code) = extract_text_section(&bytes).context("extracting .text")?;
    let gt = load_gt(gt_path)?;
    let post = run_soft(base, code, entropy, dassa).context("running Soft")?;
    let entry = read_e_entry(&bytes);
    let mapped = if entry >= base && entry < base + code.len() as u64 { entry } else { base };
    let sup = Superset::new(base, code).map_err(|e| anyhow!("building superset: {e:?}"))?;
    let pmap: HashMap<u64, f64> = post.iter().copied().collect();
    let resolved = if resolve {
        resolve_indirect(&sup, &bytes, mapped, &ResolveConfig { code_anchored, ..ResolveConfig::default() })
    } else {
        Vec::new()
    };
    let sc = build_soft_confirm_resolved(&sup, mapped, &pmap, &resolved, &SoftConfig { max_fn_span, ..SoftConfig::default() });
    Ok(post
        .iter()
        .map(|&(a, p)| (logit(p), logit(sc.r.get(&a).copied().unwrap_or(0.0)), f64::from(gt.contains(&a))))
        .collect())
}

/// Mode (b) — calibrated fusion. Builds the soft model on the primary binary, fits the fusion +
/// isotonic map (on this binary, or on `--fit-elf`/`--fit-gt` for a transfer test), and reports the
/// recalibrated Layer-2 posterior `P̂` against `π` on both the honest (ECE) and accurate (AUROC)
/// axes — the Theorem-1 check — plus a `P̂` risk–coverage sweep with decoy leak (§9.3).
fn run_fuse(
    args: &Args,
    bytes: &[u8],
    base: u64,
    code: &[u8],
    post: &[(u64, f64)],
    gt: &HashSet<u64>,
    func_gt: Option<&HashSet<u64>>,
) -> Result<()> {
    let entry = read_e_entry(bytes);
    let mapped = if entry >= base && entry < base + code.len() as u64 { entry } else { base };
    let sup = Superset::new(base, code).map_err(|e| anyhow!("building superset: {e:?}"))?;
    let pmap: HashMap<u64, f64> = post.iter().copied().collect();
    let resolved = resolve_edges(args, bytes, &sup, mapped, false)?;
    let sc = build_soft_confirm_resolved(
        &sup,
        mapped,
        &pmap,
        &resolved,
        &SoftConfig { max_fn_span: args.max_fn_span, ..SoftConfig::default() },
    );
    report_soft_summary(&sc, gt, entry, mapped, args.gamma);
    if let Some(fg) = func_gt {
        report_function_calibration(&sc, fg, mapped, args.decoy_from);
    }
    if args.beta0_perbin {
        report_beta0_features(&sc, &sup, mapped, resolved.len());
    }

    // Primary-binary design rows (aligned to `post`), and the fit set (self or transfer).
    let rows: Vec<(u64, f64, f64, f64)> = post
        .iter()
        .map(|&(a, p)| (a, logit(p), logit(sc.r.get(&a).copied().unwrap_or(0.0)), f64::from(gt.contains(&a))))
        .collect();
    let (fit_rows, fit_label): (Vec<(f64, f64, f64)>, String) =
        match (&args.fit_elf, &args.fit_gt) {
            (Some(e), Some(g)) => (
                fusion_rows(e, g, args.entropy, args.dassa, args.max_fn_span, args.resolve, !args.resolve_data_only)?,
                format!("transfer-fit on {}", e.display()),
            ),
            _ => (rows.iter().map(|&(_, x1, x2, y)| (x1, x2, y)).collect(), "self-fit".into()),
        };

    let fusion = Fusion::fit(&fit_rows);
    let (fa, fb, fc) = fusion.coeffs();
    // Raw fused posterior P² on the primary binary.
    let p2: Vec<(u64, f64)> =
        rows.iter().map(|&(a, x1, x2, _)| (a, fusion.p2(sigmoid(x1), sigmoid(x2)))).collect();
    // Isotonic map fit on the fit set's (P², y); applied to the primary P².
    let iso_samples: Vec<(f64, f64)> = {
        let fit_fusion = &fusion;
        fit_rows.iter().map(|&(x1, x2, y)| (fit_fusion.p2(sigmoid(x1), sigmoid(x2)), y)).collect()
    };
    let iso = IsotonicMap::fit(&iso_samples);
    let phat: Vec<(u64, f64)> = iso.apply_all(&p2);

    // Both axes: raw π (Layer-1) vs raw fusion P² vs recalibrated P̂ (Layer-2).
    let cpi = evaluate(post, gt);
    let cp2 = evaluate(&p2, gt);
    let cph = evaluate(&phat, gt);
    let auroc = |m: &Metrics| m.auroc.map(|a| format!("{a:.4}")).unwrap_or_else(|| "NA".into());
    eprintln!("  FUSE ({fit_label}): α={fa:.3} β={fb:.3} b={fc:.3}  (standardized features)");
    eprintln!(
        "  FUSE INSTR-CAL:  π    [ECE {:.4} AUROC {}]",
        cpi.ece,
        auroc(&cpi)
    );
    eprintln!(
        "  FUSE INSTR-CAL:  P²   [ECE {:.4} AUROC {}]   (raw fusion)",
        cp2.ece,
        auroc(&cp2)
    );
    eprintln!(
        "  FUSE INSTR-CAL:  P̂    [ECE {:.4} AUROC {}]   (isotonic — Theorem 1: ECE↓, AUROC=)",
        cph.ece,
        auroc(&cph)
    );
    // Machine-readable, next to the untouched `calibration,` line (honesty wall: π unchanged).
    println!("fuse_calib,pi,{:.4},{},{:.4}", cpi.ece, auroc(&cpi), cpi.base_rate);
    println!("fuse_calib,fusion,{:.4},{},{:.4}", cp2.ece, auroc(&cp2), cp2.base_rate);
    println!("fuse_calib,phat,{:.4},{},{:.4}", cph.ece, auroc(&cph), cph.base_rate);

    // P̂ risk–coverage / P–R sweep (§9.3): sweep the operating threshold on the recalibrated posterior.
    let leak_hdr = if args.decoy_from.is_some() { ",decoy_leak,mean_conf_decoy" } else { "" };
    println!("phat_tau,n_pred,tp,recall,precision,f1{leak_hdr}");
    for &t in &args.thresholds {
        let pred: Vec<u64> = phat.iter().filter(|&&(_, p)| p >= t).map(|&(a, _)| a).collect();
        let (recall, precision, f1) = score(&pred, gt);
        let leak = match args.decoy_from {
            Some(lo) => {
                let d: Vec<f64> =
                    phat.iter().filter(|&&(a, p)| a >= lo && p >= t).map(|&(_, p)| p).collect();
                let mc = if d.is_empty() { 0.0 } else { d.iter().sum::<f64>() / d.len() as f64 };
                format!(",{},{:.4}", d.len(), mc)
            }
            None => String::new(),
        };
        println!(
            "{t:.4},{},{},{recall:.4},{precision:.4},{f1:.4}{leak}",
            pred.len(),
            count_tp(&pred, gt)
        );
    }
    Ok(())
}

/// Threshold readout: predicted starts = `{ addr : P ≥ τ }`.
fn threshold_curve(post: &[(u64, f64)], gt: &HashSet<u64>, thresholds: &[f64]) {
    println!("threshold,n_pred,tp,recall,precision,f1");
    for &t in thresholds {
        let pred: Vec<u64> = post.iter().filter(|&&(_, p)| p >= t).map(|&(a, _)| a).collect();
        let (recall, precision, f1) = score(&pred, gt);
        println!("{t:.2},{},{},{recall:.4},{precision:.4},{f1:.4}", pred.len(), count_tp(&pred, gt));
    }
}

/// Cover readout: max-weight non-overlapping instruction set, swept over a global log-odds bias.
///
/// `reach = Some((r, gamma))` adds the Layer-2 reachability term `gamma·(2·r[addr] − 1)` to each
/// interval's decode weight (unreached ⇒ `r = 0` ⇒ a `−gamma` penalty). This changes only the
/// selection, never the posterior. `decoy_from = Some(lo)` adds a `decoy_leak` column counting how
/// many selected starts land at or past `lo` (the code-in-data decoy boundary) — the leak metric.
fn cover_curve(
    base: u64,
    code: &[u8],
    post: &[(u64, f64)],
    gt: &HashSet<u64>,
    biases: &[f64],
    reach: Option<&(HashMap<u64, f64>, f64)>,
    decoy_from: Option<u64>,
) -> Result<()> {
    let sup = Superset::new(base, code).map_err(|e| anyhow!("building superset: {e:?}"))?;
    let pmap: HashMap<u64, f64> = post.iter().copied().collect();

    // Candidate intervals [addr, addr+size) with base weight = log-odds(P) + reachability term,
    // sorted by end.
    let mut ivs: Vec<Iv> = Vec::new();
    for insn in sup.instructions.iter().flatten() {
        let size = insn.size as u64;
        if size == 0 {
            continue;
        }
        let p = pmap.get(&insn.address).copied().unwrap_or(0.0).clamp(1e-6, 1.0 - 1e-6);
        let mut w = (p / (1.0 - p)).ln();
        if let Some((r, gamma)) = reach {
            let ra = r.get(&insn.address).copied().unwrap_or(0.0);
            w += gamma * (2.0 * ra - 1.0);
        }
        ivs.push(Iv { start: insn.address, end: insn.address + size, w });
    }
    ivs.sort_by_key(|x| x.end);
    let ends: Vec<u64> = ivs.iter().map(|x| x.end).collect();
    // pred[j] = number of intervals whose end ≤ start of interval j (its best compatible prefix).
    let pred: Vec<usize> = ivs.iter().map(|iv| ends.partition_point(|&e| e <= iv.start)).collect();
    let n = ivs.len();

    let leak_hdr = if decoy_from.is_some() { ",decoy_leak" } else { "" };
    println!("bias,n_pred,tp,recall,precision,f1{leak_hdr}");
    for &b in biases {
        let mut dp = vec![0.0f64; n + 1];
        let mut take = vec![false; n + 1];
        for j in 1..=n {
            let incl = ivs[j - 1].w + b + dp[pred[j - 1]];
            if incl > dp[j - 1] {
                dp[j] = incl;
                take[j] = true;
            } else {
                dp[j] = dp[j - 1];
            }
        }
        let mut sel: Vec<u64> = Vec::new();
        let mut j = n;
        while j > 0 {
            if take[j] {
                sel.push(ivs[j - 1].start);
                j = pred[j - 1];
            } else {
                j -= 1;
            }
        }
        let (recall, precision, f1) = score(&sel, gt);
        let leak = match decoy_from {
            Some(lo) => format!(",{}", sel.iter().filter(|&&a| a >= lo).count()),
            None => String::new(),
        };
        println!(
            "{b:.2},{},{},{recall:.4},{precision:.4},{f1:.4}{leak}",
            sel.len(),
            count_tp(&sel, gt)
        );
    }
    Ok(())
}

struct Iv {
    start: u64,
    end: u64,
    w: f64,
}

fn count_tp(pred: &[u64], gt: &HashSet<u64>) -> usize {
    pred.iter().filter(|&&a| gt.contains(&a)).count()
}

/// (recall, precision, f1) of predicted starts against ground truth.
fn score(pred: &[u64], gt: &HashSet<u64>) -> (f64, f64, f64) {
    let tp = count_tp(pred, gt) as f64;
    let recall = tp / gt.len() as f64;
    let precision = if pred.is_empty() { 0.0 } else { tp / pred.len() as f64 };
    let f1 = if recall + precision > 0.0 {
        2.0 * recall * precision / (recall + precision)
    } else {
        0.0
    };
    (recall, precision, f1)
}

struct Args {
    binary: PathBuf,
    gt: PathBuf,
    entropy: f64,
    dassa: bool,
    cover: bool,
    reach: bool,
    confirm: bool,
    confirm_soft: bool,
    fuse: bool,
    resolve: bool,
    resolve_elf: Option<PathBuf>,
    resolve_data_only: bool,
    dump_resolved: bool,
    beta0_perbin: bool,
    gamma: f64,
    fall_decay: f64,
    anchors_all: bool,
    max_fn_span: usize,
    confirm_tail_jumps: bool,
    decoy_from: Option<u64>,
    func_gt: Option<PathBuf>,
    fit_elf: Option<PathBuf>,
    fit_gt: Option<PathBuf>,
    thresholds: Vec<f64>,
    biases: Vec<f64>,
}

impl Args {
    fn parse(mut it: impl Iterator<Item = String>) -> Result<Self> {
        const USAGE: &str =
            "usage: bench <binary> <gt> [--entropy S] [--dassa] \
             [--cover|--reach [--gamma F] [--fall-decay F] [--anchors-all] \
             |--confirm [--gamma F] [--max-fn-span N] [--confirm-tail-jumps] \
             |--confirm-soft [--gamma F] [--max-fn-span N] \
             |--fuse [--fit-elf ELF --fit-gt GT]] \
             [--resolve [--resolve-elf ELF] [--resolve-data-only] [--dump-resolved]] [--beta0-perbin] \
             [--func-gt PATH] [--decoy-from ADDR] [--thresholds a,b,..] [--biases a,b,..]";
        let mut positional = Vec::new();
        let mut entropy = 0.0;
        let mut dassa = false;
        let mut cover = false;
        let mut reach = false;
        let mut confirm = false;
        let mut confirm_soft = false;
        let mut fuse = false;
        let mut resolve = false;
        let mut resolve_elf: Option<PathBuf> = None;
        let mut resolve_data_only = false;
        let mut dump_resolved = false;
        let mut beta0_perbin = false;
        let mut gamma = 4.0;
        let mut fall_decay = 0.9;
        let mut anchors_all = false;
        let mut max_fn_span = 65536usize;
        let mut confirm_tail_jumps = false;
        let mut decoy_from: Option<u64> = None;
        let mut func_gt: Option<PathBuf> = None;
        let mut fit_elf: Option<PathBuf> = None;
        let mut fit_gt: Option<PathBuf> = None;
        let mut thresholds: Vec<f64> = (1..=19).map(|i| i as f64 / 20.0).collect();
        thresholds.push(0.99);
        let mut biases: Vec<f64> = (-10..=10).map(|i| i as f64 * 0.5).collect();

        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--entropy" => {
                    entropy = it.next().context("--entropy needs a value")?.parse().context("--entropy float")?
                }
                "--dassa" => dassa = true,
                "--cover" => cover = true,
                "--reach" => reach = true,
                "--gamma" => {
                    gamma = it.next().context("--gamma needs a value")?.parse().context("--gamma float")?
                }
                "--fall-decay" => {
                    fall_decay =
                        it.next().context("--fall-decay needs a value")?.parse().context("--fall-decay float")?
                }
                "--anchors-all" => anchors_all = true,
                "--confirm" => confirm = true,
                "--max-fn-span" => {
                    max_fn_span =
                        it.next().context("--max-fn-span needs a value")?.parse().context("--max-fn-span usize")?
                }
                "--confirm-tail-jumps" => confirm_tail_jumps = true,
                "--confirm-soft" => confirm_soft = true,
                "--fuse" => fuse = true,
                "--resolve" => resolve = true,
                "--resolve-elf" => {
                    resolve_elf = Some(PathBuf::from(it.next().context("--resolve-elf needs a path")?))
                }
                "--resolve-data-only" => resolve_data_only = true,
                "--dump-resolved" => dump_resolved = true,
                "--beta0-perbin" => beta0_perbin = true,
                "--func-gt" => {
                    func_gt = Some(PathBuf::from(it.next().context("--func-gt needs a path")?))
                }
                "--fit-elf" => {
                    fit_elf = Some(PathBuf::from(it.next().context("--fit-elf needs a path")?))
                }
                "--fit-gt" => {
                    fit_gt = Some(PathBuf::from(it.next().context("--fit-gt needs a path")?))
                }
                "--decoy-from" => {
                    let v = it.next().context("--decoy-from needs a value")?;
                    let v = v.strip_prefix("0x").map(|h| u64::from_str_radix(h, 16)).unwrap_or_else(|| v.parse());
                    decoy_from = Some(v.context("--decoy-from wants a u64 (dec or 0x hex)")?);
                }
                "--thresholds" => thresholds = parse_floats(&mut it, "--thresholds")?,
                "--biases" => biases = parse_floats(&mut it, "--biases")?,
                "-h" | "--help" => {
                    eprintln!("{USAGE}");
                    std::process::exit(0);
                }
                other if other.starts_with('-') => bail!("unexpected flag: {other}"),
                other => positional.push(PathBuf::from(other)),
            }
        }
        let [binary, gt] = positional.as_slice() else {
            bail!("{USAGE}");
        };
        Ok(Args {
            binary: binary.clone(),
            gt: gt.clone(),
            entropy,
            dassa,
            cover,
            reach,
            confirm,
            confirm_soft,
            fuse,
            resolve,
            resolve_elf,
            resolve_data_only,
            dump_resolved,
            beta0_perbin,
            gamma,
            fall_decay,
            anchors_all,
            max_fn_span,
            confirm_tail_jumps,
            decoy_from,
            func_gt,
            fit_elf,
            fit_gt,
            thresholds,
            biases,
        })
    }
}

fn parse_floats(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<Vec<f64>> {
    it.next()
        .with_context(|| format!("{flag} needs a value"))?
        .split(',')
        .map(|s| s.trim().parse::<f64>().with_context(|| format!("{flag} wants floats")))
        .collect()
}
