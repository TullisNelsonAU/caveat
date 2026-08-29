//! `udstack` — drive the uncertainty-preserving stack on a binary and report calibrated marginals.
//!
//! Modes:
//!   * `--milestone a` (default): **reproduce M2** — a two-layer stack with the bottom-up message
//!     only (`π → C → F → R`) fused once. Prints instruction `π` (L1) vs stack `P̂` (joint) on both
//!     axes; must match `bench --fuse` within noise.
//!   * `--milestone b`: **coupled relaxation** — add the top-down message (`F → R → π̂`), damped, with
//!     `(S3)` exclusion; iterate to a fixpoint. Prints convergence (sweeps, `λ`, final `‖Δ‖_∞`).
//!   * `--clamp-func ADDR[:q]`: inject online evidence (a confirmed function) before relaxing — the
//!     online-update mode (§5).
//!
//! Joint-beats-parts (§7): every mode reports L1-only (`π`), L2-only (`R`), and the joint stack
//! marginal `P̂`, so a corpus sweep can show coupling improves instruction P/R *and* function
//! calibration. Machine-readable `stack_*` lines go to stdout; human summary to stderr.
//!
//! ```text
//! udstack <binary> <gt> [--func-gt P] [--resolve-elf DATA_ELF] [--milestone a|b]
//!         [--lambda F] [--decoy-from ADDR] [--clamp-func ADDR[:q]]
//!         [--fit-elf E --fit-gt G] [--thresholds a,b,..]
//! ```

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use evalkit::{evaluate, load_gt, Metrics};
use udstack::{Kind, ObjId, Schedule, Stack};

fn main() -> Result<()> {
    let args = Args::parse(std::env::args().skip(1))?;
    let bytes = fs::read(&args.binary).with_context(|| format!("reading {}", args.binary.display()))?;
    let gt: HashSet<u64> = load_gt(&args.gt)?;
    if gt.is_empty() {
        bail!("ground truth {} is empty", args.gt.display());
    }
    let func_gt = args.func_gt.as_ref().map(|p| load_gt(p)).transpose()?;
    let resolve_bytes = args.resolve_elf.as_ref().map(fs::read).transpose()?;

    let mut stack = Stack::from_elf(
        &bytes,
        args.entropy,
        args.dassa,
        args.max_fn_span,
        resolve_bytes.as_deref(),
    )?;

    // K=3: build the L4 module layer (the condensation of the confirmed call graph). The stack then
    // relaxes at depth 3 and reports the module axis alongside instr/func. `--layers 2` (default) is
    // the untouched K=2 baseline used for the compositionality comparison.
    if args.layers >= 3 {
        stack.build_module_layer();
    }

    let sched = match args.milestone {
        Milestone::A => Schedule::bottom_up_once(),
        Milestone::B => {
            // Phase-2 knobs: let the convergence study drive `ε` tighter and lift the sweep cap so the
            // contraction tail is visible. Defaults are unchanged (`coupled` = 1e-4 / 64).
            let mut s = Schedule::coupled(args.lambda);
            if let Some(e) = args.eps {
                s.eps = e;
            }
            if let Some(m) = args.max_sweeps {
                s.max_sweeps = m;
            }
            s
        }
    };

    // §9.5 transfer: fit the whole operator (S1 pool + S2 fixpoint g_o) on a held-out binary at the
    // SAME schedule, then install both before relaxing the target — a pure cross-binary transfer.
    if let (Some(e), Some(g)) = (&args.fit_elf, &args.fit_gt) {
        let fit_bytes = fs::read(e).with_context(|| format!("reading --fit-elf {}", e.display()))?;
        let fit_gt: HashSet<u64> = load_gt(g)?;
        let fit_resolve = args.resolve_elf.as_ref().map(fs::read).transpose()?;
        let mut fit_stack = Stack::from_elf(&fit_bytes, args.entropy, args.dassa, args.max_fn_span, fit_resolve.as_deref())?;
        fit_stack.relax(sched, &fit_gt);
        stack.install_pool(fit_stack.pool_of(Kind::Instr).cloned().context("held-out pool")?);
        stack.install_cal(fit_stack.cal_of(Kind::Instr).cloned().context("held-out g_o")?);
    }

    // Online evidence (§5): clamp confirmed functions before relaxing.
    for &(addr, q) in &args.clamp_func {
        stack.clamp(ObjId::func(addr), q);
    }
    eprintln!(
        "{} : GT {} starts, entry 0x{:x}, milestone {}, λ={}, resolve={}",
        args.binary.display(),
        gt.len(),
        stack.entry(),
        match args.milestone { Milestone::A => "A (bottom-up=M2)", Milestone::B => "B (coupled)" },
        args.lambda,
        args.resolve_elf.is_some(),
    );

    // ── Active analysis (§5): greedy sequential online evidence under a strategy. ──────────────────
    // Relax once to fit the operators + freeze the calibration readout, then confirm K heads one at a
    // time by `strategy`, re-relaxing after each. Every step reports the calibrated marginals AND the
    // untouched L1 π (identical across steps ⇒ the honesty wall holds). Machine line: `stack_active`.
    if let Some((strat, k)) = &args.active {
        let fg = func_gt.clone().context("--active requires --func-gt (the query oracle)")?;
        run_active(&mut stack, sched, &gt, &fg, *strat, *k, args.query_q, args.query_cap)?;
        return Ok(());
    }

    // ── Arm B — incremental analysis (§2): an evidence stream (withheld symbols / trace hits / a
    // resolved edge) enters one item at a time as a clamp, propagating through the stack. Report the
    // calibrated instruction-map quality (AUROC / coverage / ECE) AND a committing recursive-descent
    // baseline after each item, plus the invariant π (honesty wall). Machine line: `stack_incr`.
    if let Some(counts) = args.incremental {
        let fg = func_gt.clone().context("--incremental requires --func-gt (the symbol stream)")?;
        run_incremental(&mut stack, sched, &gt, &fg, counts, args.query_q)?;
        return Ok(());
    }

    // At K=3 the higher layers need the FUNC-symbol GT to fit their (S1) pool + (S2) g_o (the same
    // held-out-GT fit L1⊗L2 uses). `relax_layered` is the K-agnostic driver; at K=2 with func_gt=None it
    // is bit-for-bit `relax`.
    let conv = stack.relax_layered(sched, &gt, func_gt.as_ref());
    if matches!(args.milestone, Milestone::B) {
        eprintln!(
            "  CONVERGE: iters={} converged={} final‖Δ‖∞={:.2e}  deltas=[{}]",
            conv.iters(),
            conv.converged,
            conv.final_delta(),
            conv.deltas.iter().map(|d| format!("{d:.1e}")).collect::<Vec<_>>().join(" "),
        );
        println!("stack_converge,{},{},{:.3e},{}", conv.iters(), conv.converged, conv.final_delta(), args.lambda);

        // Phase 2: the log-odds contraction trace and its ratio ρ ≈ limsup ‖Δ^{t+1}‖/‖Δ^{t}‖.
        if args.trace {
            let rho = conv.contraction_ratio();
            eprintln!(
                "  TRACE(λ={}, K={}): ρ={:.4} iters={} converged={}  logitΔ=[{}]",
                args.lambda, stack.depth(), rho, conv.iters(), conv.converged,
                conv.logit_deltas.iter().map(|d| format!("{d:.2e}")).collect::<Vec<_>>().join(" "),
            );
            // Machine line: ρ(λ,K) sweep row.
            println!("stack_rho,{},{},{},{},{:.6},{:.3e}", args.lambda, stack.depth(), conv.iters(), conv.converged, rho, conv.final_delta());
            // Full per-iteration log-odds trace (for plotting the geometric decay).
            let tr: Vec<String> = conv.logit_deltas.iter().map(|d| format!("{d:.6e}")).collect();
            println!("stack_trace,{},{},{}", args.lambda, stack.depth(), tr.join(","));
        }
    }

    // ── Joint-beats-parts on the instruction axis: L1-only (π) vs L2-only (R) vs joint (P̂). ──
    let auc = |m: &Metrics| m.auroc.map(|a| format!("{a:.4}")).unwrap_or_else(|| "NA".into());
    let cpi = evaluate(&stack.pi_marginals(), &gt); // L1 alone
    let r_marg: Vec<(u64, f64)> =
        stack.pi_marginals().iter().map(|&(a, _)| (a, stack.reachedness_map().get(&a).copied().unwrap_or(0.0))).collect();
    let cr = evaluate(&r_marg, &gt); // L2 alone (reachedness as instruction predictor)
    let cph = evaluate(&stack.instr_marginals(), &gt); // joint

    eprintln!(
        "  JOINT-vs-PARTS (instr): L1 π    [ECE {:.4} AUROC {}]",
        cpi.ece, auc(&cpi)
    );
    eprintln!(
        "                          L2 R    [ECE {:.4} AUROC {}]",
        cr.ece, auc(&cr)
    );
    eprintln!(
        "                          joint P̂ [ECE {:.4} AUROC {}]   (ΔAUROC vs L1 = {:+.4})",
        cph.ece, auc(&cph),
        cph.auroc.unwrap_or(0.0) - cpi.auroc.unwrap_or(0.0)
    );
    println!("stack_instr,pi,{:.4},{},{:.4}", cpi.ece, auc(&cpi), cpi.base_rate);
    println!("stack_instr,R,{:.4},{},{:.4}", cr.ece, auc(&cr), cr.base_rate);
    println!("stack_instr,phat,{:.4},{},{:.4}", cph.ece, auc(&cph), cph.base_rate);

    // ── Function-level calibration (L2 marginals vs FUNC-symbol GT), if provided. ──
    if let Some(fg) = &func_gt {
        let f_marg = stack.func_marginals();
        let cf = evaluate(&f_marg, fg);
        eprintln!(
            "  FUNC-CAL (F_h vs FUNC-GT): heads={} base_rate={:.3}  F[ECE {:.4} AUROC {}]",
            f_marg.len(), cf.base_rate, cf.ece, auc(&cf)
        );
        println!("stack_func,F,{:.4},{},{:.4}", cf.ece, auc(&cf), cf.base_rate);

        // ── Module-level axis (L4 SCC confirmations vs derived module GT), K=3 only. ──
        if stack.depth() >= 3 {
            let mg = stack.module_gt(fg);
            let m_marg = stack.module_marginals();
            let cm = evaluate(&m_marg, &mg);
            eprintln!(
                "  MODULE-CAL (F_c vs derived module-GT): comps={} real={} base_rate={:.3}  F_c[ECE {:.4} AUROC {}]",
                m_marg.len(), mg.len(), cm.base_rate, cm.ece, auc(&cm)
            );
            println!("stack_module,F,{:.4},{},{:.4}", cm.ece, auc(&cm), cm.base_rate);
        }
    }

    // ── Head diagnostics for the online-update demo: F_h per head, and a per-head body recovery. ──
    let phat = stack.instr_marginals();
    let pmap: std::collections::HashMap<u64, f64> = phat.iter().copied().collect();
    if args.dump_heads {
        // `bel_f` = the calibrated fused function marginal (`func_marginals`) — same quantity dumped at
        // K=3 in `stack_headmod`, emitted here at K=2 so the harness can measure ΔAUROC(K2→K3) on the
        // SAME per-head axis (raw `f` stays for back-compat). Works at any depth.
        let fm: std::collections::HashMap<u64, f64> = stack.func_marginals().into_iter().collect();
        for &h in stack.heads() {
            let f = stack.confirmation_map().get(&h).copied().unwrap_or(0.0);
            let bel_f = fm.get(&h).copied().unwrap_or(0.0);
            let is_real = func_gt.as_ref().map(|g| g.contains(&h)).unwrap_or(false);
            println!("stack_head,0x{h:x},{f:.4},{},{bel_f:.4}", if is_real { "real" } else { "-" });
        }
    }
    // Per-instruction calibrated marginal `bel_a = P̂_a` (and the invariant L1 `π_a`) for every
    // candidate instruction start — the confidence signal the `rewrite` crate gates on (bel ≥ τ).
    // Consumed downstream; udstack never sees the rewriter's decisions (the honesty wall runs one way).
    if args.dump_instr {
        let pi = stack.pi_marginals();
        let pimap: std::collections::HashMap<u64, f64> = pi.iter().copied().collect();
        for &(a, p) in &phat {
            let is_real = gt.contains(&a);
            println!(
                "instr_bel,0x{a:x},{p:.6},{:.6},{}",
                pimap.get(&a).copied().unwrap_or(0.0),
                if is_real { "real" } else { "-" }
            );
        }
    }
    // First-class analysis→rewrite hook: feed the calibrated instruction marginals (`phat`, the same
    // bel the `rewrite` crate gates on) straight into the confidence-gated rewriter and emit the
    // patched binary — no `--dump-instr` file round-trip. `--rewrite-tau 0` = commit-all (ungated).
    // Pure post-analysis consumer: it reads `phat`, never a belief, so the honesty wall holds.
    if let Some(out) = &args.rewrite_out {
        let bel: std::collections::HashMap<u64, f64> = phat.iter().copied().collect();
        let (patched, st) = rewrite::gated_rewrite(bytes.clone(), Some(&bel), args.rewrite_tau)
            .context("gated rewrite from calibrated marginals")?;
        std::fs::write(out, &patched).with_context(|| format!("write {}", out.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perm = std::fs::metadata(out)?.permissions();
            perm.set_mode(0o755);
            std::fs::set_permissions(out, perm)?;
        }
        eprintln!(
            "rewrite: tau={:.2} leaders={} sites={} coverage={:.4} -> {}",
            args.rewrite_tau, st.leaders, st.sites, st.coverage, out.display()
        );
    }

    // ── Discrete reachability-closure pins (FOLLOWUP_SPEC FU1) — the by-construction A_k restriction. ──
    // After the fixpoint, emit the *discrete* set of instruction offsets that the rung's evidence
    // reaches by committing recursive descent from the true anchor: seeds = program entry ∪ clamped
    // function heads (E₄/E₅ trace/oracle) ∪ M3a resolved targets (E₃). This is the fixpoint/resolver
    // OUTPUT as a membership fact — not a threshold on q — so the harness can define A_k = A_{k−1} minus
    // the objects this closure determines (reached ⇒ code; unreached decoy candidate ⇒ junk). Pure read:
    // it never touches a belief, so the calibrated posterior above is byte-identical with or without
    // this flag (the honesty wall holds). Machine lines: `pin_reach` (per offset) + `pin_resolved`.
    if args.dump_pins {
        let mut seeds: Vec<u64> = vec![stack.entry()];
        seeds.extend(args.clamp_func.iter().map(|&(a, _)| a));
        let resolved_t = stack.resolved_targets();
        seeds.extend(resolved_t.iter().copied());
        let reached = udstack::recursive_descent(stack.superset(), &seeds);
        for &(a, _) in &phat {
            println!("pin_reach,0x{a:x},{}", u8::from(reached.contains(&a)));
        }
        for t in &resolved_t {
            println!("pin_resolved,0x{t:x}");
        }
        eprintln!(
            "  PINS: {} seeds (entry + {} clamps + {} resolved) -> reached {} of {} dumped offsets",
            seeds.len(), args.clamp_func.len(), resolved_t.len(),
            phat.iter().filter(|&&(a, _)| reached.contains(&a)).count(), phat.len(),
        );
    }
    // Module-mechanism dump for the three-way audit of any function-level discrimination win: per head,
    // its raw F_h, its component id + component confirmation F_c, whether it is a real function
    // (FUNC-GT), and whether it sits in the decoy region. The audit checks that decoy heads land in
    // low-F_c (disconnected) components — i.e. the win is reachability, not a scoring artifact.
    if args.dump_modules && stack.depth() >= 3 {
        // `bel_f` is the CALIBRATED FUSED function marginal (`func_marginals`) — the axis the paper's
        // 0.889→0.925 discrimination number lives on, the one that carries the 4→2 module message. It
        // differs from the raw `f_h` (`confirmation_map`, pre-fusion): the K3 win shows up in `bel_f`,
        // not `f_h`. Dumping it per head lets the harness split the SAME axis by call-graph reachability
        // (disconnected vs self-anchoring) — the whole point of the decoy-heavy re-run.
        println!("stack_headmod,head,f_h,comp,f_c,real,in_decoy,bel_f");
        let fc = stack.component_map();
        let fm: std::collections::HashMap<u64, f64> = stack.func_marginals().into_iter().collect();
        for &h in stack.heads() {
            let f_h = stack.confirmation_map().get(&h).copied().unwrap_or(0.0);
            let comp = stack.component_of(h).unwrap_or(0);
            let f_c = fc.get(&comp).copied().unwrap_or(0.0);
            let bel_f = fm.get(&h).copied().unwrap_or(0.0);
            let real = func_gt.as_ref().map(|g| g.contains(&h)).unwrap_or(false);
            let in_decoy = args.decoy_from.map(|lo| h >= lo).unwrap_or(false);
            println!("stack_headmod,0x{h:x},{f_h:.4},0x{comp:x},{f_c:.4},{},{},{bel_f:.4}",
                u8::from(real), u8::from(in_decoy));
        }
    }
    if let Some(h) = args.report_head {
        let f = stack.confirmation_map().get(&h).copied().unwrap_or(0.0);
        let body = stack.body_of(h).unwrap_or(&[]);
        let (n, hi, sum) = body.iter().fold((0usize, 0usize, 0.0f64), |(n, hi, s), a| {
            let p = pmap.get(a).copied().unwrap_or(0.0);
            (n + 1, hi + usize::from(p >= 0.9), s + p)
        });
        let mean = if n > 0 { sum / n as f64 } else { 0.0 };
        eprintln!("  REPORT-HEAD 0x{h:x}: F_h={f:.4}  body={n}  P̂≥0.9={hi}  meanP̂={mean:.4}");
        println!("stack_report_head,0x{h:x},{f:.4},{n},{hi},{mean:.4}");
    }

    // ── P̂ risk–coverage sweep (precision/recall + decoy leak), like bench. ──
    let leak_hdr = if args.decoy_from.is_some() { ",decoy_leak,mean_conf_decoy" } else { "" };
    println!("stack_phat_tau,n_pred,tp,recall,precision,f1{leak_hdr}");
    for &t in &args.thresholds {
        let pred: Vec<u64> = phat.iter().filter(|&&(_, p)| p >= t).map(|&(a, _)| a).collect();
        let (recall, precision, f1) = score(&pred, &gt);
        let leak = match args.decoy_from {
            Some(lo) => {
                let d: Vec<f64> = phat.iter().filter(|&&(a, p)| a >= lo && p >= t).map(|&(_, p)| p).collect();
                let mc = if d.is_empty() { 0.0 } else { d.iter().sum::<f64>() / d.len() as f64 };
                format!(",{},{:.4}", d.len(), mc)
            }
            None => String::new(),
        };
        println!(
            "{t:.4},{},{},{recall:.4},{precision:.4},{f1:.4}{leak}",
            pred.len(),
            pred.iter().filter(|a| gt.contains(a)).count()
        );
    }
    Ok(())
}

/// Which head to confirm next in the active-analysis loop.
#[derive(Clone, Copy, PartialEq)]
enum Strategy {
    /// Greedy expected-information-gain (design §5): the head whose confirmation is expected to remove
    /// the most instruction-map entropy.
    Eig,
    /// The certified max-current-conditional-entropy rule (LIMITS_HIERARCHY_PROOFS, `(1−1/e)` theorem):
    /// pick `argmax_h H(X_h | E, X_queried) = argmax_h h(F_h)`, i.e. the uncertain head whose current
    /// confirmation `F_h` is nearest `0.5`. Re-ranked internally after each clamp+relax (the `f` map is
    /// re-read every step), so it runs through the SAME faithful loop as `Eig` (FOLLOWUP_SPEC FU2).
    CertEnt,
    /// Naive "confirm what the tool is least sure of": the lowest-`F_h` uncertain head.
    LowF,
    /// The highest-`F_h` uncertain head (already nearly confirmed — should buy little).
    HighF,
    /// Arbitrary order (lowest address) — an uninformed baseline.
    Addr,
}

impl Strategy {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "eig" => Strategy::Eig,
            "certent" => Strategy::CertEnt,
            "lowf" => Strategy::LowF,
            "highf" => Strategy::HighF,
            "addr" => Strategy::Addr,
            o => bail!("--active wants eig|certent|lowf|highf|addr, got {o}"),
        })
    }
    fn tag(self) -> &'static str {
        match self {
            Strategy::Eig => "eig",
            Strategy::CertEnt => "certent",
            Strategy::LowF => "lowf",
            Strategy::HighF => "highf",
            Strategy::Addr => "addr",
        }
    }
}

/// Run the greedy active-analysis loop: `k` sequential confirmations chosen by `strat`, reporting the
/// calibrated instruction marginals and the invariant L1 `π` after each. The uncertain candidate band
/// and body floor match `Stack::rank_queries`.
fn run_active(
    stack: &mut Stack,
    sched: Schedule,
    gt: &HashSet<u64>,
    func_gt: &HashSet<u64>,
    strat: Strategy,
    k: usize,
    q: f64,
    cap: usize,
) -> Result<()> {
    const LO: f64 = 0.05;
    const HI: f64 = 0.95;
    const MIN_BODY: usize = 2;

    // Baseline fixpoint: fits pool + freezes cal. Everything after re-relaxes with those frozen.
    stack.relax(sched, gt);

    // π baseline — captured once; re-checked every step to prove the honesty wall (π never mutated).
    let pi0 = evaluate(&stack.pi_marginals(), gt);

    println!("stack_active,strategy,step,head,real,f_prior,eig,entropy,ece,auroc,pi_ece,pi_auroc,hi_mass,mean_phat,tp_at_0.9,fp_at_0.9");
    let report = |stack: &Stack, step: usize, head: Option<u64>, real: i32, f_prior: f64, eig: f64| {
        let phat = stack.instr_marginals();
        let m = evaluate(&phat, gt);
        let pim = evaluate(&stack.pi_marginals(), gt);
        let (hi, sum) = phat.iter().fold((0usize, 0.0f64), |(hi, s), &(_, p)| (hi + usize::from(p >= 0.9), s + p));
        let mean = if phat.is_empty() { 0.0 } else { sum / phat.len() as f64 };
        let (tp9, fp9) = phat.iter().filter(|&&(_, p)| p >= 0.9).fold((0usize, 0usize), |(t, f), &(a, _)| {
            if gt.contains(&a) { (t + 1, f) } else { (t, f + 1) }
        });
        let auc = |m: &Metrics| m.auroc.map(|a| format!("{a:.4}")).unwrap_or_else(|| "NA".into());
        let head_s = head.map(|h| format!("0x{h:x}")).unwrap_or_else(|| "-".into());
        println!(
            "stack_active,{},{step},{head_s},{real},{f_prior:.4},{eig:.4},{:.4},{:.4},{},{:.4},{},{hi},{mean:.4},{tp9},{fp9}",
            strat.tag(), stack.instr_entropy(), m.ece, auc(&m), pim.ece, auc(&pim)
        );
    };

    report(stack, 0, None, -1, 0.0, 0.0);
    let mut clamped: HashSet<u64> = HashSet::new();
    let mut hits = 0usize; // real heads confirmed (ranking precision)

    for step in 1..=k {
        // Select the next head by strategy over the current uncertain band.
        let pick: Option<(u64, f64, f64)> = match strat {
            Strategy::Eig => stack
                .rank_queries(q, LO, HI, MIN_BODY, 1, sched, &clamped, cap)
                .into_iter()
                .next()
                .map(|qy| (qy.head, qy.f_prior, qy.eig)),
            _ => {
                let f = stack.confirmation_map();
                let mut cands: Vec<(u64, f64)> = stack
                    .heads()
                    .iter()
                    .copied()
                    .filter(|h| !clamped.contains(h))
                    .map(|h| (h, f.get(&h).copied().unwrap_or(0.0)))
                    .filter(|&(h, fv)| {
                        fv >= LO && fv <= HI && stack.body_of(h).map(|b| b.len()).unwrap_or(0) >= MIN_BODY
                    })
                    .collect();
                match strat {
                    // Certified rule: nearest F_h to 0.5 = max conditional entropy h(F_h). `f` is the
                    // CURRENT (post-relax) confirmation map, so this re-ranks internally each step.
                    Strategy::CertEnt => cands.sort_by(|a, b| {
                        (a.1 - 0.5).abs().partial_cmp(&(b.1 - 0.5).abs()).unwrap()
                    }),
                    Strategy::LowF => cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap()),
                    Strategy::HighF => cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()),
                    Strategy::Addr => cands.sort_by_key(|&(h, _)| h),
                    Strategy::Eig => unreachable!(),
                }
                cands.first().map(|&(h, fv)| (h, fv, 0.0))
            }
        };
        let Some((h, f_prior, eig)) = pick else {
            eprintln!("  [{}] step {step}: no uncertain candidate left — stopping.", strat.tag());
            break;
        };
        clamped.insert(h);
        // Oracle-truthful application: the query returns ground truth. Confirm a *real* head (inject
        // q); a *decoy* head is denied — the analyst learns it is junk and injects NO positive evidence
        // (this mechanism is confirm-only, so a wasted query leaves the marginals ≈ unchanged). This
        // keeps the demo honest: a strategy that ranks decoys first simply buys nothing.
        let real = func_gt.contains(&h);
        if real {
            hits += 1;
            stack.clamp(ObjId::func(h), q);
            stack.relax(sched, gt);
        }
        report(stack, step, Some(h), i32::from(real), f_prior, eig);
    }

    // Honesty-wall check: π after all clamps must equal the π baseline (bit-for-bit).
    let pi1 = evaluate(&stack.pi_marginals(), gt);
    let pi_moved = (pi1.ece - pi0.ece).abs() > 1e-12
        || pi1.auroc.unwrap_or(0.0) != pi0.auroc.unwrap_or(0.0);
    eprintln!(
        "  ACTIVE[{}]: {k} queries, q={q}; π invariant = {} (ECE {:.4}→{:.4}, AUROC {:?}→{:?})",
        strat.tag(),
        if pi_moved { "VIOLATED" } else { "held" },
        pi0.ece, pi1.ece, pi0.auroc, pi1.auroc
    );
    let cf = evaluate(&stack.func_marginals(), func_gt);
    eprintln!(
        "  ACTIVE[{}]: ranking precision {hits}/{k} real; final F_h [ECE {:.4} AUROC {:?}]",
        strat.tag(), cf.ece, cf.auroc
    );
    Ok(())
}

/// Arm B — the incremental workflow (INTERACTIVE_APP_SPEC §2). An evidence stream of three kinds —
/// `sym` (a withheld function symbol → `clamp(Func)`), `trace` (a dynamic-trace instruction hit →
/// `clamp(Instr)`), `edge` (a resolved indirect edge → `clamp(Func)`, the M3a mechanism) — enters one
/// item at a time. After each, we relax with the FROZEN operators and report:
///   * the stack's calibrated instruction map on the **held-out** domain (all instructions except the
///     ones a trace hit pinned directly — so trace evidence is never self-scored): AUROC, coverage
///     (true code recovered at P̂≥0.9), ECE;
///   * a committing recursive-descent baseline seeded by `entry ∪ confirmed-so-far`, scored on the same
///     domain with hard 0/1 labels: AUROC = (TPR+TNR)/2, coverage = recall, ECE of the hard labels;
///   * the invariant π (AUROC/ECE) — constant ⇒ the honesty wall holds under evidence.
/// The claim: the stack's quality rises while ECE stays bounded (calibration maintained at every step),
/// where the committing baseline can only flip hard decisions and stays miscalibrated.
fn run_incremental(
    stack: &mut Stack,
    sched: Schedule,
    gt: &HashSet<u64>,
    func_gt: &HashSet<u64>,
    counts: (usize, usize, usize),
    q: f64,
) -> Result<()> {
    let (n_sym, n_trace, n_edge) = counts;

    // Baseline fixpoint: fit pool + freeze cal. Everything after re-relaxes frozen.
    stack.relax(sched, gt);
    let f0 = stack.confirmation_map().clone();

    // Build the three evidence streams from GT-by-construction facts.
    // sym / edge: real function heads that are currently UNCERTAIN (a query on an already-confirmed head
    // teaches nothing). Split into two disjoint pools by parity for sym vs edge, address order.
    let mut uncertain_heads: Vec<u64> = stack
        .heads()
        .iter()
        .copied()
        .filter(|h| func_gt.contains(h) && {
            let f = f0.get(h).copied().unwrap_or(0.0);
            f > 0.05 && f < 0.95 && stack.body_of(*h).map(|b| b.len()).unwrap_or(0) >= 2
        })
        .collect();
    uncertain_heads.sort_unstable();
    let syms: Vec<u64> = uncertain_heads.iter().step_by(1).copied().take(n_sym).collect();
    // edges: uncertain real heads NOT used as syms (take from the tail).
    let edges: Vec<u64> = uncertain_heads.iter().rev().filter(|h| !syms.contains(h)).copied().take(n_edge).collect();

    // trace hits: real instruction starts the posterior is currently UNSURE about (π < 0.5) — a trace
    // teaches the most there. These are pinned directly, so they are EXCLUDED from the scoring domain.
    let pi = stack.pi_marginals();
    let mut trace_cands: Vec<u64> = pi.iter().filter(|&&(a, p)| p < 0.5 && gt.contains(&a)).map(|&(a, _)| a).collect();
    trace_cands.sort_unstable();
    let traces: Vec<u64> = trace_cands.into_iter().take(n_trace).collect();

    // Round-robin interleave so the arms are mixed, as a real evidence feed would be.
    let mut stream: Vec<(&str, u64)> = Vec::new();
    let (mut i, mut j, mut k) = (0usize, 0usize, 0usize);
    while i < syms.len() || j < traces.len() || k < edges.len() {
        if i < syms.len() {
            stream.push(("sym", syms[i]));
            i += 1;
        }
        if j < traces.len() {
            stream.push(("trace", traces[j]));
            j += 1;
        }
        if k < edges.len() {
            stream.push(("edge", edges[k]));
            k += 1;
        }
    }

    // Scoring domain = every instruction EXCEPT the trace-pinned ones (held-out ⇒ trace is honest).
    let trace_set: HashSet<u64> = traces.iter().copied().collect();
    let eval_domain: Vec<u64> = pi.iter().map(|&(a, _)| a).filter(|a| !trace_set.contains(a)).collect();
    let pos_total = eval_domain.iter().filter(|a| gt.contains(a)).count();

    // Committing baseline seeds: entry + everything confirmed so far.
    let mut seeds: Vec<u64> = vec![stack.entry()];

    println!("stack_incr,step,kind,n_ev,st_auroc,st_cov,st_ece,base_auroc,base_cov,base_ece,pi_auroc,pi_ece");
    let emit = |stack: &Stack, step: usize, kind: &str, n_ev: usize, seeds: &[u64]| {
        // Stack calibrated marginals on the held-out domain.
        let phat: Vec<(u64, f64)> = stack
            .instr_marginals()
            .into_iter()
            .filter(|(a, _)| !trace_set.contains(a))
            .collect();
        let m = evaluate(&phat, gt);
        let tp9 = phat.iter().filter(|&&(a, p)| p >= 0.9 && gt.contains(&a)).count();
        let st_cov = if pos_total > 0 { tp9 as f64 / pos_total as f64 } else { 0.0 };

        // Committing recursive-descent baseline on the same held-out domain.
        let pred = udstack::recursive_descent(stack.superset(), seeds);
        let (mut tp, mut fp, mut tn, mut fnn) = (0usize, 0usize, 0usize, 0usize);
        for &a in &eval_domain {
            let is_code = gt.contains(&a);
            match (pred.contains(&a), is_code) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fnn += 1,
                (false, false) => tn += 1,
            }
        }
        let tpr = if tp + fnn > 0 { tp as f64 / (tp + fnn) as f64 } else { 0.0 };
        let tnr = if tn + fp > 0 { tn as f64 / (tn + fp) as f64 } else { 0.0 };
        let base_auroc = 0.5 * (tpr + tnr); // AUROC of a hard {0,1} predictor
        let base_cov = tpr; // recall of true code
        // ECE of the hard detector (p ∈ {0,1}): bin at 1 → |1 − precision|; bin at 0 → |0 − P(code|0)|.
        let n = eval_domain.len() as f64;
        let (n1, n0) = ((tp + fp) as f64, (tn + fnn) as f64);
        let prec = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let code_given0 = if tn + fnn > 0 { fnn as f64 / (tn + fnn) as f64 } else { 0.0 };
        let base_ece = if n > 0.0 { n1 / n * (1.0 - prec).abs() + n0 / n * code_given0 } else { 0.0 };

        // π on the held-out domain (honesty wall — must be constant across steps).
        let pim_v: Vec<(u64, f64)> = stack.pi_marginals().into_iter().filter(|(a, _)| !trace_set.contains(a)).collect();
        let pim = evaluate(&pim_v, gt);
        let auc = |x: Option<f64>| x.map(|v| format!("{v:.4}")).unwrap_or_else(|| "NA".into());
        println!(
            "stack_incr,{step},{kind},{n_ev},{},{st_cov:.4},{:.4},{base_auroc:.4},{base_cov:.4},{base_ece:.4},{},{:.4}",
            auc(m.auroc), m.ece, auc(pim.auroc), pim.ece
        );
    };

    emit(stack, 0, "-", 0, &seeds);
    for (step, &(kind, addr)) in stream.iter().enumerate() {
        match kind {
            "trace" => stack.clamp(ObjId::instr(addr), q),
            _ => stack.clamp(ObjId::func(addr), q), // sym + edge both enter at the function/edge layer
        }
        seeds.push(addr);
        stack.relax(sched, gt);
        emit(stack, step + 1, kind, step + 1, &seeds);
    }

    // Honesty wall: π on the held-out domain unchanged end-to-end.
    let pi_end: Vec<(u64, f64)> = stack.pi_marginals().into_iter().filter(|(a, _)| !trace_set.contains(a)).collect();
    let base_pi: Vec<(u64, f64)> = pi.iter().copied().filter(|(a, _)| !trace_set.contains(a)).collect();
    let linf = base_pi.iter().zip(&pi_end).map(|(&(_, a), &(_, b))| (a - b).abs()).fold(0.0f64, f64::max);
    eprintln!(
        "  INCREMENTAL: {} evidence items ({} sym / {} trace / {} edge); ‖π_end − π_0‖∞ = {:.2e} (honesty wall {})",
        stream.len(), syms.len(), traces.len(), edges.len(), linf,
        if linf < 1e-12 { "held" } else { "VIOLATED" }
    );
    Ok(())
}

fn score(pred: &[u64], gt: &HashSet<u64>) -> (f64, f64, f64) {
    let tp = pred.iter().filter(|&&a| gt.contains(&a)).count() as f64;
    let recall = tp / gt.len() as f64;
    let precision = if pred.is_empty() { 0.0 } else { tp / pred.len() as f64 };
    let f1 = if recall + precision > 0.0 { 2.0 * recall * precision / (recall + precision) } else { 0.0 };
    (recall, precision, f1)
}

#[derive(Clone, Copy)]
enum Milestone {
    A,
    B,
}

struct Args {
    binary: PathBuf,
    gt: PathBuf,
    func_gt: Option<PathBuf>,
    resolve_elf: Option<PathBuf>,
    entropy: f64,
    dassa: bool,
    max_fn_span: usize,
    milestone: Milestone,
    layers: usize,
    lambda: f64,
    eps: Option<f64>,
    max_sweeps: Option<usize>,
    trace: bool,
    decoy_from: Option<u64>,
    clamp_func: Vec<(u64, f64)>,
    fit_elf: Option<PathBuf>,
    fit_gt: Option<PathBuf>,
    thresholds: Vec<f64>,
    dump_heads: bool,
    dump_modules: bool,
    dump_instr: bool,
    dump_pins: bool,
    report_head: Option<u64>,
    active: Option<(Strategy, usize)>,
    query_q: f64,
    query_cap: usize,
    incremental: Option<(usize, usize, usize)>,
    // First-class rewrite hook: emit a confidence-gated rewrite of the analysed binary directly
    // from the calibrated marginals, no `--dump-instr` round-trip. `--rewrite-tau 0` = commit-all.
    rewrite_out: Option<PathBuf>,
    rewrite_tau: f64,
}

impl Args {
    fn parse(mut it: impl Iterator<Item = String>) -> Result<Self> {
        const USAGE: &str = "usage: udstack <binary> <gt> [--func-gt P] [--resolve-elf DATA_ELF] \
             [--milestone a|b] [--layers 2|3] [--lambda F] [--decoy-from ADDR] [--clamp-func ADDR[:q]] \
             [--fit-elf E --fit-gt G] [--entropy S] [--dassa] [--max-fn-span N] [--thresholds a,b,..]";
        let mut positional = Vec::new();
        let mut func_gt = None;
        let mut resolve_elf = None;
        let mut entropy = 0.0;
        let mut dassa = false;
        let mut max_fn_span = 65536usize;
        let mut milestone = Milestone::A;
        let mut layers = 2usize;
        let mut lambda = 0.5;
        let mut eps: Option<f64> = None;
        let mut max_sweeps: Option<usize> = None;
        let mut trace = false;
        let mut decoy_from = None;
        let mut clamp_func = Vec::new();
        let mut fit_elf = None;
        let mut fit_gt = None;
        let mut thresholds: Vec<f64> = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let mut dump_heads = false;
        let mut dump_modules = false;
        let mut dump_instr = false;
        let mut dump_pins = false;
        let mut rewrite_out: Option<PathBuf> = None;
        let mut rewrite_tau: f64 = 0.95;
        let mut report_head: Option<u64> = None;
        let mut active: Option<(Strategy, usize)> = None;
        let mut query_q = 0.99f64;
        let mut query_cap = 16usize; // EIG exact-eval shortlist size (0 = score every band candidate)
        let mut incremental: Option<(usize, usize, usize)> = None;

        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--func-gt" => func_gt = Some(PathBuf::from(it.next().context("--func-gt path")?)),
                "--resolve-elf" => resolve_elf = Some(PathBuf::from(it.next().context("--resolve-elf path")?)),
                "--entropy" => entropy = it.next().context("--entropy value")?.parse().context("--entropy float")?,
                "--dassa" => dassa = true,
                "--max-fn-span" => max_fn_span = it.next().context("--max-fn-span value")?.parse().context("usize")?,
                "--milestone" => {
                    milestone = match it.next().context("--milestone a|b")?.as_str() {
                        "a" | "A" => Milestone::A,
                        "b" | "B" => Milestone::B,
                        o => bail!("--milestone wants a|b, got {o}"),
                    }
                }
                "--layers" => {
                    layers = it.next().context("--layers 2|3")?.parse().context("--layers usize")?;
                    if !(2..=3).contains(&layers) {
                        bail!("--layers wants 2 or 3, got {layers}");
                    }
                }
                "--lambda" => lambda = it.next().context("--lambda value")?.parse().context("--lambda float")?,
                "--eps" => eps = Some(it.next().context("--eps value")?.parse().context("--eps float")?),
                "--max-sweeps" => max_sweeps = Some(it.next().context("--max-sweeps value")?.parse().context("--max-sweeps usize")?),
                "--trace" => trace = true,
                "--decoy-from" => {
                    let v = it.next().context("--decoy-from value")?;
                    let v = v.strip_prefix("0x").map(|h| u64::from_str_radix(h, 16)).unwrap_or_else(|| v.parse());
                    decoy_from = Some(v.context("--decoy-from u64")?);
                }
                "--clamp-func" => {
                    let v = it.next().context("--clamp-func ADDR[:q]")?;
                    let (a, q) = match v.split_once(':') {
                        Some((a, q)) => (a, q.parse::<f64>().context("clamp q float")?),
                        None => (v.as_str(), 0.99),
                    };
                    let addr = a.strip_prefix("0x").map(|h| u64::from_str_radix(h, 16)).unwrap_or_else(|| a.parse());
                    clamp_func.push((addr.context("--clamp-func addr")?, q));
                }
                "--fit-elf" => fit_elf = Some(PathBuf::from(it.next().context("--fit-elf path")?)),
                "--fit-gt" => fit_gt = Some(PathBuf::from(it.next().context("--fit-gt path")?)),
                "--active" => {
                    let v = it.next().context("--active STRAT:K")?;
                    let (s, k) = v.split_once(':').context("--active wants STRAT:K")?;
                    active = Some((Strategy::parse(s)?, k.parse::<usize>().context("--active K usize")?));
                }
                "--query-q" => query_q = it.next().context("--query-q value")?.parse().context("--query-q float")?,
                "--query-cap" => query_cap = it.next().context("--query-cap value")?.parse().context("--query-cap usize")?,
                "--incremental" => {
                    let v = it.next().context("--incremental SYM,TRACE,EDGE")?;
                    let parts: Vec<usize> = v.split(',').map(|s| s.trim().parse::<usize>()).collect::<Result<_, _>>().context("--incremental wants SYM,TRACE,EDGE")?;
                    let [s, t, e] = parts.as_slice() else { bail!("--incremental wants three counts SYM,TRACE,EDGE") };
                    incremental = Some((*s, *t, *e));
                }
                "--dump-heads" => dump_heads = true,
                "--dump-modules" => dump_modules = true,
                "--dump-instr" => dump_instr = true,
                "--dump-pins" => dump_pins = true,
                "--rewrite" => {
                    rewrite_out = Some(PathBuf::from(it.next().context("--rewrite OUT.elf")?));
                }
                "--rewrite-tau" => {
                    rewrite_tau = it.next().context("--rewrite-tau F")?.parse().context("--rewrite-tau F")?;
                }
                "--report-head" => {
                    let v = it.next().context("--report-head ADDR")?;
                    let v = v.strip_prefix("0x").map(|h| u64::from_str_radix(h, 16)).unwrap_or_else(|| v.parse());
                    report_head = Some(v.context("--report-head u64")?);
                }
                "--thresholds" => {
                    thresholds = it
                        .next()
                        .context("--thresholds value")?
                        .split(',')
                        .map(|s| s.trim().parse::<f64>().context("threshold float"))
                        .collect::<Result<_>>()?
                }
                "-h" | "--help" => {
                    eprintln!("{USAGE}");
                    std::process::exit(0);
                }
                o if o.starts_with('-') => bail!("unexpected flag: {o}"),
                o => positional.push(PathBuf::from(o)),
            }
        }
        let [binary, gt] = positional.as_slice() else { bail!("{USAGE}") };
        Ok(Args {
            binary: binary.clone(),
            gt: gt.clone(),
            func_gt,
            resolve_elf,
            entropy,
            dassa,
            max_fn_span,
            milestone,
            layers,
            lambda,
            eps,
            max_sweeps,
            trace,
            decoy_from,
            clamp_func,
            fit_elf,
            fit_gt,
            thresholds,
            dump_heads,
            dump_modules,
            dump_instr,
            dump_pins,
            report_head,
            active,
            query_q,
            query_cap,
            incremental,
            rewrite_out,
            rewrite_tau,
        })
    }
}
