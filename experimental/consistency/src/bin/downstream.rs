//! `downstream` — a real analysis task, corrupted by stale calibration and recovered by the switch.
//!
//! The switching probe reported ECE, and an earlier cut of this binary reported an accept-at-τ
//! precision on instruction starts. Both are still numbers *about the calibration*. A reviewer is
//! entitled to ask what breaks downstream when the number moves, and neither answers that. This
//! binary answers it on a task that exists outside this project: **function-boundary recovery**.
//! Every stripped-binary tool starts there — you cannot name a function, diff two builds, sign a
//! CFG, or hand anything to a decompiler until you know where the functions begin.
//!
//! The task. Recover the set of function head addresses in a stripped `.text`. The rule is the one
//! real tools bootstrap from, and the one the engine already believes internally (`analysis.rs`
//! treats a call target as a procedure boundary and refuses to run a fall-through factor across it):
//! an address is a function head if something confidently *calls* it directly. "Confidently" is
//! where calibration enters, and it is the whole point — both the call site and its target have to
//! clear τ under the calibrated posterior, so a stale map does not merely misreport a confidence, it
//! moves the recovered boundary set. That is a corruption you can see without reading a metric.
//!
//! Three arms over the same held-out split and the same bank as the switching probe:
//!   (a) stale  — always-benign map, the naive deployment. The corruption.
//!   (b) switch — the GT-free consistency switch, signature-selected, no ground truth at test time.
//!                Reported twice: the bare threshold rule and the abstention-guarded rule (shipped).
//!   (c) oracle — true regime known, correct map applied. The ceiling.
//! For each arm × regime × τ we grade the recovered head set and report boundary precision, recall
//! and F1. The claim we are testing, stated so it can fail: under the stale map on obfuscated input
//! F1 degrades, the detector fires against the clean-fit null, and the switch recovers F1 toward the
//! oracle.
//!
//! **Ground truth is never a disassembler.** Function heads come from the unstripped original's
//! `.symtab` `STT_FUNC` entries — real symbol-table rows — materialized by `gen_boundary_gt.py`,
//! which hard-gates on the stripped/unstripped `.text` vma+size matching so the symbols are known to
//! describe the binary the engine actually reads. `gen-gt`'s `fn_min`/`fn_max` are the other
//! sanctioned source and follow the identical rule (DWARF `low_pc` ∪ `STT_FUNC`); they have simply
//! never been run over these corpora. No disassembly, ours or anyone's, is consulted for truth.
//!
//! Two honesty notes that shape how the table reads, both structural and neither hidden in a
//! footnote:
//!
//! 1. **Packed has no positive boundary truth.** The packed GT is UPX's own `b_info` chain, which
//!    proves a window is *compressed data* — it yields negatives and never positives, and the
//!    original heads do not survive compression as addresses. So on packed we report how many heads
//!    an arm invented inside the provable-data window (every one is false by construction) and leave
//!    recall and F1 empty. We do not manufacture packed positives to fill a cell.
//!
//! 2. **The predictor has a ceiling below 1.0 by construction.** A direct-call rule cannot recover a
//!    head that nothing directly calls — `_start`, indirect-only callees, unused statics. That is a
//!    property of the task rule, it is identical across all four arms, and so it cannot manufacture
//!    the arm *differences* this probe is about. We report absolute F1 against the full symtab head
//!    set anyway, and carry the reach-restricted F1 (heads that are the target of some direct call
//!    in the superset, gate or no gate) beside it so the ceiling is visible rather than argued.
//!
//! Bank, classifier, split and engine settings all come from `consistency::*`, the same machinery
//! `bin/switching.rs` runs. The `Meta` row is unchanged from the previous cut on purpose: it mirrors
//! `switching`'s columns exactly so `verify_ab.sh` can diff the two and catch shared-module drift.
//!
//! ```text
//! downstream \
//!   --clean-bins DIR --clean-gt DIR --func-gt-root DIR \
//!   --desync-level LABEL BINS GT ...  --packed-spec LABEL ELF UPXGT ... \
//!   [--tigress-level LABEL BINS GT]... [--benign-holdout LABEL BINS GT]... \
//!   --n-clean-fit N --n-clean-holdout N --n-desync-fit N --n-desync-holdout N \
//!   --n-packed-fit N --n-packed-holdout N --n-tig-holdout N \
//!   [--tau 0.5 --tau 0.7 --tau 0.9] [--entropy-strength 1.0] [--chainfwd-strength 0.5] [--seed 1] \
//!   --out boundaries.csv --meta meta.csv --summary summary.json
//! ```

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use consistency::{
    build_jobs, fit_bank, global_and_spatial, packed_data_window, region_entropy, Bank, CorpusSpec,
    HoldoutJob, Regime, SignatureClassifier,
};
use evalkit::{evaluate, load_gt, run_soft_with_cavity_cfg};
use probdisasm::{extract_text_section as extract_text, Superset};

/// The arms the analyst could be running under. `Stale` is what ships today; `Oracle` is the
/// unattainable ceiling; the two switch arms are the GT-free thing we actually propose.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    Stale,
    SwitchRule,
    SwitchGuard,
    Oracle,
}

impl Arm {
    fn tag(self) -> &'static str {
        match self {
            Arm::Stale => "stale",
            Arm::SwitchRule => "switch_rule",
            Arm::SwitchGuard => "switch_guard",
            Arm::Oracle => "oracle",
        }
    }
    const ALL: [Arm; 4] = [Arm::Stale, Arm::SwitchRule, Arm::SwitchGuard, Arm::Oracle];
}

/// One (binary, arm, τ) function-boundary recovery, graded against the symtab head set.
#[derive(Clone, Debug)]
struct Recovery {
    name: String,
    regime: Regime,
    sublabel: String,
    arm: Arm,
    /// Which regime's config this arm applied (for the switch arms: what the GT-free rule picked).
    pick: Regime,
    tau: f64,
    /// True heads from `.symtab`, restricted to `.text`. The recall denominator.
    n_gt: usize,
    /// True heads that *are* the target of some direct call in the superset, gate ignored — the
    /// structural ceiling of this predictor, identical across arms.
    n_reach: usize,
    n_pred: usize,
    tp: usize,
    fp: usize,
    fn_: usize,
    /// Packed only: heads this arm invented inside UPX's provable-data window, and how many call
    /// targets land there at all. Every invented head there is false by construction.
    n_window: usize,
    win_pred: usize,
}

impl Recovery {
    const CSV_HEADER: &'static str =
        "name,regime,sublabel,arm,pick,tau,n_gt,n_reach,n_pred,tp,fp,fn,n_window,win_pred";

    fn to_csv(&self) -> String {
        format!(
            "{},{},{},{},{},{:.2},{},{},{},{},{},{},{},{}",
            self.name,
            self.regime.tag(),
            self.sublabel,
            self.arm.tag(),
            self.pick.tag(),
            self.tau,
            self.n_gt,
            self.n_reach,
            self.n_pred,
            self.tp,
            self.fp,
            self.fn_,
            self.n_window,
            self.win_pred,
        )
    }
}

/// Per-binary bookkeeping that mirrors `switching`'s CSV columns exactly. This is the drift guard:
/// `verify_ab.sh` diffs these against a `switching` run over the same tiny corpus, so if the shared
/// module ever diverges from `bin/switching.rs` the check fails loudly. Deliberately unchanged by the
/// boundary-task rewrite — the task moved, the calibration bookkeeping did not.
struct Meta {
    name: String,
    regime: Regime,
    sublabel: String,
    n: usize,
    code_bytes: usize,
    base_rate: f64,
    ece_always_benign: f64,
    ece_oracle: f64,
    region_ent: f64,
    mmae_pick: Regime,
    mmae_nis_pick: Regime,
    clf_pick: Regime,
    rule_pick: Regime,
    guard_pick: Regime,
    s_glob_benign_eng: f64,
    s_spat_benign_eng: f64,
    s_glob_packed_eng: f64,
    s_glob_obf_eng: f64,
    nis_benign_eng: f64,
    nis_packed_eng: f64,
    nis_obf_eng: f64,
}

impl Meta {
    const CSV_HEADER: &'static str = "name,regime,sublabel,n,code_bytes,base_rate,\
ece_always_benign,ece_oracle,region_ent,\
mmae_pick,mmae_nis_pick,clf_pick,rule_pick,guard_pick,\
s_glob_benign_eng,s_spat_benign_eng,s_glob_packed_eng,s_glob_obf_eng,\
nis_benign_eng,nis_packed_eng,nis_obf_eng";

    fn to_csv(&self) -> String {
        format!(
            "{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            self.name,
            self.regime.tag(),
            self.sublabel,
            self.n,
            self.code_bytes,
            self.base_rate,
            self.ece_always_benign,
            self.ece_oracle,
            self.region_ent,
            self.mmae_pick.tag(),
            self.mmae_nis_pick.tag(),
            self.clf_pick.tag(),
            self.rule_pick.tag(),
            self.guard_pick.tag(),
            self.s_glob_benign_eng,
            self.s_spat_benign_eng,
            self.s_glob_packed_eng,
            self.s_glob_obf_eng,
            self.nis_benign_eng,
            self.nis_packed_eng,
            self.nis_obf_eng,
        )
    }
}

fn main() -> Result<()> {
    let args = Args::parse(std::env::args().skip(1))?;

    let (fit_jobs, hold_jobs) = build_jobs(&args.spec)?;
    eprintln!(
        "fit: {} benign / {} obf / {} packed   held-out: {}   taus: {:?}",
        fit_jobs.iter().filter(|j| j.regime == Regime::Benign).count(),
        fit_jobs.iter().filter(|j| j.regime == Regime::Obfuscated).count(),
        fit_jobs.iter().filter(|j| j.regime == Regime::Packed).count(),
        hold_jobs.len(),
        args.taus,
    );

    // ── Pass 1: the same bank and the same classifier the switching probe uses ──
    let bank = fit_bank(&fit_jobs, args.entropy_strength, args.chainfwd_strength)?;
    let clf = SignatureClassifier::train(&fit_jobs, args.entropy_strength, args.chainfwd_strength)?;
    eprintln!(
        "clean-fit null: S_glob_hi={:.4}  S_spat_hi={:.4}  pack_ent_lo={:.4}",
        clf.glob_hi, clf.spat_hi, clf.pack_ent_lo,
    );

    // ── Pass 2: recover function boundaries, per held-out binary, four arms ──
    eprintln!("── pass 2: function-boundary recovery, four arms ──");
    // Resume on the meta file: it is written only after that binary's recovery rows are flushed, so
    // a name present there is a binary whose rows are all on disk.
    let done: HashSet<String> = read_done_keys(&args.meta);
    let mut rec_csv = open_csv_append(&args.out, Recovery::CSV_HEADER)?;
    let mut meta_csv = open_csv_append(&args.meta, Meta::CSV_HEADER)?;
    let mut recoveries: Vec<Recovery> = read_existing_recoveries(&args.out);
    let mut metas: Vec<Meta> = Vec::new();

    for job in &hold_jobs {
        let key = format!("{}|{}", job.name, job.sublabel);
        if done.contains(&key) {
            eprintln!("  resume {} [{}] (from CSV)", job.name, job.sublabel);
            continue;
        }
        let (rows, meta) = evaluate_holdout(job, &bank, &clf, &args)?;
        for r in &rows {
            writeln!(rec_csv, "{}", r.to_csv())?;
        }
        rec_csv.flush()?;
        writeln!(meta_csv, "{}", meta.to_csv())?;
        meta_csv.flush()?;
        // A one-line trace at the headline threshold, so a long run is auditable as it goes.
        let hi = args.taus.last().copied().unwrap_or(0.9);
        let at = |arm: Arm| rows.iter().find(|r| r.arm == arm && (r.tau - hi).abs() < 1e-9);
        if let (Some(s), Some(g), Some(o)) = (at(Arm::Stale), at(Arm::SwitchGuard), at(Arm::Oracle)) {
            eprintln!(
                "  {} [{}/{}] τ={hi}: F1 stale={} switch={} oracle={}  (pick {} → {})",
                job.name, job.regime.tag(), job.sublabel,
                fmt_f1(s), fmt_f1(g), fmt_f1(o),
                meta.guard_pick.tag(), job.regime.tag(),
            );
        }
        recoveries.extend(rows);
        metas.push(meta);
    }

    let summary = summarize(&recoveries, &metas, &clf, &args.taus);
    summary.print();
    fs::write(&args.summary, summary.to_json())
        .with_context(|| format!("writing {}", args.summary.display()))?;
    eprintln!("wrote {}, {} and {}", args.out.display(), args.meta.display(), args.summary.display());
    Ok(())
}

fn fmt_f1(r: &Recovery) -> String {
    match f1(r) {
        Some(v) => format!("{v:.3}"),
        None => "n/a".into(),
    }
}

/// Boundary precision: of the heads this arm recovered, the fraction that are real symtab heads.
/// `None` when the arm recovered nothing — an empty head set has no precision, and folding it in as
/// 0 (or as 1) would be a silent lie either way.
fn precision(r: &Recovery) -> Option<f64> {
    let denom = r.tp + r.fp;
    if denom == 0 { None } else { Some(r.tp as f64 / denom as f64) }
}

/// Boundary recall over the full symtab head set. `None` on packed, where the GT proves data and
/// never code, so there are no known heads to have missed.
fn recall(r: &Recovery) -> Option<f64> {
    if r.n_gt == 0 { None } else { Some(r.tp as f64 / r.n_gt as f64) }
}

fn harmonic(p: Option<f64>, r: Option<f64>) -> Option<f64> {
    match (p, r) {
        (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    }
}

fn f1(r: &Recovery) -> Option<f64> {
    harmonic(precision(r), recall(r))
}

// ── The four-arm recovery for one binary ───────────────────────────────────────

fn evaluate_holdout(
    job: &HoldoutJob,
    bank: &Bank,
    clf: &SignatureClassifier,
    args: &Args,
) -> Result<(Vec<Recovery>, Meta)> {
    let bytes = fs::read(&job.bin).with_context(|| format!("reading {}", job.bin.display()))?;
    let (base, code) = extract_text(&bytes)?;

    // Run each config's engine; the three runs are sequential and each converged graph is dropped
    // before the next, so peak memory stays one binary's factor graph.
    let mut posts: [Vec<(u64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut s_glob = [0.0f64; 3];
    let mut s_spat = [0.0f64; 3];
    let mut nis = [0.0f64; 3];
    for r in Regime::ALL {
        let (ent, cfw) = r.engine(args.entropy_strength, args.chainfwd_strength);
        let (post, cav) = run_soft_with_cavity_cfg(base, code, ent, cfw, false)
            .with_context(|| format!("engine[{}] on {}", r.tag(), job.name))?;
        let s = global_and_spatial(&cav);
        s_glob[r.idx()] = s.mean_surprise;
        s_spat[r.idx()] = s.moran;
        nis[r.idx()] = s.mean_nis;
        posts[r.idx()] = post;
    }

    // GT-free selections, read off the benign-engine signature only.
    let sig_glob = s_glob[Regime::Benign.idx()];
    let sig_spat = s_spat[Regime::Benign.idx()];
    let region_ent = region_entropy(code);
    let mmae_pick = argmin_regime(&s_glob);
    let mmae_nis_pick = argmin_regime(&nis);
    let clf_pick = clf.classify(sig_glob, sig_spat);
    let rule_pick = clf.classify_rule(sig_glob, sig_spat);
    let guard_pick = clf.classify_guard(sig_glob, sig_spat, region_ent);

    let arm_pick = |arm: Arm| match arm {
        Arm::Stale => Regime::Benign,
        Arm::SwitchRule => rule_pick,
        Arm::SwitchGuard => guard_pick,
        Arm::Oracle => job.regime,
    };

    // The call graph of the candidate space. Built once and shared by every arm and every τ: it is a
    // pure capstone decode of the bytes, so it carries no calibration and cannot itself differ
    // between arms. What differs between arms is only which of these call edges clear the gate.
    let sup = Superset::new(base, code).map_err(|e| anyhow::anyhow!("superset on {}: {e}", job.name))?;
    let lo_hi = (base, base + code.len() as u64);
    let call_edges: Vec<(u64, u64)> = sup
        .iter_valid()
        .filter(|i| i.is_call())
        .filter_map(|i| i.branch_target.map(|t| (i.address, t)))
        .filter(|&(_, t)| t >= lo_hi.0 && t < lo_hi.1)
        .collect();
    let reachable: BTreeSet<u64> = call_edges.iter().map(|&(_, t)| t).collect();
    drop(sup);

    // Ground truth for this binary. Never a disassembly of the input — `.symtab` FUNC heads for
    // benign+obfuscated, UPX's own b_info window for packed.
    let head_gt = match job.regime {
        Regime::Packed => None,
        _ => Some(load_boundary_gt(&args.func_gt_root, &job.sublabel, &job.name, lo_hi)?),
    };
    let insn_gt = match job.regime {
        Regime::Packed => None,
        _ => Some(load_gt(job.gt.as_ref().unwrap())?),
    };
    let window = match job.regime {
        Regime::Packed => Some(packed_data_window(job.packed_gt.as_ref().unwrap())?),
        _ => None,
    };
    let n_reach = head_gt
        .as_ref()
        .map(|g| reachable.iter().filter(|t| g.contains(t)).count())
        .unwrap_or(0);
    // Packed denominator: call targets landing inside the provable-data window at all, gate ignored.
    // Against this, `win_pred` reads as "how much of the rubble did this arm mistake for functions".
    let n_window = match window {
        Some((lo, hi)) => reachable.iter().filter(|&&t| t >= lo && t < hi).count(),
        None => 0,
    };

    let mut rows = Vec::new();
    for arm in Arm::ALL {
        let pick = arm_pick(arm);
        let cal = bank.map(pick).apply_all(&posts[pick.idx()]);
        let conf: HashMap<u64, f64> = cal.iter().copied().collect();
        for &tau in &args.taus {
            let pred = recover_boundaries(&call_edges, &conf, tau);
            rows.push(grade_recovery(job, arm, pick, tau, &pred, head_gt.as_ref(), n_reach, n_window, window));
        }
    }

    // Meta row: the switching-probe columns, so the A/B drift check has something to diff.
    let (base_rate, n) = match job.regime {
        Regime::Packed => (0.0, posts[Regime::Benign.idx()].len()),
        _ => {
            let g = insn_gt.as_ref().unwrap();
            let bp = &posts[Regime::Benign.idx()];
            let br = bp.iter().filter(|&&(a, _)| g.contains(&a)).count() as f64 / bp.len().max(1) as f64;
            (br, bp.len())
        }
    };
    let ece = |pick: Regime| -> f64 {
        let cal = bank.map(pick).apply_all(&posts[pick.idx()]);
        match window {
            Some((lo, hi)) => {
                let ps: Vec<f64> =
                    cal.iter().filter(|&&(a, _)| a >= lo && a < hi).map(|&(_, p)| p).collect();
                if ps.is_empty() { 0.0 } else { ps.iter().sum::<f64>() / ps.len() as f64 }
            }
            None => evaluate(&cal, insn_gt.as_ref().unwrap()).ece,
        }
    };
    let meta = Meta {
        name: job.name.clone(),
        regime: job.regime,
        sublabel: job.sublabel.clone(),
        n,
        code_bytes: code.len(),
        base_rate,
        ece_always_benign: ece(Regime::Benign),
        ece_oracle: ece(job.regime),
        region_ent,
        mmae_pick,
        mmae_nis_pick,
        clf_pick,
        rule_pick,
        guard_pick,
        s_glob_benign_eng: s_glob[Regime::Benign.idx()],
        s_spat_benign_eng: s_spat[Regime::Benign.idx()],
        s_glob_packed_eng: s_glob[Regime::Packed.idx()],
        s_glob_obf_eng: s_glob[Regime::Obfuscated.idx()],
        nis_benign_eng: nis[Regime::Benign.idx()],
        nis_packed_eng: nis[Regime::Packed.idx()],
        nis_obf_eng: nis[Regime::Obfuscated.idx()],
    };

    Ok((rows, meta))
}

/// The task rule. A head is recovered when a direct call reaches it and *both* ends of that call
/// clear τ under this arm's calibrated posterior.
///
/// Gating both ends is not belt-and-braces. Gating only the target would let a junk decode deep in a
/// desynchronized region nominate a head purely because its bytes happen to look like a plausible
/// entry; gating only the call site would admit a confident call into rubble. The pair is what makes
/// the recovered set a function of the calibration rather than of capstone.
fn recover_boundaries(
    call_edges: &[(u64, u64)],
    conf: &HashMap<u64, f64>,
    tau: f64,
) -> BTreeSet<u64> {
    call_edges
        .iter()
        .filter(|&&(c, t)| {
            // An address the engine never surfaced as a candidate has no calibrated confidence, and
            // absence is not evidence of code — it fails the gate rather than defaulting through it.
            conf.get(&c).is_some_and(|&p| p >= tau) && conf.get(&t).is_some_and(|&p| p >= tau)
        })
        .map(|&(_, t)| t)
        .collect()
}

/// Grade one arm's recovered head set at one threshold.
fn grade_recovery(
    job: &HoldoutJob,
    arm: Arm,
    pick: Regime,
    tau: f64,
    pred: &BTreeSet<u64>,
    head_gt: Option<&BTreeSet<u64>>,
    n_reach: usize,
    n_window: usize,
    window: Option<(u64, u64)>,
) -> Recovery {
    let n_pred = pred.len();
    let (n_gt, tp, fp, fn_) = match head_gt {
        Some(g) => {
            let tp = pred.iter().filter(|t| g.contains(t)).count();
            (g.len(), tp, n_pred - tp, g.len() - tp)
        }
        // Packed: the GT proves data. There are no known heads, so tp = 0 and the recall and F1 cells
        // stay empty on purpose.
        None => (0, 0, 0, 0),
    };
    let win_pred = match window {
        Some((lo, hi)) => pred.iter().filter(|&&t| t >= lo && t < hi).count(),
        None => 0,
    };
    // On packed, the false heads we can *prove* are exactly the ones inside the data window.
    let fp = if window.is_some() { win_pred } else { fp };
    Recovery {
        name: job.name.clone(),
        regime: job.regime,
        sublabel: job.sublabel.clone(),
        arm,
        pick,
        tau,
        n_gt,
        n_reach,
        n_pred,
        tp,
        fp,
        fn_,
        n_window,
        win_pred,
    }
}

/// Load the symtab head set for one held-out binary, restricted to `.text`.
///
/// Missing GT is a hard error, not a skipped binary: a boundary probe that silently drops the
/// binaries it could not grade reports the F1 of whatever happened to survive.
fn load_boundary_gt(
    root: &Path,
    sublabel: &str,
    name: &str,
    (lo, hi): (u64, u64),
) -> Result<BTreeSet<u64>> {
    let path = root.join(sublabel).join(format!("{name}.func.gt"));
    let text = fs::read_to_string(&path)
        .with_context(|| format!("reading function-boundary GT {} (run gen_boundary_gt.py)", path.display()))?;
    let mut set = BTreeSet::new();
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let a = u64::from_str_radix(t.trim_start_matches("0x"), 16)
            .with_context(|| format!("bad address {t:?} in {}", path.display()))?;
        if a >= lo && a < hi {
            set.insert(a);
        }
    }
    Ok(set)
}

fn argmin_regime(stat: &[f64; 3]) -> Regime {
    let mut best = Regime::Benign;
    let mut best_v = f64::INFINITY;
    for r in Regime::ALL {
        if stat[r.idx()] < best_v {
            best_v = stat[r.idx()];
            best = r;
        }
    }
    best
}

// ── Aggregation ────────────────────────────────────────────────────────────────

/// A pooled (micro-averaged) cell of the headline table: one (regime, arm, τ).
///
/// Micro, not macro, is the headline: an analyst works through one pile of recovered heads, and what
/// they experience is the fraction of *that pile* that is junk — a big binary contributing more heads
/// should weigh more. The per-binary macro mean is carried alongside for the appendix.
struct Cell {
    n_bins: usize,
    tp: usize,
    fp: usize,
    fn_: usize,
    n_pred: usize,
    n_gt: usize,
    n_reach: usize,
    n_window: usize,
    win_pred: usize,
    macro_f1: Vec<f64>,
    /// Binaries where this arm recovered nothing at all — precision undefined, recall 0.
    n_empty: usize,
}

impl Cell {
    fn new() -> Self {
        Cell {
            n_bins: 0, tp: 0, fp: 0, fn_: 0, n_pred: 0, n_gt: 0, n_reach: 0, n_window: 0,
            win_pred: 0, macro_f1: Vec::new(), n_empty: 0,
        }
    }
    fn push(&mut self, r: &Recovery) {
        self.n_bins += 1;
        self.tp += r.tp;
        self.fp += r.fp;
        self.fn_ += r.fn_;
        self.n_pred += r.n_pred;
        self.n_gt += r.n_gt;
        self.n_reach += r.n_reach;
        self.n_window += r.n_window;
        self.win_pred += r.win_pred;
        if r.n_pred == 0 {
            self.n_empty += 1;
        }
        if let Some(v) = f1(r) {
            self.macro_f1.push(v);
        }
    }
    /// Micro precision. On packed this is 0 whenever anything inside the provable-data window was
    /// nominated: those heads are provably data, and there are no provable positives to offset them.
    fn micro_precision(&self) -> Option<f64> {
        let denom = self.tp + self.fp;
        if denom == 0 { None } else { Some(self.tp as f64 / denom as f64) }
    }
    fn micro_recall(&self) -> Option<f64> {
        if self.n_gt == 0 { None } else { Some(self.tp as f64 / self.n_gt as f64) }
    }
    fn micro_recall_reach(&self) -> Option<f64> {
        if self.n_reach == 0 { None } else { Some(self.tp as f64 / self.n_reach as f64) }
    }
    fn micro_f1(&self) -> Option<f64> {
        harmonic(self.micro_precision(), self.micro_recall())
    }
    fn micro_f1_reach(&self) -> Option<f64> {
        harmonic(self.micro_precision(), self.micro_recall_reach())
    }
}

/// What the detector saw, pooled per reporting group. This is the middle term of the claim — the
/// corruption and the recovery are only interesting if the thing that triggers the switch fired for a
/// legible reason, so the signature stats travel with the F1 table rather than in a separate file.
struct Detect {
    n: usize,
    s_glob: Vec<f64>,
    s_spat: Vec<f64>,
    /// Held-out binaries whose signature cleared the clean-fit null on each axis.
    n_glob_fire: usize,
    n_spat_fire: usize,
    /// GT-free selection correctness, rule and guard, against the true regime.
    n_rule_correct: usize,
    n_guard_correct: usize,
}

impl Detect {
    fn new() -> Self {
        Detect { n: 0, s_glob: Vec::new(), s_spat: Vec::new(), n_glob_fire: 0, n_spat_fire: 0,
                 n_rule_correct: 0, n_guard_correct: 0 }
    }
}

struct Summary {
    taus: Vec<f64>,
    /// (regime-or-group label, arm, τ) → cell.
    cells: Vec<((String, Arm, f64), Cell)>,
    /// The clean-fit null the detector is read against, and the per-group detector behaviour.
    glob_hi: f64,
    spat_hi: f64,
    detect: Vec<(String, Detect)>,
}

/// Group a row into a reporting bucket. The three headline regimes are benign / packed / desync;
/// Tigress and the legitimate-VM rows are split out so neither dilutes a regime headline (Tigress is
/// the known blind spot, legit-VM is the false-positive gate).
fn group_of(sublabel: &str, regime: Regime) -> String {
    if sublabel.starts_with("tig") {
        "tigress".into()
    } else if sublabel.starts_with("vm") {
        "legit_vm".into()
    } else {
        match regime {
            Regime::Benign => "benign".into(),
            Regime::Packed => "packed".into(),
            Regime::Obfuscated => "desync".into(),
        }
    }
}

const GROUPS: [&str; 5] = ["benign", "packed", "desync", "tigress", "legit_vm"];

fn summarize(
    recoveries: &[Recovery],
    metas: &[Meta],
    clf: &SignatureClassifier,
    taus: &[f64],
) -> Summary {
    let mut map: HashMap<(String, String, u64), Cell> = HashMap::new();
    let key = |g: &str, a: Arm, t: f64| (g.to_string(), a.tag().to_string(), (t * 1e6) as u64);
    for r in recoveries {
        let g = group_of(&r.sublabel, r.regime);
        map.entry(key(&g, r.arm, r.tau)).or_insert_with(Cell::new).push(r);
    }
    let mut cells = Vec::new();
    for g in GROUPS {
        for arm in Arm::ALL {
            for &t in taus {
                if let Some(c) = map.remove(&key(g, arm, t)) {
                    cells.push(((g.to_string(), arm, t), c));
                }
            }
        }
    }

    let mut dmap: HashMap<String, Detect> = HashMap::new();
    for m in metas {
        let d = dmap.entry(group_of(&m.sublabel, m.regime)).or_insert_with(Detect::new);
        d.n += 1;
        d.s_glob.push(m.s_glob_benign_eng);
        d.s_spat.push(m.s_spat_benign_eng);
        d.n_glob_fire += usize::from(m.s_glob_benign_eng > clf.glob_hi);
        d.n_spat_fire += usize::from(m.s_spat_benign_eng > clf.spat_hi);
        d.n_rule_correct += usize::from(m.rule_pick == m.regime);
        d.n_guard_correct += usize::from(m.guard_pick == m.regime);
    }
    let detect = GROUPS.iter().filter_map(|g| dmap.remove(*g).map(|d| (g.to_string(), d))).collect();

    Summary { taus: taus.to_vec(), cells, glob_hi: clf.glob_hi, spat_hi: clf.spat_hi, detect }
}

fn opt(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:.4}"),
        None => "n/a".into(),
    }
}

fn opt_json(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:.6}"),
        None => "null".into(),
    }
}

fn delta(a: Option<f64>, b: Option<f64>) -> String {
    match (a, b) {
        (Some(a), Some(b)) => format!("{:+.4}", a - b),
        _ => "n/a".into(),
    }
}

impl Summary {
    fn get(&self, g: &str, arm: Arm, tau: f64) -> Option<&Cell> {
        self.cells
            .iter()
            .find(|((gg, a, t), _)| gg == g && *a == arm && (t - tau).abs() < 1e-9)
            .map(|(_, c)| c)
    }

    fn print(&self) {
        println!("\n═══════════ DOWNSTREAM TASK: function-boundary recovery (symtab GT) ═══════════");
        println!("micro-averaged over held-out binaries; precision n/a ⇒ arm recovered nothing;");
        println!("recall/F1 n/a ⇒ regime GT proves data only (packed), so there are no known heads.");
        println!("F1_reach restricts recall to heads some direct call targets — the rule's ceiling.\n");
        println!("  group     τ     arm              prec     recall       F1   F1_reach   heads   win-FP");
        for g in GROUPS {
            for &tau in &self.taus {
                for arm in Arm::ALL {
                    let Some(c) = self.get(g, arm, tau) else { continue };
                    println!(
                        "  {:<9} {:<5.2} {:<16} {:>6}  {:>9}  {:>7}  {:>9}  {:>6}  {:>6}",
                        g, tau, arm.tag(),
                        opt(c.micro_precision()), opt(c.micro_recall()),
                        opt(c.micro_f1()), opt(c.micro_f1_reach()),
                        c.n_pred, c.win_pred,
                    );
                }
            }
        }

        // The middle term: did the detector actually fire, and against what null.
        println!("\n— The detector, read against the clean-fit null (S_glob>{:.4}, S_spat>{:.4}) —",
                 self.glob_hi, self.spat_hi);
        println!("  group      n   mean S_glob   mean S_spat   glob fires   spat fires   rule acc   guard acc");
        for (g, d) in &self.detect {
            println!(
                "  {:<9} {:>3}   {:>11.4}   {:>11.4}   {:>4}/{:<5} {:>4}/{:<5}   {:>7.3}   {:>8.3}",
                g, d.n,
                consistency::mean(&d.s_glob), consistency::mean(&d.s_spat),
                d.n_glob_fire, d.n, d.n_spat_fire, d.n,
                d.n_rule_correct as f64 / d.n.max(1) as f64,
                d.n_guard_correct as f64 / d.n.max(1) as f64,
            );
        }

        // The headline: the drop the stale map costs, and how much of it the switch buys back.
        let hi = self.taus.last().copied().unwrap_or(0.9);
        println!("\n— Corrupt → recover at τ={hi}: boundary F1 —");
        println!("  group      stale    switch    oracle   drop(stale−oracle)  recovery(switch−stale)");
        for g in GROUPS {
            let (s, w, o) = (
                self.get(g, Arm::Stale, hi),
                self.get(g, Arm::SwitchGuard, hi),
                self.get(g, Arm::Oracle, hi),
            );
            if let (Some(s), Some(w), Some(o)) = (s, w, o) {
                println!(
                    "  {:<9} {:>7}  {:>8}  {:>8}  {:>18}  {:>21}",
                    g,
                    opt(s.micro_f1()), opt(w.micro_f1()), opt(o.micro_f1()),
                    delta(s.micro_f1(), o.micro_f1()),
                    delta(w.micro_f1(), s.micro_f1()),
                );
            }
        }
        println!("═══════════════════════════════════════════════════════════════════════════════");
    }

    fn to_json(&self) -> String {
        let mut s = String::from("{\n");
        s.push_str(&format!(
            "  \"clean_fit_null\": {{\"s_glob_hi\": {:.6}, \"s_spat_hi\": {:.6}}},\n",
            self.glob_hi, self.spat_hi
        ));
        s.push_str("  \"detect\": [\n");
        for (i, (g, d)) in self.detect.iter().enumerate() {
            s.push_str(&format!(
                "    {{\"group\": \"{}\", \"n\": {}, \"mean_s_glob\": {:.6}, \"mean_s_spat\": {:.6}, \
\"n_glob_fire\": {}, \"n_spat_fire\": {}, \"rule_accuracy\": {:.6}, \"guard_accuracy\": {:.6}}}{}\n",
                g, d.n, consistency::mean(&d.s_glob), consistency::mean(&d.s_spat),
                d.n_glob_fire, d.n_spat_fire,
                d.n_rule_correct as f64 / d.n.max(1) as f64,
                d.n_guard_correct as f64 / d.n.max(1) as f64,
                if i + 1 == self.detect.len() { "" } else { "," },
            ));
        }
        s.push_str("  ],\n  \"cells\": [\n");
        for (i, ((g, arm, tau), c)) in self.cells.iter().enumerate() {
            s.push_str(&format!(
                "    {{\"group\": \"{}\", \"arm\": \"{}\", \"tau\": {:.2}, \"n_bins\": {}, \
\"precision\": {}, \"recall\": {}, \"f1\": {}, \"recall_reach\": {}, \"f1_reach\": {}, \
\"macro_f1\": {}, \"tp\": {}, \"fp\": {}, \"fn\": {}, \"n_pred\": {}, \"n_gt\": {}, \"n_reach\": {}, \
\"n_window\": {}, \"win_pred\": {}, \"n_empty\": {}}}{}\n",
                g, arm.tag(), tau, c.n_bins,
                opt_json(c.micro_precision()),
                opt_json(c.micro_recall()),
                opt_json(c.micro_f1()),
                opt_json(c.micro_recall_reach()),
                opt_json(c.micro_f1_reach()),
                opt_json(if c.macro_f1.is_empty() { None } else { Some(consistency::mean(&c.macro_f1)) }),
                c.tp, c.fp, c.fn_, c.n_pred, c.n_gt, c.n_reach, c.n_window, c.win_pred, c.n_empty,
                if i + 1 == self.cells.len() { "" } else { "," },
            ));
        }
        s.push_str("  ],\n  \"_end\": true\n}\n");
        s
    }
}

// ── CSV resume / IO ────────────────────────────────────────────────────────────

fn read_done_keys(meta: &Path) -> HashSet<String> {
    let mut set = HashSet::new();
    let Ok(text) = fs::read_to_string(meta) else {
        return set;
    };
    for line in text.lines().skip(1) {
        let c: Vec<&str> = line.split(',').collect();
        if c.len() >= 3 {
            set.insert(format!("{}|{}", c[0], c[2]));
        }
    }
    set
}

fn read_existing_recoveries(path: &Path) -> Vec<Recovery> {
    let mut out = Vec::new();
    let Ok(text) = fs::read_to_string(path) else {
        return out;
    };
    for line in text.lines().skip(1) {
        let c: Vec<&str> = line.split(',').collect();
        if c.len() < 14 {
            continue;
        }
        let (Some(regime), Some(pick)) = (consistency::parse_regime(c[1]), consistency::parse_regime(c[4]))
        else {
            continue;
        };
        let arm = match c[3] {
            "stale" => Arm::Stale,
            "switch_rule" => Arm::SwitchRule,
            "switch_guard" => Arm::SwitchGuard,
            "oracle" => Arm::Oracle,
            _ => continue,
        };
        let p = |i: usize| c[i].parse::<usize>().ok();
        let (Some(n_gt), Some(n_reach), Some(n_pred), Some(tp), Some(fp), Some(fn_), Some(n_window), Some(win_pred)) =
            (p(6), p(7), p(8), p(9), p(10), p(11), p(12), p(13))
        else {
            continue;
        };
        let Ok(tau) = c[5].parse::<f64>() else { continue };
        out.push(Recovery {
            name: c[0].to_string(),
            regime,
            sublabel: c[2].to_string(),
            arm,
            pick,
            tau,
            n_gt,
            n_reach,
            n_pred,
            tp,
            fp,
            fn_,
            n_window,
            win_pred,
        });
    }
    out
}

fn open_csv_append(path: &Path, header: &str) -> Result<fs::File> {
    let exists = path.exists();
    let mut f = fs::OpenOptions::new().create(true).append(true).open(path)?;
    if !exists {
        writeln!(f, "{header}")?;
    }
    Ok(f)
}

// ── CLI ────────────────────────────────────────────────────────────────────────

struct Args {
    spec: CorpusSpec,
    /// Root of the symtab function-boundary GT tree: `ROOT/<sublabel>/<name>.func.gt`.
    func_gt_root: PathBuf,
    taus: Vec<f64>,
    entropy_strength: f64,
    chainfwd_strength: f64,
    out: PathBuf,
    meta: PathBuf,
    summary: PathBuf,
}

impl Args {
    fn parse(mut it: impl Iterator<Item = String>) -> Result<Self> {
        const USAGE: &str = "usage: downstream --clean-bins DIR --clean-gt DIR --func-gt-root DIR \
[--desync-level LABEL BINS GT]... [--packed-spec LABEL ELF UPXGT]... \
[--tigress-level LABEL BINS GT]... [--benign-holdout LABEL BINS GT]... --n-clean-fit N \
--n-clean-holdout N --n-desync-fit N --n-desync-holdout N --n-packed-fit N --n-packed-holdout N \
--n-tig-holdout N [--tau F]... [--entropy-strength F] [--chainfwd-strength F] [--seed S] \
--out CSV --meta CSV --summary JSON";
        let mut clean_bins = None;
        let mut clean_gt = None;
        let mut func_gt_root = None;
        let mut desync_levels = Vec::new();
        let mut packed_specs = Vec::new();
        let mut tigress_levels = Vec::new();
        let mut benign_holdout = Vec::new();
        let mut n_clean_fit = 20usize;
        let mut n_clean_holdout = 25usize;
        let mut n_desync_fit = 40usize;
        let mut n_desync_holdout = 30usize;
        let mut n_packed_fit = 9usize;
        let mut n_packed_holdout = 8usize;
        let mut n_tig_holdout = 27usize;
        let mut taus: Vec<f64> = Vec::new();
        let mut entropy_strength = 1.0f64;
        let mut chainfwd_strength = 0.5f64;
        let mut seed = 1u64;
        let mut out = None;
        let mut meta = None;
        let mut summary = None;
        while let Some(a) = it.next() {
            let mut next = |flag: &str| it.next().with_context(|| format!("{flag} needs a value"));
            match a.as_str() {
                "--clean-bins" => clean_bins = Some(PathBuf::from(next("--clean-bins")?)),
                "--clean-gt" => clean_gt = Some(PathBuf::from(next("--clean-gt")?)),
                "--func-gt-root" => func_gt_root = Some(PathBuf::from(next("--func-gt-root")?)),
                "--desync-level" => {
                    let label = next("--desync-level")?;
                    let bins = PathBuf::from(next("--desync-level bins")?);
                    let gt = PathBuf::from(next("--desync-level gt")?);
                    desync_levels.push((label, bins, gt));
                }
                "--packed-spec" => {
                    let label = next("--packed-spec")?;
                    let elf = PathBuf::from(next("--packed-spec elf")?);
                    let gt = PathBuf::from(next("--packed-spec gt")?);
                    packed_specs.push((label, elf, gt));
                }
                "--tigress-level" => {
                    let label = next("--tigress-level")?;
                    let bins = PathBuf::from(next("--tigress-level bins")?);
                    let gt = PathBuf::from(next("--tigress-level gt")?);
                    tigress_levels.push((label, bins, gt));
                }
                "--benign-holdout" => {
                    let label = next("--benign-holdout")?;
                    let bins = PathBuf::from(next("--benign-holdout bins")?);
                    let gt = PathBuf::from(next("--benign-holdout gt")?);
                    benign_holdout.push((label, bins, gt));
                }
                "--n-clean-fit" => n_clean_fit = next("--n-clean-fit")?.parse()?,
                "--n-clean-holdout" => n_clean_holdout = next("--n-clean-holdout")?.parse()?,
                "--n-desync-fit" => n_desync_fit = next("--n-desync-fit")?.parse()?,
                "--n-desync-holdout" => n_desync_holdout = next("--n-desync-holdout")?.parse()?,
                "--n-packed-fit" => n_packed_fit = next("--n-packed-fit")?.parse()?,
                "--n-packed-holdout" => n_packed_holdout = next("--n-packed-holdout")?.parse()?,
                "--n-tig-holdout" => n_tig_holdout = next("--n-tig-holdout")?.parse()?,
                "--tau" => taus.push(next("--tau")?.parse()?),
                "--entropy-strength" => entropy_strength = next("--entropy-strength")?.parse()?,
                "--chainfwd-strength" => chainfwd_strength = next("--chainfwd-strength")?.parse()?,
                "--seed" => seed = next("--seed")?.parse()?,
                "--out" => out = Some(PathBuf::from(next("--out")?)),
                "--meta" => meta = Some(PathBuf::from(next("--meta")?)),
                "--summary" => summary = Some(PathBuf::from(next("--summary")?)),
                "-h" | "--help" => {
                    eprintln!("{USAGE}");
                    std::process::exit(0);
                }
                other => bail!("unexpected argument: {other}\n{USAGE}"),
            }
        }
        if taus.is_empty() {
            taus = vec![0.5, 0.7, 0.9];
        }
        taus.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for &t in &taus {
            if !(0.0..=1.0).contains(&t) {
                bail!("--tau must be in [0,1], got {t}");
            }
        }
        Ok(Args {
            spec: CorpusSpec {
                clean_bins: clean_bins.context(USAGE)?,
                clean_gt: clean_gt.context(USAGE)?,
                desync_levels,
                packed_specs,
                tigress_levels,
                benign_holdout,
                n_clean_fit,
                n_clean_holdout,
                n_desync_fit,
                n_desync_holdout,
                n_packed_fit,
                n_packed_holdout,
                n_tig_holdout,
                seed,
            },
            func_gt_root: func_gt_root.context(USAGE)?,
            taus,
            entropy_strength,
            chainfwd_strength,
            out: out.context(USAGE)?,
            meta: meta.context(USAGE)?,
            summary: summary.context(USAGE)?,
        })
    }
}
