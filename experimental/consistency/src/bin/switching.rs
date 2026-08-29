//! `switching` — Paper 2's payoff: consistency-driven calibration-map switching.
//!
//! The credibility run proved the cavity-surprise statistic *detects* drift. A reviewer's fair
//! objection: a scalar drift monitor (mean-π) tracks ECE about as well — so what does the surprise
//! buy us that a scalar can't? This binary is the answer. The scalar says "something's off"; it
//! can't *act*. We turn the detector into a self-calibrating capability: a bank of regime-specific
//! `(engine setting, calibration map)` configs, and a ground-truth-free rule that picks the config
//! most consistent with the binary in front of us — the navigation MMAE / IMM analogue. Then we
//! show, on held-out binaries with GT, that the switched calibration is restored on obfuscated /
//! packed code, approaching the oracle and beating the stale always-benign map — without ever
//! seeing the regime.
//!
//! Two GT-free selection rules, both implemented, honest about which wins:
//!   1. MMAE-direct  — run each config's engine, take the one with the lowest consistency S
//!                     (most consistent = best-fitting model), apply its map. True MMAE; works only
//!                     if a regime-matched engine setting actually lowers S on its own regime.
//!   2. Signature-classifier — the credibility signatures are regime-diagnostic (desync raises
//!                     S_glob; packers collapse S_glob but keep S_spat ~2× clean). Nearest-centroid
//!                     on the (ln S_glob, ln S_spat) signature classifies the regime, then applies
//!                     that regime's map.
//!
//! Three arms, on held-out GT binaries:
//!   (a) always-benign map — the naive baseline; goes stale under obfuscation (the problem).
//!   (b) oracle map        — true regime known, correct config applied — the ceiling.
//!   (c) consistency-selected map (ours, GT-free) — the switch.
//! The win: on obfuscated/packed held-out, (c) ≈ (b) and (c) ≪ (a).
//!
//! Standing rules honored: one binary in memory at a time (`--jobs 1` semantics; the three engine
//! runs per held-out binary are sequential — each converged graph is freed before the next);
//! resumable CSV (an existing row for a binary is reused); GT read only from the pre-supplied `.gt`
//! files / UPX `b_info` windows — *never* a disassembly of the input under test.
//!
//! ```text
//! switching \
//!   --clean-bins DIR --clean-gt DIR \
//!   --desync-level LABEL BINS GT ...  (repeatable; the obfuscated-desync corpus) \
//!   --packed-spec  LABEL ELF  UPXGT ...(repeatable; the packed corpus) \
//!   --tigress-level LABEL BINS GT ... (repeatable; held-out semantic-obfuscation limit probe) \
//!   --n-clean-fit N --n-clean-holdout N --n-desync-fit N --n-desync-holdout N \
//!   --n-packed-fit N --n-packed-holdout N --n-tig-holdout N \
//!   [--entropy-strength 1.0] [--chainfwd-strength 0.5] [--seed 1] \
//!   --out results.csv --summary summary.json
//! ```

use std::collections::HashMap;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use evalkit::{evaluate, load_gt, run_soft_with_cavity_cfg, IsotonicMap};
use probdisasm::{extract_text_section as extract_text, CavityStat};

// ── Regimes and the config bank ────────────────────────────────────────────────

/// The three regimes the bank covers. Tigress binaries carry the `Obfuscated` *true* regime (they
/// are obfuscated code) but are tracked by sub-label so we can show the honest blind spot: the
/// surprise statistic does not fire on semantic obfuscation that preserves clean decoding.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Regime {
    Benign,
    Packed,
    Obfuscated,
}

impl Regime {
    fn tag(self) -> &'static str {
        match self {
            Regime::Benign => "benign",
            Regime::Packed => "packed",
            Regime::Obfuscated => "obfuscated",
        }
    }
    /// The engine setting `(entropy_prior_strength, chainfwd_strength)` for this regime, given the
    /// two tunable strengths. Benign = both off (the untouched shared engine); packed = entropy
    /// prior (pull high-entropy payload → data); obfuscated = chainfwd prior (pull chain-consistent
    /// bytes → code). These are exactly the knobs `AnalysisConfig` already exposes.
    fn engine(self, entropy: f64, chainfwd: f64) -> (f64, f64) {
        match self {
            Regime::Benign => (0.0, 0.0),
            Regime::Packed => (entropy, 0.0),
            Regime::Obfuscated => (0.0, chainfwd),
        }
    }
    const ALL: [Regime; 3] = [Regime::Benign, Regime::Packed, Regime::Obfuscated];
    fn idx(self) -> usize {
        match self {
            Regime::Benign => 0,
            Regime::Packed => 1,
            Regime::Obfuscated => 2,
        }
    }
}

/// One held-out binary queued for the three-arm evaluation.
struct HoldoutJob {
    name: String,
    bin: PathBuf,
    /// Instruction-start GT for benign/obfuscated. `None` for packed (window from `packed_gt`).
    gt: Option<PathBuf>,
    /// Packed only: the `.upxgt` provable-data window.
    packed_gt: Option<PathBuf>,
    /// The true regime (the oracle knows this; the GT-free rules must recover it).
    regime: Regime,
    /// Corpus sub-label (e.g. "d2_heavy", "upx_nrv", "tigL"). Lets one run cover many intensities
    /// and separates the Tigress blind-spot rows in the report.
    sublabel: String,
}

/// One fit binary (contributes to a map and/or the classifier centroid).
struct FitJob {
    name: String,
    bin: PathBuf,
    gt: Option<PathBuf>,
    packed_gt: Option<PathBuf>,
    regime: Regime,
}

// ── Per-held-out-binary record ─────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct Record {
    name: String,
    regime: Regime,
    sublabel: String,
    n: usize,
    code_bytes: usize,
    base_rate: f64,
    // The four arms' true post-hoc ECE.
    ece_always_benign: f64,
    ece_oracle: f64,
    ece_mmae: f64,
    ece_clf: f64,
    ece_rule: f64,
    /// Abstention-guard arm: the rule + region-entropy gate on the packed route.
    ece_guard: f64,
    /// Region entropy (max sliding-window byte entropy) — the guard's discriminator, recorded for
    /// the audit.
    region_ent: f64,
    // GT-free selections (what regime each rule picked).
    mmae_pick: Regime,
    mmae_nis_pick: Regime,
    clf_pick: Regime,
    rule_pick: Regime,
    guard_pick: Regime,
    // Signature under the benign engine + the per-config consistency statistics (for the MMAE trace
    // and the honest "did a regime-matched engine lower S?" audit).
    s_glob_benign_eng: f64,
    s_spat_benign_eng: f64,
    s_glob_packed_eng: f64,
    s_glob_obf_eng: f64,
    nis_benign_eng: f64,
    nis_packed_eng: f64,
    nis_obf_eng: f64,
}

impl Record {
    const CSV_HEADER: &'static str = "name,regime,sublabel,n,code_bytes,base_rate,\
ece_always_benign,ece_oracle,ece_mmae,ece_clf,ece_rule,ece_guard,region_ent,\
mmae_pick,mmae_nis_pick,clf_pick,rule_pick,guard_pick,\
s_glob_benign_eng,s_spat_benign_eng,s_glob_packed_eng,s_glob_obf_eng,\
nis_benign_eng,nis_packed_eng,nis_obf_eng";

    fn to_csv(&self) -> String {
        format!(
            "{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            self.name,
            self.regime.tag(),
            self.sublabel,
            self.n,
            self.code_bytes,
            self.base_rate,
            self.ece_always_benign,
            self.ece_oracle,
            self.ece_mmae,
            self.ece_clf,
            self.ece_rule,
            self.ece_guard,
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

    fn from_csv(line: &str) -> Option<Record> {
        let c: Vec<&str> = line.split(',').collect();
        if c.len() < 25 {
            return None;
        }
        let regime = parse_regime(c[1])?;
        Some(Record {
            name: c[0].to_string(),
            regime,
            sublabel: c[2].to_string(),
            n: c[3].parse().ok()?,
            code_bytes: c[4].parse().ok()?,
            base_rate: c[5].parse().ok()?,
            ece_always_benign: c[6].parse().ok()?,
            ece_oracle: c[7].parse().ok()?,
            ece_mmae: c[8].parse().ok()?,
            ece_clf: c[9].parse().ok()?,
            ece_rule: c[10].parse().ok()?,
            ece_guard: c[11].parse().ok()?,
            region_ent: c[12].parse().ok()?,
            mmae_pick: parse_regime(c[13])?,
            mmae_nis_pick: parse_regime(c[14])?,
            clf_pick: parse_regime(c[15])?,
            rule_pick: parse_regime(c[16])?,
            guard_pick: parse_regime(c[17])?,
            s_glob_benign_eng: c[18].parse().ok()?,
            s_spat_benign_eng: c[19].parse().ok()?,
            s_glob_packed_eng: c[20].parse().ok()?,
            s_glob_obf_eng: c[21].parse().ok()?,
            nis_benign_eng: c[22].parse().ok()?,
            nis_packed_eng: c[23].parse().ok()?,
            nis_obf_eng: c[24].parse().ok()?,
        })
    }
}

fn parse_regime(s: &str) -> Option<Regime> {
    match s {
        "benign" => Some(Regime::Benign),
        "packed" => Some(Regime::Packed),
        "obfuscated" => Some(Regime::Obfuscated),
        _ => None,
    }
}

fn main() -> Result<()> {
    let args = Args::parse(std::env::args().skip(1))?;

    // ── Assemble the corpus groups, deterministic seeded order → fit / held-out split ──
    let clean = deterministic_order(list_bins_with_gt(&args.clean_bins, &args.clean_gt)?, args.seed);
    if clean.len() < args.n_clean_fit + args.n_clean_holdout {
        bail!(
            "need {} clean bins (fit {} + holdout {}), found {}",
            args.n_clean_fit + args.n_clean_holdout,
            args.n_clean_fit,
            args.n_clean_holdout,
            clean.len()
        );
    }

    let mut fit_jobs: Vec<FitJob> = Vec::new();
    let mut hold_jobs: Vec<HoldoutJob> = Vec::new();

    for (i, (name, bin, gt)) in clean.into_iter().enumerate() {
        if i < args.n_clean_fit {
            fit_jobs.push(FitJob { name, bin, gt: Some(gt), packed_gt: None, regime: Regime::Benign });
        } else if i < args.n_clean_fit + args.n_clean_holdout {
            hold_jobs.push(HoldoutJob {
                name, bin, gt: Some(gt), packed_gt: None, regime: Regime::Benign, sublabel: "clean".into(),
            });
        }
    }

    // Desync (obfuscated by junk-insertion): pool all levels, deterministic split. FIT bins feed the
    // obfuscated map AND the obfuscated classifier centroid.
    let mut desync_all: Vec<(String, PathBuf, PathBuf, String)> = Vec::new();
    for (label, bins_dir, gt_dir) in &args.desync_levels {
        let ds = list_bins_with_gt(bins_dir, gt_dir)
            .with_context(|| format!("desync level {label}: {}", bins_dir.display()))?;
        for (name, bin, gt) in ds {
            desync_all.push((name, bin, gt, label.clone()));
        }
    }
    // Deterministic order over the pooled set (seed decorrelates from level/alpha order).
    desync_all.sort_by_key(|(name, _, _, _)| splitmix64(args.seed ^ fnv1a(name.as_bytes())));
    for (i, (name, bin, gt, label)) in desync_all.into_iter().enumerate() {
        if i < args.n_desync_fit {
            fit_jobs.push(FitJob { name, bin, gt: Some(gt), packed_gt: None, regime: Regime::Obfuscated });
        } else if i < args.n_desync_fit + args.n_desync_holdout {
            hold_jobs.push(HoldoutJob {
                name, bin, gt: Some(gt), packed_gt: None, regime: Regime::Obfuscated, sublabel: label,
            });
        }
    }

    // Packed: deterministic split. FIT bins feed the packed map (negative-window) + packed centroid.
    let mut packed_all: Vec<(String, PathBuf, PathBuf, String)> = args
        .packed_specs
        .iter()
        .map(|(label, elf, gt)| (format!("{}__{}", file_stem(elf), label), elf.clone(), gt.clone(), label.clone()))
        .collect();
    packed_all.sort_by_key(|(name, _, _, _)| splitmix64(args.seed ^ fnv1a(name.as_bytes())));
    for (i, (name, elf, gt, label)) in packed_all.into_iter().enumerate() {
        if i < args.n_packed_fit {
            fit_jobs.push(FitJob { name, bin: elf, gt: None, packed_gt: Some(gt), regime: Regime::Packed });
        } else if i < args.n_packed_fit + args.n_packed_holdout {
            hold_jobs.push(HoldoutJob {
                name, bin: elf, gt: None, packed_gt: Some(gt), regime: Regime::Packed, sublabel: label,
            });
        }
    }

    // Packed held-out ONLY — structurally-distinct non-UPX packers (Ezuri AES crypter, kiteshield
    // RC4 loader, gzexe, …). True regime = Packed. Deliberately NOT in the fit set: the bank's packed
    // map + centroid are fit on UPX alone, so these probe whether the UPX-fit packed regime and the
    // GT-free signature rule GENERALIZE across packer families. Each carries its own format-exact
    // provable-data NEGATIVE window (per-packer carver, never a disassembler of the input).
    for (label, elf, gt) in &args.packed_holdout {
        hold_jobs.push(HoldoutJob {
            name: format!("{}__{}", file_stem(elf), label),
            bin: elf.clone(),
            gt: None,
            packed_gt: Some(gt.clone()),
            regime: Regime::Packed,
            sublabel: label.clone(),
        });
    }

    // Tigress: held-out ONLY — the semantic-obfuscation blind-spot probe. True regime = Obfuscated.
    // Deliberately NOT used to fit the obfuscated map or centroid: its benign-like signature would
    // pull the obfuscated centroid toward clean, and keeping it purely held-out makes the honest
    // limit ("S is blind to Tigress") clean to state.
    let mut tig_all: Vec<(String, PathBuf, PathBuf, String)> = Vec::new();
    for (label, bins_dir, gt_dir) in &args.tigress_levels {
        let tg = list_bins_with_gt(bins_dir, gt_dir)
            .with_context(|| format!("tigress level {label}: {}", bins_dir.display()))?;
        for (name, bin, gt) in tg {
            tig_all.push((name, bin, gt, label.clone()));
        }
    }
    tig_all.sort_by_key(|(name, _, _, _)| splitmix64(args.seed ^ fnv1a(name.as_bytes())));
    for (name, bin, gt, label) in tig_all.into_iter().take(args.n_tig_holdout) {
        hold_jobs.push(HoldoutJob {
            name, bin, gt: Some(gt), packed_gt: None, regime: Regime::Obfuscated, sublabel: label,
        });
    }

    // Benign-holdout extras — the legitimate-VM false-positive gate (p05_vm baseline / virtualized,
    // vmbig). True regime = Benign: these decode cleanly and are already well-calibrated under the
    // benign map, so the correct GT-free action is to stay benign. Held-out only (never fit); the
    // guard must abstain (route benign) despite their VM dispatch loops tripping the spatial rule.
    for (label, bins_dir, gt_dir) in &args.benign_holdout {
        let bl = list_bins_with_gt(bins_dir, gt_dir)
            .with_context(|| format!("benign-holdout {label}: {}", bins_dir.display()))?;
        for (name, bin, gt) in bl {
            hold_jobs.push(HoldoutJob {
                name, bin, gt: Some(gt), packed_gt: None, regime: Regime::Benign, sublabel: label.clone(),
            });
        }
    }

    eprintln!(
        "fit: {} benign / {} obf / {} packed   held-out: {}",
        fit_jobs.iter().filter(|j| j.regime == Regime::Benign).count(),
        fit_jobs.iter().filter(|j| j.regime == Regime::Obfuscated).count(),
        fit_jobs.iter().filter(|j| j.regime == Regime::Packed).count(),
        hold_jobs.len(),
    );

    // ── Pass 1: fit the config bank + train the signature classifier ──
    let bank = fit_bank(&fit_jobs, args.entropy_strength, args.chainfwd_strength)?;
    let clf = SignatureClassifier::train(&fit_jobs, args.entropy_strength, args.chainfwd_strength)?;

    // ── Pass 2: three-arm evaluation over held-out binaries; stream to CSV (resumable) ──
    eprintln!("── pass 2: three-arm held-out evaluation ──");
    let mut done: HashMap<String, Record> = read_existing_csv(&args.out);
    let mut csv = open_csv_append(&args.out)?;
    // Selective-disassembly per-address dump (fresh file, header once). Only packed holdouts with a
    // provable-data window emit rows; a resumed (already-in-CSV) binary is not re-run, so run the
    // selective demo on a fresh --out to guarantee every packed binary streams its posteriors.
    let mut dump = match &args.selective_dump {
        Some(p) => {
            let mut f = fs::File::create(p).with_context(|| format!("creating {}", p.display()))?;
            writeln!(f, "name,sublabel,arm,pick,addr,posterior")?;
            Some(f)
        }
        None => None,
    };
    let mut records: Vec<Record> = Vec::new();
    for job in &hold_jobs {
        let key = format!("{}|{}", job.name, job.sublabel);
        if let Some(rec) = done.remove(&key) {
            eprintln!("  resume {} [{}] (from CSV)", job.name, job.sublabel);
            records.push(rec);
            continue;
        }
        let rec = evaluate_holdout(job, &bank, &clf, args.entropy_strength, args.chainfwd_strength, dump.as_mut())?;
        writeln!(csv, "{}", rec.to_csv())?;
        csv.flush()?;
        eprintln!(
            "  {} [{}/{}]: a={:.4} oracle={:.4} rule={:.4}({}) guard={:.4}({}) ent={:.2}",
            rec.name, rec.regime.tag(), rec.sublabel,
            rec.ece_always_benign, rec.ece_oracle,
            rec.ece_rule, rec.rule_pick.tag(), rec.ece_guard, rec.guard_pick.tag(), rec.region_ent,
        );
        records.push(rec);
    }

    // ── Verdict ──
    let summary = analyze(&records, &clf);
    summary.print();
    fs::write(&args.summary, summary.to_json(&records))
        .with_context(|| format!("writing {}", args.summary.display()))?;
    eprintln!("wrote {} and {}", args.out.display(), args.summary.display());
    Ok(())
}

// ── The config bank: three (engine, map) pairs ─────────────────────────────────

struct Bank {
    /// Calibration maps indexed by `Regime::idx()`.
    maps: [IsotonicMap; 3],
}

impl Bank {
    fn map(&self, r: Regime) -> &IsotonicMap {
        &self.maps[r.idx()]
    }
}

/// Fit each regime's calibration map on its FIT binaries, each under its own engine setting. The
/// benign/obfuscated maps are ordinary isotonic fits on (posterior, instruction-start-label). The
/// packed map is fit on the provable-data window only (all labels 0), so it learns to pull the
/// packed regime's posteriors down — the honest packed calibration (the payload is data).
fn fit_bank(fit: &[FitJob], entropy: f64, chainfwd: f64) -> Result<Bank> {
    eprintln!("── pass 1a: fitting the calibration-map bank ──");
    let mut pools: [Vec<(f64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for job in fit {
        let (ent, cfw) = job.regime.engine(entropy, chainfwd);
        let bytes = fs::read(&job.bin).with_context(|| format!("reading {}", job.bin.display()))?;
        let (base, code) = extract_text(&bytes)?;
        let (post, _cav) = run_soft_with_cavity_cfg(base, code, ent, cfw, false)
            .with_context(|| format!("engine on {}", job.name))?;
        match job.regime {
            Regime::Packed => {
                // Negative window only: every candidate here is provable data (label 0).
                let (lo, hi) = packed_data_window(job.packed_gt.as_ref().unwrap())?;
                for &(a, p) in &post {
                    if a >= lo && a < hi {
                        pools[Regime::Packed.idx()].push((p, 0.0));
                    }
                }
            }
            _ => {
                let gt = load_gt(job.gt.as_ref().unwrap())?;
                for &(a, p) in &post {
                    pools[job.regime.idx()].push((p, if gt.contains(&a) { 1.0 } else { 0.0 }));
                }
            }
        }
        eprintln!("  fit map[{}] += {} ({} candidates)", job.regime.tag(), job.name, post.len());
        // bytes / post / cav dropped here — one binary in memory at a time.
    }
    let maps = [
        IsotonicMap::fit(&pools[0]),
        IsotonicMap::fit(&pools[1]),
        IsotonicMap::fit(&pools[2]),
    ];
    for r in Regime::ALL {
        eprintln!("  map[{}] fit on {} pooled candidates", r.tag(), pools[r.idx()].len());
    }
    Ok(Bank { maps })
}

// ── The signature classifier (nearest-centroid on (ln S_glob, ln S_spat)) ──────

/// A GT-free regime classifier: nearest-centroid in the standardized 2-D signature space
/// (`ln S_glob`, `ln S_spat`). Trained on the FIT binaries' benign-engine signatures — the
/// credibility run characterized these exact features (desync raises S_glob; packers collapse
/// S_glob but keep S_spat ~2× clean; clean sits low on both). Test-time it reads only the two
/// scalars off a binary's benign-engine cavity pass — no ground truth.
struct SignatureClassifier {
    /// Per-regime centroid in standardized feature space, indexed by `Regime::idx()`.
    centroids: [(f64, f64); 3],
    /// Present-in-training flag (a regime with no fit bins gets no centroid).
    present: [bool; 3],
    // Standardization params (mean/std of ln S_glob, ln S_spat over all fit bins).
    mu_g: f64,
    sd_g: f64,
    mu_s: f64,
    sd_s: f64,
    /// The raw per-regime (mean ln S_glob, mean ln S_spat) for the report table.
    raw_centroids: [(f64, f64); 3],
    // ── Threshold-rule variant (benign-default; encodes the credibility signatures directly) ──
    /// S_glob above this ⇒ obfuscated (desync raises S_glob ~25×; clean/packed stay low).
    glob_hi: f64,
    /// S_spat above this (with S_glob below `glob_hi`) ⇒ packed (packers keep S_spat ~2× clean but
    /// collapse S_glob). Below both ⇒ benign — the default, so an ambiguous binary is never routed
    /// to the calibration-destroying packed map without a clear spatial signal.
    spat_hi: f64,
    /// **Abstention guard.** The packed route (S_spat > spat_hi) also demands the flagged region be
    /// genuinely packed-like: region entropy above this threshold. A Tigress-virtualized binary
    /// trips the spatial threshold on its dispatch loop but is normal-entropy real code — it decodes
    /// cleanly and is already well-calibrated under the benign map — so the guard makes it fall
    /// through to benign (abstain) rather than get its calibration destroyed by the packed-suppress
    /// map. Trained ground-truth-free from the fit split: the midpoint of the gap between the
    /// highest non-packed region entropy and the lowest packed region entropy. `NaN` if the fit
    /// split doesn't separate (honest failure — reported, not papered over).
    pack_ent_lo: f64,
}

/// A small epsilon so `ln` of a zero/near-zero spatial statistic stays finite.
const LN_EPS: f64 = 1e-3;

impl SignatureClassifier {
    fn train(fit: &[FitJob], entropy: f64, chainfwd: f64) -> Result<Self> {
        eprintln!("── pass 1b: training the signature classifier (benign-engine signatures) ──");
        // Collect (regime, ln S_glob, ln S_spat) for every fit binary, under the BENIGN engine —
        // the signature is read from the default-engine cavity pass (what the credibility study
        // characterized), regardless of the binary's regime. Keep the raw clean-regime values too,
        // for the benign-null thresholds of the rule variant.
        let mut feats: Vec<(Regime, f64, f64)> = Vec::new();
        let mut clean_glob: Vec<f64> = Vec::new();
        let mut clean_spat: Vec<f64> = Vec::new();
        // Region-entropy pools for the abstention-guard threshold: packed vs everything else.
        let mut packed_ent: Vec<f64> = Vec::new();
        let mut nonpacked_ent: Vec<f64> = Vec::new();
        for job in fit {
            let bytes = fs::read(&job.bin).with_context(|| format!("reading {}", job.bin.display()))?;
            let (base, code) = extract_text(&bytes)?;
            let (_post, cav) = run_soft_with_cavity_cfg(base, code, 0.0, 0.0, false)
                .with_context(|| format!("benign engine on {}", job.name))?;
            let s = global_and_spatial(&cav);
            feats.push((job.regime, (s.mean_surprise.max(LN_EPS)).ln(), (s.moran.max(LN_EPS)).ln()));
            if job.regime == Regime::Benign {
                clean_glob.push(s.mean_surprise);
                clean_spat.push(s.moran);
            }
            let re = region_entropy(code);
            if job.regime == Regime::Packed {
                packed_ent.push(re);
            } else {
                nonpacked_ent.push(re);
            }
        }
        let _ = (entropy, chainfwd); // signature is benign-engine only; strengths unused here.
        if feats.is_empty() {
            bail!("no fit binaries to train the classifier");
        }
        let gs: Vec<f64> = feats.iter().map(|f| f.1).collect();
        let ss: Vec<f64> = feats.iter().map(|f| f.2).collect();
        let (mu_g, sd_g) = mean_std(&gs);
        let (mu_s, sd_s) = mean_std(&ss);
        let sd_g = if sd_g > 1e-9 { sd_g } else { 1.0 };
        let sd_s = if sd_s > 1e-9 { sd_s } else { 1.0 };

        let mut sum = [(0.0, 0.0); 3];
        let mut cnt = [0usize; 3];
        let mut raw_sum = [(0.0, 0.0); 3];
        for (r, g, s) in &feats {
            let i = r.idx();
            sum[i].0 += (g - mu_g) / sd_g;
            sum[i].1 += (s - mu_s) / sd_s;
            raw_sum[i].0 += g;
            raw_sum[i].1 += s;
            cnt[i] += 1;
        }
        let mut centroids = [(0.0, 0.0); 3];
        let mut raw_centroids = [(0.0, 0.0); 3];
        let mut present = [false; 3];
        for i in 0..3 {
            if cnt[i] > 0 {
                centroids[i] = (sum[i].0 / cnt[i] as f64, sum[i].1 / cnt[i] as f64);
                raw_centroids[i] = (raw_sum[i].0 / cnt[i] as f64, raw_sum[i].1 / cnt[i] as f64);
                present[i] = true;
            }
        }
        for r in Regime::ALL {
            if present[r.idx()] {
                eprintln!(
                    "  centroid[{}]: mean ln S_glob={:.3} ln S_spat={:.3} (n={})",
                    r.tag(), raw_centroids[r.idx()].0, raw_centroids[r.idx()].1, cnt[r.idx()]
                );
            }
        }

        // Rule thresholds from the clean-fit (benign) null. Obfuscated fires well above the clean
        // S_glob tail (desync raises it ~25×, so a wide 2.5× margin over clean p95 still separates
        // packed's mild ~1.1 S_glob); packed needs S_spat above the clean p95 spatial tail. Fall
        // back to sane absolutes if there are too few clean bins to estimate a percentile.
        let glob_hi = if clean_glob.len() >= 3 { percentile(&clean_glob, 0.95) * 2.5 } else { 2.0 };
        let spat_hi = if clean_spat.len() >= 3 { percentile(&clean_spat, 0.95) } else { 0.12 };
        eprintln!("  rule thresholds: glob_hi={glob_hi:.3} (obf if S_glob>this)  spat_hi={spat_hi:.4} (packed if S_spat>this)");

        // Abstention-guard threshold, trained GT-free from the fit split. Packed payloads are
        // compressed → near-max region entropy; real code (benign / desync / obfuscated) sits far
        // lower. If the two pools separate, put the threshold at the midpoint of the gap. If they
        // overlap, entropy cannot gate packed from real code without hurting one of them — set NaN
        // so the guard is inert and the honest failure is visible in the log and the report.
        let (pack_ent_lo, ent_gap) = if packed_ent.is_empty() || nonpacked_ent.is_empty() {
            (f64::NAN, f64::NAN)
        } else {
            let np_hi = nonpacked_ent.iter().cloned().fold(f64::MIN, f64::max);
            let pk_lo = packed_ent.iter().cloned().fold(f64::MAX, f64::min);
            if pk_lo > np_hi {
                ((np_hi + pk_lo) / 2.0, pk_lo - np_hi)
            } else {
                (f64::NAN, pk_lo - np_hi)
            }
        };
        let np_hi = nonpacked_ent.iter().cloned().fold(f64::MIN, f64::max);
        let pk_lo = packed_ent.iter().cloned().fold(f64::MAX, f64::min);
        eprintln!(
            "  abstention guard: non-packed region-entropy max={np_hi:.3}, packed min={pk_lo:.3}, gap={ent_gap:.3} \
             ⇒ pack_ent_lo={pack_ent_lo:.3} (packed route requires region_ent>this)"
        );

        Ok(SignatureClassifier {
            centroids, present, mu_g, sd_g, mu_s, sd_s, raw_centroids, glob_hi, spat_hi, pack_ent_lo,
        })
    }

    /// Threshold-rule classify: benign-default signature rule. Obfuscated if the global surprise is
    /// clearly elevated; else packed if the spatial statistic exceeds the clean null; else benign.
    fn classify_rule(&self, s_glob: f64, s_spat: f64) -> Regime {
        if s_glob > self.glob_hi {
            Regime::Obfuscated
        } else if s_spat > self.spat_hi {
            Regime::Packed
        } else {
            Regime::Benign
        }
    }

    /// Guarded threshold-rule classify: the same rule, but the packed route now also demands the
    /// flagged region be genuinely packed-like (region entropy above `pack_ent_lo`). A binary whose
    /// spatial statistic is elevated but whose code is normal-entropy (a Tigress VM interpreter, a
    /// legitimate bytecode VM) falls through to benign — abstain — rather than being routed to the
    /// calibration-destroying packed-suppress map. If `pack_ent_lo` is NaN (fit split didn't
    /// separate) the guard is inert and this reduces to `classify_rule`.
    fn classify_guard(&self, s_glob: f64, s_spat: f64, region_ent: f64) -> Regime {
        // NaN threshold ⇒ guard inert ⇒ packed route fires on the spatial signal alone (old rule).
        let packed_like = self.pack_ent_lo.is_nan() || region_ent > self.pack_ent_lo;
        if s_glob > self.glob_hi {
            Regime::Obfuscated
        } else if s_spat > self.spat_hi && packed_like {
            Regime::Packed
        } else {
            Regime::Benign
        }
    }

    /// Classify a binary from its benign-engine (S_glob, S_spat) signature.
    fn classify(&self, s_glob: f64, s_spat: f64) -> Regime {
        let g = ((s_glob.max(LN_EPS)).ln() - self.mu_g) / self.sd_g;
        let s = ((s_spat.max(LN_EPS)).ln() - self.mu_s) / self.sd_s;
        let mut best = Regime::Benign;
        let mut best_d = f64::INFINITY;
        for r in Regime::ALL {
            if !self.present[r.idx()] {
                continue;
            }
            let (cg, cs) = self.centroids[r.idx()];
            let d = (g - cg).powi(2) + (s - cs).powi(2);
            if d < best_d {
                best_d = d;
                best = r;
            }
        }
        best
    }
}

// ── The three-arm held-out evaluation for one binary ───────────────────────────

/// Run one held-out binary under all three engine configs, grade the four arms, and record the
/// GT-free selections. The three engine runs are sequential — each converged graph is freed before
/// the next, so peak memory is a single binary's factor graph (the standing memory guarantee).
fn evaluate_holdout(
    job: &HoldoutJob,
    bank: &Bank,
    clf: &SignatureClassifier,
    entropy: f64,
    chainfwd: f64,
    dump: Option<&mut fs::File>,
) -> Result<Record> {
    let bytes = fs::read(&job.bin).with_context(|| format!("reading {}", job.bin.display()))?;
    let (base, code) = extract_text(&bytes)?;

    // Run each config's engine; keep only the (small) posterior + aggregate stats we need.
    let mut posts: [Vec<(u64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut s_glob = [0.0f64; 3];
    let mut s_spat = [0.0f64; 3];
    let mut nis = [0.0f64; 3];
    for r in Regime::ALL {
        let (ent, cfw) = r.engine(entropy, chainfwd);
        let (post, cav) = run_soft_with_cavity_cfg(base, code, ent, cfw, false)
            .with_context(|| format!("engine[{}] on {}", r.tag(), job.name))?;
        let s = global_and_spatial(&cav);
        s_glob[r.idx()] = s.mean_surprise;
        s_spat[r.idx()] = s.moran;
        nis[r.idx()] = s.mean_nis;
        posts[r.idx()] = post;
        // cav dropped here.
    }

    // Grade a (posteriors, applied-map) pair against this binary's ground truth. GT type depends on
    // the binary's true regime — never on which map we applied.
    let grade = |r_config: Regime| -> Result<f64> {
        let cal = bank.map(r_config).apply_all(&posts[r_config.idx()]);
        grade_ece(job, &cal)
    };

    // (a) always-benign: benign engine + benign map.
    let ece_always_benign = grade(Regime::Benign)?;
    // (b) oracle: the binary's true regime config.
    let ece_oracle = grade(job.regime)?;

    // (c1) MMAE-direct: lowest consistency S across configs (S_glob primary; NIS variant tracked).
    let mmae_pick = argmin_regime(&s_glob);
    let mmae_nis_pick = argmin_regime(&nis);
    let ece_mmae = grade(mmae_pick)?;

    // (c2) signature-classifier: nearest-centroid on the benign-engine signature.
    let sig_glob = s_glob[Regime::Benign.idx()];
    let sig_spat = s_spat[Regime::Benign.idx()];
    let clf_pick = clf.classify(sig_glob, sig_spat);
    let ece_clf = grade(clf_pick)?;
    // (c3) signature-classifier, benign-default threshold-rule variant.
    let rule_pick = clf.classify_rule(sig_glob, sig_spat);
    let ece_rule = grade(rule_pick)?;
    // (c4) abstention-guarded rule: packed route gated behind region entropy.
    let region_ent = region_entropy(code);
    let guard_pick = clf.classify_guard(sig_glob, sig_spat, region_ent);
    let ece_guard = grade(guard_pick)?;

    // Selective-disassembly dump: per-arm calibrated posterior of every candidate inside the packer's
    // provable-data window. That window is all data, so any address an arm calls code is a fabricated
    // head — this is the raw material for the requested-vs-achieved precision sweep, done offline.
    if let Some(f) = dump {
        if job.regime == Regime::Packed {
            if let Some(gt) = job.packed_gt.as_ref() {
                if let Ok((lo, hi)) = packed_data_window(gt) {
                    // Same (map, engine-posteriors) pairing grade() uses: apply regime r's map to
                    // r's own engine run. stale=benign, oracle=true regime, switch_*=rule/guard pick.
                    for (arm, pick) in [
                        ("stale", Regime::Benign),
                        ("oracle", job.regime),
                        ("switch_rule", rule_pick),
                        ("switch_guard", guard_pick),
                    ] {
                        let cal = bank.map(pick).apply_all(&posts[pick.idx()]);
                        for &(a, p) in cal.iter().filter(|&&(a, _)| a >= lo && a < hi) {
                            writeln!(f, "{},{},{},{},{:#x},{:.6}", job.name, job.sublabel, arm, pick.tag(), a, p)?;
                        }
                    }
                }
            }
        }
    }

    // base_rate / n for the report (benign-engine candidate set).
    let (base_rate, n) = match job.regime {
        Regime::Packed => (0.0, posts[Regime::Benign.idx()].len()),
        _ => {
            let gt = load_gt(job.gt.as_ref().unwrap())?;
            let bp = &posts[Regime::Benign.idx()];
            let br = bp.iter().filter(|&&(a, _)| gt.contains(&a)).count() as f64 / bp.len().max(1) as f64;
            (br, bp.len())
        }
    };

    Ok(Record {
        name: job.name.clone(),
        regime: job.regime,
        sublabel: job.sublabel.clone(),
        n,
        code_bytes: code.len(),
        base_rate,
        ece_always_benign,
        ece_oracle,
        ece_mmae,
        ece_clf,
        ece_rule,
        ece_guard,
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
    })
}

/// True post-hoc ECE of an already-calibrated posterior set against this binary's GT. Instruction-
/// start ECE for benign/obfuscated; mean-over-the-provable-data-window for packed (every address in
/// the window is a negative, so ECE against the all-zero label = mean calibrated posterior there).
fn grade_ece(job: &HoldoutJob, cal: &[(u64, f64)]) -> Result<f64> {
    match job.regime {
        Regime::Packed => {
            let (lo, hi) = packed_data_window(job.packed_gt.as_ref().unwrap())?;
            let ps: Vec<f64> = cal.iter().filter(|&&(a, _)| a >= lo && a < hi).map(|&(_, p)| p).collect();
            Ok(if ps.is_empty() { 0.0 } else { ps.iter().sum::<f64>() / ps.len() as f64 })
        }
        _ => {
            let gt = load_gt(job.gt.as_ref().unwrap())?;
            Ok(evaluate(cal, &gt).ece)
        }
    }
}

/// Index of the regime with the smallest statistic (the MMAE "most consistent" pick).
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

// ── Aggregated cavity statistics (global magnitude + spatial clustering) ────────

struct GlobalSpatial {
    mean_surprise: f64,
    mean_nis: f64,
    moran: f64,
}

/// The two GT-free statistics from the address-sorted cavity stats: `S_glob` (mean surprise, and
/// mean NIS) and `S_spat` (Moran's I of the standardized residual over address-order adjacency).
/// Mirrors the definitions in the credibility `consistency` binary so the two can't drift.
fn global_and_spatial(cav: &[(u64, CavityStat)]) -> GlobalSpatial {
    let n = cav.len();
    if n == 0 {
        return GlobalSpatial { mean_surprise: 0.0, mean_nis: 0.0, moran: 0.0 };
    }
    let mean_surprise = cav.iter().map(|(_, c)| c.surprise).sum::<f64>() / n as f64;
    let mean_nis = cav.iter().map(|(_, c)| c.nis).sum::<f64>() / n as f64;
    let resid: Vec<f64> = cav.iter().map(|(_, c)| c.residual).collect();
    GlobalSpatial { mean_surprise, mean_nis, moran: morans_i_line(&resid) }
}

/// Moran's I with a 1-D contiguity weight (neighbors = adjacent in the ordered vector).
fn morans_i_line(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 3 {
        return 0.0;
    }
    let mbar = x.iter().sum::<f64>() / n as f64;
    let denom: f64 = x.iter().map(|v| (v - mbar).powi(2)).sum();
    if denom <= 0.0 {
        return 0.0;
    }
    let mut num = 0.0;
    for i in 0..n - 1 {
        num += (x[i] - mbar) * (x[i + 1] - mbar);
    }
    (n as f64 / (n as f64 - 1.0)) * (num / denom)
}

/// Region entropy: the max sliding-window Shannon byte-entropy (bits/byte) over the code region.
/// The abstention guard's discriminator. A UPX payload is compressed data — near-max entropy
/// (measured 7.69–7.86 on the packed corpus); real machine code, including a Tigress VM interpreter
/// or a hand-written bytecode VM, sits far lower (≤6.33 across benign / desync / Tigress / legit-VM).
/// We take the *max* window, not the whole-region mean, so a small high-entropy blob embedded in real
/// code still reads as packed-like, and a decompressor stub prepended to a payload doesn't dilute the
/// payload's signal. Window 1 KiB, step ½ KiB; regions shorter than a window use the whole-region
/// entropy.
fn region_entropy(code: &[u8]) -> f64 {
    const WIN: usize = 1024;
    const STEP: usize = 512;
    if code.len() <= WIN {
        return byte_entropy(code);
    }
    let mut best = 0.0f64;
    let mut i = 0;
    while i + WIN <= code.len() {
        let e = byte_entropy(&code[i..i + WIN]);
        if e > best {
            best = e;
        }
        i += STEP;
    }
    best
}

/// Shannon entropy of a byte slice in bits/byte (0..=8).
fn byte_entropy(b: &[u8]) -> f64 {
    if b.is_empty() {
        return 0.0;
    }
    let mut counts = [0u32; 256];
    for &x in b {
        counts[x as usize] += 1;
    }
    let n = b.len() as f64;
    let mut h = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / n;
            h -= p * p.log2();
        }
    }
    h
}

fn mean_std(x: &[f64]) -> (f64, f64) {
    if x.is_empty() {
        return (0.0, 0.0);
    }
    let m = x.iter().sum::<f64>() / x.len() as f64;
    let v = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64;
    (m, v.sqrt())
}

fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().sum::<f64>() / x.len() as f64
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// `q`-quantile of `x` (0..=1), nearest-rank on a sorted copy.
fn percentile(x: &[f64], q: f64) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let mut s = x.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((s.len() as f64 - 1.0) * q).round() as usize;
    s[idx.min(s.len() - 1)]
}

// ── Verdict ────────────────────────────────────────────────────────────────────

struct Summary {
    n_holdout: usize,
    /// Per-regime arm means: [always_benign, oracle, mmae, clf, rule, guard].
    per_regime: Vec<(Regime, usize, [f64; 6])>,
    /// Selection accuracy overall and per regime.
    sel_mmae: f64,
    sel_mmae_nis: f64,
    sel_clf: f64,
    sel_rule: f64,
    sel_guard: f64,
    per_regime_sel: Vec<(Regime, f64, f64, f64, f64, f64)>, // (regime, mmae, mmae_nis, clf, rule, guard)
    /// Tigress blind-spot rows (true regime obfuscated, sublabel tig*) reported separately.
    tig_arm_means: Option<[f64; 6]>,
    tig_sel_rule: f64,
    tig_sel_guard: f64,
    tig_n: usize,
    /// Legitimate-VM false-positive gate (sublabel vm*): true regime benign; correct action = abstain.
    vm_arm_means: Option<[f64; 6]>,
    vm_sel_rule: f64,
    vm_sel_guard: f64,
    vm_n: usize,
    /// Recovery fraction per obfuscated/packed regime: (a→c) / (a→b), for rule and guard.
    recovery: Vec<(Regime, f64, f64)>, // (regime, rule_recovery, guard_recovery)
    classifier_centroids: [(f64, f64); 3],
    classifier_present: [bool; 3],
    glob_hi: f64,
    spat_hi: f64,
    pack_ent_lo: f64,
}

fn analyze(records: &[Record], clf: &SignatureClassifier) -> Summary {
    let arms = |rs: &[&Record]| -> [f64; 6] {
        [
            mean(&rs.iter().map(|r| r.ece_always_benign).collect::<Vec<_>>()),
            mean(&rs.iter().map(|r| r.ece_oracle).collect::<Vec<_>>()),
            mean(&rs.iter().map(|r| r.ece_mmae).collect::<Vec<_>>()),
            mean(&rs.iter().map(|r| r.ece_clf).collect::<Vec<_>>()),
            mean(&rs.iter().map(|r| r.ece_rule).collect::<Vec<_>>()),
            mean(&rs.iter().map(|r| r.ece_guard).collect::<Vec<_>>()),
        ]
    };

    // Per-regime arm means. Tigress (sublabel "tig*") and the legit-VM gate (sublabel "vm*") are
    // split out — Tigress from the obfuscated group (the known blind spot), the legit VMs from the
    // benign group (the FP gate) — so neither dilutes a regime headline.
    let is_tig = |r: &Record| r.sublabel.starts_with("tig");
    let is_vm = |r: &Record| r.sublabel.starts_with("vm");
    let is_special = |r: &Record| is_tig(r) || is_vm(r);
    let mut per_regime = Vec::new();
    let mut per_regime_sel = Vec::new();
    let mut recovery = Vec::new();
    for reg in Regime::ALL {
        let rs: Vec<&Record> = records
            .iter()
            .filter(|r| r.regime == reg && !is_special(r))
            .collect();
        if rs.is_empty() {
            continue;
        }
        let a = arms(&rs);
        per_regime.push((reg, rs.len(), a));
        let sel_m = frac(&rs, |r| r.mmae_pick == r.regime);
        let sel_mn = frac(&rs, |r| r.mmae_nis_pick == r.regime);
        let sel_c = frac(&rs, |r| r.clf_pick == r.regime);
        let sel_r = frac(&rs, |r| r.rule_pick == r.regime);
        let sel_g = frac(&rs, |r| r.guard_pick == r.regime);
        per_regime_sel.push((reg, sel_m, sel_mn, sel_c, sel_r, sel_g));
        // Recovery only meaningful where always-benign is genuinely stale (a > b).
        if reg != Regime::Benign {
            let denom = a[0] - a[1];
            let rec_r = if denom.abs() > 1e-9 { (a[0] - a[4]) / denom } else { 0.0 };
            let rec_g = if denom.abs() > 1e-9 { (a[0] - a[5]) / denom } else { 0.0 };
            recovery.push((reg, rec_r, rec_g));
        }
    }

    // Tigress block (held-out limit) — sel accuracy against the true (obfuscated) regime.
    let tig: Vec<&Record> = records.iter().filter(|r| is_tig(r)).collect();
    let (tig_arm_means, tig_sel_rule, tig_sel_guard, tig_n) = if tig.is_empty() {
        (None, 0.0, 0.0, 0)
    } else {
        (Some(arms(&tig)), frac(&tig, |r| r.rule_pick == r.regime), frac(&tig, |r| r.guard_pick == r.regime), tig.len())
    };

    // Legit-VM FP gate — true regime benign; sel accuracy = fraction correctly kept benign (abstained).
    let vm: Vec<&Record> = records.iter().filter(|r| is_vm(r)).collect();
    let (vm_arm_means, vm_sel_rule, vm_sel_guard, vm_n) = if vm.is_empty() {
        (None, 0.0, 0.0, 0)
    } else {
        (Some(arms(&vm)), frac(&vm, |r| r.rule_pick == Regime::Benign), frac(&vm, |r| r.guard_pick == Regime::Benign), vm.len())
    };

    // Overall selection accuracy (exclude the split-out blind-spot / FP-gate rows from the headline).
    let core: Vec<&Record> = records.iter().filter(|r| !is_special(r)).collect();
    let sel_mmae = frac(&core, |r| r.mmae_pick == r.regime);
    let sel_mmae_nis = frac(&core, |r| r.mmae_nis_pick == r.regime);
    let sel_clf = frac(&core, |r| r.clf_pick == r.regime);
    let sel_rule = frac(&core, |r| r.rule_pick == r.regime);
    let sel_guard = frac(&core, |r| r.guard_pick == r.regime);

    Summary {
        n_holdout: records.len(),
        per_regime,
        sel_mmae,
        sel_mmae_nis,
        sel_clf,
        sel_rule,
        sel_guard,
        per_regime_sel,
        tig_arm_means,
        tig_sel_rule,
        tig_sel_guard,
        tig_n,
        vm_arm_means,
        vm_sel_rule,
        vm_sel_guard,
        vm_n,
        recovery,
        classifier_centroids: clf.raw_centroids,
        classifier_present: clf.present,
        glob_hi: clf.glob_hi,
        spat_hi: clf.spat_hi,
        pack_ent_lo: clf.pack_ent_lo,
    }
}

fn frac(v: &[&Record], pred: impl Fn(&Record) -> bool) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().filter(|r| pred(r)).count() as f64 / v.len() as f64
}

impl Summary {
    fn print(&self) {
        println!("\n══════════════════ CONSISTENCY SWITCHING ══════════════════");
        println!("held-out binaries: {}", self.n_holdout);
        println!("\n— Arm ECE by regime (core; Tigress + legit-VM split out) —");
        println!("  regime        n   always-benign   oracle    rule(default)  guard(abstain)");
        for (reg, n, a) in &self.per_regime {
            println!(
                "  {:<11} {:>3}   {:>10.4}   {:>7.4}  {:>10.4}  {:>10.4}",
                reg.tag(), n, a[0], a[1], a[4], a[5]
            );
        }
        println!("\n— Selection accuracy (GT-free rule picks the true regime) —");
        println!("  overall: MMAE(S_glob)={:.2}  clf-centroid={:.2}  rule-default={:.2}  guard={:.2}",
            self.sel_mmae, self.sel_clf, self.sel_rule, self.sel_guard);
        for (reg, m, _mn, c, r, g) in &self.per_regime_sel {
            println!("    {:<11} MMAE={:.2}  clf={:.2}  rule={:.2}  guard={:.2}", reg.tag(), m, c, r, g);
        }
        println!("\n— Recovery fraction (a→c)/(a→b), where always-benign is stale —");
        for (reg, r, g) in &self.recovery {
            println!("  {:<11} rule-default={:+.2}  guard={:+.2}", reg.tag(), r, g);
        }
        if let Some(a) = self.tig_arm_means {
            println!("\n— Tigress blind-spot (held-out; n={}) —", self.tig_n);
            println!("  always-benign={:.4}  oracle={:.4}  rule={:.4}  guard={:.4}  (rule→benign sel={:.2}  guard→benign sel={:.2})",
                a[0], a[1], a[4], a[5], self.tig_sel_rule, self.tig_sel_guard);
        }
        if let Some(a) = self.vm_arm_means {
            println!("\n— Legit-VM FP gate (held-out; n={}) —", self.vm_n);
            println!("  always-benign={:.4}  rule={:.4}  guard={:.4}  (rule abstained {:.2}  guard abstained {:.2})",
                a[0], a[4], a[5], self.vm_sel_rule, self.vm_sel_guard);
        }
        println!("\n— Rule thresholds —  glob_hi={:.3} (S_glob>this ⇒ obf)  spat_hi={:.4} (S_spat>this ⇒ packed)  pack_ent_lo={:.3} (guard: region_ent>this required for packed)",
            self.glob_hi, self.spat_hi, self.pack_ent_lo);
        println!("\n— Classifier centroids (mean ln S_glob, ln S_spat) —");
        for r in Regime::ALL {
            if self.classifier_present[r.idx()] {
                let (g, s) = self.classifier_centroids[r.idx()];
                println!("  {:<11} ({:+.3}, {:+.3})", r.tag(), g, s);
            }
        }
        println!("═══════════════════════════════════════════════════════════");
    }

    fn to_json(&self, records: &[Record]) -> String {
        let mut s = String::from("{\n");
        s.push_str(&format!("  \"n_holdout\": {},\n", self.n_holdout));
        s.push_str(&format!("  \"sel_mmae_sglob\": {:.4},\n", self.sel_mmae));
        s.push_str(&format!("  \"sel_mmae_nis\": {:.4},\n", self.sel_mmae_nis));
        s.push_str(&format!("  \"sel_classifier_centroid\": {:.4},\n", self.sel_clf));
        s.push_str(&format!("  \"sel_classifier_rule\": {:.4},\n", self.sel_rule));
        s.push_str(&format!("  \"sel_guard\": {:.4},\n", self.sel_guard));
        s.push_str(&format!("  \"rule_glob_hi\": {:.4},\n", self.glob_hi));
        s.push_str(&format!("  \"rule_spat_hi\": {:.4},\n", self.spat_hi));
        s.push_str(&format!("  \"pack_ent_lo\": {:.4},\n", self.pack_ent_lo));
        // Per-regime arm means + medians.
        for (reg, n, a) in &self.per_regime {
            let rs: Vec<&Record> = records.iter().filter(|r| r.regime == *reg && !r.sublabel.starts_with("tig") && !r.sublabel.starts_with("vm")).collect();
            s.push_str(&format!(
                "  \"regime_{}\": {{\"n\": {}, \"ece_always_benign\": {:.4}, \"ece_oracle\": {:.4}, \"ece_rule\": {:.4}, \"ece_guard\": {:.4}, \"med_always_benign\": {:.4}, \"med_oracle\": {:.4}, \"med_rule\": {:.4}, \"med_guard\": {:.4}}},\n",
                reg.tag(), n, a[0], a[1], a[4], a[5],
                median(rs.iter().map(|r| r.ece_always_benign).collect()),
                median(rs.iter().map(|r| r.ece_oracle).collect()),
                median(rs.iter().map(|r| r.ece_rule).collect()),
                median(rs.iter().map(|r| r.ece_guard).collect()),
            ));
        }
        for (reg, m, mn, c, r, g) in &self.per_regime_sel {
            s.push_str(&format!(
                "  \"sel_{}\": {{\"mmae\": {:.4}, \"mmae_nis\": {:.4}, \"clf\": {:.4}, \"rule\": {:.4}, \"guard\": {:.4}}},\n",
                reg.tag(), m, mn, c, r, g
            ));
        }
        for (reg, r, g) in &self.recovery {
            s.push_str(&format!(
                "  \"recovery_{}\": {{\"rule\": {:.4}, \"guard\": {:.4}}},\n",
                reg.tag(), r, g
            ));
        }
        if let Some(a) = self.tig_arm_means {
            s.push_str(&format!(
                "  \"tigress\": {{\"n\": {}, \"ece_always_benign\": {:.4}, \"ece_oracle\": {:.4}, \"ece_rule\": {:.4}, \"ece_guard\": {:.4}, \"rule_benign_sel\": {:.4}, \"guard_benign_sel\": {:.4}}},\n",
                self.tig_n, a[0], a[1], a[4], a[5], self.tig_sel_rule, self.tig_sel_guard
            ));
        }
        if let Some(a) = self.vm_arm_means {
            s.push_str(&format!(
                "  \"legit_vm\": {{\"n\": {}, \"ece_always_benign\": {:.4}, \"ece_rule\": {:.4}, \"ece_guard\": {:.4}, \"rule_abstain\": {:.4}, \"guard_abstain\": {:.4}}},\n",
                self.vm_n, a[0], a[4], a[5], self.vm_sel_rule, self.vm_sel_guard
            ));
        }
        s.push_str("  \"_end\": true\n}\n");
        s
    }
}

// ── Corpus assembly / IO helpers ───────────────────────────────────────────────

fn file_stem(p: &Path) -> String {
    p.file_name().and_then(|s| s.to_str()).unwrap_or("?").to_string()
}

fn list_bins_with_gt(bins_dir: &Path, gt_dir: &Path) -> Result<Vec<(String, PathBuf, PathBuf)>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(bins_dir).with_context(|| format!("reading dir {}", bins_dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = file_stem(&path);
        if name.ends_with(".gt") || name.ends_with(".log") || name.ends_with(".json") || name.starts_with('.') {
            continue;
        }
        let gt = gt_dir.join(format!("{name}.gt"));
        if gt.is_file() {
            out.push((name, path, gt));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

fn deterministic_order(
    mut v: Vec<(String, PathBuf, PathBuf)>,
    seed: u64,
) -> Vec<(String, PathBuf, PathBuf)> {
    v.sort_by_key(|(name, _, _)| splitmix64(seed ^ fnv1a(name.as_bytes())));
    v
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Parse the provable-data (NEGATIVE) vaddr window out of a `.upxgt` label table — UPX's own
/// `b_info` chain, not a disassembler.
fn packed_data_window(upxgt: &Path) -> Result<(u64, u64)> {
    let text = fs::read_to_string(upxgt).with_context(|| format!("reading {}", upxgt.display()))?;
    for line in text.lines() {
        let l = line.trim();
        if l.starts_with('#') || l.is_empty() {
            continue;
        }
        let cols: Vec<&str> = l.split_whitespace().collect();
        if cols.first() == Some(&"compressed") && cols.len() >= 4 {
            return Ok((parse_hex(cols[2])?, parse_hex(cols[3])?));
        }
    }
    bail!("no `compressed` NEGATIVE row in {}", upxgt.display())
}

fn parse_hex(s: &str) -> Result<u64> {
    let t = s.trim().trim_start_matches("0x");
    u64::from_str_radix(t, 16).with_context(|| format!("bad hex {s}"))
}

// ── CSV resume ─────────────────────────────────────────────────────────────────

fn read_existing_csv(path: &Path) -> HashMap<String, Record> {
    let mut map = HashMap::new();
    let Ok(text) = fs::read_to_string(path) else {
        return map;
    };
    for line in text.lines().skip(1) {
        if let Some(rec) = Record::from_csv(line) {
            map.insert(format!("{}|{}", rec.name, rec.sublabel), rec);
        }
    }
    map
}

fn open_csv_append(path: &Path) -> Result<fs::File> {
    let exists = path.exists();
    let mut f = fs::OpenOptions::new().create(true).append(true).open(path)?;
    if !exists {
        writeln!(f, "{}", Record::CSV_HEADER)?;
    }
    Ok(f)
}

// ── CLI ────────────────────────────────────────────────────────────────────────

struct Args {
    clean_bins: PathBuf,
    clean_gt: PathBuf,
    desync_levels: Vec<(String, PathBuf, PathBuf)>,
    packed_specs: Vec<(String, PathBuf, PathBuf)>,
    packed_holdout: Vec<(String, PathBuf, PathBuf)>,
    tigress_levels: Vec<(String, PathBuf, PathBuf)>,
    benign_holdout: Vec<(String, PathBuf, PathBuf)>,
    n_clean_fit: usize,
    n_clean_holdout: usize,
    n_desync_fit: usize,
    n_desync_holdout: usize,
    n_packed_fit: usize,
    n_packed_holdout: usize,
    n_tig_holdout: usize,
    entropy_strength: f64,
    chainfwd_strength: f64,
    seed: u64,
    out: PathBuf,
    summary: PathBuf,
    /// Optional per-address dump for the selective-disassembly precision demo: for every packed
    /// held-out binary, writes the calibrated posterior of every candidate inside the packer's
    /// provable-data window under each arm (stale/oracle/switch_rule/switch_guard). Every such
    /// candidate is provable data, so any address the arm calls code there is a fabricated head.
    selective_dump: Option<PathBuf>,
}

impl Args {
    fn parse(mut it: impl Iterator<Item = String>) -> Result<Self> {
        const USAGE: &str = "usage: switching --clean-bins DIR --clean-gt DIR \
[--desync-level LABEL BINS GT]... [--packed-spec LABEL ELF UPXGT]... \
[--tigress-level LABEL BINS GT]... [--benign-holdout LABEL BINS GT]... --n-clean-fit N --n-clean-holdout N \
--n-desync-fit N --n-desync-holdout N --n-packed-fit N --n-packed-holdout N --n-tig-holdout N \
[--entropy-strength F] [--chainfwd-strength F] [--seed S] --out CSV --summary JSON";
        let mut clean_bins = None;
        let mut clean_gt = None;
        let mut desync_levels = Vec::new();
        let mut packed_specs = Vec::new();
        let mut packed_holdout = Vec::new();
        let mut tigress_levels = Vec::new();
        let mut benign_holdout = Vec::new();
        let mut n_clean_fit = 20usize;
        let mut n_clean_holdout = 25usize;
        let mut n_desync_fit = 40usize;
        let mut n_desync_holdout = 30usize;
        let mut n_packed_fit = 9usize;
        let mut n_packed_holdout = 8usize;
        let mut n_tig_holdout = 27usize;
        let mut entropy_strength = 1.0f64;
        let mut chainfwd_strength = 0.5f64;
        let mut seed = 1u64;
        let mut out = None;
        let mut summary = None;
        let mut selective_dump = None;
        while let Some(a) = it.next() {
            let mut next = |flag: &str| it.next().with_context(|| format!("{flag} needs a value"));
            match a.as_str() {
                "--clean-bins" => clean_bins = Some(PathBuf::from(next("--clean-bins")?)),
                "--clean-gt" => clean_gt = Some(PathBuf::from(next("--clean-gt")?)),
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
                "--packed-holdout" => {
                    let label = next("--packed-holdout")?;
                    let elf = PathBuf::from(next("--packed-holdout elf")?);
                    let gt = PathBuf::from(next("--packed-holdout gt")?);
                    packed_holdout.push((label, elf, gt));
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
                "--entropy-strength" => entropy_strength = next("--entropy-strength")?.parse()?,
                "--chainfwd-strength" => chainfwd_strength = next("--chainfwd-strength")?.parse()?,
                "--seed" => seed = next("--seed")?.parse()?,
                "--out" => out = Some(PathBuf::from(next("--out")?)),
                "--summary" => summary = Some(PathBuf::from(next("--summary")?)),
                "--selective-dump" => selective_dump = Some(PathBuf::from(next("--selective-dump")?)),
                "-h" | "--help" => {
                    eprintln!("{USAGE}");
                    std::process::exit(0);
                }
                other => bail!("unexpected argument: {other}\n{USAGE}"),
            }
        }
        Ok(Args {
            clean_bins: clean_bins.context(USAGE)?,
            clean_gt: clean_gt.context(USAGE)?,
            desync_levels,
            packed_specs,
            packed_holdout,
            tigress_levels,
            benign_holdout,
            n_clean_fit,
            n_clean_holdout,
            n_desync_fit,
            n_desync_holdout,
            n_packed_fit,
            n_packed_holdout,
            n_tig_holdout,
            entropy_strength,
            chainfwd_strength,
            seed,
            out: out.context(USAGE)?,
            summary: summary.context(USAGE)?,
            selective_dump,
        })
    }
}
