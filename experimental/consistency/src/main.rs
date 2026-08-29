//! `consistency` — the Paper-2 go/no-go gate.
//!
//! The question: can the cavity-belief *surprise* statistic `s_a` detect that a calibration map
//! fit on clean code has gone stale on obfuscated code — *without ever seeing ground truth*? If a
//! GT-free aggregate of `s_a` tracks the true (GT-needing) post-hoc ECE, fires on drifted binaries,
//! and stays quiet on clean ones, Paper 2 is real. If not, we say so and pivot to Paper 1.
//!
//! What this binary does, per the spec:
//!   1. Runs the Soft engine over each binary's `.text`, pulling the read-only cavity surprise
//!      (`evalkit::run_soft_with_cavity`; the engine proves the pass leaves π byte-identical).
//!   2. Per binary computes the two GT-free aggregates:
//!        - S_glob : mean per-address surprise (and mean NIS) — a magnitude, calibrated against an
//!                   empirical clean null (clean-fit 95th percentile).
//!        - S_spat : spatial clustering of the standardized residuals — Moran's I over address-order
//!                   adjacency, and max-run-length of super-threshold residuals.
//!   3. Fits an isotonic calibration map on a set of CLEAN binaries.
//!   4. On clean-holdout + obfuscated binaries with GT, computes the *true* post-hoc ECE of that
//!      clean-fit map, and asks whether the GT-free S tracks it: Spearman ρ, detector-ROC AUC,
//!      clean false-alarm rate. Confirms the expected conservative bias (loopy-cavity contamination
//!      shrinks residuals) and rules out the trivial confounds (region entropy, size).
//!
//! Standing rules honored: one binary in memory at a time; `--jobs 1` (single-threaded); rows
//! streamed to CSV and resumable (an existing CSV row for a binary is reused, not recomputed); GT
//! read only from the pre-supplied `.gt` files (symbol/objdump-on-*original* provenance — never a
//! disassembly of the obfuscated input under test).
//!
//! ```text
//! consistency \
//!   --clean-bins DIR --clean-gt DIR \
//!   --desync-bins DIR --desync-gt DIR \
//!   [--packed ELF --packed-gt UPXGT] \
//!   --n-fit N --n-holdout N --n-desync N \
//!   --out results.csv --summary summary.json
//! ```

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use evalkit::{auroc, evaluate, load_gt, run_soft_with_cavity, IsotonicMap};
use probdisasm::{extract_text_section as extract_text, CavityStat};

/// Which corpus a binary belongs to, and its role in the experiment.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Role {
    /// Clean binary used to FIT the isotonic map and to define the empirical null.
    CleanFit,
    /// Clean binary held out — measures false-alarm rate (map should stay calibrated here).
    CleanHoldout,
    /// Obfuscated: desync-cc dense. Real drift with instruction-start GT.
    Desync,
    /// Packed (UPX): the extreme. GT is the provably-data payload window (all negatives).
    Packed,
}

impl Role {
    fn tag(self) -> &'static str {
        match self {
            Role::CleanFit => "clean_fit",
            Role::CleanHoldout => "clean_holdout",
            Role::Desync => "desync",
            Role::Packed => "packed",
        }
    }
    /// Is this an obfuscated/adversarial binary (the thing we hope to detect)?
    fn is_obfuscated(self) -> bool {
        matches!(self, Role::Desync | Role::Packed)
    }
}

/// One binary queued for evaluation.
struct Job {
    name: String,
    bin: PathBuf,
    /// `.gt` for Desync/Clean (hex instruction-starts). `None` for Packed (window comes from GT file).
    gt: Option<PathBuf>,
    /// Packed only: the `.upxgt` label table (provable-data window).
    packed_gt: Option<PathBuf>,
    role: Role,
    /// Drift-intensity label for the graded corpus (e.g. "pilot","d1_med","d3_max","upx_best").
    /// Empty for clean bins. Lets one run cover the whole density ladder with a shared map.
    level: String,
}

/// The small per-binary record we keep in memory (aggregates only — never per-address).
#[derive(Clone, Debug)]
struct Record {
    name: String,
    role: Role,
    n: usize,
    base_rate: f64,
    code_bytes: usize,
    region_entropy: f64,
    /// True post-hoc ECE of the clean-fit calibration map on this binary (needs GT).
    ece_calibrated: f64,
    /// Raw (pre-calibration) ECE, for context.
    ece_raw: f64,
    /// GT-free global magnitude.
    s_glob_surprise: f64,
    s_glob_nis: f64,
    /// GT-free spatial clustering.
    s_spat_moran: f64,
    s_spat_clustered: f64,
    /// Diagnostics for the conservative-bias check.
    mean_nis: f64,
    frac_super: f64,
    /// Off-the-shelf OOD / calibration-drift baselines (the "isn't this just OOD detection?"
    /// defense). All GT-free, all scalar, none localizing. Computed on the same per-address π/φ
    /// the cavity statistics use, so the comparison is apples-to-apples.
    b_mean_pi: f64,      // mean posterior π_a
    b_pred_entropy: f64, // mean binary predictive entropy H(π_a)
    b_msp: f64,          // mean max-softmax-prob max(π_a, 1−π_a) (Hendrycks MSP OOD score)
    b_mean_abs_llr: f64, // mean |φ_a| local decode logit — a temperature/confidence proxy
    /// Graded-corpus drift-intensity label (empty for clean). Appended last for CSV back-compat.
    level: String,
}

impl Record {
    const CSV_HEADER: &'static str = "name,role,n,base_rate,code_bytes,region_entropy,\
ece_calibrated,ece_raw,s_glob_surprise,s_glob_nis,s_spat_moran,s_spat_clustered,mean_nis,frac_super,\
b_mean_pi,b_pred_entropy,b_msp,b_mean_abs_llr,level";

    fn to_csv(&self) -> String {
        format!(
            "{},{},{},{:.6},{},{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},\
{:.6},{:.6},{:.6},{:.6},{}",
            self.name,
            self.role.tag(),
            self.n,
            self.base_rate,
            self.code_bytes,
            self.region_entropy,
            self.ece_calibrated,
            self.ece_raw,
            self.s_glob_surprise,
            self.s_glob_nis,
            self.s_spat_moran,
            self.s_spat_clustered,
            self.mean_nis,
            self.frac_super,
            self.b_mean_pi,
            self.b_pred_entropy,
            self.b_msp,
            self.b_mean_abs_llr,
            self.level,
        )
    }
}

// ── Super-threshold residual definition (the spatial "innovation" event) ─────────
/// A standardized residual with |z| above this counts as a super-threshold event (~2σ).
const RESID_THR: f64 = 2.0;

fn main() -> Result<()> {
    let args = Args::parse(std::env::args().skip(1))?;

    // Strip-dump mode: emit per-address cavity stats for one binary (feeds the residual-cluster
    // figure). Runs a single binary and exits — independent of the main experiment.
    if let (Some(bin), Some(out)) = (&args.strip_bin, &args.strip_out) {
        return dump_strip(bin, args.strip_gt.as_deref(), out);
    }

    // ── Assemble the groups (deterministic; seed decorrelates from alphabetical order) ──
    let clean = list_bins_with_gt(&args.clean_bins, &args.clean_gt)?;
    let clean = deterministic_order(clean, args.seed);
    if clean.len() < args.n_fit + args.n_holdout {
        bail!(
            "need {} clean bins (fit {} + holdout {}), found {}",
            args.n_fit + args.n_holdout,
            args.n_fit,
            args.n_holdout,
            clean.len()
        );
    }
    let mut jobs: Vec<Job> = Vec::new();
    for (i, (name, bin, gt)) in clean.into_iter().enumerate() {
        let role = if i < args.n_fit {
            Role::CleanFit
        } else if i < args.n_fit + args.n_holdout {
            Role::CleanHoldout
        } else {
            continue; // extra clean bins unused
        };
        jobs.push(Job { name, bin, gt: Some(gt), packed_gt: None, role, level: String::new() });
    }
    // Desync levels: the single --desync-bins/--desync-gt pair (labeled "pilot") plus any repeatable
    // --desync-level LABEL BINDIR GTDIR. All share the one clean-fit map, so ECE is comparable across
    // the ladder. The graded set is the credibility experiment (subtle drift, non-saturated ROC).
    let mut levels: Vec<(String, PathBuf, PathBuf)> = Vec::new();
    if !args.desync_bins.as_os_str().is_empty() {
        levels.push(("pilot".into(), args.desync_bins.clone(), args.desync_gt.clone()));
    }
    levels.extend(args.desync_levels.iter().cloned());
    for (label, bins_dir, gt_dir) in &levels {
        let ds = list_bins_with_gt(bins_dir, gt_dir)
            .with_context(|| format!("desync level {label}: {}", bins_dir.display()))?;
        let ds = deterministic_order(ds, args.seed);
        for (name, bin, gt) in ds.into_iter().take(args.n_desync) {
            jobs.push(Job { name, bin, gt: Some(gt), packed_gt: None, role: Role::Desync, level: label.clone() });
        }
    }
    // Packed: single --packed/--packed-gt (labeled "upx") plus repeatable --packed-spec LABEL ELF GT.
    if let (Some(pb), Some(pg)) = (&args.packed, &args.packed_gt) {
        jobs.push(Job { name: file_stem(pb), bin: pb.clone(), gt: None,
            packed_gt: Some(pg.clone()), role: Role::Packed, level: "upx".into() });
    }
    for (label, elf, gt) in &args.packed_specs {
        jobs.push(Job { name: format!("{}__{}", file_stem(elf), label), bin: elf.clone(),
            gt: None, packed_gt: Some(gt.clone()), role: Role::Packed, level: label.clone() });
    }

    eprintln!(
        "queued {} binaries: {} fit / {} holdout / {} desync / {} packed",
        jobs.len(),
        jobs.iter().filter(|j| j.role == Role::CleanFit).count(),
        jobs.iter().filter(|j| j.role == Role::CleanHoldout).count(),
        jobs.iter().filter(|j| j.role == Role::Desync).count(),
        jobs.iter().filter(|j| j.role == Role::Packed).count(),
    );

    // ── Pass 1: fit the isotonic map on CLEAN-FIT binaries (pooled posterior,label) ──
    // One binary in memory at a time. We fit the map before computing any calibrated ECE.
    eprintln!("── pass 1: fitting isotonic map on clean-fit binaries ──");
    let mut pooled: Vec<(f64, f64)> = Vec::new();
    for job in jobs.iter().filter(|j| j.role == Role::CleanFit) {
        let bytes = fs::read(&job.bin).with_context(|| format!("reading {}", job.bin.display()))?;
        let (base, code) = extract_text(&bytes)?;
        let (post, _cav) = run_soft_with_cavity(base, code, 0.0, false)
            .with_context(|| format!("Soft on {}", job.name))?;
        let gt = load_gt(job.gt.as_ref().unwrap())?;
        for (a, p) in &post {
            pooled.push((*p, if gt.contains(a) { 1.0 } else { 0.0 }));
        }
        eprintln!("  fit += {} ({} candidates)", job.name, post.len());
        // bytes / post / cav dropped here.
    }
    let map = IsotonicMap::fit(&pooled);
    eprintln!("map fit on {} pooled candidates", pooled.len());
    drop(pooled);

    // ── Pass 2: compute per-binary aggregates for ALL binaries; stream to CSV (resumable) ──
    eprintln!("── pass 2: per-binary cavity statistics + true ECE ──");
    let mut done: HashMap<String, Record> = read_existing_csv(&args.out);
    let mut csv = open_csv_append(&args.out)?;
    let mut records: Vec<Record> = Vec::new();
    for job in &jobs {
        // Resume: reuse a previously computed row rather than recomputing (memory-safe + fast).
        if let Some(rec) = done.remove(&format!("{}|{}", job.name, job.level)) {
            eprintln!("  resume {} (from CSV)", job.name);
            records.push(rec);
            continue;
        }
        let rec = evaluate_binary(job, &map)?;
        writeln!(csv, "{}", rec.to_csv())?;
        csv.flush()?;
        eprintln!(
            "  {} [{}]: n={} ece_cal={:.4} S_surp={:.4} moran={:.3}",
            rec.name, rec.role.tag(), rec.n, rec.ece_calibrated, rec.s_glob_surprise, rec.s_spat_moran
        );
        records.push(rec);
    }

    // ── Aggregate verdict statistics ──
    let summary = analyze(&records);
    summary.print();
    let json = summary.to_json(&records);
    fs::write(&args.summary, json).with_context(|| format!("writing {}", args.summary.display()))?;
    eprintln!("wrote {} and {}", args.out.display(), args.summary.display());
    Ok(())
}

/// Per-address dump for one binary → the residual-cluster strip figure. Columns:
/// `offset,addr,cavity_q,local_m,residual,surprise,event,label`. `event` is the within-binary
/// μ+2σ surprise outlier flag; `label` is GT (1=true instruction start) if a `.gt` is given, else -1.
fn dump_strip(bin: &Path, gt: Option<&Path>, out: &Path) -> Result<()> {
    let bytes = fs::read(bin).with_context(|| format!("reading {}", bin.display()))?;
    let (base, code) = extract_text(&bytes)?;
    let (_post, cav) = run_soft_with_cavity(base, code, 0.0, false)?;
    let gt_set = match gt {
        Some(p) => Some(load_gt(p)?),
        None => None,
    };
    let surprises: Vec<f64> = cav.iter().map(|(_, c)| c.surprise).collect();
    let mu = mean(&surprises);
    let var = surprises.iter().map(|s| (s - mu).powi(2)).sum::<f64>() / surprises.len().max(1) as f64;
    let thr = mu + 2.0 * var.sqrt();
    let mut s = String::from("offset,addr,cavity_q,local_m,residual,surprise,event,label\n");
    use std::fmt::Write as _;
    for (addr, c) in &cav {
        let label = match &gt_set {
            Some(g) => {
                if g.contains(addr) {
                    1
                } else {
                    0
                }
            }
            None => -1,
        };
        let event = if c.surprise > thr { 1 } else { 0 };
        writeln!(
            s,
            "{},{:#x},{:.5},{:.5},{:.5},{:.5},{},{}",
            addr - base,
            addr,
            c.cavity_code_prob,
            c.local_code_prob,
            c.residual,
            c.surprise,
            event,
            label
        )
        .ok();
    }
    fs::write(out, s).with_context(|| format!("writing {}", out.display()))?;
    eprintln!("wrote per-address strip for {} ({} addresses) → {}", bin.display(), cav.len(), out.display());
    Ok(())
}

/// Run one binary end-to-end and produce its aggregate record.
fn evaluate_binary(job: &Job, map: &IsotonicMap) -> Result<Record> {
    let bytes = fs::read(&job.bin).with_context(|| format!("reading {}", job.bin.display()))?;
    let (base, code) = extract_text(&bytes)?;
    let (post, cav) = run_soft_with_cavity(base, code, 0.0, false)
        .with_context(|| format!("Soft on {}", job.name))?;

    // Ground truth → per-binary true ECE of the clean-fit map. GT provenance is the *original*
    // symbol/objdump table for this binary, never a disassembly of the (possibly obfuscated) input.
    let (ece_raw, ece_cal, base_rate) = match job.role {
        Role::Packed => {
            // Packed: no instruction-start GT and the payload is provably data. The whole compressed
            // window is negatives, so ECE against the all-zero label = mean calibrated posterior there.
            let (lo, hi) = packed_data_window(job.packed_gt.as_ref().unwrap())?;
            let region: Vec<(u64, f64)> = post.iter().cloned().filter(|&(a, _)| a >= lo && a < hi).collect();
            let cal: Vec<(u64, f64)> = map.apply_all(&region);
            let empty = HashSet::new();
            let raw = mean_over_negatives(&region);
            let ece = mean_over_negatives(&cal);
            // base_rate over the window is 0 by construction; `evaluate` on empty-gt gives ECE=mean p.
            let _ = evaluate(&cal, &empty);
            (raw, ece, 0.0)
        }
        _ => {
            let gt = load_gt(job.gt.as_ref().unwrap())?;
            let cal = map.apply_all(&post);
            let m_raw = evaluate(&post, &gt);
            let m_cal = evaluate(&cal, &gt);
            (m_raw.ece, m_cal.ece, m_cal.base_rate)
        }
    };

    let s = spatial_and_global(&cav);
    let b = ood_baselines(&post, &cav);
    Ok(Record {
        name: job.name.clone(),
        role: job.role,
        n: post.len(),
        base_rate,
        code_bytes: code.len(),
        region_entropy: shannon_entropy(code),
        ece_calibrated: ece_cal,
        ece_raw,
        s_glob_surprise: s.mean_surprise,
        s_glob_nis: s.mean_nis, // same as mean_nis; kept as the S_glob-NIS variant name
        s_spat_moran: s.moran,
        s_spat_clustered: s.clustered_frac,
        mean_nis: s.mean_nis,
        frac_super: s.frac_super,
        b_mean_pi: b.0,
        b_pred_entropy: b.1,
        b_msp: b.2,
        b_mean_abs_llr: b.3,
        level: job.level.clone(),
    })
}

/// Off-the-shelf OOD / calibration-drift baselines, GT-free and scalar. These are the "generic drift
/// monitor" a reviewer will ask about: mean confidence, predictive entropy, max-softmax-probability
/// (Hendrycks & Gimpel's MSP), and a local-decode temperature proxy. Computed on the same per-address
/// posterior (π) and local logit (φ) the cavity statistics use. None of these localizes — they are
/// single scalars with no spatial structure, which is the point of the comparison.
fn ood_baselines(post: &[(u64, f64)], cav: &[(u64, CavityStat)]) -> (f64, f64, f64, f64) {
    let n = post.len();
    if n == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mut sum_pi = 0.0;
    let mut sum_h = 0.0;
    let mut sum_msp = 0.0;
    for &(_, p) in post {
        let p = p.clamp(1e-12, 1.0 - 1e-12);
        sum_pi += p;
        sum_h += -(p * p.ln() + (1.0 - p) * (1.0 - p).ln());
        sum_msp += p.max(1.0 - p);
    }
    let sum_abs_llr: f64 = cav.iter().map(|(_, c)| c.llr_local.abs()).sum();
    (
        sum_pi / n as f64,
        sum_h / n as f64,
        sum_msp / n as f64,
        sum_abs_llr / cav.len().max(1) as f64,
    )
}

/// Aggregated cavity statistics over one binary.
struct SpatialGlobal {
    mean_surprise: f64,
    mean_nis: f64,
    moran: f64,
    /// Fraction of super-threshold surprise events that sit in a contiguous run of ≥3 — the
    /// innovation-whiteness statistic. ~0 under a well-specified (spatially exchangeable) model;
    /// obfuscation makes surprise cluster, driving this up. Scale-free in [0,1].
    clustered_frac: f64,
    /// Event rate: fraction of addresses that are within-binary surprise outliers (> μ+2σ).
    frac_super: f64,
}

/// Compute the global magnitude and spatial-clustering statistics from the (address-sorted)
/// cavity stats. Address order in `cav` is the spatial axis for the clustering statistics.
fn spatial_and_global(cav: &[(u64, CavityStat)]) -> SpatialGlobal {
    let n = cav.len();
    if n == 0 {
        return SpatialGlobal { mean_surprise: 0.0, mean_nis: 0.0, moran: 0.0, clustered_frac: 0.0, frac_super: 0.0 };
    }
    let surprises: Vec<f64> = cav.iter().map(|(_, c)| c.surprise).collect();
    let nis: Vec<f64> = cav.iter().map(|(_, c)| c.nis).collect();
    let resid: Vec<f64> = cav.iter().map(|(_, c)| c.residual).collect();
    let mean_surprise = mean(&surprises);
    let mean_nis = mean(&nis);

    // Moran's I over address-order adjacency (consecutive candidates are neighbors). Scale-free
    // spatial autocorrelation of the standardized residual; > 0 means high residuals cluster.
    let moran = morans_i_line(&resid);

    // Super-threshold *surprise* events, defined within-binary as > μ+2σ so the threshold is
    // comparable across binaries of any size or base surprise level (the raw NIS residual explodes
    // near cavity 0/1, so we cannot use a fixed cutoff on it). Under a spatially-exchangeable model
    // these ~2.3% of addresses fall at random and rarely chain; obfuscation lines them up in
    // contiguous runs. We report the fraction of events that live in a run of length ≥ 3.
    let mu = mean_surprise;
    let var = surprises.iter().map(|s| (s - mu).powi(2)).sum::<f64>() / n as f64;
    let thr = mu + 2.0 * var.sqrt();
    let events: Vec<bool> = surprises.iter().map(|&s| s > thr).collect();
    let total_events = events.iter().filter(|&&e| e).count();
    // Walk runs of true; count events that belong to a run of length ≥ 3.
    let mut clustered = 0usize;
    let mut i = 0;
    while i < n {
        if events[i] {
            let mut j = i;
            while j + 1 < n && events[j + 1] {
                j += 1;
            }
            let run = j - i + 1;
            if run >= 3 {
                clustered += run;
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    let _ = resid.iter().filter(|r| r.abs() > RESID_THR).count(); // RESID_THR retained for diagnostics
    SpatialGlobal {
        mean_surprise,
        mean_nis,
        moran,
        clustered_frac: if total_events == 0 { 0.0 } else { clustered as f64 / total_events as f64 },
        frac_super: total_events as f64 / n as f64,
    }
}

/// Moran's I with a 1-D contiguity weight (neighbors = adjacent in the ordered vector).
fn morans_i_line(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 3 {
        return 0.0;
    }
    let mbar = mean(x);
    let denom: f64 = x.iter().map(|v| (v - mbar).powi(2)).sum();
    if denom <= 0.0 {
        return 0.0;
    }
    let mut num = 0.0;
    for i in 0..n - 1 {
        num += (x[i] - mbar) * (x[i + 1] - mbar);
    }
    // W = 2(n-1) for the symmetric consecutive weight; the num above is the one-directional sum.
    (n as f64 / (n as f64 - 1.0)) * (num / denom)
}

fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().sum::<f64>() / x.len() as f64
}

fn mean_over_negatives(post: &[(u64, f64)]) -> f64 {
    if post.is_empty() {
        return 0.0;
    }
    post.iter().map(|&(_, p)| p).sum::<f64>() / post.len() as f64
}

/// Shannon entropy (bits) of the byte-value distribution over a region.
fn shannon_entropy(bytes: &[u8]) -> f64 {
    if bytes.is_empty() {
        return 0.0;
    }
    let mut counts = [0u32; 256];
    for &b in bytes {
        counts[b as usize] += 1;
    }
    let len = bytes.len() as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / len;
            -p * p.log2()
        })
        .sum()
}

/// Parse the provable-data (NEGATIVE) vaddr window out of a `.upxgt` label table. We take the
/// "compressed" row's `vaddr_start vaddr_end` — its provenance is UPX's own b_info chain, not a
/// disassembler.
fn packed_data_window(upxgt: &Path) -> Result<(u64, u64)> {
    let text = fs::read_to_string(upxgt).with_context(|| format!("reading {}", upxgt.display()))?;
    for line in text.lines() {
        let l = line.trim();
        if l.starts_with('#') || l.is_empty() {
            continue;
        }
        let cols: Vec<&str> = l.split_whitespace().collect();
        // field label vaddr_start vaddr_end file_start file_end notes...
        if cols.first() == Some(&"compressed") && cols.len() >= 4 {
            let lo = parse_hex(cols[2])?;
            let hi = parse_hex(cols[3])?;
            return Ok((lo, hi));
        }
    }
    bail!("no `compressed` NEGATIVE row in {}", upxgt.display())
}

fn parse_hex(s: &str) -> Result<u64> {
    let t = s.trim().trim_start_matches("0x");
    u64::from_str_radix(t, 16).with_context(|| format!("bad hex {s}"))
}

// ── Corpus assembly ──────────────────────────────────────────────────────────────

fn file_stem(p: &Path) -> String {
    p.file_name().and_then(|s| s.to_str()).unwrap_or("?").to_string()
}

/// List `(name, bin_path, gt_path)` for every binary in `bins_dir` that has a matching
/// `<name>.gt` in `gt_dir`. Non-file / ext-carrying entries in `bins_dir` are skipped.
fn list_bins_with_gt(bins_dir: &Path, gt_dir: &Path) -> Result<Vec<(String, PathBuf, PathBuf)>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(bins_dir).with_context(|| format!("reading dir {}", bins_dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = file_stem(&path);
        // Skip obvious non-binaries.
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

/// Deterministic seeded order: sort by a splitmix64 hash of (seed, name). Decorrelates the
/// fit/holdout split from alphabetical tool order without any RNG state.
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

// ── CSV resume ──────────────────────────────────────────────────────────────────

fn read_existing_csv(path: &Path) -> HashMap<String, Record> {
    let mut map = HashMap::new();
    let Ok(text) = fs::read_to_string(path) else {
        return map;
    };
    for line in text.lines().skip(1) {
        if let Some(rec) = Record::from_csv(line) {
            map.insert(format!("{}|{}", rec.name, rec.level), rec);
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

impl Record {
    fn from_csv(line: &str) -> Option<Record> {
        let c: Vec<&str> = line.split(',').collect();
        if c.len() < 18 {
            return None;
        }
        let role = match c[1] {
            "clean_fit" => Role::CleanFit,
            "clean_holdout" => Role::CleanHoldout,
            "desync" => Role::Desync,
            "packed" => Role::Packed,
            _ => return None,
        };
        Some(Record {
            name: c[0].to_string(),
            role,
            n: c[2].parse().ok()?,
            base_rate: c[3].parse().ok()?,
            code_bytes: c[4].parse().ok()?,
            region_entropy: c[5].parse().ok()?,
            ece_calibrated: c[6].parse().ok()?,
            ece_raw: c[7].parse().ok()?,
            s_glob_surprise: c[8].parse().ok()?,
            s_glob_nis: c[9].parse().ok()?,
            s_spat_moran: c[10].parse().ok()?,
            s_spat_clustered: c[11].parse().ok()?,
            mean_nis: c[12].parse().ok()?,
            frac_super: c[13].parse().ok()?,
            b_mean_pi: c[14].parse().ok()?,
            b_pred_entropy: c[15].parse().ok()?,
            b_msp: c[16].parse().ok()?,
            b_mean_abs_llr: c[17].parse().ok()?,
            level: c.get(18).map(|s| s.to_string()).unwrap_or_default(),
        })
    }
}

// ── Statistics ───────────────────────────────────────────────────────────────────

/// Fractional (tie-averaged) ranks of `x`.
fn ranks(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && x[idx[j + 1]] == x[idx[i]] {
            j += 1;
        }
        let avg = ((i + 1) + (j + 1)) as f64 / 2.0;
        for &k in &idx[i..=j] {
            r[k] = avg;
        }
        i = j + 1;
    }
    r
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let ma = mean(a);
    let mb = mean(b);
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..n {
        let x = a[i] - ma;
        let y = b[i] - mb;
        num += x * y;
        da += x * x;
        db += y * y;
    }
    if da <= 0.0 || db <= 0.0 {
        return 0.0;
    }
    num / (da.sqrt() * db.sqrt())
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    pearson(&ranks(a), &ranks(b))
}

/// Partial Spearman of (s, ece) controlling for confounds: rank-transform everything, least-squares
/// residualize ranked s and ranked ece on [1, conf_ranks...], then Pearson of the residuals.
fn partial_spearman(s: &[f64], ece: &[f64], confounds: &[Vec<f64>]) -> f64 {
    let rs = ranks(s);
    let re = ranks(ece);
    let rc: Vec<Vec<f64>> = confounds.iter().map(|c| ranks(c)).collect();
    let res_s = residualize(&rs, &rc);
    let res_e = residualize(&re, &rc);
    pearson(&res_s, &res_e)
}

/// Residual of `y` after least-squares regression on [intercept, cols...]. Normal equations with a
/// tiny Gaussian solve — dimensions are 1..=3 here.
fn residualize(y: &[f64], cols: &[Vec<f64>]) -> Vec<f64> {
    let n = y.len();
    let k = cols.len() + 1; // + intercept
    // Design matrix rows: [1, col0_i, col1_i, ...]
    let x = |i: usize, j: usize| -> f64 {
        if j == 0 {
            1.0
        } else {
            cols[j - 1][i]
        }
    };
    // Normal equations: (XtX) beta = Xty
    let mut xtx = vec![vec![0.0; k]; k];
    let mut xty = vec![0.0; k];
    for i in 0..n {
        for a in 0..k {
            xty[a] += x(i, a) * y[i];
            for b in 0..k {
                xtx[a][b] += x(i, a) * x(i, b);
            }
        }
    }
    let beta = solve(xtx, xty);
    (0..n)
        .map(|i| {
            let pred: f64 = (0..k).map(|a| beta[a] * x(i, a)).sum();
            y[i] - pred
        })
        .collect()
}

/// Gaussian elimination with partial pivoting for a small dense system. Returns zeros on singularity.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-12 {
            return vec![0.0; n];
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in col + 1..n {
            let f = a[r][col] / a[col][col];
            for c in col..n {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut xr = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in i + 1..n {
            s -= a[i][j] * xr[j];
        }
        xr[i] = s / a[i][i];
    }
    xr
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ── Verdict ──────────────────────────────────────────────────────────────────────

struct Summary {
    n_bins: usize,
    // best S statistic name + its correlation
    rho_surprise: f64,
    rho_nis: f64,
    rho_moran: f64,
    rho_maxrun: f64,
    // detector
    ece_drift_thr: f64,
    n_drifted: usize,
    auc_surprise: Option<f64>,
    auc_moran: Option<f64>,
    // null / firing
    null_p95: f64,
    false_alarm: f64,
    sensitivity: f64,
    // conservative bias
    clean_mean_surprise: f64,
    obf_mean_surprise: f64,
    clean_mean_nis: f64,
    obf_mean_nis: f64,
    /// The conservative signature: among genuinely-drifted obfuscated binaries in the *milder* half
    /// of the drift range, the fraction that FAIL to fire at the clean-95th null. A contaminated
    /// (leak-biased-small) detector misses the mild tail before it ever false-alarms.
    mild_drift_miss_rate: f64,
    // confounds
    rho_entropy_ece: f64,
    rho_size_ece: f64,
    partial_rho_surprise: f64,
    partial_rho_moran: f64,
    // group ECE medians
    clean_holdout_ece_med: f64,
    desync_ece_med: f64,
    verdict_go: bool,
    reasons: Vec<String>,
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

fn analyze(records: &[Record]) -> Summary {
    // Correlate over binaries that have a real instruction-start ECE (exclude packed's
    // window-mean, which is a different measurement — reported separately in the doc).
    let corr: Vec<&Record> = records.iter().filter(|r| r.role != Role::Packed).collect();
    let ece: Vec<f64> = corr.iter().map(|r| r.ece_calibrated).collect();
    let surp: Vec<f64> = corr.iter().map(|r| r.s_glob_surprise).collect();
    let nis: Vec<f64> = corr.iter().map(|r| r.s_glob_nis).collect();
    let moran: Vec<f64> = corr.iter().map(|r| r.s_spat_moran).collect();
    let maxrun: Vec<f64> = corr.iter().map(|r| r.s_spat_clustered).collect();
    let entropy: Vec<f64> = corr.iter().map(|r| r.region_entropy).collect();
    let logsize: Vec<f64> = corr.iter().map(|r| (r.n as f64 + 1.0).ln()).collect();

    let rho_surprise = spearman(&surp, &ece);
    let rho_nis = spearman(&nis, &ece);
    let rho_moran = spearman(&moran, &ece);
    let rho_maxrun = spearman(&maxrun, &ece);

    // Drift label: ECE above the clean-holdout 90th percentile (adaptive, honest).
    let mut clean_ho_ece: Vec<f64> = records
        .iter()
        .filter(|r| r.role == Role::CleanHoldout)
        .map(|r| r.ece_calibrated)
        .collect();
    clean_ho_ece.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let ece_drift_thr = percentile(&clean_ho_ece, 0.90).max(0.02);

    let labels: Vec<f64> = corr
        .iter()
        .map(|r| if r.ece_calibrated > ece_drift_thr { 1.0 } else { 0.0 })
        .collect();
    let n_drifted = labels.iter().filter(|&&l| l > 0.5).count();
    let auc_surprise = auroc(&surp.iter().cloned().zip(labels.iter().cloned()).collect::<Vec<_>>());
    let auc_moran = auroc(&moran.iter().cloned().zip(labels.iter().cloned()).collect::<Vec<_>>());

    // Empirical null from clean-FIT S_glob (surprise); fire above its 95th percentile.
    let mut fit_s: Vec<f64> = records
        .iter()
        .filter(|r| r.role == Role::CleanFit)
        .map(|r| r.s_glob_surprise)
        .collect();
    fit_s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let null_p95 = percentile(&fit_s, 0.95);

    let holdout: Vec<&Record> = records.iter().filter(|r| r.role == Role::CleanHoldout).collect();
    let obf: Vec<&Record> = records.iter().filter(|r| r.role.is_obfuscated()).collect();
    let false_alarm = frac(&holdout, |r| r.s_glob_surprise > null_p95);
    // Sensitivity: of obfuscated binaries whose true ECE is actually drifted, how many fire?
    let drifted_obf: Vec<&&Record> = obf.iter().filter(|r| r.ece_calibrated > ece_drift_thr).collect();
    let sensitivity = if drifted_obf.is_empty() {
        0.0
    } else {
        drifted_obf.iter().filter(|r| r.s_glob_surprise > null_p95).count() as f64 / drifted_obf.len() as f64
    };

    // Conservative-bias check. The loopy cavity is contaminated — e_a leaks back around loops, so
    // q_a is pulled toward the local decode and the surprise it induces is biased *small*. We do not
    // claim an unbiased detector; we confirm the direction from its operating characteristic: it
    // misses the milder end of the drift range (at a false-alarm-controlled null) before it ever
    // false-alarms on clean. `mild_drift_miss_rate` quantifies that miss on the lower half of the
    // truly-drifted obfuscated binaries. The magnitude columns (clean vs obf surprise/NIS) are
    // reported as raw diagnostics — both rise under drift, but neither is asserted to hit a
    // well-specified target (the superset's near-deterministic overlap nodes inflate NIS far above
    // the exchangeable-Bernoulli E=0.5, so 0.5 is not the right yardstick here).
    let clean_mean_surprise = mean(&holdout.iter().map(|r| r.s_glob_surprise).collect::<Vec<_>>());
    let obf_mean_surprise = mean(&obf.iter().map(|r| r.s_glob_surprise).collect::<Vec<_>>());
    let clean_mean_nis = mean(
        &records
            .iter()
            .filter(|r| r.role == Role::CleanFit || r.role == Role::CleanHoldout)
            .map(|r| r.mean_nis)
            .collect::<Vec<_>>(),
    );
    let obf_mean_nis = mean(&obf.iter().map(|r| r.mean_nis).collect::<Vec<_>>());
    // Mild-drift miss rate: split truly-drifted obfuscated bins at their median ECE; of the milder
    // half, how many stay below the firing null.
    let mut drift_eces: Vec<f64> = drifted_obf.iter().map(|r| r.ece_calibrated).collect();
    drift_eces.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mild_drift_miss_rate = if drift_eces.len() < 2 {
        0.0
    } else {
        let med = drift_eces[drift_eces.len() / 2];
        let mild: Vec<_> = drifted_obf.iter().filter(|r| r.ece_calibrated <= med).collect();
        if mild.is_empty() {
            0.0
        } else {
            mild.iter().filter(|r| r.s_glob_surprise <= null_p95).count() as f64 / mild.len() as f64
        }
    };

    // Confounds.
    let rho_entropy_ece = spearman(&entropy, &ece);
    let rho_size_ece = spearman(&logsize, &ece);
    let confs = vec![entropy.clone(), logsize.clone()];
    let partial_rho_surprise = partial_spearman(&surp, &ece, &confs);
    let partial_rho_moran = partial_spearman(&moran, &ece, &confs);

    let clean_holdout_ece_med = median(holdout.iter().map(|r| r.ece_calibrated).collect());
    let desync_ece_med = median(
        records
            .iter()
            .filter(|r| r.role == Role::Desync)
            .map(|r| r.ece_calibrated)
            .collect(),
    );

    // ── GO / NO-GO ──
    let best_rho = rho_surprise.max(rho_moran).max(rho_nis).max(rho_maxrun);
    let best_auc = auc_surprise.unwrap_or(0.5).max(auc_moran.unwrap_or(0.5));
    let best_partial = partial_rho_surprise.max(partial_rho_moran);
    let mut reasons = Vec::new();
    let c_rho = best_rho >= 0.40;
    let c_auc = best_auc >= 0.70;
    let c_fire = sensitivity > 0.0;
    let c_fa = false_alarm <= 0.15;
    let c_partial = best_partial >= 0.25;
    reasons.push(format!("best Spearman ρ(S,ECE)={best_rho:.3} (≥0.40? {c_rho})"));
    reasons.push(format!("detector AUC={best_auc:.3} (≥0.70? {c_auc})"));
    reasons.push(format!("fires on drifted (sensitivity={sensitivity:.2}>0? {c_fire})"));
    reasons.push(format!("clean false-alarm={false_alarm:.2} (≤0.15? {c_fa})"));
    reasons.push(format!("partial ρ after confounds={best_partial:.3} (≥0.25? {c_partial})"));
    let verdict_go = c_rho && c_auc && c_fire && c_fa && c_partial;

    Summary {
        n_bins: records.len(),
        rho_surprise,
        rho_nis,
        rho_moran,
        rho_maxrun,
        ece_drift_thr,
        n_drifted,
        auc_surprise,
        auc_moran,
        null_p95,
        false_alarm,
        sensitivity,
        clean_mean_surprise,
        obf_mean_surprise,
        clean_mean_nis,
        obf_mean_nis,
        mild_drift_miss_rate,
        rho_entropy_ece,
        rho_size_ece,
        partial_rho_surprise,
        partial_rho_moran,
        clean_holdout_ece_med,
        desync_ece_med,
        verdict_go,
        reasons,
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
        println!("\n══════════════════ CONSISTENCY GO/NO-GO ══════════════════");
        println!("binaries scored (excl. packed for ρ): {}", self.n_bins);
        println!("\n— GT-free S vs true post-hoc ECE (Spearman ρ) —");
        println!("  S_glob mean-surprise : ρ = {:+.3}", self.rho_surprise);
        println!("  S_glob mean-NIS      : ρ = {:+.3}", self.rho_nis);
        println!("  S_spat Moran's I     : ρ = {:+.3}", self.rho_moran);
        println!("  S_spat clustered-frac: ρ = {:+.3}", self.rho_maxrun);
        println!("\n— Detector (drift = ECE > {:.3}; {} drifted) —", self.ece_drift_thr, self.n_drifted);
        println!("  ROC AUC (surprise)   : {:?}", self.auc_surprise.map(round3));
        println!("  ROC AUC (Moran)      : {:?}", self.auc_moran.map(round3));
        println!("  clean-fit null p95   : {:.4}", self.null_p95);
        println!("  clean false-alarm    : {:.3}", self.false_alarm);
        println!("  sensitivity (drifted): {:.3}", self.sensitivity);
        println!("\n— Conservative-bias check (loopy-cavity contamination shrinks surprise) —");
        println!("  mild-drift miss rate : {:.3}  (>0 ⇒ conservative: misses mild drift)", self.mild_drift_miss_rate);
        println!("  clean mean surprise  : {:.4}", self.clean_mean_surprise);
        println!("  obf   mean surprise  : {:.4}", self.obf_mean_surprise);
        println!("  clean mean NIS       : {:.4}  (raw diagnostic, not vs 0.5)", self.clean_mean_nis);
        println!("  obf   mean NIS       : {:.4}", self.obf_mean_nis);
        println!("\n— Confound control —");
        println!("  ρ(entropy, ECE)      : {:+.3}", self.rho_entropy_ece);
        println!("  ρ(log size, ECE)     : {:+.3}", self.rho_size_ece);
        println!("  partial ρ(surprise,ECE|entropy,size): {:+.3}", self.partial_rho_surprise);
        println!("  partial ρ(Moran,ECE|entropy,size)   : {:+.3}", self.partial_rho_moran);
        println!("\n— Group ECE medians —");
        println!("  clean holdout        : {:.4}", self.clean_holdout_ece_med);
        println!("  desync               : {:.4}", self.desync_ece_med);
        println!("\n— Verdict criteria —");
        for r in &self.reasons {
            println!("  · {r}");
        }
        println!("\n  >>> {} <<<", if self.verdict_go { "GO" } else { "NO-GO" });
        println!("═══════════════════════════════════════════════════════════");
    }

    fn to_json(&self, records: &[Record]) -> String {
        let mut s = String::new();
        s.push_str("{\n");
        s.push_str(&format!("  \"verdict\": \"{}\",\n", if self.verdict_go { "GO" } else { "NO-GO" }));
        s.push_str(&format!("  \"n_bins\": {},\n", self.n_bins));
        s.push_str(&format!("  \"rho_surprise\": {:.4},\n", self.rho_surprise));
        s.push_str(&format!("  \"rho_nis\": {:.4},\n", self.rho_nis));
        s.push_str(&format!("  \"rho_moran\": {:.4},\n", self.rho_moran));
        s.push_str(&format!("  \"rho_maxrun\": {:.4},\n", self.rho_maxrun));
        s.push_str(&format!("  \"ece_drift_thr\": {:.4},\n", self.ece_drift_thr));
        s.push_str(&format!("  \"n_drifted\": {},\n", self.n_drifted));
        s.push_str(&format!("  \"auc_surprise\": {:.4},\n", self.auc_surprise.unwrap_or(f64::NAN)));
        s.push_str(&format!("  \"auc_moran\": {:.4},\n", self.auc_moran.unwrap_or(f64::NAN)));
        s.push_str(&format!("  \"null_p95\": {:.4},\n", self.null_p95));
        s.push_str(&format!("  \"false_alarm\": {:.4},\n", self.false_alarm));
        s.push_str(&format!("  \"sensitivity\": {:.4},\n", self.sensitivity));
        s.push_str(&format!("  \"clean_mean_surprise\": {:.4},\n", self.clean_mean_surprise));
        s.push_str(&format!("  \"obf_mean_surprise\": {:.4},\n", self.obf_mean_surprise));
        s.push_str(&format!("  \"clean_mean_nis\": {:.4},\n", self.clean_mean_nis));
        s.push_str(&format!("  \"obf_mean_nis\": {:.4},\n", self.obf_mean_nis));
        s.push_str(&format!("  \"mild_drift_miss_rate\": {:.4},\n", self.mild_drift_miss_rate));
        s.push_str(&format!("  \"rho_entropy_ece\": {:.4},\n", self.rho_entropy_ece));
        s.push_str(&format!("  \"rho_size_ece\": {:.4},\n", self.rho_size_ece));
        s.push_str(&format!("  \"partial_rho_surprise\": {:.4},\n", self.partial_rho_surprise));
        s.push_str(&format!("  \"partial_rho_moran\": {:.4},\n", self.partial_rho_moran));
        s.push_str(&format!("  \"clean_holdout_ece_med\": {:.4},\n", self.clean_holdout_ece_med));
        s.push_str(&format!("  \"desync_ece_med\": {:.4},\n", self.desync_ece_med));
        // Per-group means for the doc table.
        for role in [Role::CleanFit, Role::CleanHoldout, Role::Desync, Role::Packed] {
            let g: Vec<&Record> = records.iter().filter(|r| r.role == role).collect();
            if g.is_empty() {
                continue;
            }
            s.push_str(&format!(
                "  \"group_{}\": {{\"n\": {}, \"ece_cal_mean\": {:.4}, \"s_surprise_mean\": {:.4}, \"moran_mean\": {:.4}, \"mean_nis\": {:.4}}},\n",
                role.tag(),
                g.len(),
                mean(&g.iter().map(|r| r.ece_calibrated).collect::<Vec<_>>()),
                mean(&g.iter().map(|r| r.s_glob_surprise).collect::<Vec<_>>()),
                mean(&g.iter().map(|r| r.s_spat_moran).collect::<Vec<_>>()),
                mean(&g.iter().map(|r| r.mean_nis).collect::<Vec<_>>()),
            ));
        }
        s.push_str("  \"_end\": true\n}\n");
        s
    }
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

// ── CLI ──────────────────────────────────────────────────────────────────────────

struct Args {
    clean_bins: PathBuf,
    clean_gt: PathBuf,
    desync_bins: PathBuf,
    desync_gt: PathBuf,
    packed: Option<PathBuf>,
    packed_gt: Option<PathBuf>,
    /// Graded-drift ladder: repeatable (label, bins-dir, gt-dir). One run, one map, many levels.
    desync_levels: Vec<(String, PathBuf, PathBuf)>,
    /// Multi-packer slice: repeatable (label, packed-elf, upxgt).
    packed_specs: Vec<(String, PathBuf, PathBuf)>,
    n_fit: usize,
    n_holdout: usize,
    n_desync: usize,
    seed: u64,
    out: PathBuf,
    summary: PathBuf,
    // Strip-dump mode (single binary → per-address figure data), independent of the experiment.
    strip_bin: Option<PathBuf>,
    strip_out: Option<PathBuf>,
    strip_gt: Option<PathBuf>,
}

impl Args {
    fn parse(mut it: impl Iterator<Item = String>) -> Result<Self> {
        const USAGE: &str = "usage: consistency --clean-bins DIR --clean-gt DIR --desync-bins DIR \
--desync-gt DIR [--packed ELF --packed-gt UPXGT] --n-fit N --n-holdout N --n-desync N \
[--seed S] --out results.csv --summary summary.json";
        let mut clean_bins = None;
        let mut clean_gt = None;
        let mut desync_bins = None;
        let mut desync_gt = None;
        let mut packed = None;
        let mut packed_gt = None;
        let mut desync_levels: Vec<(String, PathBuf, PathBuf)> = Vec::new();
        let mut packed_specs: Vec<(String, PathBuf, PathBuf)> = Vec::new();
        let mut n_fit = 30usize;
        let mut n_holdout = 30usize;
        let mut n_desync = 40usize;
        let mut seed = 1u64;
        let mut out = None;
        let mut summary = None;
        let mut strip_bin = None;
        let mut strip_out = None;
        let mut strip_gt = None;
        while let Some(a) = it.next() {
            let mut next = |flag: &str| it.next().with_context(|| format!("{flag} needs a value"));
            match a.as_str() {
                "--clean-bins" => clean_bins = Some(PathBuf::from(next("--clean-bins")?)),
                "--clean-gt" => clean_gt = Some(PathBuf::from(next("--clean-gt")?)),
                "--desync-bins" => desync_bins = Some(PathBuf::from(next("--desync-bins")?)),
                "--desync-gt" => desync_gt = Some(PathBuf::from(next("--desync-gt")?)),
                "--packed" => packed = Some(PathBuf::from(next("--packed")?)),
                "--packed-gt" => packed_gt = Some(PathBuf::from(next("--packed-gt")?)),
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
                "--n-fit" => n_fit = next("--n-fit")?.parse()?,
                "--n-holdout" => n_holdout = next("--n-holdout")?.parse()?,
                "--n-desync" => n_desync = next("--n-desync")?.parse()?,
                "--seed" => seed = next("--seed")?.parse()?,
                "--out" => out = Some(PathBuf::from(next("--out")?)),
                "--summary" => summary = Some(PathBuf::from(next("--summary")?)),
                "--strip-bin" => strip_bin = Some(PathBuf::from(next("--strip-bin")?)),
                "--strip-out" => strip_out = Some(PathBuf::from(next("--strip-out")?)),
                "--strip-gt" => strip_gt = Some(PathBuf::from(next("--strip-gt")?)),
                "-h" | "--help" => {
                    eprintln!("{USAGE}");
                    std::process::exit(0);
                }
                other => bail!("unexpected argument: {other}\n{USAGE}"),
            }
        }
        // In strip-dump mode the experiment paths are not required.
        let strip_mode = strip_bin.is_some() && strip_out.is_some();
        let req = |o: Option<PathBuf>| -> Result<PathBuf> {
            if strip_mode {
                Ok(o.unwrap_or_default())
            } else {
                o.context(USAGE)
            }
        };
        // desync via the single-pair flags OR the repeatable --desync-level; at least one required
        // outside strip mode.
        let have_desync = desync_bins.is_some() || !desync_levels.is_empty();
        if !strip_mode && !have_desync {
            bail!("need --desync-bins/--desync-gt or at least one --desync-level\n{USAGE}");
        }
        Ok(Args {
            clean_bins: req(clean_bins)?,
            clean_gt: req(clean_gt)?,
            desync_bins: desync_bins.unwrap_or_default(),
            desync_gt: desync_gt.unwrap_or_default(),
            packed,
            packed_gt,
            desync_levels,
            packed_specs,
            n_fit,
            n_holdout,
            n_desync,
            seed,
            out: req(out)?,
            summary: req(summary)?,
            strip_bin,
            strip_out,
            strip_gt,
        })
    }
}
