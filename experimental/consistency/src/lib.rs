//! Shared machinery for the Paper-2 regime-switching experiments: the regime enum, the
//! `(engine setting, calibration map)` config bank, the GT-free signature classifier (+ the
//! abstention guard), the cavity statistics, and the deterministic corpus split.
//!
//! Why this exists. `bin/switching.rs` is the landed three-arm ECE probe and I don't want to touch
//! it — its CSV and summary are the numbers the paper quotes. `bin/downstream.rs` needs *exactly*
//! the same bank, the same classifier, and the same fit/held-out split, because the whole point of
//! the downstream-decision experiment is to show what the *already-reported* calibration drift does
//! to a real analyst decision. So the shared pieces live here, lifted verbatim out of
//! `bin/switching.rs`.
//!
//! That leaves two copies of this logic in the tree. The guard against drift is numeric, not
//! stylistic: `downstream` re-emits the columns `switching` emits (the benign-engine signature, the
//! four selection picks, always-benign/oracle ECE), and `docs/downstream_decision/verify_ab.sh`
//! runs both binaries over a tiny corpus and diffs those columns. If this file ever drifts from
//! `bin/switching.rs`, that diff fails. Do not "clean up" the duplication by editing one side only.
//!
//! Nothing here touches the engine — `probdisasm` / `evalkit` are reused as-is.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use evalkit::{load_gt, run_soft_with_cavity_cfg, IsotonicMap};
use probdisasm::{extract_text_section as extract_text, CavityStat};

// ── Regimes and the config bank ────────────────────────────────────────────────

/// The three regimes the bank covers. Tigress binaries carry the `Obfuscated` *true* regime (they
/// are obfuscated code) but are tracked by sub-label so we can show the honest blind spot: the
/// surprise statistic does not fire on semantic obfuscation that preserves clean decoding.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Regime {
    Benign,
    Packed,
    Obfuscated,
}

impl Regime {
    pub fn tag(self) -> &'static str {
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
    pub fn engine(self, entropy: f64, chainfwd: f64) -> (f64, f64) {
        match self {
            Regime::Benign => (0.0, 0.0),
            Regime::Packed => (entropy, 0.0),
            Regime::Obfuscated => (0.0, chainfwd),
        }
    }
    pub const ALL: [Regime; 3] = [Regime::Benign, Regime::Packed, Regime::Obfuscated];
    pub fn idx(self) -> usize {
        match self {
            Regime::Benign => 0,
            Regime::Packed => 1,
            Regime::Obfuscated => 2,
        }
    }
}

pub fn parse_regime(s: &str) -> Option<Regime> {
    match s {
        "benign" => Some(Regime::Benign),
        "packed" => Some(Regime::Packed),
        "obfuscated" => Some(Regime::Obfuscated),
        _ => None,
    }
}

/// One held-out binary queued for the multi-arm evaluation.
pub struct HoldoutJob {
    pub name: String,
    pub bin: PathBuf,
    /// Instruction-start GT for benign/obfuscated. `None` for packed (window from `packed_gt`).
    pub gt: Option<PathBuf>,
    /// Packed only: the `.upxgt` provable-data window.
    pub packed_gt: Option<PathBuf>,
    /// The true regime (the oracle knows this; the GT-free rules must recover it).
    pub regime: Regime,
    /// Corpus sub-label (e.g. "d2_heavy", "upx_nrv", "tigL").
    pub sublabel: String,
}

/// One fit binary (contributes to a map and/or the classifier centroid).
pub struct FitJob {
    pub name: String,
    pub bin: PathBuf,
    pub gt: Option<PathBuf>,
    pub packed_gt: Option<PathBuf>,
    pub regime: Regime,
}

pub struct Bank {
    /// Calibration maps indexed by `Regime::idx()`.
    pub maps: [IsotonicMap; 3],
}

impl Bank {
    pub fn map(&self, r: Regime) -> &IsotonicMap {
        &self.maps[r.idx()]
    }
}

/// Fit each regime's calibration map on its FIT binaries, each under its own engine setting. The
/// benign/obfuscated maps are ordinary isotonic fits on (posterior, instruction-start-label). The
/// packed map is fit on the provable-data window only (all labels 0), so it learns to pull the
/// packed regime's posteriors down — the honest packed calibration (the payload is data).
pub fn fit_bank(fit: &[FitJob], entropy: f64, chainfwd: f64) -> Result<Bank> {
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
/// (`ln S_glob`, `ln S_spat`), plus the benign-default threshold rule and the abstention guard.
/// Trained on the FIT binaries' benign-engine signatures. Test-time it reads only the two scalars
/// off a binary's benign-engine cavity pass — no ground truth.
pub struct SignatureClassifier {
    /// Per-regime centroid in standardized feature space, indexed by `Regime::idx()`.
    pub centroids: [(f64, f64); 3],
    /// Present-in-training flag (a regime with no fit bins gets no centroid).
    pub present: [bool; 3],
    pub mu_g: f64,
    pub sd_g: f64,
    pub mu_s: f64,
    pub sd_s: f64,
    /// The raw per-regime (mean ln S_glob, mean ln S_spat) for the report table.
    pub raw_centroids: [(f64, f64); 3],
    /// S_glob above this ⇒ obfuscated (desync raises S_glob ~25×; clean/packed stay low).
    pub glob_hi: f64,
    /// S_spat above this (with S_glob below `glob_hi`) ⇒ packed. Below both ⇒ benign — the default,
    /// so an ambiguous binary is never routed to the calibration-destroying packed map without a
    /// clear spatial signal.
    pub spat_hi: f64,
    /// **Abstention guard.** The packed route also demands the flagged region be genuinely
    /// packed-like: region entropy above this threshold. A Tigress-virtualized binary trips the
    /// spatial threshold on its dispatch loop but is normal-entropy real code, so the guard makes it
    /// fall through to benign (abstain). Trained ground-truth-free from the fit split: the midpoint
    /// of the gap between the highest non-packed region entropy and the lowest packed one. `NaN` if
    /// the fit split doesn't separate (honest failure — reported, not papered over).
    pub pack_ent_lo: f64,
}

/// A small epsilon so `ln` of a zero/near-zero spatial statistic stays finite.
pub const LN_EPS: f64 = 1e-3;

impl SignatureClassifier {
    pub fn train(fit: &[FitJob], entropy: f64, chainfwd: f64) -> Result<Self> {
        eprintln!("── pass 1b: training the signature classifier (benign-engine signatures) ──");
        let mut feats: Vec<(Regime, f64, f64)> = Vec::new();
        let mut clean_glob: Vec<f64> = Vec::new();
        let mut clean_spat: Vec<f64> = Vec::new();
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

        let glob_hi = if clean_glob.len() >= 3 { percentile(&clean_glob, 0.95) * 2.5 } else { 2.0 };
        let spat_hi = if clean_spat.len() >= 3 { percentile(&clean_spat, 0.95) } else { 0.12 };
        eprintln!("  rule thresholds: glob_hi={glob_hi:.3} (obf if S_glob>this)  spat_hi={spat_hi:.4} (packed if S_spat>this)");

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

    /// Threshold-rule classify: benign-default signature rule.
    pub fn classify_rule(&self, s_glob: f64, s_spat: f64) -> Regime {
        if s_glob > self.glob_hi {
            Regime::Obfuscated
        } else if s_spat > self.spat_hi {
            Regime::Packed
        } else {
            Regime::Benign
        }
    }

    /// Guarded threshold-rule classify: the same rule, but the packed route also demands the flagged
    /// region be genuinely packed-like (region entropy above `pack_ent_lo`). If `pack_ent_lo` is NaN
    /// (fit split didn't separate) the guard is inert and this reduces to `classify_rule`.
    pub fn classify_guard(&self, s_glob: f64, s_spat: f64, region_ent: f64) -> Regime {
        let packed_like = self.pack_ent_lo.is_nan() || region_ent > self.pack_ent_lo;
        if s_glob > self.glob_hi {
            Regime::Obfuscated
        } else if s_spat > self.spat_hi && packed_like {
            Regime::Packed
        } else {
            Regime::Benign
        }
    }

    /// Classify a binary from its benign-engine (S_glob, S_spat) signature (nearest centroid).
    pub fn classify(&self, s_glob: f64, s_spat: f64) -> Regime {
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

// ── Aggregated cavity statistics (global magnitude + spatial clustering) ────────

pub struct GlobalSpatial {
    pub mean_surprise: f64,
    pub mean_nis: f64,
    pub moran: f64,
}

/// The two GT-free statistics from the address-sorted cavity stats: `S_glob` (mean surprise, and
/// mean NIS) and `S_spat` (Moran's I of the standardized residual over address-order adjacency).
pub fn global_and_spatial(cav: &[(u64, CavityStat)]) -> GlobalSpatial {
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
pub fn morans_i_line(x: &[f64]) -> f64 {
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
/// Window 1 KiB, step ½ KiB; regions shorter than a window use the whole-region entropy.
pub fn region_entropy(code: &[u8]) -> f64 {
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
pub fn byte_entropy(b: &[u8]) -> f64 {
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

// ── Small numeric helpers ──────────────────────────────────────────────────────

pub fn mean_std(x: &[f64]) -> (f64, f64) {
    if x.is_empty() {
        return (0.0, 0.0);
    }
    let m = x.iter().sum::<f64>() / x.len() as f64;
    let v = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64;
    (m, v.sqrt())
}

pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().sum::<f64>() / x.len() as f64
}

pub fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// `q`-quantile of `x` (0..=1), nearest-rank on a sorted copy.
pub fn percentile(x: &[f64], q: f64) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let mut s = x.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((s.len() as f64 - 1.0) * q).round() as usize;
    s[idx.min(s.len() - 1)]
}

// ── Corpus assembly / IO helpers ───────────────────────────────────────────────

pub fn file_stem(p: &Path) -> String {
    p.file_name().and_then(|s| s.to_str()).unwrap_or("?").to_string()
}

pub fn list_bins_with_gt(bins_dir: &Path, gt_dir: &Path) -> Result<Vec<(String, PathBuf, PathBuf)>> {
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

pub fn deterministic_order(
    mut v: Vec<(String, PathBuf, PathBuf)>,
    seed: u64,
) -> Vec<(String, PathBuf, PathBuf)> {
    v.sort_by_key(|(name, _, _)| splitmix64(seed ^ fnv1a(name.as_bytes())));
    v
}

pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Parse the provable-data (NEGATIVE) vaddr window out of a `.upxgt` label table — UPX's own
/// `b_info` chain, not a disassembler.
pub fn packed_data_window(upxgt: &Path) -> Result<(u64, u64)> {
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

pub fn parse_hex(s: &str) -> Result<u64> {
    let t = s.trim().trim_start_matches("0x");
    u64::from_str_radix(t, 16).with_context(|| format!("bad hex {s}"))
}

// ── The deterministic fit / held-out split ─────────────────────────────────────

/// Everything the split needs. Mirrors `bin/switching.rs`'s CLI surface so the two binaries build
/// byte-identical fit and held-out sets from the same flags and seed.
pub struct CorpusSpec {
    pub clean_bins: PathBuf,
    pub clean_gt: PathBuf,
    pub desync_levels: Vec<(String, PathBuf, PathBuf)>,
    pub packed_specs: Vec<(String, PathBuf, PathBuf)>,
    pub tigress_levels: Vec<(String, PathBuf, PathBuf)>,
    pub benign_holdout: Vec<(String, PathBuf, PathBuf)>,
    pub n_clean_fit: usize,
    pub n_clean_holdout: usize,
    pub n_desync_fit: usize,
    pub n_desync_holdout: usize,
    pub n_packed_fit: usize,
    pub n_packed_holdout: usize,
    pub n_tig_holdout: usize,
    pub seed: u64,
}

/// Build the fit and held-out job lists. Lifted verbatim from `bin/switching.rs::main` so the two
/// experiments grade the *same* held-out binaries under the *same* bank — the downstream-decision
/// result has to be a re-reading of the already-published calibration numbers, not a new split.
pub fn build_jobs(spec: &CorpusSpec) -> Result<(Vec<FitJob>, Vec<HoldoutJob>)> {
    let clean = deterministic_order(list_bins_with_gt(&spec.clean_bins, &spec.clean_gt)?, spec.seed);
    if clean.len() < spec.n_clean_fit + spec.n_clean_holdout {
        bail!(
            "need {} clean bins (fit {} + holdout {}), found {}",
            spec.n_clean_fit + spec.n_clean_holdout,
            spec.n_clean_fit,
            spec.n_clean_holdout,
            clean.len()
        );
    }

    let mut fit_jobs: Vec<FitJob> = Vec::new();
    let mut hold_jobs: Vec<HoldoutJob> = Vec::new();

    for (i, (name, bin, gt)) in clean.into_iter().enumerate() {
        if i < spec.n_clean_fit {
            fit_jobs.push(FitJob { name, bin, gt: Some(gt), packed_gt: None, regime: Regime::Benign });
        } else if i < spec.n_clean_fit + spec.n_clean_holdout {
            hold_jobs.push(HoldoutJob {
                name, bin, gt: Some(gt), packed_gt: None, regime: Regime::Benign, sublabel: "clean".into(),
            });
        }
    }

    // Desync (obfuscated by junk-insertion): pool all levels, deterministic split.
    let mut desync_all: Vec<(String, PathBuf, PathBuf, String)> = Vec::new();
    for (label, bins_dir, gt_dir) in &spec.desync_levels {
        let ds = list_bins_with_gt(bins_dir, gt_dir)
            .with_context(|| format!("desync level {label}: {}", bins_dir.display()))?;
        for (name, bin, gt) in ds {
            desync_all.push((name, bin, gt, label.clone()));
        }
    }
    desync_all.sort_by_key(|(name, _, _, _)| splitmix64(spec.seed ^ fnv1a(name.as_bytes())));
    for (i, (name, bin, gt, label)) in desync_all.into_iter().enumerate() {
        if i < spec.n_desync_fit {
            fit_jobs.push(FitJob { name, bin, gt: Some(gt), packed_gt: None, regime: Regime::Obfuscated });
        } else if i < spec.n_desync_fit + spec.n_desync_holdout {
            hold_jobs.push(HoldoutJob {
                name, bin, gt: Some(gt), packed_gt: None, regime: Regime::Obfuscated, sublabel: label,
            });
        }
    }

    // Packed: deterministic split.
    let mut packed_all: Vec<(String, PathBuf, PathBuf, String)> = spec
        .packed_specs
        .iter()
        .map(|(label, elf, gt)| (format!("{}__{}", file_stem(elf), label), elf.clone(), gt.clone(), label.clone()))
        .collect();
    packed_all.sort_by_key(|(name, _, _, _)| splitmix64(spec.seed ^ fnv1a(name.as_bytes())));
    for (i, (name, elf, gt, label)) in packed_all.into_iter().enumerate() {
        if i < spec.n_packed_fit {
            fit_jobs.push(FitJob { name, bin: elf, gt: None, packed_gt: Some(gt), regime: Regime::Packed });
        } else if i < spec.n_packed_fit + spec.n_packed_holdout {
            hold_jobs.push(HoldoutJob {
                name, bin: elf, gt: None, packed_gt: Some(gt), regime: Regime::Packed, sublabel: label,
            });
        }
    }

    // Tigress: held-out ONLY — the semantic-obfuscation blind-spot probe.
    let mut tig_all: Vec<(String, PathBuf, PathBuf, String)> = Vec::new();
    for (label, bins_dir, gt_dir) in &spec.tigress_levels {
        let tg = list_bins_with_gt(bins_dir, gt_dir)
            .with_context(|| format!("tigress level {label}: {}", bins_dir.display()))?;
        for (name, bin, gt) in tg {
            tig_all.push((name, bin, gt, label.clone()));
        }
    }
    tig_all.sort_by_key(|(name, _, _, _)| splitmix64(spec.seed ^ fnv1a(name.as_bytes())));
    for (name, bin, gt, label) in tig_all.into_iter().take(spec.n_tig_holdout) {
        hold_jobs.push(HoldoutJob {
            name, bin, gt: Some(gt), packed_gt: None, regime: Regime::Obfuscated, sublabel: label,
        });
    }

    // Benign-holdout extras — the legitimate-VM false-positive gate.
    for (label, bins_dir, gt_dir) in &spec.benign_holdout {
        let bl = list_bins_with_gt(bins_dir, gt_dir)
            .with_context(|| format!("benign-holdout {label}: {}", bins_dir.display()))?;
        for (name, bin, gt) in bl {
            hold_jobs.push(HoldoutJob {
                name, bin, gt: Some(gt), packed_gt: None, regime: Regime::Benign, sublabel: label.clone(),
            });
        }
    }

    Ok((fit_jobs, hold_jobs))
}
