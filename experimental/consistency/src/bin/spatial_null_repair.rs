//! Re-threshold every committed signature table under the **size-aware spatial null**.
//!
//! The published spatial gate was a flat `S_spat > 0.1052`. `S_spat` is Moran's I over
//! address-ordered residuals — a lag-one autocorrelation whose dispersion falls as `1/sqrt(n)` in
//! the candidate count `n`. A flat line across a widening noise cone is far too tight for small
//! binaries, so the gate becomes size-aware:
//!
//! ```text
//! T(n) = mu + 1.645 * c / sqrt(n)
//! ```
//!
//! `mu` and `c` are estimated **once**, on the 20 `role == clean_fit` rows of the credibility
//! table, and then held fixed: `mu` is the mean of `s_spat_moran`, `c` the population standard
//! deviation of `(s_spat_moran - mu) * sqrt(n)`. They are never refit on an evaluation corpus —
//! the whole point of the result is that the repair was estimated on the fit split and scored out
//! of sample.
//!
//! This binary re-runs **no inference**. Every corpus already records per-binary `n` and the
//! benign-engine `S_glob` / `S_spat`, so the exercise is pure re-thresholding. Decisions are taken
//! by the shipped `SignatureClassifier::{classify_rule, classify_guard}` — the classifier is simply
//! rebuilt per row with `spat_hi = T(n)` — so the rule here cannot drift from the one the paper
//! describes. As a guard against exactly that drift, every corpus that recorded a `rule_pick` /
//! `guard_pick` column is replayed at the old flat threshold first and checked against the recorded
//! decision; any mismatch is a hard error.
//!
//! Output is a per-row TSV of decisions. All aggregation lives in `analyze_repair.py`, so no
//! reported number is computed twice or copied by hand.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use consistency::{percentile, Regime, SignatureClassifier};

/// The published flat spatial gate, and the global gate that this repair leaves untouched. Both are
/// re-derived below from the clean-fit split and cross-checked against these recorded values.
const REC_GLOB_HI: f64 = 2.514702;
const REC_SPAT_HI: f64 = 0.105178;

/// The abstention guard's region-entropy floor, trained ground-truth-free from the fit split
/// (non-packed max 6.651, packed min 7.687 -> midpoint). Unaffected by the spatial repair; carried
/// forward verbatim from the run of record so the guarded arm stays comparable.
const PACK_ENT_LO: f64 = 7.1688;

/// The one-sided 95% normal quantile the gate is built on — a nominal-5% test.
const Z95: f64 = 1.645;

// ── tiny CSV reader ────────────────────────────────────────────────────────────

struct Table {
    cols: HashMap<String, usize>,
    rows: Vec<Vec<String>>,
}

impl Table {
    fn load(path: &Path) -> Result<Table> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let header = lines.next().context("empty CSV")?;
        let names: Vec<&str> = header.split(',').collect();
        let cols: HashMap<String, usize> =
            names.iter().enumerate().map(|(i, n)| (n.trim().to_string(), i)).collect();
        let mut rows = Vec::new();
        for (i, line) in lines.enumerate() {
            let f: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();
            if f.len() != names.len() {
                bail!(
                    "{}: row {} has {} fields, header has {} — refusing to guess",
                    path.display(), i + 2, f.len(), names.len()
                );
            }
            rows.push(f);
        }
        Ok(Table { cols, rows })
    }

    fn has(&self, col: &str) -> bool {
        self.cols.contains_key(col)
    }

    fn get<'a>(&self, row: &'a [String], col: &str) -> Result<&'a str> {
        let i = *self.cols.get(col).with_context(|| format!("missing column `{col}`"))?;
        Ok(row[i].as_str())
    }

    fn num(&self, row: &[String], col: &str) -> Result<f64> {
        let raw = self.get(row, col)?;
        raw.parse::<f64>().with_context(|| format!("column `{col}` = {raw:?} is not a number"))
    }
}

// ── the size-aware gate ────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct SpatialNull {
    mu: f64,
    c: f64,
}

impl SpatialNull {
    /// The unfloored size-aware gate `T(n) = mu + 1.645 * c / sqrt(n)`.
    fn t(&self, n: f64) -> f64 {
        self.mu + Z95 * self.c / n.sqrt()
    }

    /// The **floored** gate `T'(n) = max(FLAT, T(n))` — the recommended operating point.
    ///
    /// `T(n)` crosses below the published flat gate at `n_cross`, and above that count it fires
    /// *more* than the flat gate did. Nothing was ever measured to be wrong there: the wild corpus
    /// puts the large-`n` false-alarm rate at 0.030 under the flat gate already. Loosening above
    /// the crossover would be an extrapolation of a two-parameter model, fit over the narrow `n`
    /// range the 20 clean-fit binaries span, into a regime with no evidence of a defect. The floor
    /// keeps the correction strictly one-sided — it only ever *raises* the bar.
    fn t_floored(&self, n: f64, flat: f64) -> f64 {
        self.t(n).max(flat)
    }

    /// The candidate count at which `T(n)` crosses the flat gate. Above it the floor binds.
    fn crossover(&self, flat: f64) -> f64 {
        (Z95 * self.c / (flat - self.mu)).powi(2)
    }
}

/// A classifier carrying only the three decision thresholds. The centroid fields are unused by
/// `classify_rule` / `classify_guard`, which is all this binary calls.
fn gate(glob_hi: f64, spat_hi: f64, pack_ent_lo: f64) -> SignatureClassifier {
    SignatureClassifier {
        centroids: [(0.0, 0.0); 3],
        present: [false; 3],
        mu_g: 0.0,
        sd_g: 1.0,
        mu_s: 0.0,
        sd_s: 1.0,
        raw_centroids: [(0.0, 0.0); 3],
        glob_hi,
        spat_hi,
        pack_ent_lo,
    }
}

/// Population standard deviation (the `train`-side convention: divide by N, not N-1).
fn pstdev(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n).sqrt()
}

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

// ── corpus specs ───────────────────────────────────────────────────────────────

/// A signature table to re-threshold. `regime_col` names the ground-truth regime column; for the
/// credibility table that is `role`, which is remapped through `role_map`.
struct Corpus {
    label: &'static str,
    path: PathBuf,
    regime_col: &'static str,
    n_col: &'static str,
    glob_col: &'static str,
    spat_col: &'static str,
    ent_col: &'static str,
    role_map: bool,
    /// Keep only rows whose `status` column equals this. The wild census records binaries it
    /// declined to analyse (`too_large:<bytes>`); those carry no signature and are not scored.
    status_ok: Option<&'static str>,
    /// Corpora with no regime column because every member has the same known ground truth — the
    /// wild Debian census is all stock, hence all `benign`.
    fixed_regime: Option<&'static str>,
}

/// `clean_holdout -> benign`, `desync -> obfuscated`, `packed -> packed`. `clean_fit` is the split
/// the thresholds are estimated on and is never scored — it is emitted tagged so the aggregator can
/// exclude it explicitly rather than silently.
fn map_role(role: &str) -> &str {
    match role {
        "clean_holdout" => "benign",
        "clean_fit" => "clean_fit",
        "desync" => "obfuscated",
        "packed" => "packed",
        other => other,
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let suite: PathBuf = args.next().context("usage: spatial_null_repair <upd-suite> <upd-suite-regime> <out.tsv>")?.into();
    let regime: PathBuf = args.next().context("missing <upd-suite-regime>")?.into();
    let out: PathBuf = args.next().context("missing <out.tsv>")?.into();

    let cred_path = suite.join("docs/consistency_credibility/credibility.csv");

    // ── step 1: estimate the null on the clean-fit split, once ────────────────
    let cred = Table::load(&cred_path)?;
    let fit: Vec<&Vec<String>> =
        cred.rows.iter().filter(|r| cred.get(r, "role").map(|v| v == "clean_fit").unwrap_or(false)).collect();
    if fit.len() != 20 {
        bail!("expected 20 clean_fit rows, found {}", fit.len());
    }
    let fit_spat: Vec<f64> = fit.iter().map(|r| cred.num(r, "s_spat_moran")).collect::<Result<_>>()?;
    let fit_glob: Vec<f64> = fit.iter().map(|r| cred.num(r, "s_glob_surprise")).collect::<Result<_>>()?;
    let fit_n: Vec<f64> = fit.iter().map(|r| cred.num(r, "n")).collect::<Result<_>>()?;

    let mu = fit_spat.iter().sum::<f64>() / fit_spat.len() as f64;
    let scaled: Vec<f64> =
        fit_spat.iter().zip(&fit_n).map(|(s, n)| (s - mu) * n.sqrt()).collect();
    let c = pstdev(&scaled);
    let null = SpatialNull { mu, c };

    eprintln!("── spatial null, estimated on the 20 clean_fit rows ──");
    eprintln!("  mu = {mu:.6}   c = {c:.6}");
    if !approx(mu, 0.0692, 5e-5) || !approx(c, 4.034, 5e-4) {
        bail!("null constants do not reproduce the derived mu=0.0692 / c=4.034 (got {mu:.6} / {c:.6})");
    }
    for (n, want) in [(32000.0, 0.106), (4000.0, 0.174), (500.0, 0.366)] {
        let got = null.t(n);
        eprintln!("  T({n:>7.0}) = {got:.6}   (expected ~{want})");
        if !approx(got, want, 1e-3) {
            bail!("sanity check failed: T({n}) = {got:.6}, expected ~{want}");
        }
    }

    // ── step 2: reproduce the published flat thresholds from the same split ───
    let glob_hi = percentile(&fit_glob, 0.95) * 2.5;
    let flat_spat_hi = percentile(&fit_spat, 0.95);
    eprintln!("── published flat gate, re-derived ──");
    eprintln!("  glob_hi = {glob_hi:.6} (recorded {REC_GLOB_HI})   spat_hi = {flat_spat_hi:.6} (recorded {REC_SPAT_HI})");
    if !approx(glob_hi, REC_GLOB_HI, 1e-3) || !approx(flat_spat_hi, REC_SPAT_HI, 1e-4) {
        bail!("failed to reproduce the recorded flat thresholds from the clean-fit split");
    }

    // ── step 3: re-threshold every corpus ─────────────────────────────────────
    let corpora = vec![
        Corpus {
            label: "credibility",
            path: cred_path.clone(),
            regime_col: "role",
            n_col: "n",
            glob_col: "s_glob_surprise",
            spat_col: "s_spat_moran",
            ent_col: "region_entropy",
            role_map: true,
            status_ok: None,
            fixed_regime: None,
        },
        Corpus {
            label: "breadth_main",
            path: regime.join("docs/packer_breadth/breadth_main.csv"),
            regime_col: "regime",
            n_col: "n",
            glob_col: "s_glob_benign_eng",
            spat_col: "s_spat_benign_eng",
            ent_col: "region_ent",
            role_map: false,
            status_ok: None,
            fixed_regime: None,
        },
        Corpus {
            label: "breadth_ezuri",
            path: regime.join("docs/packer_breadth/breadth_ezuri.csv"),
            regime_col: "regime",
            n_col: "n",
            glob_col: "s_glob_benign_eng",
            spat_col: "s_spat_benign_eng",
            ent_col: "region_ent",
            role_map: false,
            status_ok: None,
            fixed_regime: None,
        },
        Corpus {
            label: "switching_core",
            path: regime.join("docs/consistency_switching/switching.csv"),
            regime_col: "regime",
            n_col: "n",
            glob_col: "s_glob_benign_eng",
            spat_col: "s_spat_benign_eng",
            ent_col: "region_ent",
            role_map: false,
            status_ok: None,
            fixed_regime: None,
        },
        Corpus {
            label: "corpus_expansion",
            path: regime.join("docs/corpus_expansion/expanded.csv"),
            regime_col: "regime",
            n_col: "n",
            glob_col: "s_glob_benign_eng",
            spat_col: "s_spat_benign_eng",
            ent_col: "region_ent",
            role_map: false,
            status_ok: None,
            fixed_regime: None,
        },
        Corpus {
            label: "boundaries_meta",
            path: regime.join("docs/downstream_decision/boundaries_meta.csv"),
            regime_col: "regime",
            n_col: "n",
            glob_col: "s_glob_benign_eng",
            spat_col: "s_spat_benign_eng",
            ent_col: "region_ent",
            role_map: false,
            status_ok: None,
            fixed_regime: None,
        },
        Corpus {
            label: "abstention_guard",
            path: regime.join("docs/abstention_guard/guard.csv"),
            regime_col: "regime",
            n_col: "n",
            glob_col: "s_glob_benign_eng",
            spat_col: "s_spat_benign_eng",
            ent_col: "region_ent",
            role_map: false,
            status_ok: None,
            fixed_regime: None,
        },
        // The wild census: 1095 analysable stock Debian binaries, no obfuscation anywhere, so every
        // alarm is a false alarm. This is the only corpus that reaches down to n in the hundreds,
        // which is exactly where the flat gate was shown to be too tight.
        Corpus {
            label: "wild_debian",
            path: regime.join("docs/realworld_fire_rate/firerate.csv"),
            regime_col: "status",
            n_col: "n_cand",
            glob_col: "s_glob",
            spat_col: "s_spat",
            ent_col: "region_ent",
            role_map: false,
            status_ok: Some("ok"),
            fixed_regime: Some("benign"),
        },
    ];

    let mut f = fs::File::create(&out).with_context(|| format!("creating {}", out.display()))?;
    writeln!(
        f,
        "corpus\tname\tsublabel\ttrue_regime\tn\ts_glob\ts_spat\tregion_ent\tt_new\tt_flr\tfloor_binds\t\
         old_rule\tnew_rule\tflr_rule\told_guard\tnew_guard\tflr_guard\t\
         old_spat_only\tnew_spat_only\tflr_spat_only\trec_rule\trec_guard"
    )?;

    let mut replay_checked = 0usize;
    for cor in &corpora {
        if !cor.path.exists() {
            bail!("corpus `{}` not found at {}", cor.label, cor.path.display());
        }
        let t = Table::load(&cor.path)?;

        // Refuse to proceed on any corpus that cannot support pure re-thresholding.
        for req in [cor.n_col, cor.spat_col, cor.glob_col] {
            if !t.has(req) {
                bail!(
                    "corpus `{}` ({}) has no `{}` column — cannot re-threshold without re-inference",
                    cor.label, cor.path.display(), req
                );
            }
        }
        let has_ent = t.has(cor.ent_col);
        let has_rec_rule = t.has("rule_pick");
        let has_rec_guard = t.has("guard_pick");

        let mut kept = 0usize;
        for row in &t.rows {
            // Rows the census declined to analyse carry no signature at all — skip before any
            // parse, rather than letting a sentinel reach the gate.
            if let Some(want) = cor.status_ok {
                if t.get(row, "status")? != want {
                    continue;
                }
            }
            kept += 1;
            let name = t.get(row, "name")?.to_string();
            let sublabel = if t.has("sublabel") { t.get(row, "sublabel")?.to_string() } else { "-".into() };
            let true_regime = match cor.fixed_regime {
                Some(r) => r,
                None => {
                    let raw = t.get(row, cor.regime_col)?;
                    if cor.role_map { map_role(raw) } else { raw }
                }
            };

            let n = t.num(row, cor.n_col)?;
            if !(n > 0.0) {
                bail!("{}: {} has non-positive candidate count n={}", cor.label, name, n);
            }
            let sg = t.num(row, cor.glob_col)?;
            let ss = t.num(row, cor.spat_col)?;
            let ent = if has_ent { t.num(row, cor.ent_col)? } else { f64::NAN };

            let t_new = null.t(n);
            let t_flr = null.t_floored(n, flat_spat_hi);
            let floor_binds = t_new < flat_spat_hi;
            let old = gate(glob_hi, flat_spat_hi, PACK_ENT_LO);
            let new = gate(glob_hi, t_new, PACK_ENT_LO);
            let flr = gate(glob_hi, t_flr, PACK_ENT_LO);

            let old_rule = old.classify_rule(sg, ss);
            let new_rule = new.classify_rule(sg, ss);
            let flr_rule = flr.classify_rule(sg, ss);

            // Spatial-only arm of the routing ablation: the global axis is switched off, which the
            // ablation models as "that axis never fires" — i.e. an infinite `glob_hi`. Still the
            // shipped rule, just with one gate disabled.
            let old_spat_only = gate(f64::INFINITY, flat_spat_hi, PACK_ENT_LO).classify_rule(sg, ss);
            let new_spat_only = gate(f64::INFINITY, t_new, PACK_ENT_LO).classify_rule(sg, ss);
            let flr_spat_only = gate(f64::INFINITY, t_flr, PACK_ENT_LO).classify_rule(sg, ss);
            // With no region-entropy column the guard cannot be evaluated; emit `-` rather than
            // silently substituting the bare rule.
            let (old_guard, new_guard, flr_guard) = if has_ent {
                (
                    old.classify_guard(sg, ss, ent).tag().to_string(),
                    new.classify_guard(sg, ss, ent).tag().to_string(),
                    flr.classify_guard(sg, ss, ent).tag().to_string(),
                )
            } else {
                ("-".to_string(), "-".to_string(), "-".to_string())
            };

            // ── subset invariant ───────────────────────────────────────────────
            // `T'(n) = max(FLAT, T(n)) >= FLAT` everywhere, so the floored gate can only ever fire
            // on a strict subset of what the flat gate fired on. A binary that fires under `T'` but
            // not under the flat gate would be a bug in the gate, not a finding — so it aborts.
            if t_flr < flat_spat_hi {
                bail!(
                    "invariant violated in {}: {} has T'(n)={t_flr} below the flat gate {flat_spat_hi}",
                    cor.label, name
                );
            }
            if flr_spat_only != Regime::Benign && old_spat_only == Regime::Benign {
                bail!(
                    "invariant violated in {}: {} fires under the floored gate (T'={t_flr}) but not \
                     under the flat gate ({flat_spat_hi}); S_spat={ss}, n={n}",
                    cor.label, name
                );
            }
            // The same subset property has to hold for the full rule and for the guarded rule: the
            // floored gate must never move a binary *away* from the flat gate's benign default.
            if flr_rule != old_rule && old_rule == Regime::Benign {
                bail!(
                    "invariant violated in {}: {} routes {} under the floored gate but benign under \
                     the flat gate; S_glob={sg}, S_spat={ss}, n={n}",
                    cor.label, name, flr_rule.tag()
                );
            }
            if has_ent && flr_guard != old_guard && old_guard == "benign" {
                bail!(
                    "invariant violated in {}: {} guards to {} under the floored gate but benign \
                     under the flat gate; S_glob={sg}, S_spat={ss}, n={n}",
                    cor.label, name, flr_guard
                );
            }

            let rec_rule = if has_rec_rule { t.get(row, "rule_pick")?.to_string() } else { "-".into() };
            let rec_guard = if has_rec_guard { t.get(row, "guard_pick")?.to_string() } else { "-".into() };

            // Replay check: the shipped rule at the old flat gate must reproduce what was recorded.
            if has_rec_rule && rec_rule != old_rule.tag() {
                bail!(
                    "replay mismatch in {}: {} recorded rule_pick={} but the shipped rule at the \
                     flat gate gives {} (S_glob={sg}, S_spat={ss})",
                    cor.label, name, rec_rule, old_rule.tag()
                );
            }
            if has_rec_guard && has_ent && rec_guard != old_guard {
                bail!(
                    "replay mismatch in {}: {} recorded guard_pick={} but the shipped guard at the \
                     flat gate gives {} (S_glob={sg}, S_spat={ss}, region_ent={ent})",
                    cor.label, name, rec_guard, old_guard
                );
            }
            if has_rec_rule {
                replay_checked += 1;
            }

            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{:.0}\t{:.6}\t{:.6}\t{}\t{:.6}\t{:.6}\t{}\t\
                 {}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                cor.label, name, sublabel, true_regime, n, sg, ss,
                if has_ent { format!("{ent:.6}") } else { "-".into() },
                t_new, t_flr, floor_binds,
                old_rule.tag(), new_rule.tag(), flr_rule.tag(),
                old_guard, new_guard, flr_guard,
                old_spat_only.tag(), new_spat_only.tag(), flr_spat_only.tag(),
                rec_rule, rec_guard
            )?;
        }
        eprintln!(
            "  {:<18} {:>4} rows{}  <- {}",
            cor.label, kept,
            if kept == t.rows.len() { String::new() } else { format!(" (of {}, status-filtered)", t.rows.len()) },
            cor.path.display()
        );
    }

    eprintln!("── replay check: {replay_checked} recorded decisions reproduced exactly at the flat gate ──");
    eprintln!("wrote {}", out.display());

    // Emit the constants alongside the decisions so the aggregator never hard-codes them.
    let meta = out.with_extension("meta.tsv");
    let mut m = fs::File::create(&meta)?;
    writeln!(m, "key\tvalue")?;
    writeln!(m, "mu\t{mu:.10}")?;
    writeln!(m, "c\t{c:.10}")?;
    writeln!(m, "z95\t{Z95}")?;
    writeln!(m, "glob_hi\t{glob_hi:.10}")?;
    writeln!(m, "flat_spat_hi\t{flat_spat_hi:.10}")?;
    writeln!(m, "pack_ent_lo\t{PACK_ENT_LO:.10}")?;
    writeln!(m, "n_fit\t{}", fit.len())?;
    writeln!(m, "replay_checked\t{replay_checked}")?;
    writeln!(m, "n_crossover\t{:.10}", null.crossover(flat_spat_hi))?;
    writeln!(m, "fit_n_lo\t{:.0}", fit_n.iter().cloned().fold(f64::MAX, f64::min))?;
    writeln!(m, "fit_n_hi\t{:.0}", fit_n.iter().cloned().fold(f64::MIN, f64::max))?;
    for n in [500.0, 4000.0, 32000.0] {
        writeln!(m, "T_{n:.0}\t{:.10}", null.t(n))?;
        writeln!(m, "Tflr_{n:.0}\t{:.10}", null.t_floored(n, flat_spat_hi))?;
    }
    eprintln!("wrote {}", meta.display());

    let _ = Regime::ALL;
    Ok(())
}
