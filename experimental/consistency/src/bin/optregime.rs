//! `optregime` — the regime-adaptive calibration probe, benign edition.
//!
//! The switching binary proved the cavity signature can steer a calibration-map bank across
//! *adversarial* regimes (clean / packed / desync). Fair question a reviewer will ask: is that a
//! quirk of obfuscation, or is calibration-map switching a general answer to distribution shift? The
//! cleanest benign shift we have is optimization level — same source, four gcc `-O` settings, so the
//! regime is the *only* variable. If (1) a map fit on the common case (gcc O2) drifts when applied to
//! O0/O1/O3 held-out, and (2) the ground-truth-free `(S_glob, S_spat)` signature picks the right
//! per-regime map without labels, then the self-calibration story generalizes past obfuscation. If
//! either fails, that's an honest NO-GO and the paper stays scoped to obfuscation.
//!
//! This reuses the engine and the calibration/consistency machinery verbatim (`evalkit`'s
//! `IsotonicMap`, `run_soft_with_cavity_cfg`, `evaluate`, and the same `S_glob`/`S_spat` cavity
//! statistics the switching binary uses) — it does NOT reimplement the engine. The only new thing is
//! that the regimes are opt levels and every regime runs under the *same* (default) engine: there is
//! no benign per-regime engine knob, and there shouldn't be — we are testing whether the calibration
//! map alone needs to move, selected by the signature alone.
//!
//! Frame: this is calibration *maintenance*, not compiler/opt identification. A purpose-built
//! classifier would beat us at guessing the opt level; the deliverable is "keep the posterior honest
//! for whatever regime is in front of us, GT-free."
//!
//! Two selection arms, both GT-free, plus the mandatory confound arm:
//!   * signature   — nearest-centroid on the standardized `(ln S_glob, ln S_spat)` signature. The
//!                    calibration-relevant one: it reads surprise structure, not bulk size.
//!   * size/entropy — nearest-centroid on standardized `(ln code_bytes, .text byte-entropy)`. The
//!                    confound baseline. If this selects as well as the signature, the signature adds
//!                    nothing here and we say so (Gate: signature must be additive over size+entropy).
//!
//! Three ECE arms on held-out binaries (instruction-level ECE, the paper's metric):
//!   (a) always-default — apply the O2 map to everything (the stale baseline / the problem).
//!   (b) oracle         — apply the true regime's map (uses the label; the ceiling).
//!   (c) switched       — apply the signature-selected map (ours, GT-free).
//! plus a fourth, (d) size/entropy-switched, for the confound audit.
//!
//! Standing rules honored: GT only from the pre-supplied `.gt` (gen-gt instruction stream), never a
//! disassembly of the input; one binary in memory at a time (both passes are serial, each converged
//! graph freed before the next); resumable holdout CSV. Split is by *program*, so a program's four
//! opt builds never straddle the fit/holdout line (no leakage through shared source).
//!
//! ```text
//! optregime \
//!   --level O0 BINS GT  --level O1 BINS GT  --level O2 BINS GT  --level O3 BINS GT \
//!   --default O2  --n-fit 12  --seed 1 \
//!   --out results.csv --summary summary.json
//! ```

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use evalkit::{evaluate, extract_text, load_gt, run_soft_with_cavity_cfg, IsotonicMap};
use probdisasm::CavityStat;

// ── One binary = one (program, opt level) ──────────────────────────────────────

/// A single compiled binary: program `stem` built at optimization `level`, with its instruction-start
/// GT. Same `stem` appears at every level; the split keys on `stem` so all four builds share a fate.
#[derive(Clone)]
struct Bin {
    stem: String,
    level: String,
    level_idx: usize,
    bin: PathBuf,
    gt: PathBuf,
}

/// The per-binary features + arm ECEs we record. Features (through `s_spat`) are computed for every
/// binary; the arm ECEs and picks are meaningful only for held-out rows (fit rows carry `NA`).
#[derive(Clone, Debug)]
struct Record {
    stem: String,
    level: String,
    split: &'static str, // "fit" | "holdout"
    n: usize,
    code_bytes: usize,
    entropy: f64, // .text byte Shannon entropy, bits/byte
    base_rate: f64,
    s_glob: f64,
    s_spat: f64,
    // Held-out only (NA on fit rows):
    ece_raw: OptF,     // uncalibrated
    ece_default: OptF, // (a) always-default (O2) map
    ece_oracle: OptF,  // (b) true-regime map
    ece_sig: OptF,     // (c) signature-selected map
    ece_se: OptF,      // (d) size/entropy-selected map
    sig_pick: OptStr,  // signature classifier's regime pick
    se_pick: OptStr,   // size/entropy classifier's regime pick
}

/// A tiny "value or NA" wrapper so fit rows serialize cleanly and the Python side reads real NAs.
#[derive(Clone, Copy, Debug)]
struct OptF(Option<f64>);
#[derive(Clone, Debug)]
struct OptStr(Option<String>);

impl std::fmt::Display for OptF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(v) => write!(f, "{v:.6}"),
            None => write!(f, "NA"),
        }
    }
}
impl std::fmt::Display for OptStr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Some(s) => write!(f, "{s}"),
            None => write!(f, "NA"),
        }
    }
}

impl Record {
    const CSV_HEADER: &'static str = "stem,level,split,n,code_bytes,entropy,base_rate,s_glob,s_spat,\
ece_raw,ece_default,ece_oracle,ece_sig,ece_se,sig_pick,se_pick";

    fn to_csv(&self) -> String {
        format!(
            "{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{},{}",
            self.stem,
            self.level,
            self.split,
            self.n,
            self.code_bytes,
            self.entropy,
            self.base_rate,
            self.s_glob,
            self.s_spat,
            self.ece_raw,
            self.ece_default,
            self.ece_oracle,
            self.ece_sig,
            self.ece_se,
            self.sig_pick,
            self.se_pick,
        )
    }
}

fn main() -> Result<()> {
    let args = Args::parse(std::env::args().skip(1))?;
    let default_idx = args
        .levels
        .iter()
        .position(|(l, _, _)| *l == args.default)
        .with_context(|| format!("--default {} is not among the --level labels", args.default))?;

    // ── Assemble the binary set: intersection of program stems present at EVERY level ──
    let mut per_level: Vec<Vec<(String, PathBuf, PathBuf)>> = Vec::new();
    for (label, bins, gt) in &args.levels {
        per_level.push(
            list_bins_with_gt(bins, gt).with_context(|| format!("level {label}: {}", bins.display()))?,
        );
    }
    let mut common: BTreeSet<String> = per_level[0].iter().map(|(s, _, _)| s.clone()).collect();
    for pl in &per_level[1..] {
        let here: BTreeSet<String> = pl.iter().map(|(s, _, _)| s.clone()).collect();
        common = common.intersection(&here).cloned().collect();
    }
    let mut programs: Vec<String> = common.into_iter().collect();
    if programs.len() < args.n_fit + 1 {
        bail!("only {} common programs; need > n_fit={}", programs.len(), args.n_fit);
    }
    // Deterministic program-level shuffle → first n_fit programs FIT, rest HELD-OUT.
    programs.sort_by_key(|s| splitmix64(args.seed ^ fnv1a(s.as_bytes())));
    let fit_progs: BTreeSet<String> = programs.iter().take(args.n_fit).cloned().collect();

    // Build the flat binary list, tagged fit/holdout, one entry per (program, level).
    let lookup: Vec<HashMap<String, (PathBuf, PathBuf)>> = per_level
        .iter()
        .map(|pl| pl.iter().map(|(s, b, g)| (s.clone(), (b.clone(), g.clone()))).collect())
        .collect();
    let mut fit_bins: Vec<Bin> = Vec::new();
    let mut hold_bins: Vec<Bin> = Vec::new();
    for stem in &programs {
        for (li, (label, _, _)) in args.levels.iter().enumerate() {
            let (b, g) = lookup[li][stem].clone();
            let bin = Bin { stem: stem.clone(), level: label.clone(), level_idx: li, bin: b, gt: g };
            if fit_progs.contains(stem) {
                fit_bins.push(bin);
            } else {
                hold_bins.push(bin);
            }
        }
    }
    eprintln!(
        "levels: {}  programs: {} ({} fit / {} holdout)  binaries: {} fit / {} holdout  default={}",
        args.levels.len(),
        programs.len(),
        fit_progs.len(),
        programs.len() - fit_progs.len(),
        fit_bins.len(),
        hold_bins.len(),
        args.default,
    );

    // ── Pass 1: fit the per-regime map bank + train both nearest-centroid classifiers ──
    // Always re-run pass 1 (it is cheap and it holds no state on disk); the bank/classifiers are the
    // in-memory product it feeds pass 2.
    let n_levels = args.levels.len();
    let mut done: HashMap<String, Record> = read_existing_csv(&args.out);
    let mut csv = open_csv_append(&args.out)?;
    let mut records: Vec<Record> = Vec::new();

    eprintln!("── pass 1: fitting the calibration-map bank + centroids ──");
    let mut pools: Vec<Vec<(f64, f64)>> = vec![Vec::new(); n_levels];
    let mut sig_feats: Vec<(usize, f64, f64)> = Vec::new(); // (level_idx, ln s_glob, ln s_spat)
    let mut se_feats: Vec<(usize, f64, f64)> = Vec::new(); // (level_idx, ln code_bytes, entropy)
    for b in &fit_bins {
        let feat = run_features(b)?;
        for &(a, p) in &feat.post {
            pools[b.level_idx].push((p, if feat.gt_set.contains(&a) { 1.0 } else { 0.0 }));
        }
        sig_feats.push((b.level_idx, ln_eps(feat.s_glob), ln_eps(feat.s_spat)));
        se_feats.push((b.level_idx, (feat.code_bytes as f64).ln(), feat.entropy));
        let key = format!("{}|{}", b.stem, b.level);
        let rec = Record {
            stem: b.stem.clone(),
            level: b.level.clone(),
            split: "fit",
            n: feat.post.len(),
            code_bytes: feat.code_bytes,
            entropy: feat.entropy,
            base_rate: feat.base_rate,
            s_glob: feat.s_glob,
            s_spat: feat.s_spat,
            ece_raw: OptF(Some(evaluate(&feat.post, &feat.gt_set).ece)),
            ece_default: OptF(None),
            ece_oracle: OptF(None),
            ece_sig: OptF(None),
            ece_se: OptF(None),
            sig_pick: OptStr(None),
            se_pick: OptStr(None),
        };
        if !done.contains_key(&key) {
            writeln!(csv, "{}", rec.to_csv())?;
            csv.flush()?;
        }
        records.push(rec);
        // feat (post, cav, gt) dropped here — one binary in memory at a time.
    }
    let bank: Vec<IsotonicMap> = pools.iter().map(|p| IsotonicMap::fit(p)).collect();
    for (li, (label, _, _)) in args.levels.iter().enumerate() {
        eprintln!("  map[{label}] fit on {} pooled candidates", pools[li].len());
    }
    let sig_clf = Centroid::train(&sig_feats, n_levels);
    let se_clf = Centroid::train(&se_feats, n_levels);
    eprintln!("  signature centroids (std space): {}", sig_clf.describe(&args));
    eprintln!("  size/entropy centroids (std space): {}", se_clf.describe(&args));

    // ── Pass 2: three-arm (+confound) held-out evaluation, streamed to CSV (resumable) ──
    eprintln!("── pass 2: held-out evaluation ──");
    for b in &hold_bins {
        let key = format!("{}|{}", b.stem, b.level);
        if let Some(rec) = done.remove(&key) {
            eprintln!("  resume {} [{}] (from CSV)", b.stem, b.level);
            records.push(rec);
            continue;
        }
        let feat = run_features(b)?;
        let ece_of = |m: &IsotonicMap| evaluate(&m.apply_all(&feat.post), &feat.gt_set).ece;
        let sig_pick = sig_clf.classify(ln_eps(feat.s_glob), ln_eps(feat.s_spat));
        let se_pick = se_clf.classify((feat.code_bytes as f64).ln(), feat.entropy);
        let rec = Record {
            stem: b.stem.clone(),
            level: b.level.clone(),
            split: "holdout",
            n: feat.post.len(),
            code_bytes: feat.code_bytes,
            entropy: feat.entropy,
            base_rate: feat.base_rate,
            s_glob: feat.s_glob,
            s_spat: feat.s_spat,
            ece_raw: OptF(Some(evaluate(&feat.post, &feat.gt_set).ece)),
            ece_default: OptF(Some(ece_of(&bank[default_idx]))),
            ece_oracle: OptF(Some(ece_of(&bank[b.level_idx]))),
            ece_sig: OptF(Some(ece_of(&bank[sig_pick]))),
            ece_se: OptF(Some(ece_of(&bank[se_pick]))),
            sig_pick: OptStr(Some(args.levels[sig_pick].0.clone())),
            se_pick: OptStr(Some(args.levels[se_pick].0.clone())),
        };
        writeln!(csv, "{}", rec.to_csv())?;
        csv.flush()?;
        eprintln!(
            "  {} [{}]: raw={:.4} default={:.4} oracle={:.4} sig={:.4}({}) se={:.4}({})",
            b.stem, b.level,
            rec.ece_raw, rec.ece_default, rec.ece_oracle,
            rec.ece_sig, args.levels[sig_pick].0, rec.ece_se, args.levels[se_pick].0,
        );
        records.push(rec);
    }

    // ── Summary (the report tables come from the CSV in Python; this is the eyeball check) ──
    let summary = summarize(&records, &args);
    summary.print();
    fs::write(&args.summary, summary.to_json())
        .with_context(|| format!("writing {}", args.summary.display()))?;
    eprintln!("wrote {} and {}", args.out.display(), args.summary.display());
    Ok(())
}

// ── Per-binary engine pass + features ──────────────────────────────────────────

struct Features {
    post: Vec<(u64, f64)>,
    gt_set: std::collections::HashSet<u64>,
    s_glob: f64,
    s_spat: f64,
    code_bytes: usize,
    entropy: f64,
    base_rate: f64,
}

/// Run the default engine on one binary and pull everything downstream needs: posteriors, the two
/// cavity signature scalars, `.text` size + byte-entropy, and the GT/base-rate. Default engine only
/// (`entropy_prior=0, chainfwd=0`) — the benign regimes carry no per-regime engine knob by design.
fn run_features(b: &Bin) -> Result<Features> {
    let bytes = fs::read(&b.bin).with_context(|| format!("reading {}", b.bin.display()))?;
    let (base, code) = extract_text(&bytes)?;
    let (post, cav) = run_soft_with_cavity_cfg(base, code, 0.0, 0.0, false)
        .with_context(|| format!("engine on {} [{}]", b.stem, b.level))?;
    let (s_glob, s_spat) = global_and_spatial(&cav);
    let entropy = byte_entropy(code);
    let gt_set = load_gt(&b.gt)?;
    let base_rate =
        post.iter().filter(|&&(a, _)| gt_set.contains(&a)).count() as f64 / post.len().max(1) as f64;
    Ok(Features {
        code_bytes: code.len(),
        post,
        gt_set,
        s_glob,
        s_spat,
        entropy,
        base_rate,
    })
}

/// Shannon entropy of the byte histogram, bits/byte (0..=8). The size/entropy confound baseline's
/// second feature — cheap, GT-free, exactly the kind of trivial statistic the audit must rule out.
fn byte_entropy(code: &[u8]) -> f64 {
    if code.is_empty() {
        return 0.0;
    }
    let mut hist = [0u64; 256];
    for &b in code {
        hist[b as usize] += 1;
    }
    let n = code.len() as f64;
    let mut h = 0.0;
    for &c in &hist {
        if c > 0 {
            let p = c as f64 / n;
            h -= p * p.log2();
        }
    }
    h
}

// ── The two cavity signature scalars (identical defn to the switching binary) ───

/// `S_glob` = mean cavity surprise; `S_spat` = Moran's I of the standardized residual over
/// address-order adjacency. Same definitions the credibility/switching code uses so the signature
/// can't drift between experiments.
fn global_and_spatial(cav: &[(u64, CavityStat)]) -> (f64, f64) {
    let n = cav.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean_surprise = cav.iter().map(|(_, c)| c.surprise).sum::<f64>() / n as f64;
    let resid: Vec<f64> = cav.iter().map(|(_, c)| c.residual).collect();
    (mean_surprise, morans_i_line(&resid))
}

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

const LN_EPS: f64 = 1e-3;
fn ln_eps(v: f64) -> f64 {
    v.max(LN_EPS).ln()
}

// ── Nearest-centroid classifier over N regimes (the same mechanism as switching) ─

/// Nearest-centroid in standardized 2-D feature space. `train` standardizes both feature dims over
/// all fit binaries, then averages per regime. `classify` standardizes the query and returns the
/// closest present centroid's regime index. This is the exact selection mechanism the obfuscation
/// switching binary uses — here fed either the signature or the size/entropy features.
struct Centroid {
    centroids: Vec<Option<(f64, f64)>>,
    mu_a: f64,
    sd_a: f64,
    mu_b: f64,
    sd_b: f64,
    raw: Vec<Option<(f64, f64)>>, // raw (mean feat_a, mean feat_b) per regime, for the report
}

impl Centroid {
    fn train(feats: &[(usize, f64, f64)], n_levels: usize) -> Self {
        let a: Vec<f64> = feats.iter().map(|f| f.1).collect();
        let bb: Vec<f64> = feats.iter().map(|f| f.2).collect();
        let (mu_a, sd_a0) = mean_std(&a);
        let (mu_b, sd_b0) = mean_std(&bb);
        let sd_a = if sd_a0 > 1e-9 { sd_a0 } else { 1.0 };
        let sd_b = if sd_b0 > 1e-9 { sd_b0 } else { 1.0 };
        let mut sum = vec![(0.0, 0.0); n_levels];
        let mut raw_sum = vec![(0.0, 0.0); n_levels];
        let mut cnt = vec![0usize; n_levels];
        for &(li, va, vb) in feats {
            sum[li].0 += (va - mu_a) / sd_a;
            sum[li].1 += (vb - mu_b) / sd_b;
            raw_sum[li].0 += va;
            raw_sum[li].1 += vb;
            cnt[li] += 1;
        }
        let centroids = (0..n_levels)
            .map(|i| (cnt[i] > 0).then(|| (sum[i].0 / cnt[i] as f64, sum[i].1 / cnt[i] as f64)))
            .collect();
        let raw = (0..n_levels)
            .map(|i| (cnt[i] > 0).then(|| (raw_sum[i].0 / cnt[i] as f64, raw_sum[i].1 / cnt[i] as f64)))
            .collect();
        Centroid { centroids, mu_a, sd_a, mu_b, sd_b, raw }
    }

    fn classify(&self, va: f64, vb: f64) -> usize {
        let a = (va - self.mu_a) / self.sd_a;
        let b = (vb - self.mu_b) / self.sd_b;
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for (i, c) in self.centroids.iter().enumerate() {
            if let Some((ca, cb)) = c {
                let d = (a - ca).powi(2) + (b - cb).powi(2);
                if d < best_d {
                    best_d = d;
                    best = i;
                }
            }
        }
        best
    }

    fn describe(&self, args: &Args) -> String {
        let mut s = String::new();
        for (i, r) in self.raw.iter().enumerate() {
            if let Some((a, b)) = r {
                s.push_str(&format!("{}=({a:+.3},{b:+.3}) ", args.levels[i].0));
            }
        }
        s
    }
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
        0.0
    } else {
        x.iter().sum::<f64>() / x.len() as f64
    }
}

// ── Summary (eyeball; the paper tables are regenerated from the CSV in Python) ──

struct Summary {
    levels: Vec<String>,
    per_regime: Vec<(String, usize, [f64; 5])>, // (level, n, [raw, default, oracle, sig, se])
    sel_sig: Vec<(String, f64)>,
    sel_se: Vec<(String, f64)>,
    sel_sig_overall: f64,
    sel_se_overall: f64,
}

fn summarize(records: &[Record], args: &Args) -> Summary {
    let hold: Vec<&Record> = records.iter().filter(|r| r.split == "holdout").collect();
    let arms = |rs: &[&Record]| -> [f64; 5] {
        let col = |f: fn(&Record) -> OptF| -> f64 {
            mean(&rs.iter().filter_map(|r| f(r).0).collect::<Vec<_>>())
        };
        [
            col(|r| r.ece_raw),
            col(|r| r.ece_default),
            col(|r| r.ece_oracle),
            col(|r| r.ece_sig),
            col(|r| r.ece_se),
        ]
    };
    let mut per_regime = Vec::new();
    let mut sel_sig = Vec::new();
    let mut sel_se = Vec::new();
    for (label, _, _) in &args.levels {
        let rs: Vec<&Record> = hold.iter().filter(|r| &r.level == label).copied().collect();
        if rs.is_empty() {
            continue;
        }
        per_regime.push((label.clone(), rs.len(), arms(&rs)));
        let ss = frac(&rs, |r| r.sig_pick.0.as_deref() == Some(label.as_str()));
        let se = frac(&rs, |r| r.se_pick.0.as_deref() == Some(label.as_str()));
        sel_sig.push((label.clone(), ss));
        sel_se.push((label.clone(), se));
    }
    let sel_sig_overall = frac(&hold, |r| r.sig_pick.0.as_deref() == Some(r.level.as_str()));
    let sel_se_overall = frac(&hold, |r| r.se_pick.0.as_deref() == Some(r.level.as_str()));
    Summary {
        levels: args.levels.iter().map(|(l, _, _)| l.clone()).collect(),
        per_regime,
        sel_sig,
        sel_se,
        sel_sig_overall,
        sel_se_overall,
    }
}

fn frac(v: &[&Record], pred: impl Fn(&Record) -> bool) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().filter(|r| pred(r)).count() as f64 / v.len() as f64
    }
}

impl Summary {
    fn print(&self) {
        println!("\n════════════════ OPT-REGIME CALIBRATION PROBE ════════════════");
        println!("levels: {}", self.levels.join(" "));
        println!("\n— Held-out instruction-level ECE by regime —");
        println!("  regime   n     raw    default(O2)  oracle    sig      size/ent");
        for (lvl, n, a) in &self.per_regime {
            println!(
                "  {:<7} {:>3}  {:>7.4}  {:>9.4}  {:>7.4}  {:>7.4}  {:>7.4}",
                lvl, n, a[0], a[1], a[2], a[3], a[4]
            );
        }
        println!("\n— GT-free selection accuracy (pick the true regime) —");
        print!("  signature   overall={:.2} | ", self.sel_sig_overall);
        for (l, s) in &self.sel_sig {
            print!("{l}={s:.2} ");
        }
        println!();
        print!("  size/entropy overall={:.2} | ", self.sel_se_overall);
        for (l, s) in &self.sel_se {
            print!("{l}={s:.2} ");
        }
        println!("\n──────────────────────────────────────────────────────────────");
        println!("(Phase-A drift, three-arm table, and the confound audit are");
        println!(" regenerated from the CSV by analyze_optregime.py.)");
    }

    fn to_json(&self) -> String {
        let mut s = String::from("{\n");
        s.push_str(&format!("  \"levels\": [{}],\n", self.levels.iter().map(|l| format!("\"{l}\"")).collect::<Vec<_>>().join(", ")));
        s.push_str(&format!("  \"sel_sig_overall\": {:.4},\n", self.sel_sig_overall));
        s.push_str(&format!("  \"sel_se_overall\": {:.4},\n", self.sel_se_overall));
        for (lvl, n, a) in &self.per_regime {
            s.push_str(&format!(
                "  \"regime_{lvl}\": {{\"n\": {n}, \"ece_raw\": {:.4}, \"ece_default\": {:.4}, \"ece_oracle\": {:.4}, \"ece_sig\": {:.4}, \"ece_se\": {:.4}}},\n",
                a[0], a[1], a[2], a[3], a[4]
            ));
        }
        s.push_str("  \"_end\": true\n}\n");
        s
    }
}

// ── Corpus assembly / IO (mirrors switching's helpers) ─────────────────────────

fn list_bins_with_gt(bins_dir: &Path, gt_dir: &Path) -> Result<Vec<(String, PathBuf, PathBuf)>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(bins_dir).with_context(|| format!("reading dir {}", bins_dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
        if name.is_empty() || name.starts_with('.') || name.ends_with(".gt") {
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

fn read_existing_csv(path: &Path) -> HashMap<String, Record> {
    let mut map = HashMap::new();
    let Ok(text) = fs::read_to_string(path) else {
        return map;
    };
    for line in text.lines().skip(1) {
        let c: Vec<&str> = line.split(',').collect();
        if c.len() < 16 {
            continue;
        }
        let opt_f = |s: &str| OptF(if s == "NA" { None } else { s.parse().ok() });
        let opt_s = |s: &str| OptStr(if s == "NA" { None } else { Some(s.to_string()) });
        let rec = Record {
            stem: c[0].to_string(),
            level: c[1].to_string(),
            split: if c[2] == "fit" { "fit" } else { "holdout" },
            n: c[3].parse().unwrap_or(0),
            code_bytes: c[4].parse().unwrap_or(0),
            entropy: c[5].parse().unwrap_or(0.0),
            base_rate: c[6].parse().unwrap_or(0.0),
            s_glob: c[7].parse().unwrap_or(0.0),
            s_spat: c[8].parse().unwrap_or(0.0),
            ece_raw: opt_f(c[9]),
            ece_default: opt_f(c[10]),
            ece_oracle: opt_f(c[11]),
            ece_sig: opt_f(c[12]),
            ece_se: opt_f(c[13]),
            sig_pick: opt_s(c[14]),
            se_pick: opt_s(c[15]),
        };
        map.insert(format!("{}|{}", rec.stem, rec.level), rec);
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
    levels: Vec<(String, PathBuf, PathBuf)>,
    default: String,
    n_fit: usize,
    seed: u64,
    out: PathBuf,
    summary: PathBuf,
}

impl Args {
    fn parse(mut it: impl Iterator<Item = String>) -> Result<Self> {
        const USAGE: &str = "usage: optregime --level LABEL BINS GT [--level ...] \
--default LABEL --n-fit N [--seed S] --out CSV --summary JSON";
        let mut levels = Vec::new();
        let mut default = None;
        let mut n_fit = 12usize;
        let mut seed = 1u64;
        let mut out = None;
        let mut summary = None;
        while let Some(a) = it.next() {
            let mut next = |flag: &str| it.next().with_context(|| format!("{flag} needs a value"));
            match a.as_str() {
                "--level" => {
                    let label = next("--level")?;
                    let bins = PathBuf::from(next("--level bins")?);
                    let gt = PathBuf::from(next("--level gt")?);
                    levels.push((label, bins, gt));
                }
                "--default" => default = Some(next("--default")?),
                "--n-fit" => n_fit = next("--n-fit")?.parse()?,
                "--seed" => seed = next("--seed")?.parse()?,
                "--out" => out = Some(PathBuf::from(next("--out")?)),
                "--summary" => summary = Some(PathBuf::from(next("--summary")?)),
                "-h" | "--help" => {
                    eprintln!("{USAGE}");
                    std::process::exit(0);
                }
                other => bail!("unexpected argument: {other}\n{USAGE}"),
            }
        }
        if levels.len() < 2 {
            bail!("need >= 2 --level regimes\n{USAGE}");
        }
        Ok(Args {
            default: default.context(USAGE)?,
            levels,
            n_fit,
            seed,
            out: out.context(USAGE)?,
            summary: summary.context(USAGE)?,
        })
    }
}
