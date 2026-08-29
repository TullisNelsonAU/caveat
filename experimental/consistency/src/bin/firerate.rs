//! `firerate` — the label-free fire-rate probe on wild third-party binaries.
//!
//! Every binary in every experiment we have published so far was compiled or transformed by us. That
//! is the paper's biggest exposure: our clean null was fit on coreutils, our desync and packed arms
//! are our own transforms, and a reviewer is entitled to ask whether any of it survives contact with
//! software we did not build. This probe answers that, and the reason it is possible at all is that
//! it needs **no ground truth**.
//!
//! The argument is simple. We cannot measure ECE without labels. We *can* measure how often the
//! detector fires, and on what. On binaries we have every reason to believe are benign — stock
//! Debian packages, the exact bytes shipped to users — every firing is a false alarm. That yields a
//! real-world false-alarm rate with no GT whatsoever, and it is the number a deployer actually
//! cares about.
//!
//! Four things are recorded per binary, all label-free:
//!   1. Whether it fires at the *published detection nulls* (`S_glob` 1.01, `S_spat` 0.105).
//!   2. The routing decision the *bare rule* would take, and what the *guard* does to it. This is the
//!      strongest available evidence for the abstention guard: how often would an unguarded system
//!      have switched calibration maps on ordinary software, and how often does the guard stop it?
//!   3. The raw `(S_glob, S_spat)` distribution, so we can ask whether the coreutils-fit null is
//!      representative of software in the wild or merely narrow.
//!   4. Binaries firing *both* prongs, flagged for hand inspection — genuinely packed shipped
//!      software is plausible and would be a real finding rather than an error.
//!
//! Nothing is tuned here and nothing may be. The deliverable is an unbiased estimate, not a good
//! one; if the wild false-alarm rate is materially worse than the 0.12 we report on our own corpus,
//! that is the headline result.
//!
//! Machinery reused, not reimplemented: the engine (`probdisasm` via `evalkit::run_soft_with_cavity_cfg`),
//! the two cavity scalars (`global_and_spatial`), the region-entropy guard feature (`region_entropy`),
//! and the decision functions themselves (`SignatureClassifier::{classify_rule, classify_guard}`).
//! The classifier is built by *struct literal* from the published thresholds rather than retrained —
//! there is no ground truth here to train on, and reusing the real decision functions means this
//! probe cannot drift from the rule the paper describes.
//!
//! Serial and memory-safe by construction: one binary is read, analyzed, scored, and dropped before
//! the next is opened. The CSV is flushed per row and the run is resumable, so a kill mid-corpus
//! leaves a usable file.
//!
//! ```text
//! firerate --bins DIR [--out firerate.csv] [--summary firerate_summary.json]
//!          [--max-code-bytes 3000000] [--limit N]
//! ```

use std::collections::HashSet;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use consistency::{global_and_spatial, region_entropy, Regime, SignatureClassifier};
use evalkit::run_soft_with_cavity_cfg;
use probdisasm::extract_text_section as extract_text;

// ── The published thresholds. Copied here as constants, never refit. ───────────

/// Detection null, `S_glob`: the clean-fit 95th percentile from the credibility run (n=45 clean),
/// `experimental/adversary/src/main.rs:35`. Above this ⇒ flagged.
const DET_GLOB_HI: f64 = 1.01;
/// Detection null, `S_spat`: same provenance, `experimental/adversary/src/main.rs:36`.
const DET_SPAT_HI: f64 = 0.105;

/// Routing rule, `S_glob`. Note this is *far* laxer than the detection null: the switching fit
/// inflates the clean p95 by 2.5× (`lib.rs:257`). Published value from `docs/corpus_expansion/expanded.json`.
const RULE_GLOB_HI: f64 = 2.5147;
/// Routing rule, `S_spat`. The spatial arm is *not* inflated — it is the same clean p95, which is why
/// the routing and detection spatial bars nearly coincide (0.1052 vs 0.105).
const RULE_SPAT_HI: f64 = 0.1052;
/// Abstention guard: the packed route additionally demands region entropy above this. Published
/// value from the same fit (`docs/corpus_expansion/expanded.json`, `docs/abstention_guard/run.log`).
const PACK_ENT_LO: f64 = 7.1688;

/// Build the decision object from the published thresholds. The centroid fields are unused by
/// `classify_rule`/`classify_guard`; they are filled with NaN so that any accidental future call to
/// the nearest-centroid `classify()` produces obvious garbage rather than a plausible-looking lie.
fn published_classifier() -> SignatureClassifier {
    SignatureClassifier {
        centroids: [(f64::NAN, f64::NAN); 3],
        present: [false; 3],
        mu_g: f64::NAN,
        sd_g: f64::NAN,
        mu_s: f64::NAN,
        sd_s: f64::NAN,
        raw_centroids: [(f64::NAN, f64::NAN); 3],
        glob_hi: RULE_GLOB_HI,
        spat_hi: RULE_SPAT_HI,
        pack_ent_lo: PACK_ENT_LO,
    }
}

// ── CSV row ───────────────────────────────────────────────────────────────────

const HEADER: &str = "name,status,code_bytes,n_cand,entropy,region_ent,s_glob,s_spat,s_nis,\
fire_glob,fire_spat,fire_any,fire_both,rule_pick,guard_pick,guard_vetoed,secs";

struct Row {
    name: String,
    status: String,
    code_bytes: usize,
    n_cand: usize,
    entropy: f64,
    region_ent: f64,
    s_glob: f64,
    s_spat: f64,
    s_nis: f64,
    fire_glob: bool,
    fire_spat: bool,
    rule_pick: Regime,
    guard_pick: Regime,
    secs: f64,
}

impl Row {
    /// A row for a binary the engine could not process at all. Recorded rather than dropped: silent
    /// exclusion of the hard cases would bias the fire rate, and by which direction we could not say.
    fn failed(name: &str, status: &str, secs: f64) -> Self {
        Row {
            name: name.to_string(),
            status: status.to_string(),
            code_bytes: 0,
            n_cand: 0,
            entropy: f64::NAN,
            region_ent: f64::NAN,
            s_glob: f64::NAN,
            s_spat: f64::NAN,
            s_nis: f64::NAN,
            fire_glob: false,
            fire_spat: false,
            rule_pick: Regime::Benign,
            guard_pick: Regime::Benign,
            secs,
        }
    }

    fn write(&self, w: &mut fs::File) -> Result<()> {
        let f = |v: f64| if v.is_nan() { "NA".to_string() } else { format!("{v:.6}") };
        let ok = self.status == "ok";
        writeln!(
            w,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.2}",
            self.name,
            self.status,
            self.code_bytes,
            self.n_cand,
            f(self.entropy),
            f(self.region_ent),
            f(self.s_glob),
            f(self.s_spat),
            f(self.s_nis),
            if ok { self.fire_glob as u8 as i32 } else { -1 },
            if ok { self.fire_spat as u8 as i32 } else { -1 },
            if ok { (self.fire_glob || self.fire_spat) as u8 as i32 } else { -1 },
            if ok { (self.fire_glob && self.fire_spat) as u8 as i32 } else { -1 },
            if ok { self.rule_pick.tag() } else { "NA" },
            if ok { self.guard_pick.tag() } else { "NA" },
            // The guard "vetoed" exactly when it downgraded the bare rule's pick.
            if ok && self.rule_pick != self.guard_pick { 1 } else if ok { 0 } else { -1 },
            self.secs
        )?;
        w.flush()?;
        Ok(())
    }
}

// ── One binary ────────────────────────────────────────────────────────────────

fn measure(path: &Path, name: &str, max_code: usize, clf: &SignatureClassifier) -> Row {
    let t0 = Instant::now();
    let el = |t: Instant| t.elapsed().as_secs_f64();

    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) => return Row::failed(name, &format!("read_error:{e}"), el(t0)),
    };
    // `extract_text` is fallible on anything that is not a well-formed ELF with a .text section; a
    // wild corpus will contain some. Catch rather than abort — optregime's `?`-propagation is right
    // for a curated corpus and wrong here.
    let (base, code) = match extract_text(&bytes) {
        Ok(v) => v,
        Err(e) => return Row::failed(name, &format!("no_text:{e}"), el(t0)),
    };
    if code.is_empty() {
        return Row::failed(name, "empty_text", el(t0));
    }
    // Memory guard. The superset graph is per-byte, so peak RSS scales with .text; a few Debian
    // binaries (rustc, clang, the Go toolchain) are large enough to threaten the box. Skipping is
    // recorded as its own status so the report can state exactly what was excluded and why, rather
    // than quietly reporting a rate over the easy binaries only.
    if code.len() > max_code {
        return Row::failed(name, &format!("too_large:{}", code.len()), el(t0));
    }

    let (post, cav) = match run_soft_with_cavity_cfg(base, code, 0.0, 0.0, false) {
        Ok(v) => v,
        Err(e) => return Row::failed(name, &format!("engine_error:{e}"), el(t0)),
    };
    if cav.is_empty() {
        return Row::failed(name, "no_candidates", el(t0));
    }

    let gs = global_and_spatial(&cav);
    let region_ent = region_entropy(code);
    let entropy = byte_entropy(code);

    let (s_glob, s_spat) = (gs.mean_surprise, gs.moran);
    Row {
        name: name.to_string(),
        status: "ok".to_string(),
        code_bytes: code.len(),
        n_cand: post.len(),
        entropy,
        region_ent,
        s_glob,
        s_spat,
        s_nis: gs.mean_nis,
        fire_glob: s_glob > DET_GLOB_HI,
        fire_spat: s_spat > DET_SPAT_HI,
        rule_pick: clf.classify_rule(s_glob, s_spat),
        guard_pick: clf.classify_guard(s_glob, s_spat, region_ent),
        secs: el(t0),
    }
}

/// Shannon entropy of the `.text` byte histogram, bits/byte. Same definition the other probes use.
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

// ── Resume support (same idiom as optregime/switching) ────────────────────────

/// Names already recorded, so a resumed run skips them.
///
/// `too_large` rows are deliberately *not* treated as done: that status is a property of the
/// `--max-code-bytes` budget of the run that wrote it, not of the binary. Excluding them from the
/// skip set means raising the cap and re-running picks up exactly the previously-skipped binaries
/// and nothing else. (Duplicate rows for those names can then appear in the CSV; `write_summary`
/// and the analyzer both take the *last* row per name, so the newer verdict wins.)
fn read_existing(path: &Path) -> HashSet<String> {
    let mut seen = HashSet::new();
    if let Ok(txt) = fs::read_to_string(path) {
        for line in txt.lines().skip(1) {
            let mut it = line.split(',');
            let (Some(name), Some(status)) = (it.next(), it.next()) else { continue };
            if name.is_empty() || status.starts_with("too_large") {
                continue;
            }
            seen.insert(name.to_string());
        }
    }
    seen
}

fn open_csv_append(path: &Path) -> Result<fs::File> {
    let fresh = !path.exists();
    let mut f = fs::OpenOptions::new().create(true).append(true).open(path)?;
    if fresh {
        writeln!(f, "{HEADER}")?;
        f.flush()?;
    }
    Ok(f)
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let mut bins: Option<PathBuf> = None;
    let mut out = PathBuf::from("firerate.csv");
    let mut summary = PathBuf::from("firerate_summary.json");
    let mut max_code: usize = 3_000_000;
    let mut limit: usize = usize::MAX;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--bins" => { bins = Some(PathBuf::from(&args[i + 1])); i += 2; }
            "--out" => { out = PathBuf::from(&args[i + 1]); i += 2; }
            "--summary" => { summary = PathBuf::from(&args[i + 1]); i += 2; }
            "--max-code-bytes" => { max_code = args[i + 1].parse()?; i += 2; }
            "--limit" => { limit = args[i + 1].parse()?; i += 2; }
            other => bail!("unknown arg {other}"),
        }
    }
    let bins = bins.context("--bins DIR is required")?;

    let mut paths: Vec<PathBuf> = fs::read_dir(&bins)
        .with_context(|| format!("reading {}", bins.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    paths.sort();

    let clf = published_classifier();
    let done = read_existing(&out);
    let mut csv = open_csv_append(&out)?;

    eprintln!(
        "── firerate: {} binaries in {} ({} already done) ──",
        paths.len(),
        bins.display(),
        done.len()
    );
    eprintln!(
        "   detection nulls: S_glob>{DET_GLOB_HI}  S_spat>{DET_SPAT_HI}\n   \
         routing rule:    S_glob>{RULE_GLOB_HI}  S_spat>{RULE_SPAT_HI}  guard region_ent>{PACK_ENT_LO}"
    );

    let mut rows: Vec<Row> = Vec::new();
    let mut n = 0usize;
    for p in &paths {
        if n >= limit {
            break;
        }
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if done.contains(&name) {
            continue;
        }
        n += 1;
        let row = measure(p, &name, max_code, &clf);
        eprintln!(
            "[{n}] {name}: {} S_glob={:.4} S_spat={:.4} rule={} guard={} ({:.1}s)",
            row.status,
            row.s_glob,
            row.s_spat,
            row.rule_pick.tag(),
            row.guard_pick.tag(),
            row.secs
        );
        row.write(&mut csv)?;
        rows.push(row);
        // `bytes`, the superset, the converged graph and the cavity vector are all dropped at the end
        // of `measure` — one binary in memory at a time, same discipline as optregime.
    }

    write_summary(&summary, &out)?;
    eprintln!("── wrote {} and {} ──", out.display(), summary.display());
    Ok(())
}

/// Recompute the headline rates from the *whole* CSV (including rows carried over from a previous
/// resumed run) so the summary always describes the complete corpus, not just this invocation.
fn write_summary(path: &Path, csv_path: &Path) -> Result<()> {
    let txt = fs::read_to_string(csv_path)?;
    let mut total = 0usize;
    let mut ok = 0usize;
    let (mut fg, mut fs_, mut fa, mut fb) = (0usize, 0usize, 0usize, 0usize);
    let (mut r_benign, mut r_packed, mut r_obf) = (0usize, 0usize, 0usize);
    let (mut g_benign, mut g_packed, mut g_obf) = (0usize, 0usize, 0usize);
    let mut vetoed = 0usize;
    let mut globs: Vec<f64> = Vec::new();
    let mut spats: Vec<f64> = Vec::new();

    // Last row per binary wins — see `read_existing`: raising --max-code-bytes and resuming appends
    // a fresh verdict for a name that previously read `too_large`.
    let mut latest: std::collections::BTreeMap<&str, &str> = std::collections::BTreeMap::new();
    for line in txt.lines().skip(1) {
        if let Some(name) = line.split(',').next() {
            if !name.is_empty() {
                latest.insert(name, line);
            }
        }
    }

    for line in latest.values() {
        let c: Vec<&str> = line.split(',').collect();
        if c.len() < 17 {
            continue;
        }
        total += 1;
        if c[1] != "ok" {
            continue;
        }
        ok += 1;
        if c[9] == "1" { fg += 1; }
        if c[10] == "1" { fs_ += 1; }
        if c[11] == "1" { fa += 1; }
        if c[12] == "1" { fb += 1; }
        match c[13] { "benign" => r_benign += 1, "packed" => r_packed += 1, "obfuscated" => r_obf += 1, _ => {} }
        match c[14] { "benign" => g_benign += 1, "packed" => g_packed += 1, "obfuscated" => g_obf += 1, _ => {} }
        if c[15] == "1" { vetoed += 1; }
        if let Ok(v) = c[6].parse::<f64>() { globs.push(v); }
        if let Ok(v) = c[7].parse::<f64>() { spats.push(v); }
    }

    globs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    spats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |v: &[f64], q: f64| -> f64 {
        if v.is_empty() { return f64::NAN; }
        let i = ((v.len() - 1) as f64 * q).round() as usize;
        v[i]
    };
    let rate = |x: usize| if ok > 0 { x as f64 / ok as f64 } else { f64::NAN };

    let mut s = String::from("{\n");
    s.push_str(&format!("  \"n_total\": {total},\n"));
    s.push_str(&format!("  \"n_analyzed\": {ok},\n"));
    s.push_str(&format!("  \"det_glob_hi\": {DET_GLOB_HI},\n"));
    s.push_str(&format!("  \"det_spat_hi\": {DET_SPAT_HI},\n"));
    s.push_str(&format!("  \"rule_glob_hi\": {RULE_GLOB_HI},\n"));
    s.push_str(&format!("  \"rule_spat_hi\": {RULE_SPAT_HI},\n"));
    s.push_str(&format!("  \"pack_ent_lo\": {PACK_ENT_LO},\n"));
    s.push_str(&format!("  \"fire_glob\": {fg},\n  \"fire_glob_rate\": {:.4},\n", rate(fg)));
    s.push_str(&format!("  \"fire_spat\": {fs_},\n  \"fire_spat_rate\": {:.4},\n", rate(fs_)));
    s.push_str(&format!("  \"fire_any\": {fa},\n  \"fire_any_rate\": {:.4},\n", rate(fa)));
    s.push_str(&format!("  \"fire_both\": {fb},\n  \"fire_both_rate\": {:.4},\n", rate(fb)));
    s.push_str(&format!("  \"rule_benign\": {r_benign},\n  \"rule_packed\": {r_packed},\n  \"rule_obf\": {r_obf},\n"));
    s.push_str(&format!("  \"guard_benign\": {g_benign},\n  \"guard_packed\": {g_packed},\n  \"guard_obf\": {g_obf},\n"));
    s.push_str(&format!("  \"rule_switch_rate\": {:.4},\n", rate(r_packed + r_obf)));
    s.push_str(&format!("  \"guard_switch_rate\": {:.4},\n", rate(g_packed + g_obf)));
    s.push_str(&format!("  \"guard_vetoed\": {vetoed},\n  \"guard_veto_rate\": {:.4},\n", rate(vetoed)));
    s.push_str(&format!("  \"s_glob_p50\": {:.4},\n  \"s_glob_p95\": {:.4},\n  \"s_glob_max\": {:.4},\n",
        pct(&globs, 0.50), pct(&globs, 0.95), globs.last().copied().unwrap_or(f64::NAN)));
    s.push_str(&format!("  \"s_spat_p50\": {:.4},\n  \"s_spat_p95\": {:.4},\n  \"s_spat_max\": {:.4}\n",
        pct(&spats, 0.50), pct(&spats, 0.95), spats.last().copied().unwrap_or(f64::NAN)));
    s.push_str("}\n");
    fs::write(path, s)?;
    Ok(())
}
