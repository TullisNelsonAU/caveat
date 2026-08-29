//! `honesty` — a confidence-versus-entropy probe for the Soft disassembler.
//!
//! The question it answers: when Soft is handed bytes that *aren't* code — the compressed
//! payload of a packed binary, an encrypted blob, embedded data — does its confidence
//! actually back off, or does it keep calling random-looking bytes "instructions"?
//!
//! We use local byte-entropy as a label-free stand-in for "this region is compressed."
//! Real machine code has structure: opcodes, registers, and immediates repeat, so a window
//! of code lands well under the 8-bit ceiling. Compressed or encrypted data is close to
//! uniform and sits right against it. So we slide a window across the code region, bin every
//! candidate address by the entropy around it, and watch Soft's confident-code rate as the
//! entropy climbs.
//!
//! An honest model's confident-code rate should fall off a cliff in the high-entropy bins —
//! it has no business calling compressed bytes code. If the rate stays flat instead, Soft is
//! over-committing on data, and that flat tail is precisely the behavior an uncertainty-
//! technique fix would target.
//!
//! ```text
//! honesty <binary> [--window N] [--bins N] [--csv out.csv]
//! ```

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use probdisasm::{
    extract_all_hints, extract_hint_pairs, extract_text_section, Analysis, AnalysisConfig,
    AnalysisMode, Superset,
};

/// Posteriors at or above this count as "confident code"; at or below [`DATA`] as "confident
/// data". The band between is the model hedging — the honest answer over ambiguous bytes.
const CODE: f64 = 0.9;
const DATA: f64 = 0.1;

/// We treat windows below this entropy as "structured" (looks like code) and at or above
/// [`COMPRESSED_BITS`] as "compressed" (looks like data) when summarizing the two ends.
const STRUCTURED_BITS: f64 = 5.0;
const COMPRESSED_BITS: f64 = 7.0;

/// The Shannon-entropy ceiling for a byte stream: log2(256).
const MAX_ENTROPY_BITS: f64 = 8.0;

fn main() -> Result<()> {
    let args = Args::parse(std::env::args().skip(1))?;

    let bytes =
        fs::read(&args.binary).with_context(|| format!("reading {}", args.binary.display()))?;

    // Pull out the code region. Thanks to the segment fallback in `extract_text_section`,
    // this now succeeds on headerless / packed inputs (e.g. a UPX'd binary) instead of
    // erroring on the missing `.text` section — which is the whole point of probing here.
    let (base, code) = extract_text_section(&bytes)?;
    // Print exactly what we're feeding Soft. On packed / headerless input the region comes
    // from the executable-segment fallback, and knowing its base+size is how we tell whether
    // we're actually looking at the compressed payload or just the unpacker stub.
    eprintln!(
        "region: base=0x{base:x}  {} bytes of code  (entropy_prior_strength={})",
        code.len(),
        args.entropy_strength
    );
    let posteriors = run_soft(base, code, args.entropy_strength).context("running the Soft model")?;

    let curve = Curve::build(&posteriors, base, code, args.half_window, args.bins);
    curve.report(&args.binary.display().to_string());

    // The numbers that actually matter: with ground truth we report the two axes separately —
    // calibration (honest: does P mean what it says) and discrimination (accurate: does it rank
    // code above data). The entropy histogram above can't tell an accuracy gain from a
    // calibration gain from plain over-suppression; these can.
    if let Some(gt_path) = &args.gt {
        let gt = load_gt(gt_path)?;
        report_calibration(&posteriors, &gt);
    }

    if let Some(path) = &args.csv {
        curve
            .write_csv(path)
            .with_context(|| format!("writing {}", path.display()))?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

/// Run the Soft model over a code region and hand back its per-address posteriors.
///
/// This is deliberately the same pipeline as [`probdisasm::disassemble`], unrolled so we keep
/// `base` and `code` in hand — we need the raw bytes to measure the entropy around each
/// address, which the convenience wrapper doesn't expose.
fn run_soft(base: u64, code: &[u8], entropy_strength: f64) -> Result<Vec<(u64, f64)>> {
    let superset = Superset::new(base, code)?;
    let priors = extract_all_hints(&superset);
    let pairs = extract_hint_pairs(&superset);
    // `AnalysisConfig::default()` defaults `mode` to Hard (the Miller baseline), so we have
    // to ask for Soft explicitly — otherwise we'd be probing the baseline, not our engine.
    // `entropy_strength` is the knob under test: 0.0 reproduces the old behavior.
    let config = AnalysisConfig {
        mode: AnalysisMode::Soft,
        entropy_prior_strength: entropy_strength,
        ..AnalysisConfig::default()
    };
    let mut analysis = Analysis::new(&superset);
    analysis.run_with_config(&priors, &pairs, &config);
    Ok(analysis.sorted_posteriors())
}

/// Shannon entropy, in bits, of the byte-value distribution in `window`. Returns 0 when every
/// byte is identical and approaches [`MAX_ENTROPY_BITS`] as the distribution flattens toward
/// uniform — which is what compressed and encrypted data look like.
fn shannon_entropy(window: &[u8]) -> f64 {
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

/// One entropy bucket and the posterior statistics of the candidates that fell into it.
#[derive(Clone, Default)]
struct Bin {
    candidates: usize,
    sum_posterior: f64,
    confident_code: usize,
    confident_data: usize,
}

impl Bin {
    fn add(&mut self, posterior: f64) {
        self.candidates += 1;
        self.sum_posterior += posterior;
        if posterior >= CODE {
            self.confident_code += 1;
        } else if posterior <= DATA {
            self.confident_data += 1;
        }
    }

    fn mean_posterior(&self) -> f64 {
        self.sum_posterior / self.candidates.max(1) as f64
    }

    fn code_rate(&self) -> f64 {
        self.confident_code as f64 / self.candidates.max(1) as f64
    }

    fn data_rate(&self) -> f64 {
        self.confident_data as f64 / self.candidates.max(1) as f64
    }
}

/// The confidence-versus-entropy curve: every candidate bucketed by the local byte-entropy
/// around it, with each bucket's Soft posterior summary.
struct Curve {
    bins: Vec<Bin>,
    /// Entropy span of a single bucket, in bits (`MAX_ENTROPY_BITS / bins.len()`).
    bin_width_bits: f64,
}

impl Curve {
    fn build(
        posteriors: &[(u64, f64)],
        base: u64,
        code: &[u8],
        half_window: usize,
        n_bins: usize,
    ) -> Self {
        let bin_width_bits = MAX_ENTROPY_BITS / n_bins as f64;
        let mut bins = vec![Bin::default(); n_bins];

        for &(addr, posterior) in posteriors {
            // Map the address back into the code slice, skipping anything outside it.
            let Some(offset) = addr.checked_sub(base).map(|o| o as usize) else {
                continue;
            };
            if offset >= code.len() {
                continue;
            }

            let lo = offset.saturating_sub(half_window);
            let hi = (offset + half_window).min(code.len());
            let entropy = shannon_entropy(&code[lo..hi]);

            // Entropy can touch exactly the ceiling, so clamp the top edge into the last bin.
            let idx = ((entropy / bin_width_bits) as usize).min(n_bins - 1);
            bins[idx].add(posterior);
        }

        Self {
            bins,
            bin_width_bits,
        }
    }

    /// Confident-code rate (and candidate count) aggregated over the bins whose center
    /// entropy satisfies `include` — used to summarize the structured vs. compressed ends.
    fn code_rate_where(&self, include: impl Fn(f64) -> bool) -> (f64, usize) {
        let (mut code, mut total) = (0usize, 0usize);
        for (i, bin) in self.bins.iter().enumerate() {
            let center = (i as f64 + 0.5) * self.bin_width_bits;
            if include(center) {
                code += bin.confident_code;
                total += bin.candidates;
            }
        }
        let rate = if total == 0 {
            0.0
        } else {
            code as f64 / total as f64
        };
        (rate, total)
    }

    fn report(&self, name: &str) {
        let total: usize = self.bins.iter().map(|b| b.candidates).sum();
        println!("honesty curve for {name}  ({total} candidates)");
        println!(
            "  {:>11}  {:>8}  {:>6}  {:>9}  {:>9}",
            "entropy(b)", "count", "meanP", "code>=.9", "data<=.1"
        );
        for (i, bin) in self.bins.iter().enumerate() {
            if bin.candidates == 0 {
                continue;
            }
            let lo = i as f64 * self.bin_width_bits;
            let hi = lo + self.bin_width_bits;
            println!(
                "  {lo:4.1}–{hi:<4.1}  {:>8}  {:>6.3}  {:>8.1}%  {:>8.1}%",
                bin.candidates,
                bin.mean_posterior(),
                100.0 * bin.code_rate(),
                100.0 * bin.data_rate(),
            );
        }

        // The headline: compare confident-code at the two ends. Honest behavior is a steep
        // drop from the structured end to the compressed end.
        let (structured, n_lo) = self.code_rate_where(|e| e < STRUCTURED_BITS);
        let (compressed, n_hi) = self.code_rate_where(|e| e >= COMPRESSED_BITS);
        println!();
        println!(
            "  structured  (entropy < {STRUCTURED_BITS:.0}b): confident-code {:>5.1}%  (n={n_lo})",
            100.0 * structured
        );
        println!(
            "  compressed (entropy ≥ {COMPRESSED_BITS:.0}b): confident-code {:>5.1}%  (n={n_hi})",
            100.0 * compressed
        );
        println!("  → {}", verdict(structured, compressed, n_hi));
    }

    fn write_csv(&self, path: &Path) -> Result<()> {
        use std::fmt::Write as _;
        let mut out = String::from("entropy_lo,entropy_hi,count,mean_posterior,code_rate,data_rate\n");
        for (i, bin) in self.bins.iter().enumerate() {
            let lo = i as f64 * self.bin_width_bits;
            let hi = lo + self.bin_width_bits;
            writeln!(
                out,
                "{lo:.3},{hi:.3},{},{:.6},{:.6},{:.6}",
                bin.candidates,
                bin.mean_posterior(),
                bin.code_rate(),
                bin.data_rate(),
            )
            .expect("writing to a String cannot fail");
        }
        fs::write(path, out)?;
        Ok(())
    }
}

/// Turn the two end-rates into a one-line read. The honest-uncertainty claim is an *absolute*
/// one — Soft should put almost no confident-code over bytes that look compressed — so we judge
/// on the compressed rate itself and quote the structured rate alongside for contrast. (Judging
/// on a structured/compressed ratio breaks when a heavily-packed binary has ~no structured code
/// to use as a baseline.) The thresholds are judgement calls, not load-bearing science.
fn verdict(structured: f64, compressed: f64, n_compressed: usize) -> String {
    if n_compressed == 0 {
        return "no compressed-looking region in this binary to judge".into();
    }
    let (s, c) = (100.0 * structured, 100.0 * compressed);
    let read = if compressed <= 0.05 {
        "Soft stays honest on compressed bytes"
    } else if compressed >= 0.15 {
        "Soft over-commits on compressed bytes — the fix target"
    } else {
        "partial honesty — a tail of over-commitment on compressed bytes"
    };
    format!("{read} (confident-code {c:.1}% compressed vs {s:.1}% structured)")
}

/// Load a ground-truth file: one hex instruction-start address per line (the `.gt` format used
/// across the project). Anything that isn't valid hex is skipped.
fn load_gt(path: &Path) -> Result<HashSet<u64>> {
    let text = fs::read_to_string(path).with_context(|| format!("reading gt {}", path.display()))?;
    let mut set = HashSet::new();
    for line in text.lines() {
        let t = line.trim().trim_start_matches("0x");
        if !t.is_empty() {
            if let Ok(addr) = u64::from_str_radix(t, 16) {
                set.insert(addr);
            }
        }
    }
    Ok(set)
}

/// Report the two axes against ground truth, kept deliberately separate so we can tell which one
/// a lever actually moved:
///   * HONEST   — ECE and Brier-reliability: does a stated 0.5 land at a 50% real code-rate?
///   * ACCURATE — AUROC and Brier-resolution: does the score rank true instructions above data?
/// A change can buy one at the other's expense, so we never collapse them into one "better".
fn report_calibration(posteriors: &[(u64, f64)], gt: &HashSet<u64>) {
    // Label every candidate by truth: 1 if its address is a real instruction start, else 0.
    let pairs: Vec<(f64, f64)> = posteriors
        .iter()
        .map(|&(addr, p)| (p, if gt.contains(&addr) { 1.0 } else { 0.0 }))
        .collect();
    let n = pairs.len();
    if n == 0 {
        println!("calibration: no candidates to score");
        return;
    }
    let base_rate = pairs.iter().map(|&(_, y)| y).sum::<f64>() / n as f64;
    let brier = pairs.iter().map(|&(p, y)| (p - y).powi(2)).sum::<f64>() / n as f64;
    let (ece, reliability, resolution) = ece_and_decomposition(&pairs, 10);
    let uncertainty = base_rate * (1.0 - base_rate);

    println!("calibration vs GT  (n={n}, base-rate code={base_rate:.3})");
    println!("  HONEST   : ECE {ece:.4}   Brier {brier:.4}   reliability {reliability:.4} (lower = better)");
    match auroc(&pairs) {
        Some(a) => println!(
            "  ACCURATE : AUROC {a:.4}   resolution {resolution:.4} (higher = better)   [uncertainty {uncertainty:.4}]"
        ),
        None => println!("  ACCURATE : AUROC n/a (need both code and data present in GT)"),
    }
}

/// Equal-width 10-bin ECE, plus the Murphy/Brier reliability and resolution terms. Reliability is
/// the squared calibration gap (the honest axis); resolution is how far each bin's accuracy moves
/// from the base rate (the discriminative signal). They satisfy Brier = reliability − resolution +
/// uncertainty up to binning, so this one pass splits honesty from accuracy.
fn ece_and_decomposition(pairs: &[(f64, f64)], n_bins: usize) -> (f64, f64, f64) {
    let n = pairs.len() as f64;
    let base = pairs.iter().map(|&(_, y)| y).sum::<f64>() / n;
    let mut conf_sum = vec![0.0; n_bins];
    let mut acc_sum = vec![0.0; n_bins];
    let mut count = vec![0usize; n_bins];
    for &(p, y) in pairs {
        let b = ((p * n_bins as f64) as usize).min(n_bins - 1);
        conf_sum[b] += p;
        acc_sum[b] += y;
        count[b] += 1;
    }
    let (mut ece, mut reliability, mut resolution) = (0.0, 0.0, 0.0);
    for b in 0..n_bins {
        if count[b] == 0 {
            continue;
        }
        let nb = count[b] as f64;
        let conf = conf_sum[b] / nb;
        let acc = acc_sum[b] / nb;
        let w = nb / n;
        ece += w * (conf - acc).abs();
        reliability += w * (conf - acc).powi(2);
        resolution += w * (acc - base).powi(2);
    }
    (ece, reliability, resolution)
}

/// AUROC via the rank-sum (Mann–Whitney) identity, averaging ranks within ties. Returns None when
/// a class is empty (AUROC is undefined then). It's invariant to any monotone re-scaling of the
/// scores, so it measures pure discrimination — the accuracy axis, untouched by calibration.
fn auroc(pairs: &[(f64, f64)]) -> Option<f64> {
    let n_pos = pairs.iter().filter(|&&(_, y)| y > 0.5).count();
    let n_neg = pairs.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return None;
    }
    let mut idx: Vec<usize> = (0..pairs.len()).collect();
    idx.sort_by(|&a, &b| {
        pairs[a]
            .0
            .partial_cmp(&pairs[b].0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Assign average ranks (1-based) so ties contribute 0.5 to the AUROC, as they should.
    let mut ranks = vec![0.0f64; pairs.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && pairs[idx[j + 1]].0 == pairs[idx[i]].0 {
            j += 1;
        }
        let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        for &k in &idx[i..=j] {
            ranks[k] = avg_rank;
        }
        i = j + 1;
    }
    let sum_pos_ranks: f64 = pairs
        .iter()
        .zip(&ranks)
        .filter(|((_, y), _)| *y > 0.5)
        .map(|(_, &r)| r)
        .sum();
    let auc =
        (sum_pos_ranks - (n_pos * (n_pos + 1)) as f64 / 2.0) / (n_pos as f64 * n_neg as f64);
    Some(auc)
}

/// Parsed command line. Kept tiny on purpose — when this graduates out of `experimental/` it
/// can grow a real arg parser, but a hand-rolled one keeps the dependency surface honest here.
struct Args {
    binary: PathBuf,
    half_window: usize,
    bins: usize,
    csv: Option<PathBuf>,
    /// Strength of the engine's entropy-aware data prior; 0.0 leaves the engine unchanged.
    entropy_strength: f64,
    /// Optional ground-truth `.gt` file; when present we report ECE / Brier / AUROC.
    gt: Option<PathBuf>,
}

/// Bytes of context on each side of a candidate when measuring local entropy. The window has
/// to be wide enough that compressed data can actually reach ~8 bits: a W-byte window holds at
/// most W distinct values, so its entropy ceiling is log2(W). A 64-byte window would top out at
/// 6 bits and never look "compressed" — so we default to 128 each side (256 total), which spans
/// the full 0..8-bit range.
const DEFAULT_HALF_WINDOW: usize = 128;
const DEFAULT_BINS: usize = 16;

impl Args {
    fn parse(mut it: impl Iterator<Item = String>) -> Result<Self> {
        const USAGE: &str =
            "usage: honesty <binary> [--window N] [--bins N] [--entropy-strength F] [--gt file.gt] [--csv out.csv]";

        let mut binary = None;
        let mut half_window = DEFAULT_HALF_WINDOW;
        let mut bins = DEFAULT_BINS;
        let mut csv = None;
        let mut entropy_strength = 0.0;
        let mut gt = None;

        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--window" => half_window = (parse_next(&mut it, "--window")? / 2).max(1),
                "--bins" => bins = parse_next(&mut it, "--bins")?,
                "--entropy-strength" => entropy_strength = parse_next_f64(&mut it, "--entropy-strength")?,
                "--gt" => gt = Some(PathBuf::from(take_next(&mut it, "--gt")?)),
                "--csv" => csv = Some(PathBuf::from(take_next(&mut it, "--csv")?)),
                "-h" | "--help" => {
                    eprintln!("{USAGE}");
                    std::process::exit(0);
                }
                other if !other.starts_with('-') && binary.is_none() => {
                    binary = Some(PathBuf::from(other))
                }
                other => bail!("unexpected argument: {other}"),
            }
        }

        let binary = binary.context(USAGE)?;
        if bins == 0 {
            bail!("--bins must be at least 1");
        }
        if entropy_strength < 0.0 {
            bail!("--entropy-strength must be non-negative");
        }
        Ok(Self {
            binary,
            half_window,
            bins,
            csv,
            entropy_strength,
            gt,
        })
    }
}

/// Take the next argument, erroring with the flag name if it's missing.
fn take_next(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    it.next().with_context(|| format!("{flag} needs a value"))
}

/// Take the next argument and parse it as a `usize`, with a flag-aware error message.
fn parse_next(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<usize> {
    take_next(it, flag)?
        .parse()
        .with_context(|| format!("{flag} expects a non-negative integer"))
}

/// Take the next argument and parse it as an `f64`, with a flag-aware error message.
fn parse_next_f64(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<f64> {
    take_next(it, flag)?
        .parse()
        .with_context(|| format!("{flag} expects a number"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_is_zero_for_constant_bytes() {
        assert_eq!(shannon_entropy(&[0xAA; 64]), 0.0);
    }

    #[test]
    fn entropy_hits_the_ceiling_for_all_distinct_bytes() {
        let all: Vec<u8> = (0..=255).collect();
        // 256 equally-likely values -> log2(256) = 8 bits, within float slop.
        assert!((shannon_entropy(&all) - MAX_ENTROPY_BITS).abs() < 1e-9);
    }

    #[test]
    fn binning_lands_high_entropy_in_the_last_bucket() {
        let code: Vec<u8> = (0..=255).collect(); // all distinct -> ~8 bits over a wide window
        let posteriors = vec![(128u64, 0.95)]; // centered so the window spans all 256 bytes
        let curve = Curve::build(&posteriors, 0, &code, 128, 16);
        assert_eq!(curve.bins.last().unwrap().confident_code, 1);
    }
}
