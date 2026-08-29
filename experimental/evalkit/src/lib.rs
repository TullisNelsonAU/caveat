//! `evalkit` — the shared machinery behind the honesty/accuracy experiments.
//!
//! One home for the pieces every front-end (`honesty`, `synth`, `sweep`) needs, so the metric
//! and engine-call definitions can't drift apart:
//!   * [`run_soft`]   — run the Soft model at a given entropy-prior strength,
//!   * [`evaluate`]    — score posteriors against ground truth on both axes (honest + accurate),
//!   * [`make_data`]   — generate a high-entropy "data half" (xorshift or real-compressed),
//!   * [`extract_text`]/[`load_gt`] — the small ELF/GT helpers.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{bail, Context, Result};
use goblin::Object;
use probdisasm::{
    extract_all_hints, extract_hint_pairs_with, Analysis, AnalysisConfig, AnalysisMode, CavityStat,
    Superset,
};

pub use probdisasm::CavityStat as EngineCavityStat;

// ── Engine ────────────────────────────────────────────────────────────────────

/// Run the Soft model over a code region and return its per-address posteriors.
///
/// Soft is requested explicitly because `AnalysisConfig::default()` defaults to the Hard/Miller
/// baseline. The two levers under study are independent: `entropy_strength` is the entropy-aware
/// data prior (0.0 = off), and `use_dassa` swaps Miller's control-flow hint weights for DASSA's
/// corrected ones. Either, both, or neither — which is what the ablation needs.
pub fn run_soft(
    base: u64,
    code: &[u8],
    entropy_strength: f64,
    use_dassa: bool,
) -> Result<Vec<(u64, f64)>> {
    let superset = Superset::new(base, code)?;
    let priors = extract_all_hints(&superset);
    let pairs = extract_hint_pairs_with(&superset, use_dassa);
    let config = AnalysisConfig {
        mode: AnalysisMode::Soft,
        entropy_prior_strength: entropy_strength,
        ..AnalysisConfig::default()
    };
    let mut analysis = Analysis::new(&superset);
    analysis.run_with_config(&priors, &pairs, &config);
    Ok(analysis.sorted_posteriors())
}

/// Run Soft and return *both* the posteriors and the read-only cavity/surprise stats from the
/// converged graph — the input the consistency detector (Paper 2) needs. Identical inference to
/// [`run_soft`]; the cavity pass is post-hoc and leaves π byte-identical (the engine test
/// `cavity_is_pi_with_local_factor_removed` proves this). Both vectors are address-sorted.
///
/// This is the benign-engine call (both data priors off). The switching experiment needs the
/// regime-specific engine settings too — see [`run_soft_with_cavity_cfg`].
pub fn run_soft_with_cavity(
    base: u64,
    code: &[u8],
    entropy_strength: f64,
    use_dassa: bool,
) -> Result<(Vec<(u64, f64)>, Vec<(u64, CavityStat)>)> {
    run_soft_with_cavity_cfg(base, code, entropy_strength, 0.0, use_dassa)
}

/// The full regime-config engine call: both data-prior knobs exposed. `entropy_strength` is the
/// entropy-aware data prior (pushes high-entropy bytes → data, the packed regime's lever);
/// `chainfwd_strength` is the forward decode-chain-consistency prior (pushes chain-consistent bytes
/// → code, the obfuscated regime's lever). Both default to 0 (the benign engine, bit-for-bit the
/// pre-knob behavior). The switching bank runs the same binary under each `(entropy, chainfwd)` pair
/// so the consistency statistic can pick the config that best fits it. Same post-hoc cavity pass;
/// π untouched. Both vectors address-sorted.
pub fn run_soft_with_cavity_cfg(
    base: u64,
    code: &[u8],
    entropy_strength: f64,
    chainfwd_strength: f64,
    use_dassa: bool,
) -> Result<(Vec<(u64, f64)>, Vec<(u64, CavityStat)>)> {
    let superset = Superset::new(base, code)?;
    let priors = extract_all_hints(&superset);
    let pairs = extract_hint_pairs_with(&superset, use_dassa);
    let config = AnalysisConfig {
        mode: AnalysisMode::Soft,
        entropy_prior_strength: entropy_strength,
        chainfwd_strength,
        ..AnalysisConfig::default()
    };
    let mut analysis = Analysis::new(&superset);
    analysis.run_with_config(&priors, &pairs, &config);
    Ok((analysis.sorted_posteriors(), analysis.sorted_cavity()))
}

// ── ELF / GT helpers ───────────────────────────────────────────────────────────

/// Return the `.text` section's virtual address and bytes from an ELF image.
pub fn extract_text(bytes: &[u8]) -> Result<(u64, &[u8])> {
    let Object::Elf(elf) = Object::parse(bytes)? else {
        bail!("input is not an ELF");
    };
    let text = elf
        .section_headers
        .iter()
        .find(|s| elf.shdr_strtab.get_at(s.sh_name) == Some(".text"))
        .context("no .text section")?;
    let range = text.file_range().context(".text has no file range")?;
    Ok((text.sh_addr, &bytes[range]))
}

/// Load a `.gt` file: one hex instruction-start address per line. Non-hex lines are skipped.
pub fn load_gt(path: &Path) -> Result<HashSet<u64>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading gt {}", path.display()))?;
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

// ── Metrics: the two axes, kept separate ────────────────────────────────────────

/// Calibration (honest) and discrimination (accurate) scored against ground truth. We never fold
/// these into one "better" — a lever can buy one at the other's expense, and we want to see that.
pub struct Metrics {
    pub n: usize,
    pub base_rate: f64,
    /// Honest axis: equal-width ECE, full Brier, and the Brier-reliability term (all lower-better).
    pub ece: f64,
    pub brier: f64,
    pub reliability: f64,
    /// Accurate axis: Brier-resolution (higher-better) and AUROC (None if a class is empty).
    pub resolution: f64,
    pub uncertainty: f64,
    pub auroc: Option<f64>,
}

/// Score posteriors against ground truth (1 = address is a real instruction start).
pub fn evaluate(posteriors: &[(u64, f64)], gt: &HashSet<u64>) -> Metrics {
    let pairs: Vec<(f64, f64)> = posteriors
        .iter()
        .map(|&(addr, p)| (p, if gt.contains(&addr) { 1.0 } else { 0.0 }))
        .collect();
    let n = pairs.len();
    if n == 0 {
        return Metrics {
            n: 0,
            base_rate: 0.0,
            ece: 0.0,
            brier: 0.0,
            reliability: 0.0,
            resolution: 0.0,
            uncertainty: 0.0,
            auroc: None,
        };
    }
    let base_rate = pairs.iter().map(|&(_, y)| y).sum::<f64>() / n as f64;
    let brier = pairs.iter().map(|&(p, y)| (p - y).powi(2)).sum::<f64>() / n as f64;
    let (ece, reliability, resolution) = ece_and_decomposition(&pairs, 10);
    Metrics {
        n,
        base_rate,
        ece,
        brier,
        reliability,
        resolution,
        uncertainty: base_rate * (1.0 - base_rate),
        auroc: auroc(&pairs),
    }
}

/// Equal-width 10-bin ECE plus the Murphy/Brier reliability and resolution terms (reliability =
/// honest axis, resolution = accurate axis; Brier = reliability − resolution + uncertainty up to
/// binning).
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

/// AUROC via the rank-sum (Mann–Whitney) identity, averaging ranks within ties. None when a class
/// is empty. Invariant to monotone re-scaling, so it isolates discrimination from calibration.
pub fn auroc(pairs: &[(f64, f64)]) -> Option<f64> {
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
    Some((sum_pos_ranks - (n_pos * (n_pos + 1)) as f64 / 2.0) / (n_pos as f64 * n_neg as f64))
}

// ── Synthetic data half ──────────────────────────────────────────────────────────

/// What to fill the "data half" of a synthetic two-sided-GT binary with.
#[derive(Clone, Copy, Debug)]
pub enum DataKind {
    /// A uniform pseudo-random stream (~8 bits/byte) — the cleanest high-entropy stand-in.
    Xorshift,
    /// DEFLATE of the source bytes — real compressed structure (~7.5–8 bits), like a packer's body.
    Compressed,
    /// Real instruction bytes tiled from the source `.text` — a *low*-entropy, self-consistent
    /// control-flow chain. This is the hard adversary: it sits below the entropy floor (so the
    /// entropy prior can't suppress it) and it fires the conv/cross CF hint-pairs (so it's the only
    /// place DASSA's corrected weights can bite). Models embedded/decoy code, not a packer body.
    Code,
}

/// Generate `n` bytes of "data" standing in for an adversarial region. [`DataKind::Compressed`]
/// DEFLATEs `source` and tiles it; [`DataKind::Code`] tiles the raw `source` instruction bytes
/// (real CF-consistent code); [`DataKind::Xorshift`] ignores `source`.
pub fn make_data(kind: DataKind, seed: u64, source: &[u8], n: usize) -> Vec<u8> {
    match kind {
        DataKind::Xorshift => xorshift_stream(seed, n),
        DataKind::Compressed => {
            let comp = deflate(source);
            if comp.is_empty() {
                // Degenerate (empty source) — fall back so we never emit a zero-entropy block.
                return xorshift_stream(seed, n);
            }
            comp.into_iter().cycle().take(n).collect()
        }
        DataKind::Code => {
            if source.is_empty() {
                return xorshift_stream(seed, n);
            }
            // Tile the real code. Start a third of the way in so the tiled copy doesn't share the
            // code half's exact prefix — a distinct but still self-consistent instruction blob.
            let skip = (source.len() / 3).min(source.len().saturating_sub(1));
            source.iter().cycle().skip(skip).take(n).copied().collect()
        }
    }
}

/// Build a minimal headerless ELF64: one R+X `PT_LOAD` segment at `vaddr` holding `payload`, no
/// section headers — the shape the engine's segment fallback handles and what packed/obfuscated
/// binaries look like in the wild. Shared so `synth` and `forge` can't drift on the layout.
pub fn build_min_elf(vaddr: u64, payload: &[u8]) -> Vec<u8> {
    const EHDR_LEN: u64 = 64;
    const PHDR_LEN: u64 = 56;
    const PAYLOAD_OFFSET: u64 = EHDR_LEN + PHDR_LEN;

    let mut elf = Vec::with_capacity(PAYLOAD_OFFSET as usize + payload.len());

    // ── ELF header (64 bytes) ──
    elf.extend_from_slice(&[0x7f, b'E', b'L', b'F', 2, 1, 1, 0]); // magic, ELF64, LE, v1, SysV
    elf.extend_from_slice(&[0u8; 8]); // rest of e_ident
    elf.extend_from_slice(&2u16.to_le_bytes()); // e_type = ET_EXEC
    elf.extend_from_slice(&62u16.to_le_bytes()); // e_machine = EM_X86_64
    elf.extend_from_slice(&1u32.to_le_bytes()); // e_version
    elf.extend_from_slice(&vaddr.to_le_bytes()); // e_entry
    elf.extend_from_slice(&EHDR_LEN.to_le_bytes()); // e_phoff
    elf.extend_from_slice(&0u64.to_le_bytes()); // e_shoff = none
    elf.extend_from_slice(&0u32.to_le_bytes()); // e_flags
    elf.extend_from_slice(&(EHDR_LEN as u16).to_le_bytes()); // e_ehsize
    elf.extend_from_slice(&(PHDR_LEN as u16).to_le_bytes()); // e_phentsize
    elf.extend_from_slice(&1u16.to_le_bytes()); // e_phnum
    elf.extend_from_slice(&0u16.to_le_bytes()); // e_shentsize
    elf.extend_from_slice(&0u16.to_le_bytes()); // e_shnum
    elf.extend_from_slice(&0u16.to_le_bytes()); // e_shstrndx

    // ── Program header: one PT_LOAD, R | X ──
    elf.extend_from_slice(&1u32.to_le_bytes()); // p_type = PT_LOAD
    elf.extend_from_slice(&5u32.to_le_bytes()); // p_flags = R | X
    elf.extend_from_slice(&PAYLOAD_OFFSET.to_le_bytes()); // p_offset
    elf.extend_from_slice(&vaddr.to_le_bytes()); // p_vaddr
    elf.extend_from_slice(&vaddr.to_le_bytes()); // p_paddr
    elf.extend_from_slice(&(payload.len() as u64).to_le_bytes()); // p_filesz
    elf.extend_from_slice(&(payload.len() as u64).to_le_bytes()); // p_memsz
    elf.extend_from_slice(&0x1000u64.to_le_bytes()); // p_align

    debug_assert_eq!(elf.len() as u64, PAYLOAD_OFFSET);
    elf.extend_from_slice(payload);
    elf
}

/// A deterministic xorshift64 byte stream — output is near-uniform, ~8 bits of entropy per byte.
fn xorshift_stream(seed: u64, n: usize) -> Vec<u8> {
    let mut state = seed | 1; // xorshift must never see a zero state
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.extend_from_slice(&state.to_le_bytes());
    }
    out.truncate(n);
    out
}

/// Raw DEFLATE of `data` at best compression — a real compressed byte stream.
fn deflate(data: &[u8]) -> Vec<u8> {
    use flate2::write::DeflateEncoder;
    use flate2::Compression;
    use std::io::Write;

    let mut enc = DeflateEncoder::new(Vec::new(), Compression::best());
    let _ = enc.write_all(data);
    enc.finish().unwrap_or_default()
}

// ── UPX ground truth: provable negatives from the packer's own format ───────────────
//
// For a UPX-packed binary we have no instruction-level ground truth — but we don't need a
// disassembler to get a *perfect negative* set. UPX compresses the original program into blocks,
// each prefixed by a `b_info` header that records its exact compressed size. Those bytes are
// compressed data the stub reads, never instructions executed in place, so every byte of every
// block is a true negative by construction. We parse the chain straight out of the file — no
// entropy, no disassembly, no inference — and hand back the exact byte range it covers. The label
// is "is this a real instruction start in the static image", and for compressed payload the answer
// is provably no.

/// One UPX compressed block, located by walking the `b_info` chain.
#[derive(Clone, Copy, Debug)]
pub struct UpxBlock {
    /// Uncompressed size recorded in the block header.
    pub sz_unc: u32,
    /// Compressed size recorded in the block header (this many payload bytes follow).
    pub sz_cpr: u32,
    /// UPX compression method id (informational).
    pub method: u8,
    /// File-offset span of the compressed payload bytes (header excluded).
    pub file_data: (usize, usize),
}

/// The pieces of a UPX-packed image we can label with certainty.
#[derive(Clone, Debug)]
pub struct UpxLayout {
    /// Load address of the executable segment our engine actually analyzes.
    pub exec_vaddr: u64,
    /// File-offset span of that executable segment.
    pub exec_file: (usize, usize),
    /// `p_filesize` from the UPX `p_info` (the original program's size) — a parse cross-check.
    pub p_filesize: u32,
    /// Every compressed block found inside the executable segment, in order.
    pub blocks: Vec<UpxBlock>,
    /// Exact compressed-payload extent within the segment, as **vaddr** `[lo, hi)`. Provable
    /// negatives: nothing here is an instruction start.
    pub compressed: (u64, u64),
    /// The single largest block, fully interior (no header/stub adjacency) — the conservative,
    /// zero-boundary-risk negative window, as vaddr `[lo, hi)`.
    pub conservative: (u64, u64),
    /// The unpacker stub: entry point to segment end, as vaddr `[lo, hi)`. Real code — we neither
    /// score it nor label it (getting its starts would need a disassembler).
    pub stub: (u64, u64),
    /// File offset of the `UPX!` `l_info` magic, for reporting.
    pub magic_file_off: usize,
}

/// Parse a UPX-packed ELF and return the byte ranges we can label with certainty.
///
/// Walks the `b_info` chain from the `UPX!` `l_info` magic, accepting blocks while they stay inside
/// the executable segment and pass a sanity bound (the garbage that follows the last real block —
/// e.g. the stub decoded as a header — fails the in-segment / ratio check and stops the walk).
pub fn parse_upx_layout(bytes: &[u8]) -> Result<UpxLayout> {
    use goblin::elf::program_header::{PF_X, PT_LOAD};

    let Object::Elf(elf) = Object::parse(bytes)? else {
        bail!("input is not an ELF");
    };
    let seg = elf
        .program_headers
        .iter()
        .find(|p| p.p_type == PT_LOAD && p.p_flags & PF_X != 0)
        .context("no executable PT_LOAD segment")?;
    let fstart = seg.p_offset as usize;
    let fend = fstart
        .checked_add(seg.p_filesz as usize)
        .filter(|&e| e <= bytes.len())
        .context("executable segment runs past end of file")?;
    let vaddr = seg.p_vaddr;
    let entry = elf.header.e_entry;
    let off2v = |o: usize| vaddr + (o - fstart) as u64;

    // Locate the `UPX!` l_info magic near the top of the segment.
    let scan_end = (fstart + 0x400).min(fend);
    let magic = bytes[fstart..scan_end]
        .windows(4)
        .position(|w| w == b"UPX!")
        .map(|i| fstart + i)
        .context("no UPX! magic in executable segment — not a UPX image?")?;

    // l_info: magic(4) + l_lsize(2) + l_version(1) + l_format(1) = 8 bytes, then p_info(12).
    let p_info = magic + 8;
    let p_filesize = u32::from_le_bytes(bytes[p_info + 4..p_info + 8].try_into().unwrap());

    // Walk the b_info chain.
    let mut off = p_info + 12;
    let mut blocks: Vec<UpxBlock> = Vec::new();
    while off + 12 <= fend {
        let sz_unc = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let sz_cpr = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
        let method = bytes[off + 8];
        let data = off + 12;
        let data_end = match data.checked_add(sz_cpr as usize) {
            Some(e) => e,
            None => break,
        };
        // A real block is non-empty, fits in the segment, and doesn't "expand" implausibly.
        let sane = sz_unc != 0
            && sz_cpr != 0
            && data_end <= fend
            && sz_cpr as u64 <= sz_unc as u64 + 4096;
        if !sane {
            break;
        }
        blocks.push(UpxBlock { sz_unc, sz_cpr, method, file_data: (data, data_end) });
        off = data_end;
        if blocks.len() > 256 {
            break;
        }
    }
    if blocks.is_empty() {
        bail!("no valid UPX b_info blocks in the executable segment");
    }

    let comp_lo = blocks[0].file_data.0;
    let comp_hi = blocks.last().unwrap().file_data.1;
    let big = blocks.iter().max_by_key(|b| b.sz_cpr).unwrap();

    Ok(UpxLayout {
        exec_vaddr: vaddr,
        exec_file: (fstart, fend),
        p_filesize,
        compressed: (off2v(comp_lo), off2v(comp_hi)),
        conservative: (off2v(big.file_data.0), off2v(big.file_data.1)),
        stub: (entry, vaddr + seg.p_filesz),
        magic_file_off: magic,
        blocks,
    })
}

/// What a posterior set looks like over a region we *know* is all data (target = 0 everywhere).
#[derive(Clone, Copy, Debug)]
pub struct NegMetrics {
    /// Candidates that fell in the known-negative window.
    pub n: usize,
    /// False-positive rate: fraction called confident code (P ≥ 0.9). The honesty headline.
    pub fp_rate: f64,
    /// Mean posterior over the window — should sit near 0 for an honest model.
    pub mean_p: f64,
    /// Brier score against the all-zero target (= mean of P²); pure reliability here.
    pub brier: f64,
    /// Largest single posterior in the window (worst-case over-commitment).
    pub max_p: f64,
}

/// Score posteriors over a provably-negative vaddr window `[lo, hi)`. Every address here is known
/// to be non-instruction, so we report over-commitment directly — no positive labels needed.
pub fn evaluate_negatives(posteriors: &[(u64, f64)], lo: u64, hi: u64) -> NegMetrics {
    let ps: Vec<f64> = posteriors
        .iter()
        .filter(|&&(a, _)| a >= lo && a < hi)
        .map(|&(_, p)| p)
        .collect();
    let n = ps.len();
    if n == 0 {
        return NegMetrics { n: 0, fp_rate: 0.0, mean_p: 0.0, brier: 0.0, max_p: 0.0 };
    }
    let fp = ps.iter().filter(|&&p| p >= 0.9).count() as f64 / n as f64;
    let mean = ps.iter().sum::<f64>() / n as f64;
    let brier = ps.iter().map(|p| p * p).sum::<f64>() / n as f64;
    let max_p = ps.iter().cloned().fold(0.0_f64, f64::max);
    NegMetrics { n, fp_rate: fp, mean_p: mean, brier, max_p }
}

// ── Isotonic recalibration (the honesty axis, post-hoc) ─────────────────────────────
//
// A monotone map from raw posteriors to calibrated probabilities, fit by pool-adjacent-violators
// (isotonic regression of label on posterior). It tightens ECE and tames worst-case over-confidence
// (the `max_p=1.0` problem) WITHOUT touching the model: because the remap is monotone, it preserves
// the ranking exactly (AUROC unchanged) while making the numbers mean what they say. This is the
// "add maps as more is discovered" idea — fit on a labeled calibration set, apply to new posteriors
// (transfer: fit on A, apply to B). It is the calibration machinery Layer 2's reachability reuses.

/// A fitted monotone recalibration map (raw posterior → calibrated probability).
#[derive(Clone, Debug)]
pub struct IsotonicMap {
    /// `(x_upper, value)` blocks, sorted by `x` ascending with non-decreasing `value`.
    points: Vec<(f64, f64)>,
}

impl IsotonicMap {
    /// Fit from `(posterior, label)` samples via pool-adjacent-violators (isotonic regression).
    pub fn fit(samples: &[(f64, f64)]) -> Self {
        let mut pts: Vec<(f64, f64)> = samples.to_vec();
        pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        // Each block: (x_upper, sum_of_labels, count). Pool adjacent blocks that violate monotonicity.
        let mut blocks: Vec<(f64, f64, f64)> = Vec::with_capacity(pts.len());
        for (x, y) in pts {
            blocks.push((x, y, 1.0));
            while blocks.len() >= 2 {
                let n = blocks.len();
                let v_last = blocks[n - 1].1 / blocks[n - 1].2;
                let v_prev = blocks[n - 2].1 / blocks[n - 2].2;
                if v_prev <= v_last {
                    break;
                }
                let (x1, s1, c1) = blocks[n - 1];
                let (_x0, s0, c0) = blocks[n - 2];
                blocks.truncate(n - 2);
                blocks.push((x1, s0 + s1, c0 + c1));
            }
        }
        IsotonicMap { points: blocks.iter().map(|&(x, s, c)| (x, s / c)).collect() }
    }

    /// The fitted PAVA blocks `(x_upper, value)` — the map's entire state. Exposed so a fitted map can
    /// be dumped to JSON and a bank of maps serialized (the tool's calibration-map bank). Sorted by
    /// `x` ascending with non-decreasing `value`.
    pub fn to_points(&self) -> Vec<(f64, f64)> {
        self.points.clone()
    }

    /// Reconstruct a map from previously-fitted PAVA points — the load inverse of [`to_points`], so a
    /// serialized bank round-trips to the exact same `apply`. Callers pass points produced by `fit`
    /// (sorted `x` ascending, `value` non-decreasing); they are used as-is, not re-validated. An empty
    /// vec is the identity map (`apply(p) = p`).
    pub fn from_points(points: Vec<(f64, f64)>) -> Self {
        IsotonicMap { points }
    }

    /// Fit from a posterior set and its positive ground-truth set.
    pub fn fit_from_gt(posteriors: &[(u64, f64)], gt: &HashSet<u64>) -> Self {
        let samples: Vec<(f64, f64)> = posteriors
            .iter()
            .map(|&(a, p)| (p, if gt.contains(&a) { 1.0 } else { 0.0 }))
            .collect();
        Self::fit(&samples)
    }

    /// Map a raw posterior to its calibrated value (piecewise-constant, clamped to `[0,1]`).
    pub fn apply(&self, p: f64) -> f64 {
        if self.points.is_empty() {
            return p;
        }
        for &(x, v) in &self.points {
            if p <= x {
                return v.clamp(0.0, 1.0);
            }
        }
        self.points.last().map(|&(_, v)| v.clamp(0.0, 1.0)).unwrap_or(p)
    }

    /// Remap a whole posterior set through the calibration map (ranking preserved).
    pub fn apply_all(&self, posteriors: &[(u64, f64)]) -> Vec<(u64, f64)> {
        posteriors.iter().map(|&(a, p)| (a, self.apply(p))).collect()
    }
}

#[cfg(test)]
mod recal_tests {
    use super::*;

    #[test]
    fn isotonic_monotone_and_sane() {
        // Step target: label 1 above 0.6, 0 below — the map should learn that shape, monotone.
        let s: Vec<(f64, f64)> = (0..=100)
            .map(|i| {
                let p = i as f64 / 100.0;
                (p, if p > 0.6 { 1.0 } else { 0.0 })
            })
            .collect();
        let m = IsotonicMap::fit(&s);
        let mut prev = -1.0;
        for &(_, v) in &m.points {
            assert!(v + 1e-9 >= prev, "calibration map must be monotone");
            prev = v;
        }
        for q in [0.0_f64, 0.3, 0.61, 0.9, 1.0] {
            assert!((0.0..=1.0).contains(&m.apply(q)), "calibrated value out of range");
        }
        assert!(m.apply(0.9) > m.apply(0.1), "map must separate high from low");
    }

    #[test]
    fn to_from_points_round_trips_apply() {
        // A fitted map dumped to points and reloaded must produce byte-identical `apply` — this is the
        // contract a serialized calibration bank relies on.
        let s: Vec<(f64, f64)> = (0..=100).map(|i| (i as f64 / 100.0, if i > 60 { 1.0 } else { 0.0 })).collect();
        let m = IsotonicMap::fit(&s);
        let m2 = IsotonicMap::from_points(m.to_points());
        for q in [0.0_f64, 0.15, 0.5, 0.61, 0.8, 1.0] {
            assert_eq!(m.apply(q), m2.apply(q), "reloaded map diverged at {q}");
        }
    }

    #[test]
    fn degenerate_all_data_map_suppresses_to_zero() {
        // The packed regime's map is fit on the provable-data window (all labels 0). PAVA collapses it
        // to ≈0 everywhere — the abstain/suppress calibration (the paper's packed → ≈0).
        let s: Vec<(f64, f64)> = (0..=100).map(|i| (i as f64 / 100.0, 0.0)).collect();
        let m = IsotonicMap::fit(&s);
        for q in [0.1_f64, 0.4, 0.7, 0.99] {
            assert!(m.apply(q) < 1e-6, "packed map must suppress {q} to ≈0, got {}", m.apply(q));
        }
    }

    #[test]
    fn empty_points_is_identity() {
        let m = IsotonicMap::from_points(Vec::new());
        for q in [0.0_f64, 0.37, 0.5, 1.0] {
            assert_eq!(m.apply(q), q, "empty bank map must be identity");
        }
    }
}
