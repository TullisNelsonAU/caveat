//! adversary — adaptive-adversary probe for Paper 2's cavity-surprise consistency detector.
//!
//! The detector fires on two ground-truth-free statistics of the converged Soft graph:
//!   * `S_glob` = mean per-address soft cavity surprise `s_a = -ln(q0*m0 + q1*m1)` — high when
//!     the local decode evidence `m` disagrees with the structural cavity belief `q`.
//!   * `S_spat` = Moran's I of the standardized residual `(m1-q1)/sqrt(q1(1-q1))` in address order
//!     — high when the disagreement is spatially *clustered* into a block.
//! Clean null (credibility run, n=45): S_glob 0.53–1.16 (p95 ~1.05), S_spat 0.03–0.11 (p95 ~0.10),
//! calibrated ECE ~0.003. Switching thresholds: glob_hi=2.57, spat_hi=0.108. Desync trips S_glob
//! (~35); packing evades S_glob (~1.1) but trips S_spat (~0.19).
//!
//! The adversary's target hole: S_glob < 1.01 AND S_spat < 0.105 AND calibrated ECE >> 0.005.
//! We build controlled constructions whose ground truth we KNOW (we emit the bytes), run them
//! through the #18 engine, and measure whether any lands in the hole — and, for the ones that
//! evade, what the evasion cost the adversary in decode cleanliness / obfuscation strength.
//!
//! GT is by construction, and by the injector's own record only. Every construction below is
//! emitted together with the exact label set it implies: the substrate's true starts come from
//! `gen-gt` (DWARF/symtab, never a disassembler's opinion), and every byte *we* append or insert
//! is labeled by the injector as it writes it — appended decoy/packed/NOP bytes are negatives,
//! desync-shifted real starts are positives at the offset the injector placed them, and the C1b
//! control's decoy positives are the donor's own `gen-gt` starts translated by the append offset.
//! No recovery, disassembly, or heuristic step ever touches the label set. The per-row
//! `gt_positives` / `adv_bytes` / `adv_positives` columns are that record, carried into the CSV so
//! the accounting is auditable without re-running.
//!
//! Determinism: every construction is a pure function of (substrate ELF, donor ELF, `--seed`,
//! `--interleave-seed`). The two seeds default to the values of the run of record; re-running with
//! the same inputs reproduces the CSV byte for byte.

use std::collections::HashSet;
use std::fs;
use std::io::Write as _;
use std::path::Path;

use anyhow::{Context, Result};
use evalkit::{evaluate, extract_text, run_soft_with_cavity, IsotonicMap};

const VADDR: u64 = 0x40_0000; // fixed load address for every construction

// ── clean-null reference (credibility run, for verdict flags) ───────────────────
// The DETECTION null: the detector fires above the clean-FIT 95th percentile of each statistic
// (consistency binary, n=20 clean-fit). This is the bar for "evades the detector." The switching
// binary's coarser thresholds (glob_hi=2.57, spat_hi=0.108) are the *repair classifier* — how it
// routes a flagged binary to a regime map — a more lenient bar we do NOT use for the evasion claim.
const GLOB_HI: f64 = 1.01; // clean-fit S_glob p95 (detection null); above ⇒ flagged
const SPAT_HI: f64 = 0.105; // clean-fit S_spat p95 (detection null); above ⇒ flagged
const ECE_MISCAL: f64 = 0.02; // calibrated ECE above this ⇒ materially miscalibrated

// ── fixed seeds of the run of record ────────────────────────────────────────────
// Two independent byte streams, so they get two seeds: the packed body (C3) and the interleave
// sweep's random half (C5, mixed with the block size k). Overridable with --seed /
// --interleave-seed; the defaults are what produced the committed CSV.
const SEED_PACK: u64 = 0x9E37_79B9_7F4A_7C15;
const SEED_INTERLEAVE: u64 = 0xDEAD_BEEF;

/// Read gen-gt `insn_max.txt` (one hex vaddr per line) as a set of vaddrs.
fn read_gt_vaddrs(path: &Path) -> Result<Vec<u64>> {
    let txt = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut v = Vec::new();
    for line in txt.lines() {
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        v.push(u64::from_str_radix(s.trim_start_matches("0x"), 16)?);
    }
    v.sort_unstable();
    Ok(v)
}

/// The `.text` bytes + the true instruction-start offsets (relative to `.text` start) of a clean
/// ELF, via gen-gt. These offsets are our portable GT: re-place the bytes anywhere and shift.
fn load_clean(elf: &Path, gt: &Path) -> Result<(Vec<u8>, Vec<u64>)> {
    let bytes = fs::read(elf).with_context(|| format!("reading {}", elf.display()))?;
    let (base, code) = extract_text(&bytes)?;
    let starts = read_gt_vaddrs(gt)?;
    let offs: Vec<u64> = starts
        .iter()
        .filter(|&&va| va >= base && (va - base) < code.len() as u64)
        .map(|&va| va - base)
        .collect();
    Ok((code.to_vec(), offs))
}

// ── xorshift + deflate stand-ins (packer body) ──────────────────────────────────
fn xorshift(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        })
        .collect()
}

// ── aggregate statistics (byte-identical to the detector's definitions) ──────────
fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        0.0
    } else {
        x.iter().sum::<f64>() / x.len() as f64
    }
}

/// Moran's I with 1-D contiguity (neighbors = adjacent in the address-sorted vector).
fn morans_i(x: &[f64]) -> f64 {
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
    (n as f64 / (n as f64 - 1.0)) * (num / denom)
}

struct Stats {
    n: usize,
    s_glob: f64,
    s_spat: f64,
    ece_raw: f64,
    ece_cal: f64,
    // region diagnostics (over the adversary-inserted window, all GT-negative for append/pack):
    region_n: usize,
    region_mean_pi_cal: f64,     // false-positive confidence the adversary induced there
    region_fp_conf: usize,       // candidates with calibrated π ≥ 0.8 in region (confident wrong)
    region_s_glob: f64,          // surprise magnitude inside the region only
    region_s_spat: f64,          // spatial clustering inside the region only
    // recovery of the TRUE code (obfuscation cost): fraction of true starts that are candidates
    // whose calibrated π ≥ 0.5. For append constructions this stays ~clean (real code untouched);
    // for desync it drops as the overlap corrupts the stream.
    true_recovered: f64,
}

/// The injector's own record of what it emitted: the label set is built here and nowhere else.
struct Injected {
    family: &'static str,
    gt_positives: usize, // size of the label set fed to `measure`
    adv_bytes: usize,    // bytes the adversary appended or inserted
    adv_positives: usize, // of those bytes, how many the injector labeled as instruction starts
}

fn measure(
    payload: &[u8],
    gt_offsets: &HashSet<u64>,
    region: Option<(u64, u64)>, // [lo,hi) vaddr of the adversary window, if any
    map: &IsotonicMap,
) -> Result<Stats> {
    // Feed the payload straight to the engine (Superset::new takes raw base+code; no ELF wrapper
    // needed, and build_min_elf emits no section headers for extract_text to find anyway).
    let (post, cav) = run_soft_with_cavity(VADDR, payload, 0.0, false)?;

    // GT as vaddrs.
    let gt: HashSet<u64> = gt_offsets.iter().map(|o| VADDR + o).collect();
    let cal = map.apply_all(&post);
    let ece_raw = evaluate(&post, &gt).ece;
    let ece_cal = evaluate(&cal, &gt).ece;

    let s_glob = mean(&cav.iter().map(|(_, c)| c.surprise).collect::<Vec<_>>());
    let resid: Vec<f64> = cav.iter().map(|(_, c)| c.residual).collect();
    let s_spat = morans_i(&resid);

    // region diagnostics
    let (mut region_n, mut region_pi_sum, mut region_fp) = (0usize, 0.0f64, 0usize);
    let cal_map: std::collections::HashMap<u64, f64> = cal.iter().cloned().collect();
    let mut region_surp = Vec::new();
    let mut region_resid = Vec::new();
    if let Some((lo, hi)) = region {
        for (a, c) in &cav {
            if *a >= lo && *a < hi {
                region_n += 1;
                let p = *cal_map.get(a).unwrap_or(&0.0);
                region_pi_sum += p;
                if p >= 0.8 {
                    region_fp += 1;
                }
                region_surp.push(c.surprise);
                region_resid.push(c.residual);
            }
        }
    }

    // true-code recovery: over GT-positive addresses, fraction whose calibrated π ≥ 0.5.
    let (mut tp_total, mut tp_hit) = (0usize, 0usize);
    for (a, p) in &cal {
        if gt.contains(a) {
            tp_total += 1;
            if *p >= 0.5 {
                tp_hit += 1;
            }
        }
    }
    let true_recovered = if tp_total == 0 { 0.0 } else { tp_hit as f64 / tp_total as f64 };

    Ok(Stats {
        n: post.len(),
        s_glob,
        s_spat,
        ece_raw,
        ece_cal,
        region_n,
        region_mean_pi_cal: if region_n == 0 { 0.0 } else { region_pi_sum / region_n as f64 },
        region_fp_conf: region_fp,
        region_s_glob: mean(&region_surp),
        region_s_spat: morans_i(&region_resid),
        true_recovered,
    })
}

/// Fit the clean isotonic map on pooled (posterior,label) from clean binaries (paper's Pass 1).
fn fit_clean_map(clean: &[(Vec<u8>, Vec<u64>)]) -> Result<IsotonicMap> {
    let mut pooled: Vec<(f64, f64)> = Vec::new();
    for (code, offs) in clean {
        let (post, _cav) = run_soft_with_cavity(VADDR, code, 0.0, false)?;
        let gt: HashSet<u64> = offs.iter().map(|o| VADDR + o).collect();
        for (a, p) in &post {
            pooled.push((*p, if gt.contains(a) { 1.0 } else { 0.0 }));
        }
    }
    Ok(IsotonicMap::fit(&pooled))
}

fn fires(s: &Stats) -> String {
    match (s.s_glob > GLOB_HI, s.s_spat > SPAT_HI) {
        (true, true) => "S_glob+S_spat".into(),
        (true, false) => "S_glob".into(),
        (false, true) => "S_spat".into(),
        (false, false) => "none".into(),
    }
}

fn print_row(tag: &str, s: &Stats) {
    let glob_flag = if s.s_glob > GLOB_HI { "FLAG" } else { "  · " };
    let spat_flag = if s.s_spat > SPAT_HI { "FLAG" } else { "  · " };
    let miscal = if s.ece_cal > ECE_MISCAL { "MISCAL" } else { "  ok  " };
    let evades = s.s_glob <= GLOB_HI && s.s_spat <= SPAT_HI;
    let verdict = if evades && s.ece_cal > ECE_MISCAL {
        "*** EVADES+MISCAL ***"
    } else if evades {
        "evades(calibrated)"
    } else {
        "detected"
    };
    println!(
        "{:<24} n={:<6} S_glob={:>7.3} [{}]  S_spat={:>6.3} [{}]  ECEcal={:>6.4} [{}]  ECEraw={:>6.4} | \
         regFPconf={:<5} regMeanPi={:>5.3} regSglob={:>6.3} regSspat={:>6.3} | trueRecov={:>5.3} | {}",
        tag,
        s.n,
        s.s_glob,
        glob_flag,
        s.s_spat,
        spat_flag,
        s.ece_cal,
        miscal,
        s.ece_raw,
        s.region_fp_conf,
        s.region_mean_pi_cal,
        s.region_s_glob,
        s.region_s_spat,
        s.true_recovered,
        verdict,
    );
}

const CSV_HEADER: &str = "construction,family,base,donor,seed,interleave_seed,n_cand,s_glob,s_spat,\
ece_cal,ece_raw,true_recov,gt_positives,adv_bytes,adv_positives,gt_source,region_n,\
region_mean_pi_cal,region_fp_conf,region_s_glob,region_s_spat,fire_glob,fire_spat,fires";

/// One CSV line per constructed binary. `gt_source` is a constant because there is only one:
/// the injector's record. It is written per row so a reader of the CSV alone can see that.
#[allow(clippy::too_many_arguments)]
fn csv_line(tag: &str, inj: &Injected, base: &str, donor: &str, seed: u64, iseed: u64, s: &Stats) -> String {
    format!(
        "{tag},{family},{base},{donor},{seed:#x},{iseed:#x},{n},{sg:.6},{sp:.6},{ec:.6},{er:.6},\
{tr:.6},{gtp},{advb},{advp},injector-record,{rn},{rpi:.6},{rfp},{rsg:.6},{rsp:.6},{fg},{fs},{fires}",
        family = inj.family,
        n = s.n,
        sg = s.s_glob,
        sp = s.s_spat,
        ec = s.ece_cal,
        er = s.ece_raw,
        tr = s.true_recovered,
        gtp = inj.gt_positives,
        advb = inj.adv_bytes,
        advp = inj.adv_positives,
        rn = s.region_n,
        rpi = s.region_mean_pi_cal,
        rfp = s.region_fp_conf,
        rsg = s.region_s_glob,
        rsp = s.region_s_spat,
        fg = u8::from(s.s_glob > GLOB_HI),
        fs = u8::from(s.s_spat > SPAT_HI),
        fires = fires(s),
    )
}

fn usage() -> ! {
    eprintln!(
        "usage: adversary [options] <base.elf> <base.insn_max> <donor.elf> <donor.insn_max> <fitspec...>\n\
         fitspec = pairs of <clean.elf> <clean.insn_max> to fit the clean isotonic map.\n\
         options:\n\
         \x20 --csv <path>            append one row per construction (writes the header if new)\n\
         \x20 --seed <u64>            packed-body seed (default {SEED_PACK:#x})\n\
         \x20 --interleave-seed <u64> interleave random-half seed (default {SEED_INTERLEAVE:#x})"
    );
    std::process::exit(2)
}

fn parse_u64(s: &str) -> u64 {
    let t = s.trim();
    let r = if let Some(h) = t.strip_prefix("0x") { u64::from_str_radix(h, 16) } else { t.parse() };
    r.unwrap_or_else(|_| {
        eprintln!("bad u64: {t}");
        std::process::exit(2)
    })
}

fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (mut csv_path, mut seed, mut iseed) = (None::<String>, SEED_PACK, SEED_INTERLEAVE);
    let mut args: Vec<String> = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--csv" => {
                csv_path = Some(argv.get(i + 1).unwrap_or_else(|| usage()).clone());
                i += 2;
            }
            "--seed" => {
                seed = parse_u64(argv.get(i + 1).unwrap_or_else(|| usage()));
                i += 2;
            }
            "--interleave-seed" => {
                iseed = parse_u64(argv.get(i + 1).unwrap_or_else(|| usage()));
                i += 2;
            }
            "-h" | "--help" => usage(),
            _ => {
                args.push(argv[i].clone());
                i += 1;
            }
        }
    }
    if args.len() < 5 {
        usage();
    }
    let base_elf = Path::new(&args[0]);
    let base_gt = Path::new(&args[1]);
    let donor_elf = Path::new(&args[2]);
    let donor_gt = Path::new(&args[3]);
    let fit_paths = &args[4..];

    let stem = |p: &Path| p.file_name().unwrap().to_string_lossy().into_owned();
    let base_name = stem(base_elf);
    let donor_name = stem(donor_elf);

    // CSV sink: append, writing the header only when the file is new/empty, so two substrate/donor
    // pairs land in one file without either run needing to know about the other.
    let mut csv = match &csv_path {
        Some(p) => {
            let fresh = fs::metadata(p).map(|m| m.len() == 0).unwrap_or(true);
            let mut f = fs::OpenOptions::new().create(true).append(true).open(p)?;
            if fresh {
                writeln!(f, "{CSV_HEADER}")?;
            }
            Some(f)
        }
        None => None,
    };

    eprintln!("── fitting clean isotonic map ──");
    let mut clean_fit = Vec::new();
    let mut i = 0;
    while i + 1 < fit_paths.len() {
        clean_fit.push(load_clean(Path::new(&fit_paths[i]), Path::new(&fit_paths[i + 1]))?);
        i += 2;
    }
    let map = fit_clean_map(&clean_fit)?;

    let (code, offs) = load_clean(base_elf, base_gt)?;
    let (donor_code, donor_offs) = load_clean(donor_elf, donor_gt)?;
    let base_starts: HashSet<u64> = offs.iter().cloned().collect();
    let code_len = code.len() as u64;
    // Length of every adversary window: match the donor code length so the constructions are
    // size-comparable (append ~equal bytes of decoy / packed / nop / deflate).
    let win = donor_code.len();

    eprintln!(
        "── base={} (.text {} B, {} true starts)  donor={} (.text {} B)  window={} B  seed={:#x} ──",
        base_name,
        code_len,
        offs.len(),
        donor_name,
        donor_code.len(),
        win,
        seed,
    );
    println!("\n# Adaptive-adversary constructions (glob_hi={GLOB_HI} spat_hi={SPAT_HI})\n");

    let mut report = |tag: &str, inj: &Injected, s: &Stats| -> Result<()> {
        print_row(tag, s);
        if let Some(f) = csv.as_mut() {
            writeln!(f, "{}", csv_line(tag, inj, &base_name, &donor_name, seed, iseed, s))?;
        }
        Ok(())
    };

    // ── C0: clean base (null check) ────────────────────────────────────────────
    {
        let s = measure(&code, &base_starts, None, &map)?;
        let inj = Injected {
            family: "clean",
            gt_positives: base_starts.len(),
            adv_bytes: 0,
            adv_positives: 0,
        };
        report("C0_clean_base", &inj, &s)?;
    }

    // Helper to build an APPEND construction: code ++ region. Region bytes are all GT-negative —
    // that is the injector's record: it emitted them, and it emitted no positive for any of them.
    let region_lo = VADDR + code_len;
    let append = |region: &[u8]| -> Result<Stats> {
        let mut payload = code.clone();
        payload.extend_from_slice(region);
        measure(&payload, &base_starts, Some((region_lo, region_lo + region.len() as u64)), &map)
    };
    let appended = |family: &'static str, bytes: usize| Injected {
        family,
        gt_positives: base_starts.len(),
        adv_bytes: bytes,
        adv_positives: 0,
    };

    // ── C1: self-consistent clean decoy (real code re-placed as DATA) ──────────
    // Prime evasion candidate: decodes clean + forms a self-consistent CFG, so m≈q (low surprise)
    // and residual≈0 (low Moran). If the engine over-commits to it, ECE rises → evasion.
    {
        let s = append(&donor_code[..win])?;
        report("C1_decoy_realcode", &appended("decoy", win), &s)?;
    }

    // ── C1b: same decoy, but GT-labeled by the DONOR's own instruction starts ──
    // The decode-level GT the detector is calibrated against (gen-gt insn_max = valid instruction
    // starts) would label the decoy's real instruction starts POSITIVE, not data. If C1's ECE was
    // genuine decode miscalibration it should persist; if it was a role-labeling artifact (decode-
    // valid code relabeled as data) it collapses back toward clean. This separates "detector blind
    // spot" from "GT definition choice." The relabeling is still by construction: the donor's own
    // gen-gt starts, translated by the exact offset the injector appended them at.
    {
        let mut payload = code.clone();
        payload.extend_from_slice(&donor_code[..win]);
        // positives = base true starts ++ donor true starts (shifted into the appended window)
        let mut gt2: HashSet<u64> = base_starts.clone();
        let mut adv_pos = 0usize;
        for &o in &donor_offs {
            if (o as usize) < win {
                gt2.insert(code_len + o);
                adv_pos += 1;
            }
        }
        let s = measure(&payload, &gt2, Some((region_lo, region_lo + win as u64)), &map)?;
        let inj = Injected {
            family: "decoy_relabeled",
            gt_positives: gt2.len(),
            adv_bytes: win,
            adv_positives: adv_pos,
        };
        report("C1b_decoy_decodeGT", &inj, &s)?;
    }

    // ── C2: NOP sled decoy ─────────────────────────────────────────────────────
    {
        let s = append(&vec![0x90u8; win])?;
        report("C2_decoy_nopsled", &appended("packing", win), &s)?;
    }

    // ── C3: packed body — uniform pseudo-random (the classic packed regime) ────
    {
        let s = append(&xorshift(seed, win))?;
        report("C3_pack_random", &appended("packing", win), &s)?;
    }

    // ── C4: packed body — DEFLATE of the real code (realistic packer output) ───
    {
        use std::io::Write;
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::best());
        // repeat source until we have >= win compressed bytes
        let mut src = Vec::new();
        while src.len() < win * 3 {
            src.extend_from_slice(&donor_code);
        }
        enc.write_all(&src).ok();
        let mut z = enc.finish().unwrap_or_default();
        z.resize(win, 0);
        let s = append(&z)?;
        report("C4_pack_deflate", &appended("packing", win), &s)?;
    }

    // ── C5: de-clustering sweep — attack S_spat by interleaving random with clean ─
    // decoy code in blocks of size k. Smaller k ⇒ residual sign alternates ⇒ Moran drops, BUT the
    // clean-code fraction (recoverable, un-packed) rises. Traces S_spat vs packed-fraction cost.
    for &k in &[512usize, 128, 32, 8] {
        let mut region = Vec::with_capacity(win);
        let rnd = xorshift(iseed ^ k as u64, win);
        let mut ri = 0usize; // index into donor code for the "clean" blocks
        let mut toggle = true;
        while region.len() < win {
            let take = k.min(win - region.len());
            if toggle {
                for j in 0..take {
                    region.push(rnd[(region.len().wrapping_add(j)) % rnd.len()]);
                }
            } else {
                for _ in 0..take {
                    region.push(donor_code[ri % donor_code.len()]);
                    ri += 1;
                }
            }
            toggle = !toggle;
        }
        let s = append(&region)?;
        report(&format!("C5_interleave_k{k}"), &appended("interleave", win), &s)?;
    }

    // ── C6: desync / overlap — corrupt the REAL stream, trace obfuscation vs detection ─
    // Insert a REX.W prefix (0x48) before a fraction ρ of true starts. The prefix forces the
    // decoder to swallow the following byte, misaligning the real instruction → local/structural
    // disagreement → surprise. GT = the shifted true starts, recorded by the injector at the exact
    // post-insertion offset as it writes each byte. As ρ rises, S_glob should rise (detected) AND
    // true-code recovery should fall (real obfuscation cost).
    let mut sorted_starts: Vec<u64> = offs.clone();
    sorted_starts.sort_unstable();
    for &rho in &[0.02f64, 0.05, 0.10, 0.20, 0.40] {
        let mut payload = Vec::with_capacity(code.len() + (code.len() as f64 * rho) as usize);
        let mut new_starts: HashSet<u64> = HashSet::new();
        // stride: insert a prefix before every ~(1/rho)-th true start
        let step = (1.0 / rho).round().max(1.0) as usize;
        let start_set: HashSet<u64> = sorted_starts.iter().cloned().collect();
        let mut count_since = 0usize;
        let mut inserted = 0usize;
        for (off, &b) in code.iter().enumerate() {
            let off = off as u64;
            if start_set.contains(&off) {
                count_since += 1;
                if count_since % step == 0 {
                    payload.push(0x48); // REX.W junk prefix
                    inserted += 1;
                }
                new_starts.insert(payload.len() as u64); // real start lands here (post-insert)
            }
            payload.push(b);
        }
        let s = measure(&payload, &new_starts, None, &map)?;
        let inj = Injected {
            family: "desync",
            gt_positives: new_starts.len(),
            adv_bytes: inserted,
            adv_positives: 0, // the junk prefixes are never labeled as starts
        };
        report(&format!("C6_desync_rho{:.2}", rho), &inj, &s)?;
    }

    eprintln!("\ndone.");
    Ok(())
}
