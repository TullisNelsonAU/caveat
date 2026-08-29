//! `upxeval` — the labeled over-commitment measurement on a real UPX-packed binary.
//!
//! This replaces the entropy-binned UPX *demonstration* with a measurement against perfect ground
//! truth. There is no instruction-level GT for a packed image, but we don't need one: UPX's own
//! `b_info` chain tells us the exact byte range of the compressed payload, and those bytes are
//! provably not instructions (compressed data the stub reads, never executed in place). So we run
//! Soft over the executable segment and report its false-positive rate over that known-negative
//! region — the honesty claim ("does confidence back off on data?") with zero reliance on entropy
//! or a disassembler's opinion.
//!
//! We report two windows: the *exact* compressed extent from the format walk, and a *conservative
//! interior* (the single largest block, far from any header/stub boundary) so the result holds even
//! if the parse were off by a byte. Both at each requested entropy-prior strength.
//!
//! ```text
//! upxeval <packed.elf> [--strengths 0,30] [--csv out.csv]
//! ```

use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use evalkit::{evaluate_negatives, parse_upx_layout, run_soft, NegMetrics};
use probdisasm::extract_text_section;

fn main() -> Result<()> {
    let args = Args::parse(std::env::args().skip(1))?;

    let bytes = fs::read(&args.path).with_context(|| format!("reading {}", args.path.display()))?;

    // Perfect negatives straight from the packer's format.
    let layout = parse_upx_layout(&bytes).context("parsing UPX layout")?;
    let (comp_lo, comp_hi) = layout.compressed;
    let (cons_lo, cons_hi) = layout.conservative;
    let unc_in_seg: u64 = layout.blocks.iter().map(|b| b.sz_unc as u64).sum();

    eprintln!("== UPX layout (provable ground truth) ==");
    eprintln!(
        "  exec segment   : vaddr 0x{:x}  file [0x{:x},0x{:x})  ({} B)",
        layout.exec_vaddr,
        layout.exec_file.0,
        layout.exec_file.1,
        layout.exec_file.1 - layout.exec_file.0,
    );
    eprintln!(
        "  UPX! magic     : file 0x{:x}   p_filesize (original) = {} B",
        layout.magic_file_off, layout.p_filesize
    );
    eprintln!("  blocks in seg  : {} (uncompressed {} B of {} B total)", layout.blocks.len(), unc_in_seg, layout.p_filesize);
    for (i, b) in layout.blocks.iter().enumerate() {
        eprintln!(
            "    [{i}] unc={:>7} cpr={:>7} method={} file[0x{:x},0x{:x})",
            b.sz_unc, b.sz_cpr, b.method, b.file_data.0, b.file_data.1
        );
    }
    eprintln!(
        "  NEGATIVE (exact compressed) : vaddr [0x{comp_lo:x},0x{comp_hi:x})  ({} B)",
        comp_hi - comp_lo
    );
    eprintln!(
        "  NEGATIVE (conservative)     : vaddr [0x{cons_lo:x},0x{cons_hi:x})  ({} B, largest block interior)",
        cons_hi - cons_lo
    );
    eprintln!(
        "  stub (real code, excluded)  : vaddr [0x{:x},0x{:x})  ({} B)\n",
        layout.stub.0,
        layout.stub.1,
        layout.stub.1.saturating_sub(layout.stub.0),
    );

    // Run Soft over the executable segment (segment fallback handles the headerless image).
    let (base, code) = extract_text_section(&bytes).context("extracting executable segment")?;
    if base != layout.exec_vaddr {
        bail!(
            "engine analyzed vaddr 0x{base:x} but UPX exec segment is 0x{:x} — mapping mismatch",
            layout.exec_vaddr
        );
    }

    println!("region,strength,n,fp_rate,mean_p,brier,max_p");
    eprintln!(
        "{:<14} {:>4}  {:>7}  {:>8}  {:>8}  {:>8}  {:>6}",
        "region", "str", "n", "fp_rate", "mean_p", "brier", "max_p"
    );
    for &s in &args.strengths {
        let post = run_soft(base, code, s, false).context("running Soft")?;
        for (label, lo, hi) in [("exact", comp_lo, comp_hi), ("conservative", cons_lo, cons_hi)] {
            let m: NegMetrics = evaluate_negatives(&post, lo, hi);
            println!(
                "{label},{s},{},{:.4},{:.4},{:.4},{:.4}",
                m.n, m.fp_rate, m.mean_p, m.brier, m.max_p
            );
            eprintln!(
                "{label:<14} {s:>4.0}  {:>7}  {:>7.2}%  {:>8.4}  {:>8.4}  {:>6.3}",
                m.n,
                100.0 * m.fp_rate,
                m.mean_p,
                m.brier,
                m.max_p
            );
        }
    }

    if let Some(csv) = &args.csv {
        // Re-run is cheap relative to clarity; just rebuild the rows we already printed to stdout.
        eprintln!("\n(CSV mirrors stdout; redirect stdout to capture: upxeval ... > {})", csv.display());
    }
    Ok(())
}

struct Args {
    path: PathBuf,
    strengths: Vec<f64>,
    csv: Option<PathBuf>,
}

impl Args {
    fn parse(mut it: impl Iterator<Item = String>) -> Result<Self> {
        const USAGE: &str = "usage: upxeval <packed.elf> [--strengths 0,30] [--csv out.csv]";
        let mut path = None;
        let mut strengths = vec![0.0, 30.0];
        let mut csv = None;
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--strengths" => {
                    strengths = it
                        .next()
                        .context("--strengths needs a value")?
                        .split(',')
                        .map(|s| s.trim().parse::<f64>().context("--strengths wants floats"))
                        .collect::<Result<_>>()?;
                }
                "--csv" => csv = Some(PathBuf::from(it.next().context("--csv needs a path")?)),
                "-h" | "--help" => {
                    eprintln!("{USAGE}");
                    std::process::exit(0);
                }
                other if other.starts_with('-') => bail!("unexpected flag: {other}"),
                other => path = Some(PathBuf::from(other)),
            }
        }
        Ok(Self {
            path: path.context(USAGE)?,
            strengths,
            csv,
        })
    }
}
