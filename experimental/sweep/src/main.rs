//! `sweep` — turn the single-binary honesty experiment into a curve across a corpus.
//!
//! For every binary in `<bin_dir>` that has a matching `<gt_dir>/<name>.gt`, it builds the
//! synthetic two-sided-GT image in memory (`[real code][high-entropy data]`), runs Soft at each
//! entropy-prior strength, and reports the two axes against ground truth — plus the confident-code
//! rate split into the code half (want it *held*) and the data half (want it *collapsed*).
//!
//! Output is a CSV on stdout so it drops straight into a plotter or spreadsheet:
//!
//! ```text
//! binary,strength,n,base_rate,ece,brier,reliability,resolution,auroc,code_conf,data_conf
//! ```
//!
//! ```text
//! sweep <bin_dir> <gt_dir> [--data N] [--strengths 0,5,15,30] [--compressed]
//!                          [--max-code N] [--max-binaries N]
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use evalkit::{evaluate, extract_text, load_gt, make_data, run_soft, DataKind};

/// Seed for the synthetic data half — fixed so the whole sweep is reproducible.
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

fn main() -> Result<()> {
    let args = Args::parse(std::env::args().skip(1))?;

    // Discover binaries that have ground truth, in a stable order.
    let mut work: Vec<(PathBuf, PathBuf, String)> = Vec::new();
    for entry in fs::read_dir(&args.bin_dir)
        .with_context(|| format!("reading {}", args.bin_dir.display()))?
    {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        if let Some(f) = &args.filter {
            if !name.contains(f.as_str()) {
                continue;
            }
        }
        let gt = args.gt_dir.join(format!("{name}.gt"));
        if gt.exists() {
            work.push((path, gt, name));
        }
    }
    work.sort_by(|a, b| a.2.cmp(&b.2));
    if args.max_binaries > 0 && work.len() > args.max_binaries {
        work.truncate(args.max_binaries);
    }
    if work.is_empty() {
        bail!("no binaries with a matching .gt under {}", args.bin_dir.display());
    }
    eprintln!(
        "sweep: {} binaries, data={} ({:?}), strengths={:?}, dassa={}",
        work.len(),
        args.data_bytes,
        args.data_kind,
        args.strengths,
        args.use_dassa
    );

    // CSV header.
    println!("binary,strength,dassa,n,base_rate,ece,brier,reliability,resolution,auroc,code_conf,data_conf");

    for (k, (bin, gt_path, name)) in work.iter().enumerate() {
        match sweep_one(bin, gt_path, name, &args) {
            Ok(rows) => {
                for row in rows {
                    println!("{row}");
                }
                eprintln!("[{}/{}] {name}", k + 1, work.len());
            }
            Err(e) => eprintln!("[{}/{}] {name} SKIP: {e:#}", k + 1, work.len()),
        }
    }
    Ok(())
}

/// Run the full strength ladder for one binary; returns one CSV row per strength.
fn sweep_one(bin: &Path, gt_path: &Path, name: &str, args: &Args) -> Result<Vec<String>> {
    let bytes = fs::read(bin)?;
    let (vaddr, code) = extract_text(&bytes)?;
    if code.len() > args.max_code {
        bail!("code section {} bytes > --max-code {}", code.len(), args.max_code);
    }

    // [real code][high-entropy data] — the data half is the "compressed payload" stand-in.
    let mut payload = code.to_vec();
    payload.extend(make_data(args.data_kind, SEED, code, args.data_bytes));
    let code_end = vaddr + code.len() as u64;

    let gt = load_gt(gt_path)?;

    let dassa = args.use_dassa as u8;
    let mut rows = Vec::with_capacity(args.strengths.len());
    for &strength in &args.strengths {
        let posteriors = run_soft(vaddr, &payload, strength, args.use_dassa)?;
        let m = evaluate(&posteriors, &gt);
        let (code_conf, data_conf) = region_confidence(&posteriors, vaddr, code_end);
        let auroc = m.auroc.map(|a| format!("{a:.4}")).unwrap_or_else(|| "NA".into());
        rows.push(format!(
            "{name},{strength},{dassa},{},{:.4},{:.4},{:.4},{:.4},{:.4},{auroc},{code_conf:.4},{data_conf:.4}",
            m.n, m.base_rate, m.ece, m.brier, m.reliability, m.resolution
        ));
    }
    Ok(rows)
}

/// Confident-code rate (P ≥ 0.9) in the code half `[vaddr, code_end)` and the data half
/// `[code_end, …)`. Code should hold near its baseline (recall preserved); data should fall.
fn region_confidence(posteriors: &[(u64, f64)], vaddr: u64, code_end: u64) -> (f64, f64) {
    let (mut code_hi, mut code_n, mut data_hi, mut data_n) = (0usize, 0usize, 0usize, 0usize);
    for &(addr, p) in posteriors {
        if addr < code_end {
            code_n += 1;
            if p >= 0.9 {
                code_hi += 1;
            }
        } else {
            data_n += 1;
            if p >= 0.9 {
                data_hi += 1;
            }
        }
        debug_assert!(addr >= vaddr);
    }
    let rate = |hi: usize, n: usize| if n == 0 { 0.0 } else { hi as f64 / n as f64 };
    (rate(code_hi, code_n), rate(data_hi, data_n))
}

/// Parsed command line.
struct Args {
    bin_dir: PathBuf,
    gt_dir: PathBuf,
    data_bytes: usize,
    data_kind: DataKind,
    strengths: Vec<f64>,
    max_code: usize,
    max_binaries: usize,
    use_dassa: bool,
    filter: Option<String>,
}

const DEFAULT_DATA_BYTES: usize = 8000;
const DEFAULT_MAX_CODE: usize = 300_000;

impl Args {
    fn parse(mut it: impl Iterator<Item = String>) -> Result<Self> {
        const USAGE: &str = "usage: sweep <bin_dir> <gt_dir> [--data N] [--strengths 0,5,15,30] \
                             [--kind code|xorshift|compressed] [--compressed] [--dassa] [--filter S] \
                             [--max-code N] [--max-binaries N]";

        let mut positional: Vec<PathBuf> = Vec::new();
        let mut data_bytes = DEFAULT_DATA_BYTES;
        let mut data_kind = DataKind::Xorshift;
        let mut strengths = vec![0.0, 5.0, 15.0, 30.0];
        let mut max_code = DEFAULT_MAX_CODE;
        let mut max_binaries = 0;
        let mut use_dassa = false;
        let mut filter = None;

        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--data" => data_bytes = take(&mut it, "--data")?.parse().context("--data int")?,
                "--strengths" => {
                    strengths = take(&mut it, "--strengths")?
                        .split(',')
                        .map(|s| s.trim().parse::<f64>().context("--strengths wants floats"))
                        .collect::<Result<_>>()?;
                }
                "--compressed" => data_kind = DataKind::Compressed,
                "--kind" => {
                    data_kind = match take(&mut it, "--kind")?.as_str() {
                        "code" => DataKind::Code,
                        "xorshift" | "random" => DataKind::Xorshift,
                        "compressed" | "deflate" => DataKind::Compressed,
                        other => bail!("--kind wants code|xorshift|compressed, got {other:?}"),
                    }
                }
                "--dassa" => use_dassa = true,
                "--filter" => filter = Some(take(&mut it, "--filter")?),
                "--max-code" => max_code = take(&mut it, "--max-code")?.parse().context("--max-code int")?,
                "--max-binaries" => {
                    max_binaries = take(&mut it, "--max-binaries")?.parse().context("--max-binaries int")?
                }
                "-h" | "--help" => {
                    eprintln!("{USAGE}");
                    std::process::exit(0);
                }
                other if other.starts_with('-') => bail!("unexpected flag: {other}"),
                other => positional.push(PathBuf::from(other)),
            }
        }

        let [bin_dir, gt_dir] = positional.as_slice() else {
            bail!("{USAGE}");
        };
        if strengths.is_empty() {
            bail!("--strengths must list at least one value");
        }
        Ok(Self {
            bin_dir: bin_dir.clone(),
            gt_dir: gt_dir.clone(),
            data_bytes,
            data_kind,
            strengths,
            max_code,
            max_binaries,
            use_dassa,
            filter,
        })
    }
}

/// Take the next argument or error with the flag name.
fn take(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    it.next().with_context(|| format!("{flag} needs a value"))
}
