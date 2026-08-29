//! `tool` — the thin CLI over [`tool::analyze`].
//!
//! ```text
//! tool <binary> [--facts f.json] [--out result.json] [--report] [--full-insns]
//! ```
//! - `--out FILE`     write the `AnalysisResult` JSON (the frontend contract).
//! - `--facts FILE`   load `KnownFacts` and fold them in as clamps/anchors.
//! - `--report`       print the human-readable trust/confidence summary (the default with no flags).
//! - `--full-insns`   include the full per-address posterior list in the JSON.
//!
//! No flags ⇒ print the report. This binary owns no analysis logic — it parses args, calls
//! `analyze`, and formats. TUI/GUI are out of scope; they consume the same JSON later.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use tool::{analyze, AnalysisResult, CalibrationBank, KnownFacts};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tool: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    // Export mode: `tool export-bank <packed-elf> <upxgt> --out bank.json` fits the packed regime's
    // isotonic map and writes a bank artifact. Handled before the normal analyze path.
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.first().map(String::as_str) == Some("export-bank") {
        return run_export(&raw[1..]);
    }

    let args = Args::parse(raw.into_iter())?;

    let facts = match &args.facts {
        Some(p) => Some(KnownFacts::load(p)?),
        None => None,
    };
    let bank = match &args.bank {
        Some(p) => Some(CalibrationBank::load(p)?),
        None => None,
    };

    let result = analyze(&args.binary, facts.as_ref(), bank.as_ref(), args.full_insns)?;

    if let Some(out) = &args.out {
        let json = serde_json::to_string_pretty(&result).context("serializing result")?;
        std::fs::write(out, json).with_context(|| format!("writing {}", out.display()))?;
        eprintln!("tool: wrote {} ({} functions, {} edges)", out.display(), result.functions.len(), result.edges.len());
    }

    // Print the report when asked, or by default when nothing was written.
    if args.report || args.out.is_none() {
        print_report(&result);
    }
    Ok(())
}

/// Human-readable summary: the regime + trust verdict, the localized distrust regions ("don't trust
/// these"), the top functions by confidence, and the top query suggestions ("confirm these next").
fn print_report(r: &AnalysisResult) {
    println!("══════════════════ ANALYSIS ══════════════════");
    println!("binary   : {}", r.binary.path);
    println!("format   : {} / {}   entry {:#x}   {} bytes .text", r.binary.format, r.binary.arch, r.binary.entry, r.binary.n_bytes);
    println!("regime   : {} ({}, conf {:.2})", r.regime.detected, r.regime.source, r.regime.confidence);
    let unc = if r.calibration.regime_uncertain { "  ⚠ regime uncertain — calibration may be stale" } else { "" };
    let switch = if r.calibration.isotonic_applied { "engine + isotonic map (map-switch)" } else { "engine only" };
    println!("calib    : selected {} via {} → {} (engine {:?}){}",
        r.calibration.selected_regime, r.calibration.classifier, r.calibration.map_applied, r.calibration.engine, unc);
    println!("           applied: {switch}");
    println!("           {}", r.calibration.note);

    println!("\n— trust ({}) —", r.trust.overall);
    println!("  S_glob (mean surprise) : {:.4}", r.trust.s_glob);
    println!("  S_spat (Moran's I)     : {:+.4}", r.trust.s_spat);
    if r.trust.distrust_regions.is_empty() {
        println!("  distrust regions       : none — surprise did not cluster");
    } else {
        println!("  distrust regions ({}) — don't trust these windows:", r.trust.distrust_regions.len());
        for d in &r.trust.distrust_regions {
            println!("    {:#x}..{:#x}  surprise {:.3}  ({})", d.addr_lo, d.addr_hi, d.surprise, d.reason);
        }
    }

    println!("\n— instructions —");
    println!("  candidates {} · mean π {:.3} · low-confidence {}", r.instructions.n, r.instructions.mean_pi, r.instructions.low_confidence_count);

    let n_conf = r.functions.iter().filter(|f| f.confidence >= 0.5).count();
    let n_decoy = r.functions.iter().filter(|f| f.flagged_decoy).count();
    println!("\n— functions ({} candidates · {} confirmed · {} flagged suspect) —", r.functions.len(), n_conf, n_decoy);
    for f in r.functions.iter().take(12) {
        let name = f.name.as_deref().unwrap_or("");
        let flag = if f.flagged_decoy { "  ⚑ suspect" } else { "" };
        println!("  {:#x}  F={:.3}  R={:.3}  {}{}", f.addr, f.confidence, f.reached_prob, name, flag);
    }
    if r.functions.len() > 12 {
        println!("  … {} more (see --out JSON)", r.functions.len() - 12);
    }

    if !r.suggestions.is_empty() {
        println!("\n— confirm next (top query gain) —");
        for s in r.suggestions.iter().take(6) {
            println!("  {:#x}  gain {:.2}  {}", s.addr, s.expected_info_gain, s.why);
        }
    }
    println!("═══════════════════════════════════════════════");
}

/// `tool export-bank [--from bank.json] [--packed <elf> <upxgt>] [--desync <bins> <gt>]… \
///   [--desync-limit N] --out bank.json` — fit the requested regime maps and write a calibration bank.
/// Run with both `--packed` and `--desync` to ship a full bank; `--from` seeds unchanged maps.
fn run_export(argv: &[String]) -> Result<()> {
    const USAGE: &str = "usage: tool export-bank [--from bank.json] [--packed <elf> <upxgt>] \
[--desync <bins> <gt>]... [--desync-limit N] --out bank.json";
    let mut from: Option<PathBuf> = None;
    let mut packed: Option<(PathBuf, PathBuf)> = None;
    let mut desync: Vec<(PathBuf, PathBuf)> = Vec::new();
    let mut desync_limit: usize = 8;
    let mut out: Option<PathBuf> = None;
    let mut it = argv.iter();
    let val = |it: &mut std::slice::Iter<String>, flag: &str| {
        it.next().cloned().with_context(|| format!("{flag} needs a value"))
    };
    while let Some(a) = it.next() {
        match a.as_str() {
            "--from" => from = Some(PathBuf::from(val(&mut it, "--from")?)),
            "--packed" => {
                let elf = PathBuf::from(val(&mut it, "--packed elf")?);
                let gt = PathBuf::from(val(&mut it, "--packed upxgt")?);
                packed = Some((elf, gt));
            }
            "--desync" => {
                let bins = PathBuf::from(val(&mut it, "--desync bins")?);
                let gt = PathBuf::from(val(&mut it, "--desync gt")?);
                desync.push((bins, gt));
            }
            "--desync-limit" => desync_limit = val(&mut it, "--desync-limit")?.parse()?,
            "--out" => out = Some(PathBuf::from(val(&mut it, "--out")?)),
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => anyhow::bail!("unexpected argument {other}\n{USAGE}"),
        }
    }
    if packed.is_none() && desync.is_empty() {
        anyhow::bail!("nothing to fit — give --packed and/or --desync\n{USAGE}");
    }
    let opts = tool::ExportOpts {
        from: from.as_deref(),
        packed: packed.as_ref().map(|(e, g)| (e.as_path(), g.as_path())),
        desync: desync.iter().map(|(b, g)| (b.as_path(), g.as_path())).collect(),
        desync_limit,
        out: &out.context(USAGE)?,
    };
    tool::export_bank(&opts)
}

// ── CLI parsing ────────────────────────────────────────────────────────────────────
struct Args {
    binary: PathBuf,
    facts: Option<PathBuf>,
    bank: Option<PathBuf>,
    out: Option<PathBuf>,
    report: bool,
    full_insns: bool,
}

impl Args {
    fn parse(mut it: impl Iterator<Item = String>) -> Result<Self> {
        const USAGE: &str = "usage: tool <binary> [--facts f.json] [--bank bank.json] [--out result.json] [--report] [--full-insns]\n       tool export-bank <packed-elf> <upxgt> --out bank.json";
        let mut binary: Option<PathBuf> = None;
        let mut facts = None;
        let mut bank = None;
        let mut out = None;
        let mut report = false;
        let mut full_insns = false;
        while let Some(a) = it.next() {
            let mut next = |flag: &str| it.next().with_context(|| format!("{flag} needs a value"));
            match a.as_str() {
                "--facts" => facts = Some(PathBuf::from(next("--facts")?)),
                "--bank" => bank = Some(PathBuf::from(next("--bank")?)),
                "--out" => out = Some(PathBuf::from(next("--out")?)),
                "--report" => report = true,
                "--full-insns" => full_insns = true,
                "-h" | "--help" => {
                    println!("{USAGE}");
                    std::process::exit(0);
                }
                other if other.starts_with("--") => anyhow::bail!("unexpected flag {other}\n{USAGE}"),
                other => {
                    if binary.is_some() {
                        anyhow::bail!("more than one binary given\n{USAGE}");
                    }
                    binary = Some(PathBuf::from(other));
                }
            }
        }
        Ok(Args {
            binary: binary.context(USAGE)?,
            facts,
            bank,
            out,
            report,
            full_insns,
        })
    }
}
