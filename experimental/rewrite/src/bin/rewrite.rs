//! `rewrite` — apply confidence-gated basic-block instrumentation to an ELF.
//!
//! usage: rewrite <in.elf> <out.elf> --mode ours|baseline [--marginals FILE] [--tau F] [--dump-sites]
//!
//!   baseline   deterministic-CFG rewriter: instrument every linear-sweep block leader (commit
//!              everywhere) — the thing that corrupts on stripped/desynced/packed code.
//!   ours       instrument only leaders whose calibrated belief `bel ≥ τ` (from `--marginals`, a
//!              `udstack --dump-instr` dump); abstain below.
//!
//! Both arms share the leader set and the patcher; only the site filter differs. Success is judged
//! behaviourally by the eval harness, never here.

use anyhow::{bail, Context, Result};
use rewrite::{build_patched, linear_leaders, plan_sites, Elf};
use std::collections::HashMap;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args = Args::parse(std::env::args().skip(1))?;
    let elf = Elf::parse(std::fs::read(&args.input).with_context(|| format!("read {:?}", args.input))?)?;

    let cs = capstone_for_leaders()?;
    let (tv, toff, tsz) = elf.text;
    let text = &elf.bytes[toff..toff + tsz];
    let leaders = linear_leaders(&cs, text, tv)?;

    let marginals = match args.mode {
        Mode::Ours => Some(load_marginals(args.marginals.as_ref().context("--marginals required for ours")?)?),
        Mode::Baseline => None,
    };
    let (sites, rej) = plan_sites(&elf, &leaders, marginals.as_ref(), args.tau)?;

    if args.dump_sites {
        for s in &sites {
            println!("site,0x{:x},{},{:.4}", s.vaddr, s.stolen_len, s.bel);
        }
    }

    let (patched, counter) = build_patched(&elf, &sites)?;
    std::fs::write(&args.output, &patched).with_context(|| format!("write {:?}", args.output))?;

    let coverage = if leaders.is_empty() { 0.0 } else { sites.len() as f64 / leaders.len() as f64 };
    println!(
        "rewrite_summary,{},{:.3},{},{},{},0x{:x},{}",
        args.mode.as_str(),
        args.tau,
        leaders.len(),
        sites.len(),
        format!("{:.4}", coverage),
        counter,
        format!(
            "reject_lowbel={} reject_notcand={} reject_nowindow={} reject_decode={}",
            rej.get("low_belief").copied().unwrap_or(0),
            rej.get("not_a_candidate").copied().unwrap_or(0),
            rej.get("no_safe_window").copied().unwrap_or(0),
            rej.get("decode_fail").copied().unwrap_or(0),
        )
    );
    Ok(())
}

fn capstone_for_leaders() -> Result<capstone::Capstone> {
    use capstone::prelude::*;
    Capstone::new()
        .x86()
        .mode(arch::x86::ArchMode::Mode64)
        .detail(true)
        .build()
        .map_err(|e| anyhow::anyhow!("capstone: {e}"))
}

/// Parse a `udstack --dump-instr` dump: `instr_bel,0x<addr>,<phat>,<pi>,<tag>` → addr ↦ phat (the
/// calibrated belief). Non-matching lines are ignored so the same file can carry other udstack rows.
fn load_marginals(path: &PathBuf) -> Result<HashMap<u64, f64>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read marginals {path:?}"))?;
    let mut m = HashMap::new();
    for ln in text.lines() {
        let p: Vec<&str> = ln.split(',').collect();
        if p.len() >= 3 && p[0] == "instr_bel" {
            let addr = u64::from_str_radix(p[1].trim_start_matches("0x"), 16).context("marginal addr")?;
            let phat: f64 = p[2].parse().context("marginal phat")?;
            m.insert(addr, phat);
        }
    }
    if m.is_empty() {
        bail!("no instr_bel rows in {path:?} — did you pass a --dump-instr dump?");
    }
    Ok(m)
}

#[derive(Clone, Copy)]
enum Mode {
    Ours,
    Baseline,
}
impl Mode {
    fn as_str(&self) -> &'static str {
        match self {
            Mode::Ours => "ours",
            Mode::Baseline => "baseline",
        }
    }
}

struct Args {
    input: PathBuf,
    output: PathBuf,
    mode: Mode,
    marginals: Option<PathBuf>,
    tau: f64,
    dump_sites: bool,
}

impl Args {
    fn parse(mut it: impl Iterator<Item = String>) -> Result<Self> {
        const USAGE: &str =
            "usage: rewrite <in.elf> <out.elf> --mode ours|baseline [--marginals FILE] [--tau F] [--dump-sites]";
        let mut pos = Vec::new();
        let mut mode = None;
        let mut marginals = None;
        let mut tau = 0.9;
        let mut dump_sites = false;
        while let Some(a) = it.next() {
            match a.as_str() {
                "--mode" => {
                    mode = Some(match it.next().context("--mode ours|baseline")?.as_str() {
                        "ours" => Mode::Ours,
                        "baseline" => Mode::Baseline,
                        o => bail!("--mode wants ours|baseline, got {o}"),
                    })
                }
                "--marginals" => marginals = Some(PathBuf::from(it.next().context("--marginals path")?)),
                "--tau" => tau = it.next().context("--tau value")?.parse().context("--tau float")?,
                "--dump-sites" => dump_sites = true,
                "-h" | "--help" => {
                    eprintln!("{USAGE}");
                    std::process::exit(0);
                }
                o if o.starts_with('-') => bail!("unexpected flag: {o}"),
                o => pos.push(PathBuf::from(o)),
            }
        }
        let [input, output] = pos.as_slice() else { bail!("{USAGE}") };
        Ok(Args {
            input: input.clone(),
            output: output.clone(),
            mode: mode.context("--mode required")?,
            marginals,
            tau,
            dump_sites,
        })
    }
}
