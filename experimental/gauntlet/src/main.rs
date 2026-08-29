//! `gauntlet` CLI — list generators, build a corpus, validate it.
//!
//! ```text
//! gauntlet list
//! gauntlet generate --seed-elf E --seed-gt G [--source S] --out DIR
//!                   [--gen ID]... [--region-bytes N] [--seed-rng S]
//! gauntlet validate DIR
//! ```

use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use gauntlet::{fnv1a_hex, standard_registry, Availability, GenConfig, Generator, Manifest, Seed};

const USAGE: &str = "usage:\n  \
    gauntlet list\n  \
    gauntlet generate --seed-elf E --seed-gt G [--source S] --out DIR [--gen ID]... \
[--region-bytes N] [--seed-rng S]\n  \
    gauntlet validate DIR";

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("list") => cmd_list(),
        Some("generate") => cmd_generate(args),
        Some("validate") => cmd_validate(args),
        Some("-h") | Some("--help") | None => {
            println!("{USAGE}");
            Ok(())
        }
        Some(other) => bail!("unknown command {other:?}\n{USAGE}"),
    }
}

fn cmd_list() -> Result<()> {
    let reg = standard_registry();
    println!("{:<22} {:<22} {:<10} description", "id", "bucket", "available");
    for g in reg.generators() {
        let avail = match g.availability() {
            Availability::Available => "yes".to_string(),
            Availability::Missing(why) => format!("no — {why}"),
        };
        println!("{:<22} {:<22} {:<10} {}", g.id(), g.bucket().as_str(), avail, g.describe());
    }
    Ok(())
}

fn cmd_generate(mut it: impl Iterator<Item = String>) -> Result<()> {
    let mut seed_elf: Option<PathBuf> = None;
    let mut seed_gt: Option<PathBuf> = None;
    let mut source: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut wanted: Vec<String> = Vec::new();
    let mut cfg = GenConfig::default();

    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--seed-elf" => seed_elf = Some(PathBuf::from(take(&mut it, "--seed-elf")?)),
            "--seed-gt" => seed_gt = Some(PathBuf::from(take(&mut it, "--seed-gt")?)),
            "--source" => source = Some(PathBuf::from(take(&mut it, "--source")?)),
            "--out" => out = Some(PathBuf::from(take(&mut it, "--out")?)),
            "--gen" => wanted.push(take(&mut it, "--gen")?),
            "--region-bytes" => cfg.region_bytes = take(&mut it, "--region-bytes")?.parse().context("--region-bytes int")?,
            "--seed-rng" => cfg.seed_rng = take(&mut it, "--seed-rng")?.parse().context("--seed-rng u64")?,
            other => bail!("unexpected arg {other:?}\n{USAGE}"),
        }
    }

    let elf = seed_elf.context("--seed-elf is required")?;
    let gt = seed_gt.context("--seed-gt is required")?;
    let out = out.context("--out is required")?;

    let mut seed = Seed::from_files(&elf, &gt)?;
    if let Some(src) = source {
        seed = seed.with_source(src);
    }
    eprintln!(
        "seed {} ({} bytes, {} instruction starts)",
        seed.name,
        seed.bytes.len(),
        seed.gt.len()
    );

    let reg = standard_registry();
    let chosen: Vec<&dyn Generator> = if wanted.is_empty() {
        reg.generators().iter().map(|b| b.as_ref()).collect()
    } else {
        let mut v = Vec::new();
        for id in &wanted {
            v.push(reg.get(id).ok_or_else(|| anyhow::anyhow!("no generator with id {id:?}"))?);
        }
        v
    };

    let mut produced: Vec<String> = Vec::new();
    for (i, g) in chosen.iter().enumerate() {
        let tag = format!("[{}/{}] {}", i + 1, chosen.len(), g.id());
        if let Availability::Missing(why) = g.availability() {
            eprintln!("{tag} skip — {why}");
            continue;
        }
        match g.generate(&seed, &cfg) {
            Ok(art) => {
                art.write(&out)?;
                eprintln!("{tag} ok → {}.elf (+ .gt .regions .manifest.json)", art.out_name);
                produced.push(art.out_name);
            }
            Err(e) => eprintln!("{tag} skip — {e:#}"),
        }
    }

    let index = serde_json::json!({ "seed": seed.name, "artifacts": produced });
    let index_path = out.join("index.json");
    fs::write(&index_path, serde_json::to_string_pretty(&index)?)
        .with_context(|| format!("writing {}", index_path.display()))?;
    eprintln!("wrote {} artifacts + {}", produced.len(), index_path.display());
    Ok(())
}

fn cmd_validate(mut it: impl Iterator<Item = String>) -> Result<()> {
    let dir = PathBuf::from(it.next().context("usage: gauntlet validate DIR")?);
    let mut total = 0usize;
    let mut ok = 0usize;
    for entry in fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        if !path.to_string_lossy().ends_with(".manifest.json") {
            continue;
        }
        total += 1;
        let m: Manifest = serde_json::from_str(&fs::read_to_string(&path)?)
            .with_context(|| format!("parsing {}", path.display()))?;
        let stem = &m.artifact;
        let elf = dir.join(format!("{stem}.elf"));
        let gt = dir.join(format!("{stem}.gt"));
        let regions = dir.join(format!("{stem}.regions"));
        let mut problems = Vec::new();
        for f in [&elf, &gt, &regions] {
            if !f.exists() {
                problems.push(format!("missing {}", f.display()));
            }
        }
        if elf.exists() {
            let h = fnv1a_hex(&fs::read(&elf)?);
            if h != m.binary_hash {
                problems.push(format!("hash mismatch ({h} != manifest {})", m.binary_hash));
            }
        }
        if problems.is_empty() {
            ok += 1;
            println!("ok   {stem}");
        } else {
            println!("FAIL {stem}: {}", problems.join("; "));
        }
    }
    println!("\n{ok}/{total} artifacts valid");
    if ok != total {
        bail!("{} artifact(s) failed validation", total - ok);
    }
    Ok(())
}

fn take(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    it.next().with_context(|| format!("{flag} needs a value"))
}
