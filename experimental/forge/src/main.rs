//! `forge` — emit an adversarial binary with a *true-instruction log* (two-sided ground truth).
//!
//! This is the desync-cc-style corpus generator for the adversarial paper. It takes a real benign
//! ELF (`.text` + a `.gt` of true instruction starts) and forges a single-segment, headerless image
//! whose layout deliberately stresses a probabilistic disassembler:
//!
//!   * the **code half** — the real `.text`, every true instruction start known (input `.gt`), and
//!   * the **decoy half** — a planted region we know contains *no* true instructions, appended past
//!     the code so nothing in the program's control flow ever reaches it.
//!
//! The interesting knob is *what the decoy half is made of*, because the two honesty levers we built
//! catch different things (the ablation on xorshift/DEFLATE showed DASSA does nothing there):
//!
//!   * `--kind xorshift` / `--kind compressed` — high-entropy payloads (packer body, encrypted blob).
//!     The entropy prior suppresses these; DASSA is inert (the CF hint-pairs never fire on them).
//!   * `--kind code` (default) — real instruction bytes tiled into the decoy half: a *low*-entropy,
//!     self-consistent control-flow chain. This is the case the entropy prior CANNOT catch (it sits
//!     below the 6-bit floor) and the only one where DASSA's corrected CF priors can bite. It models
//!     embedded/decoy code — the bytes decode cleanly and reference each other, but they are not part
//!     of the intended program, so they are honest GT negatives.
//!
//! Output is reproducible: `<out_prefix>.elf` (one R+X PT_LOAD: code ++ decoy), `<out_prefix>.gt`
//! (the true-instruction log — input GT unchanged, since the code half keeps its load address), and
//! `<out_prefix>.regions` (the two-sided split, so downstream tooling knows code vs decoy without
//! re-deriving it from lengths).
//!
//! ```text
//! forge <benign.elf> <benign.gt> <out_prefix> [--decoy-bytes N] [--kind code|xorshift|compressed]
//!       [--seed S]
//! ```

use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use evalkit::{build_min_elf, extract_text, make_data, DataKind};

/// Default decoy size — roughly a function's worth of bytes, enough to register on both axes.
const DEFAULT_DECOY_BYTES: usize = 8000;
/// Default seed, only used by the xorshift kind; fixed so the corpus reproduces byte-for-byte.
const DEFAULT_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

fn main() -> Result<()> {
    let args = Args::parse(std::env::args().skip(1))?;

    // 1. Pull the real code: load address + bytes.
    let bytes = fs::read(&args.elf).with_context(|| format!("reading {}", args.elf.display()))?;
    let (vaddr, code) =
        extract_text(&bytes).with_context(|| format!("locating .text in {}", args.elf.display()))?;

    // 2. Payload = real code, then the decoy region of the chosen kind.
    let mut payload = code.to_vec();
    payload.extend(make_data(args.kind, args.seed, code, args.decoy_bytes));
    let code_end = vaddr + code.len() as u64;
    let decoy_end = vaddr + payload.len() as u64;

    // 3. Wrap in a headerless one-segment ELF at the code's original address, so the input GT lines
    //    up byte-for-byte with the code half and needs no remapping.
    let elf = build_min_elf(vaddr, &payload);

    // 4. The true-instruction log is the input GT unchanged: every entry is in the code half, and
    //    the decoy half — by construction unreachable — contributes none.
    let gt = fs::read(&args.gt).with_context(|| format!("reading {}", args.gt.display()))?;

    // 5. Region map: explicit two-sided split so a scorer never has to guess where code ends.
    let regions = format!(
        "# region\tlabel\tvaddr_start\tvaddr_end\tkind\n\
         code\tpositive\t0x{vaddr:x}\t0x{code_end:x}\treal\n\
         decoy\tnegative\t0x{code_end:x}\t0x{decoy_end:x}\t{:?}\n",
        args.kind
    );

    let out_elf = PathBuf::from(format!("{}.elf", args.out_prefix));
    let out_gt = PathBuf::from(format!("{}.gt", args.out_prefix));
    let out_regions = PathBuf::from(format!("{}.regions", args.out_prefix));
    fs::write(&out_elf, &elf).with_context(|| format!("writing {}", out_elf.display()))?;
    fs::write(&out_gt, &gt).with_context(|| format!("writing {}", out_gt.display()))?;
    fs::write(&out_regions, regions).with_context(|| format!("writing {}", out_regions.display()))?;

    eprintln!(
        "forged {} — code 0x{vaddr:x}..0x{code_end:x} ({} B), {:?} decoy 0x{code_end:x}..0x{decoy_end:x} ({} B)\n\
         wrote {} (true-instruction log) and {}",
        out_elf.display(),
        code.len(),
        args.kind,
        args.decoy_bytes,
        out_gt.display(),
        out_regions.display(),
    );
    Ok(())
}

/// Parsed command line.
struct Args {
    elf: PathBuf,
    gt: PathBuf,
    out_prefix: String,
    decoy_bytes: usize,
    kind: DataKind,
    seed: u64,
}

impl Args {
    fn parse(mut it: impl Iterator<Item = String>) -> Result<Self> {
        const USAGE: &str = "usage: forge <benign.elf> <benign.gt> <out_prefix> \
                             [--decoy-bytes N] [--kind code|xorshift|compressed] [--seed S]";

        let mut positional: Vec<String> = Vec::new();
        let mut decoy_bytes = DEFAULT_DECOY_BYTES;
        let mut kind = DataKind::Code;
        let mut seed = DEFAULT_SEED;

        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--decoy-bytes" => {
                    decoy_bytes = take(&mut it, "--decoy-bytes")?.parse().context("--decoy-bytes int")?
                }
                "--kind" => {
                    kind = match take(&mut it, "--kind")?.as_str() {
                        "code" => DataKind::Code,
                        "xorshift" | "random" => DataKind::Xorshift,
                        "compressed" | "deflate" => DataKind::Compressed,
                        other => bail!("--kind wants code|xorshift|compressed, got {other:?}"),
                    }
                }
                "--seed" => seed = take(&mut it, "--seed")?.parse().context("--seed u64")?,
                "-h" | "--help" => {
                    eprintln!("{USAGE}");
                    std::process::exit(0);
                }
                other if other.starts_with('-') => bail!("unexpected flag: {other}"),
                other => positional.push(other.to_string()),
            }
        }

        let [elf, gt, out_prefix] = positional.as_slice() else {
            bail!("{USAGE}");
        };
        Ok(Self {
            elf: PathBuf::from(elf),
            gt: PathBuf::from(gt),
            out_prefix: out_prefix.clone(),
            decoy_bytes,
            kind,
            seed,
        })
    }
}

/// Take the next argument or error with the flag name.
fn take(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    it.next().with_context(|| format!("{flag} needs a value"))
}
