//! `synth` — build a controlled two-sided-ground-truth binary for the honesty evaluation.
//!
//! It takes a real benign ELF (whose `.text` is known code, paired with a `.gt` of instruction
//! starts) and appends a block of high-entropy bytes — a stand-in for a compressed or encrypted
//! payload — into a single executable segment. The result is one binary with two regions we
//! have ground truth for in *both* directions:
//!
//!   * the **code half** — every real instruction start is known (the input `.gt`), and
//!   * the **data half** — known to contain no instructions at all.
//!
//! That two-sided GT is what lets us measure, on adversarial-looking input, whether a posterior
//! is genuinely *honest* over the data and *accurate* over the code — not just watch a histogram
//! move. The code keeps its original load address, so the input `.gt` is valid unchanged.
//!
//! `--compressed` fills the data half with real DEFLATE output (a packer-body stand-in) instead
//! of the default uniform pseudo-random stream; both are deterministic, so the corpus reproduces.
//!
//! ```text
//! synth <benign.elf> <benign.gt> <data_bytes> <out_prefix> [--compressed]
//! ```
//! writes `<out_prefix>.elf` (one R+X PT_LOAD: code ++ data) and `<out_prefix>.gt`.

use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use evalkit::{build_min_elf, extract_text, make_data, DataKind};

/// Fixed seed for the appended stream, so identical inputs always yield identical output.
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

fn main() -> Result<()> {
    // Positional args plus an optional --compressed flag.
    let mut positional: Vec<String> = Vec::new();
    let mut kind = DataKind::Xorshift;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--compressed" => kind = DataKind::Compressed,
            flag if flag.starts_with("--") => bail!("unexpected flag: {flag}"),
            other => positional.push(other.to_string()),
        }
    }
    let [elf_path, gt_path, data_bytes, out_prefix] = positional.as_slice() else {
        bail!("usage: synth <benign.elf> <benign.gt> <data_bytes> <out_prefix> [--compressed]");
    };
    let data_len: usize = data_bytes
        .parse()
        .with_context(|| format!("data_bytes must be a non-negative integer, got {data_bytes:?}"))?;

    // 1. Pull the real code out of the input: its load address and bytes.
    let bytes = fs::read(elf_path).with_context(|| format!("reading {elf_path}"))?;
    let (text_vaddr, text_bytes) =
        extract_text(&bytes).with_context(|| format!("locating .text in {elf_path}"))?;

    // 2. Payload = real code followed by a high-entropy blob (the "compressed" data half).
    let mut payload = text_bytes.to_vec();
    payload.extend(make_data(kind, SEED, text_bytes, data_len));

    // 3. Wrap it in a minimal one-segment ELF at the code's original address, so the input GT
    //    lines up byte-for-byte with the code half and needs no remapping.
    let elf = build_min_elf(text_vaddr, &payload);
    let out_elf = PathBuf::from(format!("{out_prefix}.elf"));
    let out_gt = PathBuf::from(format!("{out_prefix}.gt"));
    fs::write(&out_elf, &elf).with_context(|| format!("writing {}", out_elf.display()))?;

    // 4. GT is the input GT unchanged — every entry sits in the code half; the data half has none.
    let gt = fs::read(gt_path).with_context(|| format!("reading {gt_path}"))?;
    fs::write(&out_gt, &gt).with_context(|| format!("writing {}", out_gt.display()))?;

    let code_end = text_vaddr + text_bytes.len() as u64;
    eprintln!(
        "wrote {} ({} code bytes @ 0x{text_vaddr:x}..0x{code_end:x}, then {data_len} {kind:?} data bytes) and {}",
        out_elf.display(),
        text_bytes.len(),
        out_gt.display(),
    );
    Ok(())
}

