//! Ground truth: typed, two-sided, and self-validating.
//!
//! A [`GroundTruth`] carries the positive labels (true instruction-start vaddrs) and a tiling of
//! the analyzed range into typed [`Region`]s. [`GroundTruth::validate`] enforces the properties that
//! make the labels trustworthy — the regions tile the range with no gaps or overlaps, and every
//! instruction start lands inside a code-bearing region. If that check fails, the artifact's GT is
//! not perfect and must not be used.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

/// Coarse truth for a span: is it code, data, or statically undeterminable (e.g. the runtime form
/// of self-modifying code, where the only honest static answer is "unknown").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegionLabel {
    /// Real instructions on the intended decode path.
    Code,
    /// Provably not instructions in the static image.
    Data,
    /// Not statically determinable (dynamic adversaries) — honest abstention is the correct answer.
    Unknown,
}

impl RegionLabel {
    /// Does this region contain real instruction starts?
    pub fn is_code(self) -> bool {
        matches!(self, RegionLabel::Code)
    }

    /// Stable lowercase tag for the `.regions` TSV.
    pub fn as_str(self) -> &'static str {
        match self {
            RegionLabel::Code => "code",
            RegionLabel::Data => "data",
            RegionLabel::Unknown => "unknown",
        }
    }
}

/// The specific adversarial nature of a span — finer than its [`RegionLabel`], for analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegionKind {
    /// Ordinary benign code (the payload).
    RealCode,
    /// Unpacker / loader stub — real code that precedes a packed body.
    StubCode,
    /// Real instructions woven to overlap (anti-disassembly).
    OverlappingCode,
    /// Code-shaped bytes placed off the intended path (decoy / dead code).
    JunkDecoy,
    /// Data embedded inside a code region (jump tables, constant pools).
    InlineData,
    /// Compressed payload (packer body).
    CompressedPayload,
    /// Encrypted payload (crypter body).
    EncryptedPayload,
    /// Bytecode for a custom VM (VM-obfuscation) — data to a native disassembler.
    VmBytecode,
    /// The static preimage of self-modifying code — not the bytes that run.
    SmcPreimage,
    /// File/format headers and metadata.
    Header,
    /// Alignment / fill bytes.
    Padding,
}

impl RegionKind {
    /// Stable lowercase tag for the `.regions` TSV.
    pub fn as_str(self) -> &'static str {
        match self {
            RegionKind::RealCode => "real_code",
            RegionKind::StubCode => "stub_code",
            RegionKind::OverlappingCode => "overlapping_code",
            RegionKind::JunkDecoy => "junk_decoy",
            RegionKind::InlineData => "inline_data",
            RegionKind::CompressedPayload => "compressed_payload",
            RegionKind::EncryptedPayload => "encrypted_payload",
            RegionKind::VmBytecode => "vm_bytecode",
            RegionKind::SmcPreimage => "smc_preimage",
            RegionKind::Header => "header",
            RegionKind::Padding => "padding",
        }
    }
}

/// A typed span of the analyzed range, `[start, end)` in vaddr.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Region {
    /// Inclusive start vaddr.
    pub start: u64,
    /// Exclusive end vaddr.
    pub end: u64,
    /// Coarse truth.
    pub label: RegionLabel,
    /// Fine-grained adversarial kind.
    pub kind: RegionKind,
    /// Free-text note for humans / debugging.
    pub note: String,
}

/// Perfect ground truth for one artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruth {
    /// True instruction-start vaddrs (the positive labels / "true-instruction log").
    pub instruction_starts: BTreeSet<u64>,
    /// Typed tiling of the analyzed range.
    pub regions: Vec<Region>,
    /// How we know these labels are true (the GT's chain of custody).
    pub provenance: String,
}

impl GroundTruth {
    /// Validate that the regions tile `[lo, hi)` contiguously with no gaps/overlaps and that every
    /// instruction start sits inside a code-bearing region. This is "GT is perfect" in code.
    pub fn validate(&self, lo: u64, hi: u64) -> Result<()> {
        if self.regions.is_empty() {
            bail!("ground truth has no regions");
        }
        let mut cursor = lo;
        for (i, r) in self.regions.iter().enumerate() {
            if r.start >= r.end {
                bail!("region {i} is empty or inverted: [0x{:x},0x{:x})", r.start, r.end);
            }
            if r.start != cursor {
                bail!(
                    "region {i} starts at 0x{:x} but expected 0x{cursor:x} (gap or overlap)",
                    r.start
                );
            }
            cursor = r.end;
        }
        if cursor != hi {
            bail!("regions end at 0x{cursor:x} but analyzed range ends at 0x{hi:x}");
        }
        // Every positive label must land inside a Code region.
        for &addr in &self.instruction_starts {
            if addr < lo || addr >= hi {
                bail!("instruction start 0x{addr:x} is outside the analyzed range");
            }
            let in_code = self
                .regions
                .iter()
                .any(|r| r.label.is_code() && addr >= r.start && addr < r.end);
            if !in_code {
                bail!("instruction start 0x{addr:x} is not inside any code region");
            }
        }
        Ok(())
    }

    /// Write the `.gt` file: one hex instruction-start vaddr per line (project format).
    pub fn write_gt(&self, path: &Path) -> Result<()> {
        let mut s = String::with_capacity(self.instruction_starts.len() * 12);
        for &addr in &self.instruction_starts {
            s.push_str(&format!("{addr:x}\n"));
        }
        fs::write(path, s).with_context(|| format!("writing {}", path.display()))
    }

    /// Write the `.regions` TSV: `start end label kind note`, one typed span per line.
    pub fn write_regions(&self, path: &Path) -> Result<()> {
        let mut s = String::from("# start\tend\tlabel\tkind\tnote\n");
        for r in &self.regions {
            s.push_str(&format!(
                "0x{:x}\t0x{:x}\t{}\t{}\t{}\n",
                r.start,
                r.end,
                r.label.as_str(),
                r.kind.as_str(),
                r.note
            ));
        }
        fs::write(path, s).with_context(|| format!("writing {}", path.display()))
    }
}
