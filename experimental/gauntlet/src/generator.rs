//! The generator abstraction: benign [`Seed`] in, adversarial [`Artifact`] out.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::gt::GroundTruth;
use crate::manifest::Manifest;

/// Which axis of the adversarial space a generator targets (see the design doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bucket {
    /// A — packing, crypters, headerless, inline data (break the parse).
    LayoutEncoding,
    /// B — junk insertion, overlapping instructions, opaque predicates (break the disassembly).
    DecodeConfusion,
    /// C — CFG flattening, substitution, VM-obfuscation (break the meaning).
    SemanticTransform,
    /// D — self-modifying / staged / polymorphic (static GT is partial; calibration is the point).
    Dynamic,
}

impl Bucket {
    /// Stable tag for manifests and reports.
    pub fn as_str(self) -> &'static str {
        match self {
            Bucket::LayoutEncoding => "A:layout-encoding",
            Bucket::DecodeConfusion => "B:decode-confusion",
            Bucket::SemanticTransform => "C:semantic-transform",
            Bucket::Dynamic => "D:dynamic",
        }
    }
}

/// A benign input program with known ground truth. Carries the compiled bytes + instruction-start
/// GT (for post-hoc transforms) and an optional source path (for compile-time obfuscators).
pub struct Seed {
    /// Human/file name, e.g. `gcc_coreutils_64_O2_ls`.
    pub name: String,
    /// Path the compiled bytes came from.
    pub path: PathBuf,
    /// Compiled ELF bytes.
    pub bytes: Vec<u8>,
    /// True instruction-start vaddrs for the seed's `.text` (from gen-gt).
    pub gt: BTreeSet<u64>,
    /// Optional source path, for compile-time obfuscators (desync-cc, Tigress, OLLVM).
    pub source: Option<PathBuf>,
}

impl Seed {
    /// Load a binary seed from a compiled ELF and its `.gt` file (hex instruction-start per line).
    pub fn from_files(elf: &Path, gt: &Path) -> Result<Self> {
        let bytes = fs::read(elf).with_context(|| format!("reading seed elf {}", elf.display()))?;
        let gt_text = fs::read_to_string(gt).with_context(|| format!("reading seed gt {}", gt.display()))?;
        let mut starts = BTreeSet::new();
        for line in gt_text.lines() {
            let t = line.trim().trim_start_matches("0x");
            if !t.is_empty() {
                if let Ok(addr) = u64::from_str_radix(t, 16) {
                    starts.insert(addr);
                }
            }
        }
        let name = elf
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "seed".into());
        Ok(Seed { name, path: elf.to_path_buf(), bytes, gt: starts, source: None })
    }

    /// Attach a source path (enables compile-time obfuscator generators).
    pub fn with_source(mut self, source: PathBuf) -> Self {
        self.source = Some(source);
        self
    }
}

/// Knobs shared across generators. Anything random flows from `seed_rng` so the corpus reproduces.
#[derive(Debug, Clone)]
pub struct GenConfig {
    /// Master RNG seed for any synthetic content.
    pub seed_rng: u64,
    /// Size of an adversarial region (decoy / payload) in bytes, where applicable.
    pub region_bytes: usize,
    /// Free-form extra parameters for specific generators.
    pub params: serde_json::Value,
}

impl Default for GenConfig {
    fn default() -> Self {
        GenConfig {
            seed_rng: 0x9E37_79B9_7F4A_7C15,
            region_bytes: 8000,
            params: serde_json::Value::Null,
        }
    }
}

/// A produced adversarial binary plus its perfect GT and provenance.
pub struct Artifact {
    /// Output file stem.
    pub out_name: String,
    /// The adversarial ELF bytes.
    pub binary: Vec<u8>,
    /// Perfect ground truth for `binary`.
    pub gt: GroundTruth,
    /// Provenance manifest.
    pub manifest: Manifest,
}

impl Artifact {
    /// Write the four sidecars into `dir`: `<stem>.elf`, `.gt`, `.regions`, `.manifest.json`.
    pub fn write(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let elf = dir.join(format!("{}.elf", self.out_name));
        let gt = dir.join(format!("{}.gt", self.out_name));
        let regions = dir.join(format!("{}.regions", self.out_name));
        let manifest = dir.join(format!("{}.manifest.json", self.out_name));
        fs::write(&elf, &self.binary).with_context(|| format!("writing {}", elf.display()))?;
        self.gt.write_gt(&gt)?;
        self.gt.write_regions(&regions)?;
        let json = serde_json::to_string_pretty(&self.manifest).context("serializing manifest")?;
        fs::write(&manifest, json).with_context(|| format!("writing {}", manifest.display()))?;
        Ok(())
    }
}

/// Whether a generator can run right now (external tools may be absent).
pub enum Availability {
    /// Ready to run.
    Available,
    /// Not runnable; the string says why (and how to fix it).
    Missing(String),
}

impl Availability {
    /// Convenience: is this generator runnable?
    pub fn is_available(&self) -> bool {
        matches!(self, Availability::Available)
    }
}

/// A single adversarial transform. Implementors are either native (in-process Rust) or thin
/// wrappers over an external tool; the trait is the same so the CLI treats them uniformly.
pub trait Generator {
    /// Stable id (e.g. `native-code-in-data`, `desync-cc`).
    fn id(&self) -> &'static str;
    /// Taxonomy bucket.
    fn bucket(&self) -> Bucket;
    /// One-line human description.
    fn describe(&self) -> &'static str;
    /// Whether this generator can run now. Native generators are always available.
    fn availability(&self) -> Availability {
        Availability::Available
    }
    /// Produce one artifact from `seed`.
    fn generate(&self, seed: &Seed, cfg: &GenConfig) -> Result<Artifact>;
}

/// A set of generators, looked up by id.
pub struct Registry {
    gens: Vec<Box<dyn Generator>>,
}

impl Registry {
    /// Empty registry.
    pub fn new() -> Self {
        Registry { gens: Vec::new() }
    }

    /// Add a generator.
    pub fn push(&mut self, g: Box<dyn Generator>) {
        self.gens.push(g);
    }

    /// All generators, in registration order.
    pub fn generators(&self) -> &[Box<dyn Generator>] {
        &self.gens
    }

    /// Look up a generator by id.
    pub fn get(&self, id: &str) -> Option<&dyn Generator> {
        self.gens.iter().find(|g| g.id() == id).map(|b| b.as_ref())
    }
}

impl Default for Registry {
    fn default() -> Self {
        Registry::new()
    }
}
