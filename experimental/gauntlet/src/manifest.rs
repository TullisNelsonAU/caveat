//! Provenance manifest: everything needed to reproduce and trust an artifact.
//!
//! Each generated binary carries a [`Manifest`] recording its seed (with a content hash), the
//! generator and its version, the exact parameters, the transform chain (for composites), external
//! tool versions, and — most importantly — `gt_provenance`, the statement of *why* its labels are
//! true. This is what lets the corpus defend "the GT is perfect" and "the adversary is real, not
//! ours" under review.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Reference to the benign seed an artifact was produced from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedRef {
    /// Seed name (e.g. `gcc_coreutils_64_O2_ls`).
    pub name: String,
    /// Seed path as given.
    pub path: String,
    /// Content hash of the seed bytes (change-detection / reproducibility).
    pub content_hash: String,
}

/// Full provenance for one generated artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Output file stem (`<artifact>.elf` / `.gt` / `.regions` / `.manifest.json`).
    pub artifact: String,
    /// Stable generator id (e.g. `native-code-in-data`, `desync-cc`).
    pub generator: String,
    /// Generator version (crate version for native, tool `--version` for external).
    pub generator_version: String,
    /// Taxonomy bucket tag.
    pub bucket: String,
    /// The seed this came from.
    pub seed: SeedRef,
    /// Generator parameters (free-form JSON so each generator records what it used).
    pub params: serde_json::Value,
    /// Ordered transform chain — one entry for a single transform, several for a composite.
    pub transform_chain: Vec<String>,
    /// Versions of any external tools involved (`tool` -> `version string`).
    pub tool_versions: BTreeMap<String, String>,
    /// Wall-clock creation time, Unix seconds.
    pub created_unix: u64,
    /// Statement of why the ground-truth labels are true.
    pub gt_provenance: String,
    /// Content hash of the produced binary.
    pub binary_hash: String,
}

/// FNV-1a 64-bit content hash, hex, prefixed with the algorithm. Not cryptographic — sufficient for
/// change-detection and reproducibility checks. (Promotion to `crates/` upgrades this to SHA-256.)
pub fn fnv1a_hex(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("fnv1a64:{h:016x}")
}

/// Current time in Unix seconds (0 if the clock is before the epoch, which it never is).
pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
