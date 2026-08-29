//! Concrete generators.
//!
//! Native generators (in-process Rust) are deterministic and fully testable offline; they reuse
//! `evalkit`'s ELF/data machinery so the suite stays DRY. External generators wrap a published tool
//! and self-report availability so the corpus is usable with any subset of tools installed.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use evalkit::{build_min_elf, extract_text, make_data, DataKind};
use serde_json::json;

use crate::generator::{Artifact, Availability, Bucket, GenConfig, Generator, Registry, Seed};
use crate::gt::{GroundTruth, Region, RegionKind, RegionLabel};
use crate::manifest::{fnv1a_hex, now_unix, Manifest, SeedRef};

/// The standard generator set. External generators appear here regardless of whether their tool is
/// installed; the CLI prints availability and skips the missing ones.
pub fn standard_registry() -> Registry {
    let mut r = Registry::new();
    r.push(Box::new(CodeInData));
    r.push(Box::new(Headerless));
    r.push(Box::new(DesyncCc));
    r
}

/// Assemble a provenance manifest with the fields every generator fills the same way.
#[allow(clippy::too_many_arguments)]
fn make_manifest(
    seed: &Seed,
    gen_id: &str,
    version: &str,
    bucket: Bucket,
    out_name: &str,
    binary: &[u8],
    gt_provenance: &str,
    params: serde_json::Value,
    transform_chain: Vec<String>,
    tool_versions: BTreeMap<String, String>,
) -> Manifest {
    Manifest {
        artifact: out_name.to_string(),
        generator: gen_id.to_string(),
        generator_version: version.to_string(),
        bucket: bucket.as_str().to_string(),
        seed: SeedRef {
            name: seed.name.clone(),
            path: seed.path.display().to_string(),
            content_hash: fnv1a_hex(&seed.bytes),
        },
        params,
        transform_chain,
        tool_versions,
        created_unix: now_unix(),
        gt_provenance: gt_provenance.to_string(),
        binary_hash: fnv1a_hex(binary),
    }
}

/// Instruction starts of the seed, restricted to `[lo, hi)` (drops any stray label defensively).
fn starts_in(seed: &Seed, lo: u64, hi: u64) -> BTreeSet<u64> {
    seed.gt.iter().copied().filter(|&a| a >= lo && a < hi).collect()
}

// ── Native: code-in-data (bucket A) ─────────────────────────────────────────────────

/// Appends a block of real, tiled instruction bytes past the entry — a low-entropy, self-consistent
/// "decoy code" region that is provably off the intended path. The hardest local-analysis case: the
/// entropy prior can't gate it (it's real code) and the negatives look exactly like the positives.
pub struct CodeInData;

impl Generator for CodeInData {
    fn id(&self) -> &'static str {
        "native-code-in-data"
    }
    fn bucket(&self) -> Bucket {
        Bucket::LayoutEncoding
    }
    fn describe(&self) -> &'static str {
        "append a code-shaped decoy region (tiled real instructions) past the entry; GT by construction"
    }
    fn generate(&self, seed: &Seed, cfg: &GenConfig) -> Result<Artifact> {
        let (vaddr, code) = extract_text(&seed.bytes).context("extracting .text from seed")?;
        let mut payload = code.to_vec();
        payload.extend(make_data(DataKind::Code, cfg.seed_rng, code, cfg.region_bytes));
        let code_end = vaddr + code.len() as u64;
        let end = vaddr + payload.len() as u64;
        let binary = build_min_elf(vaddr, &payload);

        let regions = vec![
            Region {
                start: vaddr,
                end: code_end,
                label: RegionLabel::Code,
                kind: RegionKind::RealCode,
                note: "benign payload (gen-gt instruction starts)".into(),
            },
            Region {
                start: code_end,
                end,
                label: RegionLabel::Data,
                kind: RegionKind::JunkDecoy,
                note: "tiled real code placed past the entry, on no intended path".into(),
            },
        ];
        let gt = GroundTruth {
            instruction_starts: starts_in(seed, vaddr, code_end),
            regions,
            provenance: "code half: gen-gt validated starts; decoy half: real code tiled past the \
                         entry, unreachable, so provably not instruction starts"
                .into(),
        };
        gt.validate(vaddr, end).context("validating code-in-data GT")?;

        let out_name = format!("{}__{}", seed.name, self.id());
        let manifest = make_manifest(
            seed,
            self.id(),
            env!("CARGO_PKG_VERSION"),
            self.bucket(),
            &out_name,
            &binary,
            &gt.provenance,
            json!({ "region_bytes": cfg.region_bytes, "data_kind": "code", "seed_rng": cfg.seed_rng }),
            vec![self.id().into()],
            BTreeMap::new(),
        );
        Ok(Artifact { out_name, binary, gt, manifest })
    }
}

// ── Native: headerless (bucket A) ────────────────────────────────────────────────────

/// Re-wraps the seed's real code in a minimal single-segment ELF with no section headers — the
/// headerless / stripped shape that makes tools fall back (or fail). All code, full GT.
pub struct Headerless;

impl Generator for Headerless {
    fn id(&self) -> &'static str {
        "native-headerless"
    }
    fn bucket(&self) -> Bucket {
        Bucket::LayoutEncoding
    }
    fn describe(&self) -> &'static str {
        "strip section headers: one R+X segment of real code, recovered only via segment fallback"
    }
    fn generate(&self, seed: &Seed, _cfg: &GenConfig) -> Result<Artifact> {
        let (vaddr, code) = extract_text(&seed.bytes).context("extracting .text from seed")?;
        let end = vaddr + code.len() as u64;
        let binary = build_min_elf(vaddr, code);

        let regions = vec![Region {
            start: vaddr,
            end,
            label: RegionLabel::Code,
            kind: RegionKind::RealCode,
            note: "benign payload, section headers stripped".into(),
        }];
        let gt = GroundTruth {
            instruction_starts: starts_in(seed, vaddr, end),
            regions,
            provenance: "gen-gt validated starts; single R+X segment, no section headers".into(),
        };
        gt.validate(vaddr, end).context("validating headerless GT")?;

        let out_name = format!("{}__{}", seed.name, self.id());
        let manifest = make_manifest(
            seed,
            self.id(),
            env!("CARGO_PKG_VERSION"),
            self.bucket(),
            &out_name,
            &binary,
            &gt.provenance,
            json!({ "sections": "stripped" }),
            vec![self.id().into()],
            BTreeMap::new(),
        );
        Ok(Artifact { out_name, binary, gt, manifest })
    }
}

// ── External: desync-cc (bucket B) ───────────────────────────────────────────────────

/// Anti-disassembly via desync-cc (junk insertion / overlapping instructions). desync-cc is a
/// compile-time obfuscator, so it needs a *source* seed and emits a true-instruction log we turn
/// into GT — the least-circular bucket-B ground truth (a real, published adversary supplies it).
pub struct DesyncCc;

impl Generator for DesyncCc {
    fn id(&self) -> &'static str {
        "desync-cc"
    }
    fn bucket(&self) -> Bucket {
        Bucket::DecodeConfusion
    }
    fn describe(&self) -> &'static str {
        "anti-disassembly (junk + overlapping instructions) via desync-cc; GT from its true-instruction log"
    }
    fn availability(&self) -> Availability {
        match find_tool("GAUNTLET_DESYNC_CC", "desync-cc") {
            Some(_) => Availability::Available,
            None => Availability::Missing(
                "desync-cc not found — set GAUNTLET_DESYNC_CC to its path or add it to PATH".into(),
            ),
        }
    }
    fn generate(&self, seed: &Seed, _cfg: &GenConfig) -> Result<Artifact> {
        let tool = find_tool("GAUNTLET_DESYNC_CC", "desync-cc")
            .ok_or_else(|| anyhow!("desync-cc not found; set GAUNTLET_DESYNC_CC or add it to PATH"))?;
        let source = seed.source.as_ref().ok_or_else(|| {
            anyhow!("desync-cc is a compile-time obfuscator — attach a source seed via Seed::with_source")
        })?;
        // INTEGRATION POINT (bucket B): invoke `tool` on `source` to produce the obfuscated ELF and
        // its true-instruction log, then build GT with `parse_desync_log`. Wired for availability;
        // the invocation + log path depend on your desync-cc build (see design doc §9).
        bail!(
            "desync-cc integration pending: wire `{tool}` on {} and point parse_desync_log() at its \
             true-instruction-log output",
            source.display()
        );
    }
}

/// Parse a desync-cc true-instruction log into instruction-start vaddrs.
///
/// EXPECTED FORMAT: one hex address per line. Verify this against your desync-cc build's log output
/// before trusting it — this is the single integration point for bucket-B ground truth.
pub fn parse_desync_log(log: &str) -> BTreeSet<u64> {
    let mut starts = BTreeSet::new();
    for line in log.lines() {
        let t = line.trim().trim_start_matches("0x");
        if !t.is_empty() {
            if let Ok(addr) = u64::from_str_radix(t, 16) {
                starts.insert(addr);
            }
        }
    }
    starts
}

/// Resolve an external tool: prefer `$env_var` if it points at an existing file, else search `PATH`.
fn find_tool(env_var: &str, default: &str) -> Option<String> {
    if let Ok(p) = std::env::var(env_var) {
        if !p.is_empty() && Path::new(&p).exists() {
            return Some(p);
        }
    }
    let path = std::env::var("PATH").ok()?;
    for dir in path.split(':') {
        if dir.is_empty() {
            continue;
        }
        let cand = Path::new(dir).join(default);
        if cand.exists() {
            return Some(cand.to_string_lossy().into_owned());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desync_log_parses_hex_lines() {
        let log = "0x401000\n401004\n# comment\n0x40100a\n";
        let got = parse_desync_log(log);
        assert_eq!(got.len(), 3);
        assert!(got.contains(&0x401000) && got.contains(&0x401004) && got.contains(&0x40100a));
    }
}
