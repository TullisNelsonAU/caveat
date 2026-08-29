//! `gauntlet` — a ground-truth-bearing adversarial binary corpus generator.
//!
//! Generates binaries that are *structurally* as hard to disassemble as real malware — packing,
//! anti-disassembly, obfuscation, headerless layout — but wrap **benign payloads** and ship the
//! exact instruction-level ground truth. It is the infrastructure behind the adversarial paper's
//! accuracy and calibration measurements: you can only have unimpeachable GT for something you
//! produced, so we produce the corpus and bake the labels in.
//!
//! Design tenets (see `docs/adversarial_corpus_design.md`):
//!   1. GT is perfect or it isn't GT — labels come from the production process, never a disassembler.
//!   2. Benign payload, adversarial structure — no functional malware, ever.
//!   3. The adversary is the literature's (desync-cc, Tigress, OLLVM, UPX), not ours.
//!   4. Reproducible — recorded params, tool versions, content hashes, fixed RNG.
//!
//! The unit of work is a [`Generator`]: it takes a benign [`Seed`] and emits an [`Artifact`] —
//! the adversarial binary plus its [`GroundTruth`] and a provenance [`Manifest`].

#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]

pub mod generator;
pub mod generators;
pub mod gt;
pub mod manifest;

pub use generator::{Artifact, Availability, Bucket, GenConfig, Generator, Registry, Seed};
pub use generators::standard_registry;
pub use gt::{GroundTruth, Region, RegionKind, RegionLabel};
pub use manifest::{fnv1a_hex, Manifest, SeedRef};
