//! `tool` — the serializable analysis core (`TOOL_CORE_CLI_SPEC`).
//!
//! One clean, callable, serde-serializable pass over the probabilistic-disassembly stack, plus the
//! result type every front-end reuses. The engines already exist — `probdisasm` (Soft decode +
//! cavity surprise), `probcfg` (Layer-2 function confirmation `F_h` + reachedness `R_a`), and the
//! Paper-2 consistency statistics. This crate is the *serialization boundary*: it depends on them,
//! adds `serde` itself, and defines its own result types populated from their outputs. Nothing in the
//! engine crates is touched — they stay serde-free.
//!
//! The keystone is [`AnalysisResult`]: the JSON the CLI prints, the (future) TUI renders, and the
//! (future) GUI draws. Getting that contract clean and documented is the whole point — the frontends
//! are thin *because* the result format carries everything they need.
//!
//! **Honesty wall (inherited from the engines, non-negotiable here):** Layer-2 confirmation and
//! reachedness are decode/reachability *weights*, never a silent overwrite of the per-byte posterior
//! π. The cavity surprise is a post-hoc read of the converged graph — it never fed back into π. This
//! tool only *reports* those numbers; it does not launder one axis into another.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use probcfg::{build_soft_confirm_resolved, resolve_indirect, ResolveConfig, ResolveKind,
    ResolvedEdge, SoftConfig};
use probdisasm::{CavityStat, Superset};
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// §1. The keystone — `AnalysisResult` (the JSON contract every frontend reuses).
//
// Field names are the contract: keep them stable, keep them documented. A frontend that renders a
// confidence column reads `functions[].confidence`; one that draws "don't trust this region" reads
// `trust.distrust_regions`. Everything below is populated from the engines' outputs in `analyze`.
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// The one result type. Serialize it to `--out`, render it in a TUI, draw it in a GUI — same bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// What we analyzed: path, format, arch, entry, byte count of the analyzed `.text`.
    pub binary: BinaryInfo,
    /// Which regime the engine ran in and where that choice came from (auto-detected vs `--facts`).
    pub regime: RegimeInfo,
    /// Candidate function heads with their Layer-2 confidence `F_h` and reachedness.
    pub functions: Vec<FunctionInfo>,
    /// Call-graph edges with per-edge noisy-OR confidence.
    pub edges: Vec<EdgeInfo>,
    /// Per-instruction posterior summary; the full per-address list is opt-in (`--full-insns`).
    pub instructions: InstructionSummary,
    /// The GT-free trust verdict: global magnitude, spatial clustering, and *localized* distrust.
    pub trust: TrustReport,
    /// Active-mode: the top things to confirm next, ranked by expected information gain.
    pub suggestions: Vec<Suggestion>,
    /// What calibration map was applied and why (the map-switching hook lives here).
    pub calibration: CalibrationInfo,
}

/// The analyzed binary's identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryInfo {
    pub path: String,
    /// Container format, e.g. `"elf"`.
    pub format: String,
    /// Instruction-set architecture, e.g. `"x86_64"`.
    pub arch: String,
    /// The entry point actually used (the ELF entry, or a `--facts` override).
    pub entry: u64,
    /// Bytes of the analyzed `.text` region.
    pub n_bytes: usize,
}

/// The regime the engine ran under. Until the map-switching experiment lands the engine always runs
/// benign; `detected` is a coarse auto heuristic (documented in `analyze`) unless `--facts` overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeInfo {
    /// `"benign" | "packed" | "obfuscated"` — the coarse class.
    pub detected: String,
    /// `"auto"` (heuristic) or `"override"` (from `--facts regime`).
    pub source: String,
    /// Confidence in the class in `[0,1]` (heuristic; `1.0` when overridden).
    pub confidence: f64,
}

/// One candidate function head and its Layer-2 confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionInfo {
    /// Head (entry) address of the candidate function.
    pub addr: u64,
    /// Symbol name if a `--facts` symbol pinned it, else `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// `F_h` — the confirmation-fixpoint confidence that this is a real function (eq 3 of Layer-2).
    pub confidence: f64,
    /// `R_head` — reachedness of the head instruction (noisy-OR over containing confirmed functions).
    pub reached_prob: f64,
    /// If this head also sits inside another confirmed function's body, that function's head.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub in_body_of: Option<u64>,
    /// Low-confidence AND no confirmed caller: the residual-tail signature. **Honest caveat:** a
    /// genuine appended decoy and an indirect-only *real* function are locally indistinguishable here
    /// (Layer-2 Theorem 2), so this flags "unconfirmed / suspect", not "provably fake".
    pub flagged_decoy: bool,
}

/// A call-graph edge `from → to`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeInfo {
    pub from: u64,
    pub to: u64,
    /// `"direct"` (a static direct-CALL site) or `"indirect"` (a resolved code pointer / a `--facts`
    /// clamp modeled as a resolved edge from the entry root).
    pub kind: String,
    /// Noisy-OR edge evidence `C_{from→to}` in `[0,1]`.
    pub confidence: f64,
}

/// Per-instruction posterior summary. The full per-address list is large and opt-in.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstructionSummary {
    /// Number of candidate (valid-decode) addresses.
    pub n: usize,
    /// Mean per-byte posterior π over all candidates.
    pub mean_pi: f64,
    /// Candidate addresses with π below [`LOW_PI`] (least-trusted decodes). Capped in the summary;
    /// see `low_confidence_count` for the true total.
    pub low_confidence: Vec<u64>,
    /// Total number of low-confidence addresses (may exceed the capped `low_confidence` list).
    pub low_confidence_count: usize,
    /// Full per-address `(addr, π)`, present only when the caller asked for it (`--full-insns`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_insn: Option<Vec<InsnPosterior>>,
}

/// One address's posterior (only emitted in the full per-insn mode).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsnPosterior {
    pub addr: u64,
    pub pi: f64,
}

/// The GT-free trust verdict from the cavity-surprise statistics (Paper 2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustReport {
    /// `"trustworthy"` or `"suspect"` — the overall verdict (see `analyze` for the rule).
    pub overall: String,
    /// `S_glob` — global magnitude: mean per-address surprise.
    pub s_glob: f64,
    /// `S_spat` — spatial clustering: Moran's I of the standardized residual over address order.
    pub s_spat: f64,
    /// Localized distrust: contiguous runs of high-surprise addresses ("don't trust this region").
    pub distrust_regions: Vec<DistrustRegion>,
}

/// A contiguous address window where the decode surprise clustered — the calibration is likely stale
/// here, so the confidences inside it should be trusted less.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistrustRegion {
    pub addr_lo: u64,
    pub addr_hi: u64,
    /// Human-readable reason for the flag.
    pub reason: String,
    /// Mean surprise over the run (its magnitude).
    pub surprise: f64,
}

/// One active-mode suggestion: an address whose confirmation would resolve the most uncertainty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Suggestion {
    pub addr: u64,
    /// Expected information gain (bits, heuristic): head uncertainty × body size — see `query_gain`.
    pub expected_info_gain: f64,
    /// Why this address is worth confirming.
    pub why: String,
}

/// The self-calibration record: which regime the signature classifier selected and which bank config
/// (engine setting + map) was applied. This is Paper 2's switch made live — the tool classifies the
/// regime GT-free from the (S_glob, S_spat) signature and routes to the matching calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationInfo {
    /// The regime the classifier selected (or a `--facts` override): `"benign"|"packed"|"obfuscated"`.
    pub selected_regime: String,
    /// How the regime was chosen: `"signature-rule"` (the trained threshold rule) or `"override"`.
    pub classifier: String,
    /// The bank config applied, e.g. `"packed:entropy-prior"` — the engine setting that recalibrates π.
    pub map_applied: String,
    /// The engine data-prior strengths applied `(entropy, chainfwd)`.
    pub engine: [f64; 2],
    /// `true` when a fitted (non-identity) isotonic map from the bank was applied to the marginals —
    /// the *map*-switch, on top of the engine-switch. `false` = engine config only (no bank, or the
    /// selected regime's bank map is identity).
    pub isotonic_applied: bool,
    /// `true` when the rule fell back to benign but residual drift suggests the calibration may be
    /// stale (the semantic-obfuscation blind spot — flagged, not mis-routed).
    pub regime_uncertain: bool,
    /// A human-readable note on the selection and its honest scope.
    pub note: String,
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// §2. The facts input (`--facts f.json`) — everything the user knows, mapped to clamps/anchors.
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// Everything the user knows about the binary, each field mapped to a clamp or the entry anchor in
/// [`analyze`]. An absent file ⇒ analyze with just the binary (all fields `None`/empty).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnownFacts {
    /// Known entry point. Overrides the ELF entry as the confirmation anchor (`F_entry = 1`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry: Option<u64>,
    /// Partial symbol table `addr → name`: each real function head, clamped confirmed.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub symbols: HashMap<u64, String>,
    /// Human labels `addr → "function" | "data"`. `"function"` clamps real (like a symbol); `"data"`
    /// clamps the address's confidence to 0 (known-not-code).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub annotations: HashMap<u64, String>,
    /// Executed addresses from a trace: each is clamped reached (and, if a head, clamped confirmed).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trace: Vec<u64>,
    /// Override the auto-detected regime with a known one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regime: Option<String>,
}

impl KnownFacts {
    /// Load facts from a JSON file. `serde` accepts addresses as JSON numbers *or* `"0x…"`/decimal
    /// strings (see [`de_addr`] on the untyped path); the typed fields here take numbers directly.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading facts {}", path.display()))?;
        let facts: KnownFacts = serde_json::from_str(&text)
            .with_context(|| format!("parsing facts {}", path.display()))?;
        Ok(facts)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// §2b. The calibration-map bank (`--bank bank.json`) — Paper 2's fitted per-regime isotonic maps.
//
// The switch has two halves: the *engine config* (a regime-matched decode; always applied) and the
// post-hoc *isotonic calibration map* (a GT-fit recalibration of the marginals). The maps are fit
// once, offline, on labeled corpora and serialized here as PAVA points (`evalkit::IsotonicMap::
// {to_points,from_points}`); the tool loads the bank and applies the selected regime's map to the
// marginals before assembling the result. This is what reproduces the paper's *restoration*, not just
// the routing — e.g. the packed map is the degenerate all-data fit, so it drives packed posteriors to
// ≈0 (abstain / suppress). An empty per-regime point list is the identity map (no recalibration).
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// Provenance for a serialized bank — how/where the maps were fit, so a loaded bank is auditable.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BankMeta {
    /// Free-text provenance (corpus, engine strengths, commit).
    #[serde(default)]
    pub source: String,
    /// Engine strengths the maps were fit under `(entropy, chainfwd)` — should match the bank configs.
    #[serde(default)]
    pub engine: [f64; 2],
}

/// A bank of fitted per-regime isotonic calibration maps, each stored as `evalkit::IsotonicMap` PAVA
/// points `(x_upper, value)`. An empty list ⇒ identity (no recalibration for that regime).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CalibrationBank {
    #[serde(default)]
    pub meta: BankMeta,
    /// Benign regime map — usually identity (clean code needs no recalibration).
    #[serde(default)]
    pub benign: Vec<(f64, f64)>,
    /// Packed regime map — the degenerate all-data fit (drives posteriors → ≈0, the abstain regime).
    #[serde(default)]
    pub packed: Vec<(f64, f64)>,
    /// Obfuscated (desync) regime map — non-degenerate; identity unless fit on a desync corpus.
    #[serde(default)]
    pub obfuscated: Vec<(f64, f64)>,
}

impl CalibrationBank {
    /// Load a bank from JSON.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading bank {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing bank {}", path.display()))
    }

    /// The fitted [`evalkit::IsotonicMap`] for a regime (reconstructed from its PAVA points).
    fn map(&self, r: Regime) -> evalkit::IsotonicMap {
        let pts = match r {
            Regime::Benign => &self.benign,
            Regime::Packed => &self.packed,
            Regime::Obfuscated => &self.obfuscated,
        };
        evalkit::IsotonicMap::from_points(pts.clone())
    }

    /// Is the selected regime's map a real (non-identity) recalibration?
    fn is_active(&self, r: Regime) -> bool {
        !match r {
            Regime::Benign => &self.benign,
            Regime::Packed => &self.packed,
            Regime::Obfuscated => &self.obfuscated,
        }
        .is_empty()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// §3. `analyze()` — the core entry (one function, the whole pipeline).
// ═══════════════════════════════════════════════════════════════════════════════════════════════

// ── The regime bank + signature classifier (Paper 2's self-calibrating switch) ──────────────────
//
// The `consistency` switching experiment (CONSISTENCY_SWITCHING_RESULTS, n=80 held-out, GT-free)
// turned the drift *detector* into a self-calibrating *capability*: a bank of three regime configs
// and a rule that reads a binary's (S_glob, S_spat) signature and routes to the matching config. The
// benign-default threshold rule won outright — 1.00 selection accuracy on the held-out corpus. We
// port that rule and its bank here so the tool self-calibrates live.
//
// Honest scope (non-negotiable): this routes *structural* obfuscation — packing (entropy signature)
// and anti-disassembly desync (elevated S_glob). Semantic / virtualized obfuscation (Tigress) keeps
// clean decoding, so the surprise statistic is blind to it (the graded-Tigress probe: the mid-band
// exists, S doesn't see it). The rule is benign-default precisely so an unrecognized signature is
// never mis-routed into the calibration-destroying packed/obf map; when there's residual drift the
// rule can't confidently place, we fall back to benign AND flag "regime uncertain — calibration may
// be stale" rather than pretend we recognized it.

/// The three regimes the bank covers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Regime {
    Benign,
    Packed,
    Obfuscated,
}

impl Regime {
    fn tag(self) -> &'static str {
        match self {
            Regime::Benign => "benign",
            Regime::Packed => "packed",
            Regime::Obfuscated => "obfuscated",
        }
    }
    /// The engine data-prior setting `(entropy_prior_strength, chainfwd_strength)` for this regime —
    /// exactly the bank's per-regime `AnalysisConfig` knobs. Benign = both off (the untouched shared
    /// engine); packed = entropy prior (pull high-entropy payload → data); obfuscated = chainfwd
    /// prior (pull chain-consistent bytes → code). Applying this *is* the calibration switch: it
    /// re-runs the decode under the regime-matched model, changing π toward the honest calibration.
    fn engine(self) -> (f64, f64) {
        match self {
            Regime::Benign => (0.0, 0.0),
            Regime::Packed => (ENGINE_ENTROPY_STRENGTH, 0.0),
            Regime::Obfuscated => (0.0, ENGINE_CHAINFWD_STRENGTH),
        }
    }
    fn from_tag(s: &str) -> Option<Regime> {
        match s {
            "benign" => Some(Regime::Benign),
            "packed" => Some(Regime::Packed),
            "obfuscated" => Some(Regime::Obfuscated),
            _ => None,
        }
    }
}

/// Trained obfuscated threshold: `S_glob` above this ⇒ obfuscated (desync raises mean surprise far
/// above the clean tail). From the switching run's clean-fit null (`glob_hi = clean_p95 × 2.5`).
const RULE_GLOB_HI: f64 = 2.5659;
/// Trained packed threshold: `S_spat` (Moran's I) above this, with `S_glob` below `RULE_GLOB_HI` ⇒
/// packed (packers collapse S_glob but keep the spatial statistic ~2× clean). From the switching
/// run's clean-fit spatial p95.
const RULE_SPAT_HI: f64 = 0.1084;
/// The clean-fit `S_glob` p95 the obfuscated threshold was built from (`RULE_GLOB_HI / 2.5`). Between
/// this and `RULE_GLOB_HI` is the ambiguous mid-band: some drift, not enough to route — flag uncertain.
const RULE_GLOB_NULL: f64 = RULE_GLOB_HI / 2.5;
/// Bank engine strengths (the switching run's defaults: `--entropy-strength 1.0 --chainfwd-strength 0.5`).
const ENGINE_ENTROPY_STRENGTH: f64 = 1.0;
const ENGINE_CHAINFWD_STRENGTH: f64 = 0.5;

/// The benign-default threshold rule from the switching experiment (`SignatureClassifier::
/// classify_rule`): obfuscated if S_glob clearly elevated; else packed if the spatial statistic
/// exceeds the clean null; else benign. Reads only the two GT-free scalars off the benign-engine pass.
fn classify_rule(s_glob: f64, s_spat: f64) -> Regime {
    if s_glob > RULE_GLOB_HI {
        Regime::Obfuscated
    } else if s_spat > RULE_SPAT_HI {
        Regime::Packed
    } else {
        Regime::Benign
    }
}

/// Is a benign verdict actually *uncertain*? True when the rule fell back to benign but the global
/// surprise sits in the mid-band — above the clean-fit null yet below the obfuscated threshold. That
/// is the honest signature of the semantic/virtualized-obfuscation blind spot (Tigress raises S_glob
/// into this band but not over the threshold): the tool says benign (the safe default) yet flags that
/// the calibration may be stale rather than asserting clean. (The spatial axis needs no clause here —
/// if S_spat were high enough to matter, the rule would already have routed the binary to packed.)
fn benign_is_uncertain(s_glob: f64, _s_spat: f64) -> bool {
    s_glob > RULE_GLOB_NULL
}

/// Posterior below this counts an address as a low-confidence decode.
pub const LOW_PI: f64 = 0.5;
/// `F_h` below this, with no confirmed caller, flags a head as residual-tail / suspect.
const F_SUSPECT: f64 = 0.5;
/// Cap on the `low_confidence` address list carried in the summary (the full set is opt-in).
const LOW_CONF_CAP: usize = 256;
/// A within-binary surprise event is an address whose surprise exceeds μ + this·σ.
const SURPRISE_SIGMA: f64 = 2.0;
/// A distrust region is a run of at least this many consecutive surprise events.
const MIN_RUN: usize = 3;
/// Confidence assigned to a synthetic "the user told us this is real" resolved edge.
const FACT_EDGE_Q: f64 = 0.99;

/// Run the whole pipeline on one binary and assemble the [`AnalysisResult`].
///
/// Pipeline:
///   1. `.text` from the ELF, entry from the facts (else the ELF entry).
///   2. Benign Soft decode → per-byte π + cavity surprise → the GT-free signature (S_glob, S_spat).
///   3. **Self-calibration**: classify the regime from the signature (the trained threshold rule),
///      re-run the decode under the matching bank engine config, then — if a `bank` is supplied —
///      apply that regime's fitted isotonic map to the marginals (the *map*-switch that reproduces
///      the paper's restoration, e.g. packed → ≈0). Fall back to benign (flagged uncertain) when the
///      signature isn't recognized.
///   4. Facts → clamps: known heads become synthetic resolved edges from the entry root (the M3a
///      "resolved-real caller" lift); `"data"` annotations become a suppression set.
///   5. Layer-2 confirmation fixpoint over the superset with the real + synthetic resolved edges
///      (`probcfg::build_soft_confirm_resolved`) → `F_h`, reachedness `R_a`.
///   6. Consistency detector over the cavity stats → trust (S_glob/S_spat, localized distrust).
///   7. Rank `query_gain` for the top-k suggestions.
///   8. Assemble + return.
pub fn analyze(
    binary: &Path,
    facts: Option<&KnownFacts>,
    bank: Option<&CalibrationBank>,
    full_insns: bool,
) -> Result<AnalysisResult> {
    let bytes = std::fs::read(binary).with_context(|| format!("reading {}", binary.display()))?;

    // (1) code + entry. Prefer the `.text` section; fall back to the executable `PT_LOAD` for
    // header-stripped / packed images (UPX drops section headers). The ELF header gives
    // format/arch/entry for the `binary` block.
    let (base, code) = extract_code(&bytes).context("extracting code")?;
    let hdr = elf_header(&bytes)?;
    let entry = facts.and_then(|f| f.entry).unwrap_or(hdr.entry);
    let superset = Superset::new(base, code).context("building superset")?;

    // (2) Benign Soft decode: per-byte posterior π + the read-only cavity surprise. The signature the
    // classifier reads is *always* the benign-engine one (S_glob/S_spat), exactly as the switching
    // experiment characterized it — so the classification is on the same footing as its training.
    let (benign_post, cav) = evalkit::run_soft_with_cavity(base, code, 0.0, false)
        .with_context(|| format!("Soft decode on {}", binary.display()))?;
    let stats = cavity_stats(&cav);
    let distrust = distrust_regions(&cav, &stats);

    // (3) Self-calibration: classify the regime from the (S_glob, S_spat) signature and apply the
    // matching bank config. `--facts regime` overrides the classifier (the user asserted the regime).
    let (selected, source) = match facts.and_then(|f| f.regime.as_deref()).and_then(Regime::from_tag) {
        Some(r) => (r, "override"),
        None => (classify_rule(stats.mean_surprise, stats.moran), "signature-rule"),
    };
    let regime_uncertain =
        source == "signature-rule" && selected == Regime::Benign && benign_is_uncertain(stats.mean_surprise, stats.moran);
    // Apply the selected regime's engine config: benign reuses the pass we already have; packed /
    // obfuscated re-run the decode under their data-prior knob so π is recalibrated for that regime.
    let (ent, cfw) = selected.engine();
    let engine_post: Vec<(u64, f64)> = if selected == Regime::Benign {
        benign_post
    } else {
        evalkit::run_soft_with_cavity_cfg(base, code, ent, cfw, false)
            .with_context(|| format!("regime[{}] engine on {}", selected.tag(), binary.display()))?
            .0
    };
    // Then the *map*-switch: apply the bank's fitted isotonic map for this regime to the marginals.
    // This is the second half of the bank config — the GT-fit recalibration that turns the routing
    // into the paper's restoration (packed's degenerate map drives every posterior → ≈0). Absent bank
    // or an identity map ⇒ the marginals are the engine output unchanged.
    let isotonic_applied = bank.map(|b| b.is_active(selected)).unwrap_or(false);
    let post: Vec<(u64, f64)> = match bank {
        Some(b) if isotonic_applied => b.map(selected).apply_all(&engine_post),
        _ => engine_post,
    };
    let pmap: HashMap<u64, f64> = post.iter().copied().collect();

    // (4) Facts → clamps. `"data"` annotations are a suppression set applied at assembly time. Every
    // known-real head (symbol, `"function"` annotation, or a traced address that decodes) becomes a
    // synthetic resolved edge `entry → head`: it lands in exactly the eq-1 evidence slot a recovered
    // code pointer would, so the fixpoint lifts it from the tail to the core with earned structure —
    // no posterior is overwritten.
    let data_clamp: HashSet<u64> = facts
        .map(|f| f.annotations.iter().filter(|(_, v)| v.as_str() == "data").map(|(a, _)| *a).collect())
        .unwrap_or_default();
    let name_of: HashMap<u64, String> = facts.map(|f| f.symbols.clone()).unwrap_or_default();
    let trace_set: HashSet<u64> = facts.map(|f| f.trace.iter().copied().collect()).unwrap_or_default();

    let mut fact_heads: HashSet<u64> = HashSet::new();
    if let Some(f) = facts {
        for &a in f.symbols.keys() {
            fact_heads.insert(a);
        }
        for (a, v) in &f.annotations {
            if v == "function" {
                fact_heads.insert(*a);
            }
        }
        for &a in &f.trace {
            fact_heads.insert(a);
        }
    }
    // Real indirect edges recovered from the binary's data, plus one synthetic edge per fact head.
    let mut resolved = resolve_indirect(&superset, &bytes, entry, &ResolveConfig::default());
    let synthetic_targets: HashSet<u64> = fact_heads
        .iter()
        .copied()
        .filter(|&a| a != entry && superset.at(a).is_some() && !data_clamp.contains(&a))
        .collect();
    for &t in &synthetic_targets {
        resolved.push(ResolvedEdge { g: entry, t, q: FACT_EDGE_Q, kind: ResolveKind::DataPointer });
    }

    // (5) Layer-2 confirmation fixpoint over the (now regime-calibrated) posteriors + edges.
    let sc = build_soft_confirm_resolved(&superset, entry, &pmap, &resolved, &SoftConfig::default());

    // Direct-call (g,t) pairs, so an edge can be labeled direct vs indirect at assembly.
    let direct_pairs = direct_call_pairs(&superset, &sc.heads);
    // Reverse index head → containing confirmed function (for `in_body_of`).
    let container = body_container(&sc.bodies, &sc.f);

    // (6) Assemble the regime + self-calibration record from the classifier's decision.
    let regime = RegimeInfo {
        detected: selected.tag().into(),
        source: if source == "override" { "override" } else { "auto" }.into(),
        confidence: regime_confidence(source, selected, regime_uncertain),
    };
    let calibration = build_calibration(selected, source, [ent, cfw], isotonic_applied, regime_uncertain);

    // ── Assemble functions. `flagged_decoy` = low F AND no confirmed caller (residual-tail signature;
    // honest caveat in the field doc). `"data"`-annotated heads are suppressed to confidence 0. ──
    let mut functions: Vec<FunctionInfo> = sc
        .heads
        .iter()
        .map(|&h| {
            let confirmed_caller = has_confirmed_caller(&sc, h);
            let mut confidence = sc.f.get(&h).copied().unwrap_or(0.0);
            let mut reached = sc.r.get(&h).copied().unwrap_or(0.0);
            if data_clamp.contains(&h) {
                confidence = 0.0;
                reached = 0.0;
            }
            if trace_set.contains(&h) {
                reached = 1.0; // executed ⇒ definitively reached
            }
            FunctionInfo {
                addr: h,
                name: name_of.get(&h).cloned(),
                confidence,
                reached_prob: reached,
                in_body_of: container.get(&h).copied().filter(|&g| g != h),
                flagged_decoy: h != entry && !confirmed_caller && confidence < F_SUSPECT,
            }
        })
        .collect();
    functions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

    // ── Assemble edges from the fixpoint's incoming-edge evidence. ──
    let mut edges: Vec<EdgeInfo> = Vec::new();
    for (&to, ins) in &sc.edges_into {
        for &(from, c) in ins {
            let kind = if direct_pairs.contains(&(from, to)) { "direct" } else { "indirect" };
            edges.push(EdgeInfo { from, to, kind: kind.into(), confidence: c });
        }
    }
    edges.sort_by(|a, b| (a.from, a.to).cmp(&(b.from, b.to)));

    // ── Instruction summary. ──
    // Per-insn list (when asked) is built from the *calibrated* marginals `post` — engine config plus
    // the applied bank map — so `--full-insns` reflects the real pipeline output, not a benign re-run.
    let instructions = instruction_summary(&post, full_insns);

    // ── Trust verdict. Suspect if the classifier routed the binary to a non-benign regime (a
    // recognized structural-obfuscation signature) OR the benign fallback is flagged uncertain
    // (residual drift the rule couldn't place). A confidently-benign binary is trustworthy — its
    // isolated high-surprise windows still surface as localized `distrust_regions`, but they don't
    // by themselves flip the whole-binary verdict. ──
    let overall = if selected != Regime::Benign || regime_uncertain { "suspect" } else { "trustworthy" };
    let trust = TrustReport {
        overall: overall.into(),
        s_glob: stats.mean_surprise,
        s_spat: stats.moran,
        distrust_regions: distrust,
    };

    // (7) Active-mode suggestions: top-k by expected info gain (see `query_gain`).
    let suggestions = query_gain(&sc, entry, &data_clamp, 8);

    Ok(AnalysisResult {
        binary: BinaryInfo {
            path: binary.display().to_string(),
            format: hdr.format,
            arch: hdr.arch,
            entry,
            n_bytes: code.len(),
        },
        regime,
        functions,
        edges,
        instructions,
        trust,
        suggestions,
        calibration,
    })
}

/// The analyzable code region: the `.text` section when section headers survive, else the first
/// executable `PT_LOAD` segment (packed / stripped images drop section headers but keep the loadable
/// executable segment). Returns `(base_vaddr, code_bytes)`.
fn extract_code(bytes: &[u8]) -> Result<(u64, &[u8])> {
    if let Ok((base, code)) = evalkit::extract_text(bytes) {
        return Ok((base, code));
    }
    use goblin::elf::program_header::{PF_X, PT_LOAD};
    use goblin::Object;
    let Object::Elf(elf) = Object::parse(bytes).context("parsing ELF")? else {
        anyhow::bail!("input is not an ELF");
    };
    let seg = elf
        .program_headers
        .iter()
        .find(|p| p.p_type == PT_LOAD && p.p_flags & PF_X != 0)
        .context("no .text section and no executable PT_LOAD segment")?;
    let start = seg.p_offset as usize;
    let end = start.saturating_add(seg.p_filesz as usize).min(bytes.len());
    Ok((seg.p_vaddr, &bytes[start..end]))
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// Bank export — fit the per-regime isotonic maps the same way the switching experiment does.
//
//   packed : PAVA on the provable-data window (all label 0) under the packed engine ⇒ degenerate
//            suppress (the abstain regime; the paper's packed → ≈0).
//   desync : PAVA on `(posterior, instruction-start-label)` pooled over desync bins under the
//            obfuscated engine ⇒ a NON-degenerate recalibration (the paper's lead result: on drifted
//            code the switched map restores calibration, ECE 0.074 → ≈0.025 toward the oracle).
//   benign : identity (clean code needs no recalibration).
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// Options for [`export_bank`]. Whichever regime inputs are given get (re)fit; the rest are seeded
/// from `from` (or identity). Run with both `packed` and `desync` to ship a full bank.
pub struct ExportOpts<'a> {
    /// Seed the bank from an existing one (keep its maps unless overwritten). `None` ⇒ empty/identity.
    pub from: Option<&'a Path>,
    /// Packed regime: `(packed_elf, upxgt)`.
    pub packed: Option<(&'a Path, &'a Path)>,
    /// Desync/obfuscated regime: `(bins_dir, gt_dir)` pairs, pooled across all of them.
    pub desync: Vec<(&'a Path, &'a Path)>,
    /// Cap on desync bins fit (smallest `.text` first, for bounded runtime). `0` ⇒ no cap.
    pub desync_limit: usize,
    pub out: &'a Path,
}

/// Fit the requested regime maps and write a [`CalibrationBank`] JSON. Reproduces the switching
/// experiment's fits (see the section header). `--from` lets a slow map (packed) be reused while only
/// the desync map is refit.
pub fn export_bank(opts: &ExportOpts) -> Result<()> {
    let mut bank = match opts.from {
        Some(p) => CalibrationBank::load(p)?,
        None => CalibrationBank::default(),
    };
    let mut notes: Vec<String> = Vec::new();
    if !bank.meta.source.is_empty() {
        notes.push(bank.meta.source.clone());
    }

    if let Some((elf, upxgt)) = opts.packed {
        let (pts, n) = fit_packed_map(elf, upxgt)?;
        bank.packed = pts;
        notes.push(format!(
            "packed: PAVA on {}'s data window ({} candidates, all label 0) under the packed engine \
             (entropy={:.1}) → suppress",
            elf.file_name().and_then(|s| s.to_str()).unwrap_or("?"), n, ENGINE_ENTROPY_STRENGTH,
        ));
    }

    if !opts.desync.is_empty() {
        let (pts, n_bins, n_samples) = fit_desync_map(&opts.desync, opts.desync_limit)?;
        bank.obfuscated = pts;
        notes.push(format!(
            "obfuscated/desync: PAVA on (posterior, instr-start GT) pooled over {} desync bins \
             ({} candidates) under the obfuscated engine (chainfwd={:.1}) → non-degenerate recalibration",
            n_bins, n_samples, ENGINE_CHAINFWD_STRENGTH,
        ));
    }

    // Benign stays identity unless the seed bank carried one.
    if bank.benign.is_empty() {
        notes.push("benign: identity (clean code needs no recalibration)".into());
    }
    if bank.obfuscated.is_empty() {
        notes.push("obfuscated: identity (no desync corpus given)".into());
    }
    bank.meta.source = notes.join("; ");
    bank.meta.engine = [ENGINE_ENTROPY_STRENGTH, ENGINE_CHAINFWD_STRENGTH];

    let json = serde_json::to_string_pretty(&bank).context("serializing bank")?;
    std::fs::write(opts.out, json).with_context(|| format!("writing {}", opts.out.display()))?;
    eprintln!(
        "tool: wrote bank {} — packed {} blocks, obfuscated {} blocks, benign {} blocks",
        opts.out.display(), bank.packed.len(), bank.obfuscated.len(), bank.benign.len()
    );
    Ok(())
}

/// Fit the packed suppress map: decode under the packed engine, take the provable-data window (all
/// label 0), PAVA. Returns `(simplified_points, n_window_candidates)`.
fn fit_packed_map(packed_elf: &Path, upxgt: &Path) -> Result<(Vec<(f64, f64)>, usize)> {
    let bytes = std::fs::read(packed_elf).with_context(|| format!("reading {}", packed_elf.display()))?;
    let (base, code) = extract_code(&bytes).context("extracting code")?;
    let (ent, _cfw) = Regime::Packed.engine();
    let (post, _cav) = evalkit::run_soft_with_cavity_cfg(base, code, ent, 0.0, false)
        .with_context(|| format!("packed engine on {}", packed_elf.display()))?;
    let (lo, hi) = packed_data_window(upxgt)?;
    let samples: Vec<(f64, f64)> =
        post.iter().filter(|&&(a, _)| a >= lo && a < hi).map(|&(_, p)| (p, 0.0)).collect();
    if samples.is_empty() {
        anyhow::bail!("no candidates in packed data window [{lo:#x},{hi:#x}) — wrong .upxgt?");
    }
    eprintln!("  packed: fit on {} data-window candidates", samples.len());
    Ok((simplify_map_points(evalkit::IsotonicMap::fit(&samples).to_points()), samples.len()))
}

/// Fit the desync/obfuscated recalibration map: for each desync bin (smallest `.text` first, capped by
/// `limit`), decode under the obfuscated engine, pool `(posterior, instruction-start-label)` from its
/// `.gt`, and PAVA the pool. This is the switching experiment's obfuscated-map fit — the honest,
/// non-degenerate recalibration. Returns `(simplified_points, n_bins, n_samples)`.
fn fit_desync_map(dirs: &[(&Path, &Path)], limit: usize) -> Result<(Vec<(f64, f64)>, usize, usize)> {
    let mut pairs: Vec<(PathBuf, PathBuf)> = Vec::new();
    for (bins, gt) in dirs {
        pairs.extend(list_bins_with_gt(bins, gt)?);
    }
    if pairs.is_empty() {
        anyhow::bail!("no (bin, .gt) pairs found in the given desync dirs");
    }
    // Smallest first (metadata follows symlinks), so a `--desync-limit` picks the cheapest bins.
    pairs.sort_by_key(|(b, _)| (std::fs::metadata(b).map(|m| m.len()).unwrap_or(u64::MAX), b.clone()));
    if limit > 0 && pairs.len() > limit {
        pairs.truncate(limit);
    }
    let (_ent, cfw) = Regime::Obfuscated.engine();
    let mut pooled: Vec<(f64, f64)> = Vec::new();
    for (bin, gt) in &pairs {
        let bytes = std::fs::read(bin).with_context(|| format!("reading {}", bin.display()))?;
        let (base, code) = extract_code(&bytes).with_context(|| format!("code from {}", bin.display()))?;
        let (post, _cav) = evalkit::run_soft_with_cavity_cfg(base, code, 0.0, cfw, false)
            .with_context(|| format!("obfuscated engine on {}", bin.display()))?;
        let g = evalkit::load_gt(gt)?;
        for &(a, p) in &post {
            pooled.push((p, if g.contains(&a) { 1.0 } else { 0.0 }));
        }
        eprintln!("  desync += {} ({} candidates)", bin.file_name().and_then(|s| s.to_str()).unwrap_or("?"), post.len());
    }
    Ok((simplify_map_points(evalkit::IsotonicMap::fit(&pooled).to_points()), pairs.len(), pooled.len()))
}

/// List `(bin, gt)` pairs: every file in `bins_dir` that has a matching `<name>.gt` in `gt_dir`.
fn list_bins_with_gt(bins_dir: &Path, gt_dir: &Path) -> Result<Vec<(PathBuf, PathBuf)>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(bins_dir).with_context(|| format!("reading dir {}", bins_dir.display()))? {
        let path = entry?.path();
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name.is_empty() || name.starts_with('.')
            || name.ends_with(".gt") || name.ends_with(".log") || name.ends_with(".json")
        {
            continue;
        }
        let gt = gt_dir.join(format!("{name}.gt"));
        if gt.is_file() {
            out.push((path, gt));
        }
    }
    Ok(out)
}

/// Losslessly shrink a fitted PAVA point list: collapse each run of consecutive equal-`value` blocks
/// to its last (max-`x`) representative. `IsotonicMap::apply` returns the value of the first block
/// whose `x ≥ p`, and every block in an equal-value run shares that value, so keeping only the run's
/// upper edge leaves `apply` byte-identical for all `p` — it just drops redundant interior boundaries.
/// A degenerate all-data (constant-0) map collapses from tens of thousands of blocks to a single
/// `(1.0, 0.0)`; a non-degenerate map keeps every value change.
fn simplify_map_points(points: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    let mut out: Vec<(f64, f64)> = Vec::new();
    for (x, v) in points {
        match out.last_mut() {
            Some(last) if last.1 == v => last.0 = x, // extend the equal-value run's upper edge
            _ => out.push((x, v)),
        }
    }
    out
}

/// Parse the provable-data (NEGATIVE) vaddr window from a `.upxgt` table — the `compressed` row's
/// `vaddr_start vaddr_end`. Provenance is UPX's own `b_info` chain, not a disassembler.
fn packed_data_window(upxgt: &Path) -> Result<(u64, u64)> {
    let text = std::fs::read_to_string(upxgt).with_context(|| format!("reading {}", upxgt.display()))?;
    for line in text.lines() {
        let l = line.trim();
        if l.starts_with('#') || l.is_empty() {
            continue;
        }
        let cols: Vec<&str> = l.split_whitespace().collect();
        if cols.first() == Some(&"compressed") && cols.len() >= 4 {
            let parse = |s: &str| u64::from_str_radix(s.trim_start_matches("0x"), 16);
            return Ok((parse(cols[2])?, parse(cols[3])?));
        }
    }
    anyhow::bail!("no `compressed` NEGATIVE row in {}", upxgt.display())
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// Helpers (all read-only over the engine outputs).
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// The minimal ELF-header facts the `binary` block needs.
struct ElfHeader {
    format: String,
    arch: String,
    entry: u64,
}

fn elf_header(bytes: &[u8]) -> Result<ElfHeader> {
    use goblin::Object;
    let Object::Elf(elf) = Object::parse(bytes).context("parsing ELF")? else {
        anyhow::bail!("input is not an ELF");
    };
    let arch = match elf.header.e_machine {
        goblin::elf::header::EM_X86_64 => "x86_64",
        goblin::elf::header::EM_386 => "x86",
        goblin::elf::header::EM_AARCH64 => "aarch64",
        other => return Ok(ElfHeader { format: "elf".into(), arch: format!("machine_{other}"), entry: elf.header.e_entry }),
    };
    Ok(ElfHeader { format: "elf".into(), arch: arch.into(), entry: elf.header.e_entry })
}

/// Does any *confirmed* function call `h`? (max over incoming edges of `F_g · C ≥ 0.5`.) The
/// residual-tail test: a head with no confirmed caller sits at its bare prior.
fn has_confirmed_caller(sc: &probcfg::SoftConfirm, h: u64) -> bool {
    sc.edges_into
        .get(&h)
        .map(|es| es.iter().any(|&(g, c)| sc.f.get(&g).copied().unwrap_or(0.0) * c >= 0.5))
        .unwrap_or(false)
}

/// The set of direct-CALL `(caller_head, callee)` pairs, so edges can be labeled direct vs indirect.
/// Mirrors the builder's own site collection: a call inside `g`'s body whose target is a head.
fn direct_call_pairs(sup: &Superset, heads: &[u64]) -> HashSet<(u64, u64)> {
    use probcfg::extract_function;
    let head_set: HashSet<u64> = heads.iter().copied().collect();
    let mut pairs = HashSet::new();
    for &g in heads {
        let f = extract_function(sup, g, &head_set, 65536);
        for &a in &f.body {
            if let Some(insn) = sup.at(a) {
                if insn.is_call() {
                    if let Some(t) = insn.branch_target {
                        if t != g && head_set.contains(&t) {
                            pairs.insert((g, t));
                        }
                    }
                }
            }
        }
    }
    pairs
}

/// Reverse index: for each address that is *another* confirmed function's head, the containing head.
/// Only confirmed (`F ≥ 0.5`) bodies contribute, so `in_body_of` reflects real containment.
fn body_container(bodies: &HashMap<u64, Vec<u64>>, f: &HashMap<u64, f64>) -> HashMap<u64, u64> {
    let mut out: HashMap<u64, u64> = HashMap::new();
    for (&g, body) in bodies {
        if f.get(&g).copied().unwrap_or(0.0) < 0.5 {
            continue;
        }
        for &a in body {
            out.entry(a).or_insert(g);
        }
    }
    out
}

/// Aggregated cavity statistics over one binary (the GT-free consistency read).
struct CavityAgg {
    mean_surprise: f64,
    moran: f64,
    /// Within-binary surprise-event threshold μ + [`SURPRISE_SIGMA`]·σ.
    event_thr: f64,
}

/// Compute S_glob (mean surprise), S_spat (Moran's I of the residual), and the event threshold. The
/// aggregation matches the `consistency` binary's `spatial_and_global`; we recompute it here because
/// that logic is private to its binary and we are a clean consumer of the public `CavityStat`.
fn cavity_stats(cav: &[(u64, CavityStat)]) -> CavityAgg {
    if cav.is_empty() {
        return CavityAgg { mean_surprise: 0.0, moran: 0.0, event_thr: f64::INFINITY };
    }
    let surprises: Vec<f64> = cav.iter().map(|(_, c)| c.surprise).collect();
    let resid: Vec<f64> = cav.iter().map(|(_, c)| c.residual).collect();
    let mean_surprise = mean(&surprises);
    let var = surprises.iter().map(|s| (s - mean_surprise).powi(2)).sum::<f64>() / surprises.len() as f64;
    let event_thr = mean_surprise + SURPRISE_SIGMA * var.sqrt();
    CavityAgg { mean_surprise, moran: morans_i_line(&resid), event_thr }
}

/// Localized distrust: contiguous runs (length ≥ [`MIN_RUN`]) of within-binary surprise events. Each
/// run is a region a frontend renders as "don't trust these addresses — the decode surprise clustered
/// here, so the calibration is likely stale." `cav` is address-sorted, so index order *is* the
/// spatial axis.
fn distrust_regions(cav: &[(u64, CavityStat)], stats: &CavityAgg) -> Vec<DistrustRegion> {
    let mut out = Vec::new();
    let n = cav.len();
    let mut i = 0;
    while i < n {
        if cav[i].1.surprise > stats.event_thr {
            let mut j = i;
            while j + 1 < n && cav[j + 1].1.surprise > stats.event_thr {
                j += 1;
            }
            let run = j - i + 1;
            if run >= MIN_RUN {
                let seg = &cav[i..=j];
                let s = mean(&seg.iter().map(|(_, c)| c.surprise).collect::<Vec<_>>());
                out.push(DistrustRegion {
                    addr_lo: cav[i].0,
                    addr_hi: cav[j].0,
                    reason: "clustered decode surprise — calibration likely stale in this window".into(),
                    surprise: s,
                });
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    out
}

/// Confidence in the regime classification. An override is asserted (1.0); a confidently-benign or a
/// recognized non-benign signature is high; a benign fallback flagged uncertain is low (we chose the
/// safe default but the signature wasn't clean).
fn regime_confidence(source: &str, selected: Regime, uncertain: bool) -> f64 {
    if source == "override" {
        1.0
    } else if uncertain {
        0.4
    } else if selected == Regime::Benign {
        0.9
    } else {
        0.85
    }
}

/// Assemble the [`CalibrationInfo`] self-calibration record from the classifier's decision and which
/// halves of the bank config were applied (engine always; isotonic map only when a bank is loaded).
fn build_calibration(
    selected: Regime,
    source: &str,
    engine: [f64; 2],
    isotonic_applied: bool,
    uncertain: bool,
) -> CalibrationInfo {
    // The engine half is always in the name; the map half is appended only when a fitted map applied.
    let engine_tag = match selected {
        Regime::Benign => "benign:identity",
        Regime::Packed => "packed:entropy-prior",
        Regime::Obfuscated => "obfuscated:chainfwd-prior",
    };
    let map_applied = if isotonic_applied {
        format!("{engine_tag} + isotonic-{}", selected.tag())
    } else {
        engine_tag.to_string()
    };
    let classifier = if source == "override" { "override" } else { "signature-rule" };
    let map_note = if isotonic_applied {
        " + the bank's fitted isotonic map was applied to the marginals (the map-switch, not just the \
         engine-switch)."
    } else {
        " (engine config only — no bank loaded, or the selected regime's bank map is identity)."
    };
    let note = if source == "override" {
        format!("regime overridden via --facts; applied the {} bank config{map_note}", selected.tag())
    } else if uncertain {
        format!(
            "benign fallback, but residual drift the signature rule couldn't place — regime uncertain, \
             calibration may be stale (the semantic/virtualized-obfuscation blind spot, where the \
             surprise statistic is blind). Not routed to a non-benign map on an unrecognized signature.{map_note}"
        )
    } else {
        format!(
            "signature classifier (trained threshold rule, S_glob>{RULE_GLOB_HI:.3}⇒obfuscated, \
             S_spat>{RULE_SPAT_HI:.4}⇒packed) selected {}.{map_note}",
            selected.tag()
        )
    };
    CalibrationInfo {
        selected_regime: selected.tag().into(),
        classifier: classifier.into(),
        map_applied,
        engine,
        isotonic_applied,
        regime_uncertain: uncertain,
        note,
    }
}

/// Active-mode `query_gain`: rank unconfirmed candidate heads by the uncertainty their confirmation
/// would resolve. The heuristic EIG is `H₂(F_h) · |body(h)|` — a head that is both *uncertain* (high
/// binary entropy) and *consequential* (confirming it lights a large body) is worth a human's next
/// look. Entry and confidently-confirmed heads carry ~0 gain and drop out. Returns the top `k`.
fn query_gain(sc: &probcfg::SoftConfirm, entry: u64, data_clamp: &HashSet<u64>, k: usize) -> Vec<Suggestion> {
    let mut scored: Vec<(u64, f64, usize)> = sc
        .heads
        .iter()
        .filter(|&&h| h != entry && !data_clamp.contains(&h))
        .map(|&h| {
            let f = sc.f.get(&h).copied().unwrap_or(0.0);
            let body = sc.bodies.get(&h).map(|b| b.len()).unwrap_or(0);
            (h, binary_entropy(f) * body as f64, body)
        })
        .filter(|&(_, gain, _)| gain > 1e-6)
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
        .into_iter()
        .take(k)
        .map(|(addr, gain, body)| Suggestion {
            addr,
            expected_info_gain: gain,
            why: format!(
                "uncertain head (F={:.2}) covering {} instructions — confirming it resolves the most",
                sc.f.get(&addr).copied().unwrap_or(0.0),
                body
            ),
        })
        .collect()
}

fn instruction_summary(post: &[(u64, f64)], full: bool) -> InstructionSummary {
    let n = post.len();
    let mean_pi = mean(&post.iter().map(|(_, p)| *p).collect::<Vec<_>>());
    let mut low: Vec<u64> = post.iter().filter(|(_, p)| *p < LOW_PI).map(|(a, _)| *a).collect();
    let low_count = low.len();
    low.truncate(LOW_CONF_CAP);
    InstructionSummary {
        n,
        mean_pi,
        low_confidence: low,
        low_confidence_count: low_count,
        per_insn: full.then(|| post.iter().map(|&(addr, pi)| InsnPosterior { addr, pi }).collect()),
    }
}

/// Binary (Shannon) entropy of a Bernoulli(p), in bits. Peaks at 1.0 for p = 0.5, 0 at the extremes.
fn binary_entropy(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    -(p * p.log2() + (1.0 - p) * (1.0 - p).log2())
}

/// Moran's I with a 1-D contiguity weight (neighbors = adjacent in the address-ordered vector).
/// Recomputed here (the `consistency` binary's copy is private) so `S_spat` matches the paper.
fn morans_i_line(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 3 {
        return 0.0;
    }
    let mbar = mean(x);
    let denom: f64 = x.iter().map(|v| (v - mbar).powi(2)).sum();
    if denom <= 0.0 {
        return 0.0;
    }
    let mut num = 0.0;
    for i in 0..n - 1 {
        num += (x[i] - mbar) * (x[i + 1] - mbar);
    }
    (n as f64 / (n as f64 - 1.0)) * (num / denom)
}

fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        0.0
    } else {
        x.iter().sum::<f64>() / x.len() as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// Round-trip test — the contract must survive serialize → deserialize (acceptance §5).
// ═══════════════════════════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> AnalysisResult {
        AnalysisResult {
            binary: BinaryInfo { path: "b".into(), format: "elf".into(), arch: "x86_64".into(), entry: 0x1000, n_bytes: 42 },
            regime: RegimeInfo { detected: "benign".into(), source: "auto".into(), confidence: 0.6 },
            functions: vec![FunctionInfo { addr: 0x1000, name: Some("main".into()), confidence: 1.0, reached_prob: 1.0, in_body_of: None, flagged_decoy: false }],
            edges: vec![EdgeInfo { from: 0x1000, to: 0x1020, kind: "direct".into(), confidence: 0.9 }],
            instructions: InstructionSummary { n: 10, mean_pi: 0.7, low_confidence: vec![0x1005], low_confidence_count: 1, per_insn: None },
            trust: TrustReport { overall: "trustworthy".into(), s_glob: 0.3, s_spat: 0.01, distrust_regions: vec![] },
            suggestions: vec![Suggestion { addr: 0x1020, expected_info_gain: 3.4, why: "w".into() }],
            calibration: CalibrationInfo {
                selected_regime: "benign".into(),
                classifier: "signature-rule".into(),
                map_applied: "benign:identity".into(),
                engine: [0.0, 0.0],
                isotonic_applied: false,
                regime_uncertain: false,
                note: "n".into(),
            },
        }
    }

    #[test]
    fn analysis_result_round_trips() {
        let r = sample();
        let json = serde_json::to_string(&r).expect("serialize");
        let back: AnalysisResult = serde_json::from_str(&json).expect("deserialize");
        let json2 = serde_json::to_string(&back).expect("re-serialize");
        assert_eq!(json, json2, "AnalysisResult must survive a serialize→deserialize→serialize cycle");
    }

    #[test]
    fn known_facts_parses_partial_json() {
        // A facts file need only carry what the user knows; everything else defaults.
        let f: KnownFacts = serde_json::from_str(r#"{ "entry": 4096, "symbols": { "4128": "main" } }"#).unwrap();
        assert_eq!(f.entry, Some(4096));
        assert_eq!(f.symbols.get(&4128).map(String::as_str), Some("main"));
        assert!(f.trace.is_empty() && f.regime.is_none());
    }

    #[test]
    fn signature_rule_routes_the_three_regimes() {
        // The trained benign-default threshold rule (CONSISTENCY_SWITCHING_RESULTS, 1.00 sel acc).
        // Clean sits low on both ⇒ benign; elevated S_glob ⇒ obfuscated; high S_spat alone ⇒ packed.
        assert_eq!(classify_rule(0.8, 0.05), Regime::Benign);
        assert_eq!(classify_rule(5.0, 0.05), Regime::Obfuscated);
        assert_eq!(classify_rule(1.1, 0.30), Regime::Packed);
        // Benign default is protective: an ambiguous binary is never routed to a non-benign map.
        assert_eq!(classify_rule(RULE_GLOB_HI - 0.01, RULE_SPAT_HI - 0.001), Regime::Benign);
        // …but the mid-band drift is flagged uncertain rather than asserted clean.
        assert!(benign_is_uncertain(RULE_GLOB_NULL + 0.5, 0.02));
        assert!(!benign_is_uncertain(0.5, 0.02));
    }

    #[test]
    fn calibration_bank_round_trips_and_suppresses() {
        // A packed bank (degenerate all-data map) serializes, reloads, and drives posteriors → ≈0.
        let packed = evalkit::IsotonicMap::fit(
            &(0..=50).map(|i| (i as f64 / 50.0, 0.0)).collect::<Vec<_>>(),
        )
        .to_points();
        let bank = CalibrationBank { meta: BankMeta::default(), benign: vec![], packed, obfuscated: vec![] };
        let json = serde_json::to_string(&bank).unwrap();
        let back: CalibrationBank = serde_json::from_str(&json).unwrap();
        assert!(back.is_active(Regime::Packed), "packed map should be non-identity");
        assert!(!back.is_active(Regime::Benign), "benign map should be identity");
        let cal = back.map(Regime::Packed).apply_all(&[(0x10, 0.37), (0x20, 0.9)]);
        assert!(cal.iter().all(|&(_, p)| p < 1e-6), "packed map must suppress to ≈0: {cal:?}");
        // Benign identity leaves marginals untouched.
        let same = back.map(Regime::Benign).apply_all(&[(0x10, 0.37)]);
        assert_eq!(same, vec![(0x10, 0.37)]);
    }

    #[test]
    fn binary_entropy_peaks_at_half() {
        assert!((binary_entropy(0.5) - 1.0).abs() < 1e-9);
        assert!(binary_entropy(0.5) > binary_entropy(0.1));
        assert!(binary_entropy(0.99) < 0.1);
    }
}
