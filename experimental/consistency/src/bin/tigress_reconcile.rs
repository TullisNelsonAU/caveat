//! Re-emit the Tigress arm with each decision and the inputs it was made from in the **same row**.
//!
//! Two committed CSVs describe the same 27 Tigress binaries and disagree about routing:
//! `consistency_credibility/tigress_graded.csv` (per-binary `s_spat_moran`) and
//! `downstream_decision/boundaries_meta.csv` (per-binary `rule_pick`). This binary settles the
//! disagreement by measurement rather than by argument: it rebuilds the signature for every Tigress
//! binary in **one pass, one engine, one config**, and writes the statistic and the decision side by
//! side so the pair can never drift apart again.
//!
//! Method notes:
//!
//! * The engine call is the same one `SignatureClassifier::train` uses for its benign-engine
//!   signatures — `run_soft_with_cavity_cfg(base, code, 0.0, 0.0, false)` followed by
//!   `global_and_spatial` — so what is measured here is what the shipped rule reads.
//! * Routing is taken by the shipped `classify_rule` / `classify_guard`. Nothing is reimplemented.
//! * Per row it asserts the implication that is currently in doubt: `rule_pick == packed` requires
//!   `S_spat > spat_hi` **or** `S_glob > glob_hi`. A violation aborts, because a packed pick with
//!   both statistics quiet is precisely the condition the reconciliation is chasing.
//! * The manifest records the engine commit and the `AnalysisConfig` the signatures were produced
//!   under, so a future reader can tell at a glance whether a stale file is being compared.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use consistency::{global_and_spatial, region_entropy, Regime, SignatureClassifier};
use evalkit::run_soft_with_cavity_cfg;
use probdisasm::extract_text_section as extract_text;

/// The published operating point, as recorded by every run of record in this project.
const GLOB_HI: f64 = 2.514702;
const SPAT_HI: f64 = 0.105178;
const PACK_ENT_LO: f64 = 7.1688;

/// The floored size-aware gate from the spatial-null repair, reported alongside the flat gate.
const MU: f64 = 0.069231;
const C: f64 = 4.034322;
const Z95: f64 = 1.645;

fn t_floored(n: f64) -> f64 {
    (MU + Z95 * C / n.sqrt()).max(SPAT_HI)
}

/// The three graded transforms, weakest to strongest by the de-risk drift ordering.
const LEVELS: [(&str, &str); 3] =
    [("tigL", "Virtualize"), ("tigM", "EncodeArithmetic"), ("tigH", "Flatten")];

fn gate(glob_hi: f64, spat_hi: f64) -> SignatureClassifier {
    SignatureClassifier {
        centroids: [(0.0, 0.0); 3],
        present: [false; 3],
        mu_g: 0.0,
        sd_g: 1.0,
        mu_s: 0.0,
        sd_s: 1.0,
        raw_centroids: [(0.0, 0.0); 3],
        glob_hi,
        spat_hi,
        pack_ent_lo: PACK_ENT_LO,
    }
}

/// `git -C <repo> rev-parse HEAD`, plus a dirty marker — the provenance the stale file lacked.
fn git_describe(repo: &Path) -> String {
    let head = Command::new("git")
        .args(["-C", &repo.display().to_string(), "rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    let dirty = Command::new("git")
        .args(["-C", &repo.display().to_string(), "status", "--porcelain"])
        .output()
        .ok()
        .map(|o| !String::from_utf8_lossy(&o.stdout).trim().is_empty())
        .unwrap_or(false);
    if dirty { format!("{head}-dirty") } else { head }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let tig_root: PathBuf = args
        .next()
        .context("usage: tigress_reconcile <tig_graded_root> <out.tsv>")?
        .into();
    let out: PathBuf = args.next().context("missing <out.tsv>")?.into();

    let mut f = fs::File::create(&out).with_context(|| format!("creating {}", out.display()))?;
    writeln!(
        f,
        "name\ttransform\ttransform_name\tn\tcode_bytes\tregion_ent\ts_glob\ts_spat\t\
         t_floored\trule_pick\tguard_pick\trule_pick_floored\tguard_pick_floored"
    )?;

    eprintln!("── re-running the Tigress arm: one pass, one engine, one config ──");
    eprintln!("  gates: glob_hi={GLOB_HI} spat_hi={SPAT_HI} pack_ent_lo={PACK_ENT_LO}");

    let mut total = 0usize;
    for (lvl, xform) in LEVELS {
        let bins = tig_root.join(lvl).join("bins");
        if !bins.is_dir() {
            bail!("missing Tigress binaries for {lvl} at {} — rebuild with build_tigress_graded.sh", bins.display());
        }
        let mut names: Vec<PathBuf> =
            fs::read_dir(&bins)?.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        names.sort();
        for path in names {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
            let (base, code) = extract_text(&bytes)
                .with_context(|| format!("extracting .text from {}", path.display()))?;

            // The benign engine, exactly as `SignatureClassifier::train` invokes it.
            let (_post, cav) = run_soft_with_cavity_cfg(base, code, 0.0, 0.0, false)
                .with_context(|| format!("benign engine on {name}"))?;
            let s = global_and_spatial(&cav);
            let n = cav.len() as f64;
            let ent = region_entropy(code);
            let (sg, ss) = (s.mean_surprise, s.moran);

            let flat = gate(GLOB_HI, SPAT_HI);
            let tfl = t_floored(n);
            let floored = gate(GLOB_HI, tfl);

            let rule = flat.classify_rule(sg, ss);
            let guard = flat.classify_guard(sg, ss, ent);
            let rule_f = floored.classify_rule(sg, ss);
            let guard_f = floored.classify_guard(sg, ss, ent);

            // The implication currently in doubt: a packed pick must be carried by a statistic.
            for (tag, pick, hi) in [("flat", rule, SPAT_HI), ("floored", rule_f, tfl)] {
                if pick == Regime::Packed && !(ss > hi || sg > GLOB_HI) {
                    bail!(
                        "IMPLICATION VIOLATED ({tag}) on {lvl}/{name}: rule_pick=packed but \
                         S_spat={ss} <= spat_hi={hi} and S_glob={sg} <= glob_hi={GLOB_HI}"
                    );
                }
            }

            writeln!(
                f,
                "{name}\t{lvl}\t{xform}\t{:.0}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{}",
                n,
                code.len(),
                ent,
                sg,
                ss,
                tfl,
                rule.tag(),
                guard.tag(),
                rule_f.tag(),
                guard_f.tag()
            )?;
            eprintln!(
                "  {lvl}/{name:<18} n={n:>6.0} S_glob={sg:.4} S_spat={ss:+.4} -> rule={} guard={}",
                rule.tag(),
                guard.tag()
            );
            total += 1;
        }
    }
    if total != 27 {
        bail!("expected 27 Tigress binaries, analysed {total}");
    }

    // ── manifest: the provenance the disagreement turned on ───────────────────
    let manifest = out.with_extension("manifest.tsv");
    let mut m = fs::File::create(&manifest)?;
    let home = std::env::var("HOME").unwrap_or_default();
    writeln!(m, "key\tvalue")?;
    writeln!(m, "engine_probdisasm\t{}", git_describe(Path::new(&format!("{home}/lab/projects/probdisasm"))))?;
    writeln!(m, "harness_upd_suite_regime\t{}", git_describe(Path::new(&format!("{home}/lab/projects/upd-suite-regime"))))?;
    writeln!(m, "analysis_mode\tSoft")?;
    writeln!(m, "entropy_prior_strength\t0")?;
    writeln!(m, "chainfwd_strength\t0")?;
    writeln!(m, "use_dassa\tfalse")?;
    writeln!(m, "engine_call\trun_soft_with_cavity_cfg(base, code, 0.0, 0.0, false)")?;
    writeln!(m, "glob_hi\t{GLOB_HI}")?;
    writeln!(m, "spat_hi\t{SPAT_HI}")?;
    writeln!(m, "pack_ent_lo\t{PACK_ENT_LO}")?;
    writeln!(m, "n_binaries\t{total}")?;
    writeln!(m, "tigress_seed\t20260707")?;
    eprintln!("wrote {} and {}", out.display(), manifest.display());
    Ok(())
}
