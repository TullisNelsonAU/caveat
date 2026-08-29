//! Signature + gate pass over a flat directory of binaries — no calibration bank, no ground truth.
//!
//! This is the GT-free half of the small-packed probe (paper Limitation 3): for every binary in a
//! directory it reports the candidate count `n`, the analysed code size, the region entropy, the
//! benign-engine signature `(S_glob, S_spat)`, and what the shipped `classify_rule` /
//! `classify_guard` decide under both the published flat spatial gate and the floored size-aware
//! gate `T(n) = max(FLAT, mu + z*c/sqrt(n))` from the spatial-null repair.
//!
//! It exists so the *unpacked* baselines (which have no packer data window, hence no packed GT and
//! no ECE) can be measured with exactly the engine call and exactly the gates that
//! `tigress_reconcile` and `switching` use — `run_soft_with_cavity_cfg(base, code, 0.0, 0.0, false)`
//! followed by `global_and_spatial`. Packed binaries get their ECE from `switching`; this binary is
//! about the signature and the gate only.
//!
//! Usage: small_signature <dir> <out.tsv> [label]

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use consistency::{global_and_spatial, region_entropy, SignatureClassifier};
use evalkit::run_soft_with_cavity_cfg;
use probdisasm::extract_text_section as extract_text;

/// The published operating point, as recorded by every run of record in this project.
const GLOB_HI: f64 = 2.514702;
const SPAT_HI: f64 = 0.105178;
const PACK_ENT_LO: f64 = 7.1688;

/// The floored size-aware gate from the spatial-null repair (Sec. 6.10).
const MU: f64 = 0.069231;
const C: f64 = 4.034322;
const Z95: f64 = 1.645;

fn t_floored(n: f64) -> f64 {
    (MU + Z95 * C / n.sqrt()).max(SPAT_HI)
}

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

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir: PathBuf = args.next().context("usage: small_signature <dir> <out.tsv> [label]")?.into();
    let out: PathBuf = args.next().context("missing <out.tsv>")?.into();
    let label = args.next().unwrap_or_else(|| "-".into());

    let mut paths: Vec<PathBuf> = fs::read_dir(&dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().map_or(true, |x| x != "upxgt"))
        .collect();
    paths.sort();

    let mut f = fs::File::create(&out).with_context(|| format!("creating {}", out.display()))?;
    writeln!(
        f,
        "name\tlabel\tn\tcode_bytes\tregion_ent\ts_glob\ts_spat\tt_floored\t\
         fire_flat\tfire_floored\trule_pick\tguard_pick\trule_pick_floored\tguard_pick_floored"
    )?;

    for path in &paths {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let (base, code) =
            extract_text(&bytes).with_context(|| format!("extracting .text from {}", path.display()))?;

        // The benign engine, exactly as `SignatureClassifier::train` invokes it.
        let (_post, cav) = run_soft_with_cavity_cfg(base, code, 0.0, 0.0, false)
            .with_context(|| format!("benign engine on {name}"))?;
        let s = global_and_spatial(&cav);
        let n = cav.len() as f64;
        let ent = region_entropy(code);
        let (sg, ss) = (s.mean_surprise, s.moran);

        let tfl = t_floored(n);
        let flat = gate(GLOB_HI, SPAT_HI);
        let floored = gate(GLOB_HI, tfl);

        writeln!(
            f,
            "{name}\t{label}\t{:.0}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{}\t{}\t{}",
            n,
            code.len(),
            ent,
            sg,
            ss,
            tfl,
            ss > SPAT_HI,
            ss > tfl,
            flat.classify_rule(sg, ss).tag(),
            flat.classify_guard(sg, ss, ent).tag(),
            floored.classify_rule(sg, ss).tag(),
            floored.classify_guard(sg, ss, ent).tag()
        )?;
        eprintln!(
            "  {name:<28} n={n:>6.0} S_glob={sg:.4} S_spat={ss:+.4} T(n)={tfl:.4} \
             H={ent:.3} flat={} floored={}",
            if ss > SPAT_HI { "FIRE" } else { "quiet" },
            if ss > tfl { "FIRE" } else { "quiet" }
        );
    }
    eprintln!("wrote {} ({} binaries)", out.display(), paths.len());
    Ok(())
}
