use probdisasm::{
    Analysis, AnalysisConfig, AnalysisMode, Superset, extract_all_hints, extract_hint_pairs,
    extract_text_section,
};

/// Path to a small, representative coreutils binary used as a test fixture.
/// Any x86-64 ELF will do; `cat` is small and representative.
const CAT_BINARY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../probablistic/corpus/x86_64-binaries/elf/coreutils/gcc_coreutils_64_O0_cat"
);

fn run_both_modes() -> (Vec<f64>, Vec<f64>) {
    let bytes = std::fs::read(CAT_BINARY).expect("test fixture not found — run from repo root");
    let (base, code) = extract_text_section(&bytes).expect("failed to extract .text");
    let superset = Superset::new(base, code).expect("failed to build superset");
    let hints = extract_all_hints(&superset);
    let pairs = extract_hint_pairs(&superset);

    let hard_posteriors = {
        let mut analysis = Analysis::new(&superset);
        analysis.run_with_config(
            &hints,
            &pairs,
            &AnalysisConfig {
                mode: AnalysisMode::Hard,
                msg_damp: 0.5,
                ft_eps: 0.5,
                evidence_scale: 1.0,
                transfer_log_weight: 4.0,
                unhinted_code_prob: 0.2,
                reaching_scale: 0.0,
                entropy_prior_strength: 0.0,
                entropy_floor_bits: 6.0,
                chainfwd_strength: 0.0,
            },
        );
        analysis
            .sorted_posteriors()
            .into_iter()
            .map(|(_, p)| p)
            .collect::<Vec<_>>()
    };

    let soft_posteriors = {
        let mut analysis = Analysis::new(&superset);
        analysis.run_with_config(
            &hints,
            &pairs,
            &AnalysisConfig {
                mode: AnalysisMode::Soft,
                msg_damp: 0.5,
                ft_eps: 0.5,
                evidence_scale: 1.0,
                transfer_log_weight: 4.0,
                unhinted_code_prob: 0.2,
                reaching_scale: 0.0,
                entropy_prior_strength: 0.0,
                entropy_floor_bits: 6.0,
                chainfwd_strength: 0.0,
            },
        );
        analysis
            .sorted_posteriors()
            .into_iter()
            .map(|(_, p)| p)
            .collect::<Vec<_>>()
    };

    (hard_posteriors, soft_posteriors)
}

/// Run Soft on the fixture at a given `chainfwd_strength`, returning the full posterior vector
/// (address-sorted). Everything else is the stock Soft config, so the only lever is the new prior.
fn soft_chainfwd(chainfwd_strength: f64) -> Vec<(u64, f64)> {
    let bytes = std::fs::read(CAT_BINARY).expect("test fixture not found — run from repo root");
    let (base, code) = extract_text_section(&bytes).expect("failed to extract .text");
    let superset = Superset::new(base, code).expect("failed to build superset");
    let hints = extract_all_hints(&superset);
    let pairs = extract_hint_pairs(&superset);
    let mut analysis = Analysis::new(&superset);
    analysis.run_with_config(
        &hints,
        &pairs,
        &AnalysisConfig {
            mode: AnalysisMode::Soft,
            chainfwd_strength,
            ..AnalysisConfig::default()
        },
    );
    analysis.sorted_posteriors()
}

/// THE safety guard. `chainfwd_strength = 0` must leave π identical to the pre-knob engine — the
/// whole opt-in story rests on it, because every paper on the shared engine keeps running the
/// strength-0 default. One wrinkle I had to confront honestly: Soft BP is **not** bit-for-bit
/// reproducible run-to-run. Two identical baseline runs already disagree on ~500/31k addresses by up
/// to ~1e-20 — the loopy message sums reassociate under `HashMap` iteration order (per-instance
/// `RandomState`). Literal f64-bit equality across two separate `Analysis` runs is thus impossible
/// for ANY config, chainfwd or not, and forcing determinism (BTreeMap) would reorder the sums and
/// change the frozen default — the opposite of additive. So the correct, honest guarantee is: turning
/// chainfwd on at strength 0 must add **no divergence beyond the engine's own run-to-run noise**. We
/// measure that intrinsic envelope with two baseline runs, then require the chainfwd=0 run to sit
/// inside it. A real leak of the chainfwd path into the strength-0 case moves affected posteriors by
/// O(0.01+) — orders of magnitude outside this ~1e-20 floor — and trips the assert immediately.
#[test]
fn chainfwd_zero_within_bp_noise_envelope() {
    let stock = || {
        let bytes = std::fs::read(CAT_BINARY).expect("test fixture not found — run from repo root");
        let (base, code) = extract_text_section(&bytes).expect("failed to extract .text");
        let superset = Superset::new(base, code).expect("failed to build superset");
        let hints = extract_all_hints(&superset);
        let pairs = extract_hint_pairs(&superset);
        let mut analysis = Analysis::new(&superset);
        analysis.run_with_config(
            &hints,
            &pairs,
            &AnalysisConfig { mode: AnalysisMode::Soft, ..AnalysisConfig::default() },
        );
        analysis.sorted_posteriors()
    };
    // Intrinsic BP nondeterminism: two identical baseline runs.
    let b1 = stock();
    let b2 = stock();
    // chainfwd wired in but gated off.
    let z = soft_chainfwd(0.0);

    assert_eq!(b1.len(), z.len(), "posterior vector length changed");
    let max_diff = |x: &[(u64, f64)], y: &[(u64, f64)]| {
        x.iter().zip(y).map(|(a, b)| (a.1 - b.1).abs()).fold(0.0_f64, f64::max)
    };
    let noise = max_diff(&b1, &b2);
    let chainfwd0 = max_diff(&b1, &z);
    println!(
        "BP intrinsic noise (baseline vs baseline): {noise:e}   chainfwd=0 vs baseline: {chainfwd0:e}"
    );
    // Addresses must still line up exactly (ordering is stable; only values jitter).
    for (i, (a, b)) in b1.iter().zip(&z).enumerate() {
        assert_eq!(a.0, b.0, "address mismatch at index {i}");
    }
    // chainfwd=0 must not exceed the intrinsic noise floor by more than a hair. 1e-12 is ~8 orders
    // below any real chainfwd effect and ~8 orders above the measured ~1e-20 jitter, so it separates
    // "inert" from "leaked" cleanly regardless of which HashMap seeds this run happened to draw.
    assert!(
        chainfwd0 <= noise.max(1e-12),
        "chainfwd=0 diverged from baseline by {chainfwd0:e}, beyond BP noise {noise:e} — the \
         strength-0 path is not inert"
    );
}

/// Wiring proof: a positive `chainfwd_strength` must actually move π (else the knob is dead and the
/// sweep would be measuring nothing). We only assert *some* address changed — direction/magnitude is
/// what the offline sweep + RESULTS.md quantify, not a unit test.
#[test]
fn chainfwd_positive_changes_posteriors() {
    let off = soft_chainfwd(0.0);
    let on = soft_chainfwd(1.0);
    assert_eq!(off.len(), on.len());
    let changed = off.iter().zip(&on).filter(|(a, b)| a.1.to_bits() != b.1.to_bits()).count();
    assert!(
        changed > 0,
        "chainfwd_strength=1.0 left every posterior unchanged — the prior is not wired in"
    );
    println!("chainfwd=1.0 moved {changed}/{} posteriors vs off", off.len());
}

/// Hard mode (Miller Algorithm 1) should produce a bimodal posterior distribution —
/// most values collapse near 0.0 or 1.0. Very few mid-range posteriors.
#[test]
fn hard_mode_is_bimodal() {
    let (hard, _) = run_both_modes();
    let n = hard.len() as f64;
    let mid_range = hard.iter().filter(|&&p| p > 0.1 && p < 0.9).count() as f64;
    let mid_fraction = mid_range / n;

    // Hard mode should have fewer than 10% of posteriors in (0.1, 0.9).
    // Step II + undamped Step III forces convergence toward {0, 1}.
    assert!(
        mid_fraction < 0.10,
        "Hard mode: expected < 10% mid-range posteriors, got {:.1}% ({} of {})",
        mid_fraction * 100.0,
        mid_range as usize,
        n as usize,
    );
}

/// Soft mode (proper loopy sum-product BP) should also produce a bimodal posterior
/// distribution. Hard overlap constraints + ft coupling forces most addresses to
/// near-{0,1}. Expect < 2% mid-range.
#[test]
fn soft_mode_is_bimodal() {
    let (_, soft) = run_both_modes();
    let n = soft.len() as f64;
    let mid_range = soft.iter().filter(|&&p| p > 0.1 && p < 0.9).count() as f64;
    let mid_fraction = mid_range / n;

    let near_zero = soft.iter().filter(|&&p| p < 0.1).count();
    let near_one = soft.iter().filter(|&&p| p > 0.9).count();
    println!(
        "Soft BP bimodal fraction: {:.1}% mid-range ({} of {})  P<0.1={}  P>0.9={}",
        mid_fraction * 100.0,
        mid_range as usize,
        n as usize,
        near_zero,
        near_one,
    );

    assert!(
        mid_fraction < 0.02,
        "Soft mode: expected < 2% mid-range posteriors, got {:.1}% ({} of {})",
        mid_fraction * 100.0,
        mid_range as usize,
        n as usize,
    );
}

/// Proper BP (Soft) resolves more ambiguity than the heuristic Hard propagation:
/// Soft should have strictly fewer mid-range posteriors than Hard.
/// Hard mode's bounded local propagation leaves more addresses in uncertain states.
#[test]
fn soft_has_fewer_mid_range_than_hard() {
    let (hard, soft) = run_both_modes();

    let hard_mid = hard.iter().filter(|&&p| p > 0.1 && p < 0.9).count();
    let soft_mid = soft.iter().filter(|&&p| p > 0.1 && p < 0.9).count();

    assert!(
        hard_mid > soft_mid,
        "Soft BP should have fewer mid-range posteriors than Hard (it resolves more ambiguity). \
         Hard: {hard_mid}, Soft: {soft_mid}"
    );

    println!(
        "Mid-range posteriors — Hard: {hard_mid}, Soft: {soft_mid} \
         (ratio hard/soft: {:.1}x, total={})",
        hard_mid as f64 / soft_mid.max(1) as f64,
        hard.len()
    );
}

/// Proper BP (Soft) resolves more uncertainty globally: total posterior entropy
/// should be lower for Soft than for Hard. Hard mode's heuristic propagation
/// leaves more addresses uncertain → higher entropy.
#[test]
fn hard_has_higher_entropy_than_soft() {
    let (hard, soft) = run_both_modes();

    let entropy = |posteriors: &[f64]| -> f64 {
        posteriors
            .iter()
            .map(|&p| {
                let p = p.clamp(1e-10, 1.0 - 1e-10);
                -p * p.ln() - (1.0 - p) * (1.0 - p).ln()
            })
            .sum::<f64>()
    };

    let hard_entropy = entropy(&hard);
    let soft_entropy = entropy(&soft);

    assert!(
        hard_entropy > soft_entropy,
        "Hard mode should have higher total entropy than Soft BP (Hard leaves more uncertainty). \
         Hard: {hard_entropy:.1} bits, Soft: {soft_entropy:.1} bits"
    );

    println!(
        "Total entropy — Hard: {hard_entropy:.1} bits, Soft: {soft_entropy:.1} bits \
         (ratio hard/soft: {:.2}x)",
        hard_entropy / soft_entropy.max(1e-10)
    );
}

/// The cavity hook is a read-only post-hoc pass: it must not perturb π, and the cavity belief
/// it extracts must be *exactly* the posterior with the local factor φ_a removed. We prove both
/// at once via the definitional identity
///     π_a = σ( logit(q_a) + (φ_a[1] − φ_a[0]) ),
/// since π_a ∝ φ_a · Π(incoming) and q_a ∝ Π(incoming). If reading the cavity had disturbed the
/// converged messages, or if q_a were anything other than "π minus φ", this identity would break.
#[test]
fn cavity_is_pi_with_local_factor_removed() {
    let bytes = std::fs::read(CAT_BINARY).expect("test fixture not found — run from repo root");
    let (base, code) = extract_text_section(&bytes).expect("failed to extract .text");
    let superset = Superset::new(base, code).expect("failed to build superset");
    let hints = extract_all_hints(&superset);
    let pairs = extract_hint_pairs(&superset);

    let mut analysis = Analysis::new(&superset);
    analysis.run_with_config(
        &hints,
        &pairs,
        &AnalysisConfig {
            mode: AnalysisMode::Soft,
            ..AnalysisConfig::default()
        },
    );

    let posteriors: std::collections::HashMap<u64, f64> =
        analysis.sorted_posteriors().into_iter().collect();
    let cavity = analysis.sorted_cavity();
    assert!(!cavity.is_empty(), "Soft run must populate cavity stats");

    let sigmoid = |x: f64| if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e = x.exp(); e / (1.0 + e) };
    let mut checked = 0usize;
    for (addr, c) in &cavity {
        let pi = posteriors[addr];
        // Reconstruct π from the cavity + the local LLR. Skipping addresses where the cavity is
        // pinned to a boundary (logit ±∞): there the recomposed value is exact by construction but
        // the finite-logit arithmetic below is uninformative.
        let cav_logit = (c.cavity_code_prob / (1.0 - c.cavity_code_prob)).ln();
        if !cav_logit.is_finite() {
            continue;
        }
        let recomposed = sigmoid(cav_logit + c.llr_local);
        assert!(
            (recomposed - pi).abs() < 1e-6,
            "cavity+φ must reconstruct π at {addr:#x}: got {recomposed:.9}, π={pi:.9}"
        );
        assert!((0.0..=1.0).contains(&c.cavity_code_prob));
        assert!((0.0..=1.0).contains(&c.local_code_prob));
        assert!(c.surprise >= 0.0 && c.surprise.is_finite(), "surprise must be finite ≥ 0");
        assert!(c.nis >= 0.0 && c.nis.is_finite());
        checked += 1;
    }
    assert!(checked > 100, "expected many finite-cavity addresses, checked {checked}");
    println!("cavity identity held on {checked} addresses (of {} total)", cavity.len());
}
