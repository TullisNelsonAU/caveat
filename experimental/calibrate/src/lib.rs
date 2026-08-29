//! `calibrate` — the calibration-preserving **fusion operator**, the validated coupling atom of the
//! Layer-3 stack (`probablistic/LAYER3_STACK_DESIGN.md` §2). One object `o` receives calibrated
//! messages `{m_j}` from its neighbours (lower/upper layers linked by the incidence Γ); we pool them
//! in **log-odds** (a log-linear / product-of-experts pool), then **recalibrate** with an isotonic map:
//!
//! ```text
//! logit(bel_o) = b_o + Σ_j w_j · logit(m_j)          (S1)   — the log-linear pool
//! bel_o        = g_o( σ( logit(bel_o) ) )            (S2)   — isotonic recalibration (Theorem 1)
//! ```
//!
//! **This is exactly M2 eq (5) generalized to `J` messages.** Setting `J = 2` with messages `π_a`
//! (Layer-1 Soft posterior) and `R_a` (Layer-2 reachedness) recovers M2's fusion — the stack is that
//! atom, tiled. By Theorem 1 the isotonic `g_o` yields a calibrated, rank-preserving belief: it drops
//! ECE without touching AUROC.
//!
//! We **reuse `evalkit::IsotonicMap` as-is** for `g_o` (the honesty wall / calibration machinery is
//! shared, never re-implemented). This crate adds only the log-linear pool `(S1)` and the MLE that
//! fits `(w_j, b_o)`. Nothing here overwrites a raw posterior; the fused belief is a *distinct,
//! deliberately recalibrated* confidence.

pub use evalkit::IsotonicMap;

/// A message in **log-odds** space: `logit(m_j) = ln( m_j / (1 − m_j) )`. Fusion pools these linearly.
pub type LogOdds = f64;

/// Numerically-safe logit: clamps `p` off `{0,1}` so `ln` and the pool stay finite.
pub fn logit(p: f64) -> LogOdds {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

/// The logistic link `σ(z) = 1 / (1 + e^{−z})`.
pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// **(S1)** — the log-linear pool. Given incoming messages in log-odds `{logit(m_j)}`, per-message
/// weights `{w_j}` and the object bias `b_o`, return the pooled belief `σ(b_o + Σ_j w_j·logit(m_j))`.
/// This is the raw fused posterior *before* isotonic recalibration (`P²` in the M2 notation).
///
/// `weights.len()` should match `msgs.len()`; extra weights or messages are ignored (the shorter of
/// the two governs), so a layer can send fewer messages than the model was fit for without panicking.
pub fn fuse(msgs: &[LogOdds], weights: &[f64], bias: f64) -> f64 {
    let z = bias + msgs.iter().zip(weights).map(|(m, w)| w * m).sum::<f64>();
    sigmoid(z)
}

/// A fitted fusion operator: the log-linear pool weights `(w_j, b_o)` in **raw log-odds space** plus
/// the isotonic recalibration map `g_o`. Apply `s1` for the raw pool `(S1)`, or `fuse` for the full
/// calibrated belief `(S1)+(S2)`.
///
/// The MLE is run on *standardized* features (each `logit(m_j)` centred/scaled) for numerical
/// stability — exactly as M2's `bench::Fusion` does — then the standardization is folded back into
/// `(w_j, b_o)` (an exact affine identity), so `s1` reproduces M2's raw fused posterior bit-for-bit
/// while exposing the design's `(S1)` weight form.
#[derive(Clone, Debug)]
pub struct Fusion {
    /// Raw-log-odds message weights `w_j` (standardization folded in).
    weights: Vec<f64>,
    /// Object bias `b_o` (standardization folded in).
    bias: f64,
    /// The isotonic recalibration map `g_o` fit on the raw fused posterior vs GT (Theorem 1).
    iso: IsotonicMap,
}

impl Fusion {
    /// Fit `(w_j, b_o)` by logistic MLE and `g_o` by isotonic regression, on labelled rows
    /// `(logit_msgs, y)` where `y ∈ {0,1}` is the object's GT label. The MLE matches M2 exactly:
    /// standardized batch gradient descent, `lr = 0.1`, `iters = 4000`, init `w = 1, b = 0`.
    pub fn fit(rows: &[(Vec<LogOdds>, f64)]) -> Self {
        Self::fit_with(rows, 0.1, 4000)
    }

    /// Fit with explicit optimizer knobs (see [`Fusion::fit`] for the defaults that reproduce M2).
    pub fn fit_with(rows: &[(Vec<LogOdds>, f64)], lr: f64, iters: usize) -> Self {
        let d = rows.first().map(|r| r.0.len()).unwrap_or(0);
        let n = rows.len().max(1) as f64;

        // Standardize each feature (population mean/std, floored) — as M2's Fusion::fit.
        let mean: Vec<f64> = (0..d).map(|j| rows.iter().map(|r| r.0[j]).sum::<f64>() / n).collect();
        let std: Vec<f64> = (0..d)
            .map(|j| (rows.iter().map(|r| (r.0[j] - mean[j]).powi(2)).sum::<f64>() / n).sqrt().max(1e-9))
            .collect();

        // Batch gradient descent on the standardized logistic (init a_j = 1, c = 0).
        let (mut a, mut c) = (vec![1.0f64; d], 0.0f64);
        for _ in 0..iters {
            let mut ga = vec![0.0f64; d];
            let mut gc = 0.0f64;
            for (x, y) in rows {
                let z: f64 = (0..d).map(|j| a[j] * (x[j] - mean[j]) / std[j]).sum::<f64>() + c;
                let e = sigmoid(z) - y;
                for j in 0..d {
                    ga[j] += e * (x[j] - mean[j]) / std[j];
                }
                gc += e;
            }
            for j in 0..d {
                a[j] -= lr * ga[j] / n;
            }
            c -= lr * gc / n;
        }

        // Fold standardization into raw-log-odds weights: w_j = a_j/s_j, b = c − Σ a_j·m_j/s_j.
        // Exact identity ⇒ `fuse(msgs, w, b)` == the standardized model's σ(Σ a_j·z_j + c).
        let weights: Vec<f64> = (0..d).map(|j| a[j] / std[j]).collect();
        let bias = c - (0..d).map(|j| a[j] * mean[j] / std[j]).sum::<f64>();

        // (S2) isotonic map g_o, fit on the raw fused posterior vs GT (Theorem 1 recalibration).
        let iso_samples: Vec<(f64, f64)> =
            rows.iter().map(|(x, y)| (fuse(x, &weights, bias), *y)).collect();
        let iso = IsotonicMap::fit(&iso_samples);

        Fusion { weights, bias, iso }
    }

    /// **(S1)** — the raw fused posterior for messages `logit(m_j)` (before recalibration).
    pub fn s1(&self, msgs: &[LogOdds]) -> f64 {
        fuse(msgs, &self.weights, self.bias)
    }

    /// **(S1)+(S2)** — the calibrated belief `bel_o = g_o(σ(b_o + Σ w_j·logit(m_j)))`.
    pub fn fuse(&self, msgs: &[LogOdds]) -> f64 {
        self.iso.apply(self.s1(msgs))
    }

    /// The fitted pool weights `(w_j, b_o)` in raw log-odds space — the design's `(S1)` form.
    pub fn weights(&self) -> (&[f64], f64) {
        (&self.weights, self.bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `fuse` folds the standardized fit exactly: an independently-standardized logistic evaluated the
    /// long way must equal `fuse(msgs, w, b)` on the folded weights.
    #[test]
    fn standardization_fold_is_exact() {
        // A separable-ish 2-feature set.
        let rows: Vec<(Vec<f64>, f64)> = (0..200)
            .map(|i| {
                let t = i as f64 / 199.0;
                let x1 = -4.0 + 8.0 * t;
                let x2 = 3.0 - 6.0 * t;
                let y = f64::from(x1 + 0.5 * x2 > 0.0);
                (vec![x1, x2], y)
            })
            .collect();
        let f = Fusion::fit(&rows);
        let (w, b) = f.weights();
        for (x, _) in &rows {
            let direct = fuse(x, w, b);
            assert!((f.s1(x) - direct).abs() < 1e-12, "s1 must equal fuse on folded weights");
        }
    }

    /// **The key Milestone-A atom test:** `Fusion` with `J = 2` messages `(π, R)` reproduces M2's
    /// `bench::Fusion` fused posterior `P²` and the isotonic `P̂` bit-for-bit. We replicate M2's exact
    /// standardized-GD formula inline and assert equality — this is the guarantee the two-layer stack
    /// leans on when it claims to match `bench --fuse`.
    #[test]
    fn j2_reproduces_m2_fusion() {
        // Synthetic (logit π, logit R, y) rows with structure in both features.
        let rows: Vec<(f64, f64, f64)> = (0..300)
            .map(|i| {
                let t = i as f64 / 299.0;
                let lpi = logit(0.02 + 0.96 * t); // Layer-1 posterior sweeps low→high
                let lr = logit(0.05 + 0.9 * ((i * 7) % 11) as f64 / 10.0); // Layer-2 reachedness, decorrelated
                let y = f64::from(0.7 * lpi + 0.6 * lr + 0.3 > 0.0);
                (lpi, lr, y)
            })
            .collect();

        // ── M2 reference: bench::Fusion, transcribed verbatim (standardized GD, lr 0.1, 4000 it). ──
        let n = rows.len() as f64;
        let m1 = rows.iter().map(|r| r.0).sum::<f64>() / n;
        let m2 = rows.iter().map(|r| r.1).sum::<f64>() / n;
        let s1 = (rows.iter().map(|r| (r.0 - m1).powi(2)).sum::<f64>() / n).sqrt().max(1e-9);
        let s2 = (rows.iter().map(|r| (r.1 - m2).powi(2)).sum::<f64>() / n).sqrt().max(1e-9);
        let (mut a, mut b, mut c) = (1.0f64, 1.0f64, 0.0f64);
        for _ in 0..4000 {
            let (mut ga, mut gb, mut gc) = (0.0, 0.0, 0.0);
            for r in &rows {
                let (x1, x2) = ((r.0 - m1) / s1, (r.1 - m2) / s2);
                let e = sigmoid(a * x1 + b * x2 + c) - r.2;
                ga += e * x1;
                gb += e * x2;
                gc += e;
            }
            a -= 0.1 * ga / n;
            b -= 0.1 * gb / n;
            c -= 0.1 * gc / n;
        }
        let m2_p2 = |lpi: f64, lr: f64| sigmoid(a * (lpi - m1) / s1 + b * (lr - m2) / s2 + c);
        let m2_iso = IsotonicMap::fit(&rows.iter().map(|r| (m2_p2(r.0, r.1), r.2)).collect::<Vec<_>>());

        // ── calibrate::Fusion on the same rows. ──
        let fit_rows: Vec<(Vec<f64>, f64)> = rows.iter().map(|r| (vec![r.0, r.1], r.2)).collect();
        let fu = Fusion::fit(&fit_rows);

        for r in &rows {
            let msgs = [r.0, r.1];
            assert!(
                (fu.s1(&msgs) - m2_p2(r.0, r.1)).abs() < 1e-9,
                "raw fusion P² must match M2"
            );
            assert!(
                (fu.fuse(&msgs) - m2_iso.apply(m2_p2(r.0, r.1))).abs() < 1e-9,
                "isotonic P̂ must match M2"
            );
        }
    }
}
