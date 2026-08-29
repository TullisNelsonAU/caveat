//! Hint extractors from Miller et al. and hopefully future extensions of the hint system.

use std::collections::{HashMap, HashSet};

use crate::superset::{Instruction, Superset};

/// Global cap on the number of relational hint pairs generated per binary.
/// Normal binaries produce ~10^5 pairs; this only binds on pathological dense
/// `.text` (large jump tables / data-in-code at O0) where the def-use walk would
/// otherwise generate billions of pairs and exhaust memory. Set far above any
/// normal binary's pair count, so results for sub-cap binaries are unchanged.
const HINT_PAIR_CAP: usize = 5_000_000;

/// A hint that the address that produced it plus a label for the hint type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HintKey {
    /// The address that produced this hint.
    pub source_addr: u64,
    /// The type of hint.
    pub label: HintLabel,
}

/// The type of hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HintLabel {
    /// Hint I: control-flow convergence, 1-byte displacement. Prior 1/255.
    CtrlConvRel,
    /// Hint I: control-flow convergence, 2-byte displacement. Prior 1/65535.
    CtrlConvNear,
    /// Hint I: control-flow convergence, 4-byte displacement. Prior 1/(2^32-1).
    CtrlConvLong,

    /// Hint II: control-flow crossing, 1-byte displacement.
    CtrlCrossRel,
    /// Hint II: control-flow crossing, 2-byte displacement.
    CtrlCrossNear,
    /// Hint II: control-flow crossing, 4-byte displacement.
    CtrlCrossLong,

    /// Weak control-flow hint: branch target doesn't occlude with source. Prior ~1/n.
    CtrlWeak,

    /// Hint III: register def-use. Prior 1/16.
    RegDefUse,

    /// Return-address hint: instruction immediately after a call is the return target.
    /// Emitted as a unary hint at address c + s. Prior 1e-4 (empirically near-perfect TP).
    ReturnAddr,
}

impl HintLabel {
    /// Returns the prior probability of this hint type.
    pub const fn prior(self) -> f64 {
        match self {
            // probdisasm #18 (Priyadarshan et al., ASPLOS'23) corrected hint weights.
            Self::CtrlConvRel | Self::CtrlCrossRel => 1.0 / 32.0,
            Self::CtrlConvNear | Self::CtrlCrossNear => 1.0 / 2048.0,
            Self::CtrlConvLong | Self::CtrlCrossLong => 1.0 / 32768.0,
            Self::CtrlWeak => 1.0 / 3.5,
            Self::RegDefUse => 0.5,
            Self::ReturnAddr => 1e-4,
        }
    }

    /// DASSA-corrected coincidence priors (Priyadarshan et al., ASPLOS'23, Table 2).
    ///
    /// Miller's priors assume uniform-random data, which makes the control-flow hints wildly
    /// overconfident — a 4-byte-offset "jump" in data is treated as 2^-32 rare, when DASSA
    /// measures it nearer 2^-11. DASSA's empirical data rates: short jump (1-byte offset) ≈ 2^-5,
    /// long CFT (4-byte) in 2^-11..2^-16 — we use a representative 2^-13; 2-byte is interpolated to
    /// 2^-8. The non-CFT hints (`CtrlWeak`/`RegDefUse`/`ReturnAddr`) aren't re-derived by DASSA, so
    /// they keep Miller's values.
    pub const fn dassa_prior(self) -> f64 {
        match self {
            Self::CtrlConvRel | Self::CtrlCrossRel => 1.0 / 32.0, // 2^-5  (DASSA SJ)
            Self::CtrlConvNear | Self::CtrlCrossNear => 1.0 / 256.0, // 2^-8  (interpolated)
            Self::CtrlConvLong | Self::CtrlCrossLong => 1.0 / 8192.0, // 2^-13 (DASSA LU/LC)
            Self::CtrlWeak => 1.0 / 3.5,
            Self::RegDefUse => 1.0 / 16.0,
            Self::ReturnAddr => 1e-4,
        }
    }
}

/// A pairwise relational hint: both addresses are coupled as code evidence.
///
/// `log_weight = -log(p_a) - log(p_b)` where `p_a`, `p_b` are the per-address
/// coincidence priors. Used to build `BpFactorKind::HintCoupling` factors in Soft mode.
#[derive(Debug, Clone, Copy)]
pub struct HintPair {
    /// First participant address.
    pub addr_a: u64,
    /// Second participant address.
    pub addr_b: u64,
    /// log ψ(1,1) for the corresponding HintCoupling factor.
    pub log_weight: f64,
}

/// Pre-refactor unary hints — all four families as independent unaries.
/// Phase 4 calibration comparison only; do not use in production.
pub fn extract_all_hints_legacy(superset: &Superset) -> HashMap<HintKey, f64> {
    let mut hints = HashMap::new();
    // CtrlConv unaries
    {
        let mut targets: HashMap<u64, Vec<&Instruction>> = HashMap::new();
        for insn in superset.iter_valid() {
            if !insn.is_branch() {
                continue;
            }
            if let Some(t) = insn.branch_target {
                targets.entry(t).or_default().push(insn);
            }
        }
        for branches in targets.values().filter(|b| b.len() >= 2) {
            for br in branches {
                let label = match br.size {
                    2 => HintLabel::CtrlConvRel,
                    3 | 4 => HintLabel::CtrlConvNear,
                    _ => HintLabel::CtrlConvLong,
                };
                emit_hint(&mut hints, br.address, label);
            }
        }
    }
    // CtrlCross unaries
    {
        let post: HashMap<u64, &Instruction> = superset
            .iter_valid()
            .filter(|i| i.is_branch())
            .map(|i| (i.address + i.size as u64, i))
            .collect();
        for insn in superset.iter_valid().filter(|i| i.is_branch()) {
            if let Some(tgt) = insn.branch_target {
                if let Some(&other) = post.get(&tgt) {
                    if other.address != insn.address {
                        let la = match insn.size {
                            2 => HintLabel::CtrlCrossRel,
                            3 | 4 => HintLabel::CtrlCrossNear,
                            _ => HintLabel::CtrlCrossLong,
                        };
                        let lb = match other.size {
                            2 => HintLabel::CtrlCrossRel,
                            3 | 4 => HintLabel::CtrlCrossNear,
                            _ => HintLabel::CtrlCrossLong,
                        };
                        emit_hint(&mut hints, insn.address, la);
                        emit_hint(&mut hints, other.address, lb);
                    }
                }
            }
        }
    }
    extract_weak_control_flow(superset, &mut hints);
    // RegDefUse unaries
    {
        fn walk(
            superset: &Superset,
            def_addr: u64,
            reg: u16,
            depth: usize,
            hints: &mut HashMap<HintKey, f64>,
        ) {
            let mut visited: HashSet<u64> = HashSet::new();
            let mut stack: Vec<(u64, usize)> = superset
                .successors_of(def_addr)
                .into_iter()
                .map(|s| (s, depth))
                .collect();
            while let Some((addr, rem)) = stack.pop() {
                if rem == 0 || !visited.insert(addr) {
                    continue;
                }
                let Some(insn) = superset.at(addr) else {
                    continue;
                };
                if insn.regs_read.contains(&reg) {
                    emit_hint(hints, def_addr, HintLabel::RegDefUse);
                    emit_hint(hints, addr, HintLabel::RegDefUse);
                    continue;
                }
                if insn.regs_write.contains(&reg) {
                    continue;
                }
                stack.extend(
                    superset
                        .successors_of(addr)
                        .into_iter()
                        .map(|s| (s, rem - 1)),
                );
            }
        }
        for def in superset.iter_valid() {
            for &reg in &def.regs_write {
                walk(superset, def.address, reg, 50, &mut hints);
            }
        }
    }
    hints
}

/// Unary hints only: `CtrlWeak` and `ReturnAddr`.
///
/// `CtrlConv`, `CtrlCross`, and `RegDefUse` are relational; extract them with
/// [`extract_hint_pairs`] so their coupling structure is captured as pairwise factors.
pub fn extract_all_hints(superset: &Superset) -> HashMap<HintKey, f64> {
    let mut hints = HashMap::new();
    extract_weak_control_flow(superset, &mut hints);
    extract_return_addr_hints(superset, &mut hints);
    hints
}

/// Emit a `ReturnAddr` hint at address `c + s` for every valid call instruction
/// at address `c` with size `s`, provided that `c + s` is a valid-decode address
/// inside the superset. This is a unary hint — it goes directly into `phi`, not
/// into a pairwise factor.
fn extract_return_addr_hints(superset: &Superset, hints: &mut HashMap<HintKey, f64>) {
    for insn in superset.iter_valid().filter(|i| i.is_call()) {
        let ret_addr = insn.address + insn.size as u64;
        if superset.at(ret_addr).is_some() {
            emit_hint(hints, ret_addr, HintLabel::ReturnAddr);
        }
    }
}

/// Pairwise relational hints: one `HintPair` per related address pair.
///
/// Covers `CtrlConv` (convergence), `CtrlCross` (crossing), and `RegDefUse` (def-use).
/// For convergence groups of k ≥ 3 branches, emits C(k,2) pairs covering every pair.
pub fn extract_hint_pairs(superset: &Superset) -> Vec<HintPair> {
    extract_hint_pairs_with(superset, false)
}

/// Like [`extract_hint_pairs`], but `use_dassa` selects the DASSA-corrected coincidence priors
/// (see [`HintLabel::dassa_prior`]) for the control-flow pairs instead of Miller's. Def-use pairs
/// are unchanged either way — DASSA doesn't re-derive that channel.
pub fn extract_hint_pairs_with(superset: &Superset, use_dassa: bool) -> Vec<HintPair> {
    let mut pairs = Vec::new();
    extract_conv_pairs(superset, &mut pairs, use_dassa);
    extract_cross_pairs(superset, &mut pairs, use_dassa);
    // E3 ablation 2026-06-04 RESULT: removing RegDefUse pairs was strongly harmful
    // (Brier 0.088 -> 0.144, confident-miss share -> 88%). Def-use pairs are a
    // load-bearing long-range evidence TRANSPORT channel, not a junk source. Restored.
    extract_def_use_pairs(superset, &mut pairs);
    pairs
}

fn conv_prior(insn: &Instruction, use_dassa: bool) -> f64 {
    let label = match insn.size {
        2 => HintLabel::CtrlConvRel,
        3 | 4 => HintLabel::CtrlConvNear,
        _ => HintLabel::CtrlConvLong,
    };
    if use_dassa { label.dassa_prior() } else { label.prior() }
}

/// One `HintPair` per converging branch pair (C(k,2) for k branches).
/// log_weight = -log(p_i) - log(p_j): joint coincidence prior.
fn extract_conv_pairs(superset: &Superset, pairs: &mut Vec<HintPair>, use_dassa: bool) {
    let mut targets: HashMap<u64, Vec<&Instruction>> = HashMap::new();
    for insn in superset.iter_valid() {
        if !insn.is_branch() {
            continue;
        }
        let Some(target) = insn.branch_target else {
            continue;
        };
        targets.entry(target).or_default().push(insn);
    }
    for branches in targets.values().filter(|b| b.len() >= 2) {
        if pairs.len() >= HINT_PAIR_CAP {
            break;
        }
        // E2 experiment 2026-06-04: group-normalize clique weights. A k-group
        // emits C(k,2) pairs, so each member sits in (k-1) factors and receives
        // (k-1)x its group evidence even before loopy double-counting around the
        // clique's cycles. Dividing by (k-1) makes a member's total first-order
        // group evidence one partner's worth, independent of group size.
        let norm = (branches.len() - 1) as f64;
        for i in 0..branches.len() {
            for j in (i + 1)..branches.len() {
                pairs.push(HintPair {
                    addr_a: branches[i].address,
                    addr_b: branches[j].address,
                    log_weight: (-conv_prior(branches[i], use_dassa).ln()
                        + -conv_prior(branches[j], use_dassa).ln())
                        / norm,
                });
            }
        }
    }
}

/// Extracts weak control-flow hints from the superset.
fn extract_weak_control_flow(superset: &Superset, hints: &mut HashMap<HintKey, f64>) {
    for insn in superset.iter_valid().filter(|i| i.is_branch()) {
        let Some(target) = insn.branch_target else {
            continue;
        };
        let Some(target_insn) = superset.at(target) else {
            continue;
        };

        let source_end = insn.address + insn.size as u64;
        let target_end = target + target_insn.size as u64;
        let occludes = insn.address < target_end && target < source_end;
        if occludes {
            continue;
        }

        emit_hint(hints, insn.address, HintLabel::CtrlWeak);
    }
}

fn cross_prior(insn: &Instruction, use_dassa: bool) -> f64 {
    let label = match insn.size {
        2 => HintLabel::CtrlCrossRel,
        3 | 4 => HintLabel::CtrlCrossNear,
        _ => HintLabel::CtrlCrossLong,
    };
    if use_dassa { label.dassa_prior() } else { label.prior() }
}

/// One `HintPair` per crossing pair (A's target = end of B).
fn extract_cross_pairs(superset: &Superset, pairs: &mut Vec<HintPair>, use_dassa: bool) {
    let post_branch: HashMap<u64, &Instruction> = superset
        .iter_valid()
        .filter(|insn| insn.is_branch())
        .map(|insn| (insn.address + insn.size as u64, insn))
        .collect();

    for insn in superset.iter_valid().filter(|i| i.is_branch()) {
        let Some(target) = insn.branch_target else {
            continue;
        };
        let Some(&other) = post_branch.get(&target) else {
            continue;
        };
        if other.address == insn.address {
            continue;
        }
        pairs.push(HintPair {
            addr_a: insn.address,
            addr_b: other.address,
            log_weight: -cross_prior(insn, use_dassa).ln() + -cross_prior(other, use_dassa).ln(),
        });
    }
}

/// One `HintPair` per register def-use pair (first use on each CFG path).
fn extract_def_use_pairs(superset: &Superset, pairs: &mut Vec<HintPair>) {
    const MAX_WALK_DEPTH: usize = 50;
    for def in superset.iter_valid() {
        if pairs.len() >= HINT_PAIR_CAP {
            break;
        }
        for &reg in &def.regs_write {
            walk_for_use_pairs(superset, def.address, reg, MAX_WALK_DEPTH, pairs);
        }
    }
}

fn walk_for_use_pairs(
    superset: &Superset,
    def_addr: u64,
    reg: u16,
    depth: usize,
    pairs: &mut Vec<HintPair>,
) {
    let log_w = -HintLabel::RegDefUse.prior().ln();
    let mut visited: HashSet<u64> = HashSet::new();
    let mut stack: Vec<(u64, usize)> = superset
        .successors_of(def_addr)
        .into_iter()
        .map(|s| (s, depth))
        .collect();
    while let Some((addr, remaining)) = stack.pop() {
        if pairs.len() >= HINT_PAIR_CAP {
            return;
        }
        if remaining == 0 || !visited.insert(addr) {
            continue;
        }
        let Some(insn) = superset.at(addr) else {
            continue;
        };
        if insn.regs_read.contains(&reg) {
            pairs.push(HintPair {
                addr_a: def_addr,
                addr_b: addr,
                log_weight: 2.0 * log_w,
            });
            continue;
        }
        if insn.regs_write.contains(&reg) {
            continue;
        }
        stack.extend(
            superset
                .successors_of(addr)
                .into_iter()
                .map(|s| (s, remaining - 1)),
        );
    }
}

fn emit_hint(hints: &mut HashMap<HintKey, f64>, source_addr: u64, label: HintLabel) {
    hints.insert(HintKey { source_addr, label }, label.prior());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has_pair(pairs: &[HintPair], a: u64, b: u64) -> bool {
        pairs
            .iter()
            .any(|p| (p.addr_a == a && p.addr_b == b) || (p.addr_a == b && p.addr_b == a))
    }

    #[test]
    fn test_extract_control_flow_convergence_long() {
        // 0:  e9 07 00 00 00   jmp 0xc
        // 5:  90               nop
        // 6:  e9 01 00 00 00   jmp 0xc
        let bytes: &[u8] = &[
            0xE9, 0x07, 0x00, 0x00, 0x00, 0x90, 0xE9, 0x01, 0x00, 0x00, 0x00, 0x90, 0x90,
        ];
        let superset = Superset::new(0x0, bytes).expect("failed to create superset");
        let pairs = extract_hint_pairs(&superset);
        assert!(has_pair(&pairs, 0x0, 0x6), "expected conv pair (0x0, 0x6)");
        // log_weight should be 2 * -log(1/2^32) ≈ 44.4
        let p = pairs
            .iter()
            .find(|p| has_pair(std::slice::from_ref(p), 0x0, 0x6))
            .unwrap();
        assert!((p.log_weight - 2.0 * HintLabel::CtrlConvLong.prior().ln().abs()).abs() < 1e-9);
    }

    #[test]
    fn test_extract_control_flow_convergence_rel() {
        // 0x0: eb 04   jmp 0x6
        // 0x3: eb 01   jmp 0x6
        let bytes: &[u8] = &[0xEB, 0x04, 0x90, 0xEB, 0x01, 0x90, 0x90];
        let superset = Superset::new(0x0, bytes).expect("failed to create superset");
        let pairs = extract_hint_pairs(&superset);
        assert!(has_pair(&pairs, 0x0, 0x3), "expected conv pair (0x0, 0x3)");
    }

    #[test]
    fn test_extract_control_flow_convergence_near() {
        // 0x0: 66 74 05   data16 je 0x8
        // 0x4: 66 74 01   data16 je 0x8
        let bytes: &[u8] = &[0x66, 0x74, 0x05, 0x90, 0x66, 0x74, 0x01, 0x90, 0x90];
        let superset = Superset::new(0x0, bytes).expect("failed to create superset");
        let pairs = extract_hint_pairs(&superset);
        assert!(has_pair(&pairs, 0x0, 0x4), "expected conv pair (0x0, 0x4)");
    }

    #[test]
    fn test_extract_weak_control_flow() {
        // 0x0: e9 01 00 00 00   jmp 0x6  (non-overlapping target — weak hint)
        let bytes: &[u8] = &[0xE9, 0x01, 0x00, 0x00, 0x00, 0x90, 0x90];
        let superset = Superset::new(0x0, bytes).expect("failed to create superset");
        let hints = extract_all_hints(&superset);
        assert!(hints.contains_key(&HintKey {
            source_addr: 0x0,
            label: HintLabel::CtrlWeak
        }));
    }

    #[test]
    fn test_extract_control_flow_crossing_long() {
        // 0x0: e9 05 00 00 00   jmp 0xa  <- targets end of 0x5 branch
        // 0x5: e9 00 00 00 00   jmp 0xa
        let bytes: &[u8] = &[
            0xE9, 0x05, 0x00, 0x00, 0x00, 0xE9, 0x00, 0x00, 0x00, 0x00, 0x90,
        ];
        let superset = Superset::new(0x0, bytes).expect("failed to create superset");
        let pairs = extract_hint_pairs(&superset);
        assert!(has_pair(&pairs, 0x0, 0x5), "expected cross pair (0x0, 0x5)");
    }

    #[test]
    fn test_extract_control_flow_crossing_rel() {
        // 0x0: eb 02   jmp 0x4  <- targets end of 0x2 branch
        // 0x2: eb 00   jmp 0x4
        let bytes: &[u8] = &[0xEB, 0x02, 0xEB, 0x00, 0x90];
        let superset = Superset::new(0x0, bytes).expect("failed to create superset");
        let pairs = extract_hint_pairs(&superset);
        assert!(has_pair(&pairs, 0x0, 0x2), "expected cross pair (0x0, 0x2)");
    }

    #[test]
    fn test_extract_control_flow_crossing_near() {
        // 0x0: 66 74 03   data16 je 0x6  <- 3-byte, targets 0x0+3+3=0x6 = end(0x3 branch)
        // 0x3: 66 74 00   data16 je 0x6  <- 3-byte, ends at 0x6
        // 0x6: 90         nop
        // Crossing: 0x0.target(0x6) == end(0x3)=0x6 -> pair(0x0, 0x3).
        let bytes: &[u8] = &[0x66, 0x74, 0x03, 0x66, 0x74, 0x00, 0x90];
        let superset = Superset::new(0x0, bytes).expect("failed to create superset");
        let pairs = extract_hint_pairs(&superset);
        assert!(has_pair(&pairs, 0x0, 0x3), "expected cross pair (0x0, 0x3)");
    }

    /// A single call instruction (5 bytes at 0x0) followed by a nop (1 byte at 0x5).
    /// Return address = 0x0 + 5 = 0x5. ReturnAddr hint must fire at 0x5.
    #[test]
    fn test_extract_return_addr() {
        // 0x0: e8 00 00 00 00   call 0x5  (call +0; return address = 0x5)
        // 0x5: 90               nop       (the return target)
        let bytes: &[u8] = &[0xe8, 0x00, 0x00, 0x00, 0x00, 0x90];
        let superset = Superset::new(0x0, bytes).expect("failed to create superset");

        // Confirm the call decoded at 0x0 with size 5.
        let call = superset.at(0x0).expect("expected call at 0x0");
        assert!(call.is_call(), "expected is_call() at 0x0");
        assert_eq!(call.size, 5, "call size should be 5");

        let hints = extract_all_hints(&superset);
        let key = HintKey {
            source_addr: 0x5,
            label: HintLabel::ReturnAddr,
        };
        assert!(
            hints.contains_key(&key),
            "ReturnAddr hint must fire at 0x5 (return target of call at 0x0). \
             Hints present: {:?}",
            hints.keys().collect::<Vec<_>>()
        );
        let prior = hints[&key];
        assert!(
            (prior - HintLabel::ReturnAddr.prior()).abs() < 1e-15,
            "prior={prior} expected={}",
            HintLabel::ReturnAddr.prior()
        );
    }

    #[test]
    fn test_extract_register_def_use() {
        // 0x0: b8 01 00 00 00   mov eax, 1   <- writes eax
        // 0x5: 03 d8            add ebx, eax <- reads eax
        let bytes: &[u8] = &[0xB8, 0x01, 0x00, 0x00, 0x00, 0x03, 0xD8];
        let superset = Superset::new(0x0, bytes).expect("failed to create superset");
        let pairs = extract_hint_pairs(&superset);
        assert!(
            has_pair(&pairs, 0x0, 0x5),
            "expected def-use pair (0x0, 0x5)"
        );
        let p = pairs
            .iter()
            .find(|p| has_pair(std::slice::from_ref(p), 0x0, 0x5))
            .unwrap();
        let expected_lw = 2.0 * HintLabel::RegDefUse.prior().ln().abs();
        assert!(
            (p.log_weight - expected_lw).abs() < 1e-9,
            "log_weight={} expected={}",
            p.log_weight,
            expected_lw
        );
    }
}
