#!/usr/bin/env bash
# Rebuild every adaptive-adversary construction and emit the per-binary CSV.
#
# Deterministic end to end: the constructions are a pure function of (substrate ELF, donor ELF,
# SEED, INTERLEAVE_SEED), the ground truth is the injector's own record plus gen-gt's DWARF-derived
# instruction starts, and the engine is pinned. Re-running with the same inputs reproduces
# adaptive_adversary.csv byte for byte. Nothing here reads the previous CSV.
#
# Engine of record: probdisasm `feat/chainfwd-prior` @ c62ead9.
# Two substrate/donor pairs are run so no claim rests on a single binary:
#   primary: base=sum   donor=printf
#   swap:    base=printf donor=sum
# The clean isotonic map is fit on the held-out clean pair (yes, env) in both runs — the paper's
# Pass 1, and disjoint from every substrate and donor.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

BINS="${BINS:-$HOME/lab/projects/probablistic/corpus/x86_64-binaries/elf/coreutils}"
GENGT="${GENGT:-$ROOT/target/release/gen-gt}"
PROBDISASM="${PROBDISASM:-$ROOT/../probdisasm}"
SEED="${SEED:-0x9E3779B97F4A7C15}"       # packed body (C3)
INTERLEAVE_SEED="${INTERLEAVE_SEED:-0xDEADBEEF}"  # interleave random half (C5), mixed with k

CSV="$HERE/adaptive_adversary.csv"
LOG="$HERE/run.log"
GT="$HERE/gt"   # regenerated below; gitignored, since gen-gt rebuilds it from the ELFs

command -v "$GENGT" >/dev/null 2>&1 || [ -x "$GENGT" ] || {
  echo "gen-gt not found at $GENGT (run: cargo build --release)" >&2; exit 1; }

# ── ground truth: gen-gt insn_max, DWARF+symtab, never a disassembler's opinion ──
mkdir -p "$GT"
for n in sum printf yes env; do
  elf="$BINS/gcc_coreutils_64_O1_$n"
  [ -f "$elf" ] || { echo "missing substrate $elf" >&2; exit 1; }
  "$GENGT" "$elf" "$GT/gt_$n" >/dev/null
done

cd "$ROOT"
cargo build --release -p adversary 2>&1 | tail -2

rm -f "$CSV" "$LOG"
run_pair() {  # <base> <donor>
  local b="$1" d="$2"
  ./target/release/adversary \
    --csv "$CSV" --seed "$SEED" --interleave-seed "$INTERLEAVE_SEED" \
    "$BINS/gcc_coreutils_64_O1_$b" "$GT/gt_$b/insn_max.txt" \
    "$BINS/gcc_coreutils_64_O1_$d" "$GT/gt_$d/insn_max.txt" \
    "$BINS/gcc_coreutils_64_O1_yes" "$GT/gt_yes/insn_max.txt" \
    "$BINS/gcc_coreutils_64_O1_env" "$GT/gt_env/insn_max.txt"
}
{ run_pair sum printf; run_pair printf sum; } 2>&1 | tee -a "$LOG"

# ── manifest: what the CSV is a function of. No timestamps, so re-running is a no-op on disk. ──
cat >"$HERE/run_manifest.json" <<JSON
{
  "engine": "probdisasm feat/chainfwd-prior @ $(git -C "$PROBDISASM" rev-parse --short HEAD)",
  "driver": "upd-suite experimental/adversary",
  "seed_pack": "$SEED",
  "seed_interleave": "$INTERLEAVE_SEED",
  "substrate_pairs": [
    {"base": "gcc_coreutils_64_O1_sum", "donor": "gcc_coreutils_64_O1_printf"},
    {"base": "gcc_coreutils_64_O1_printf", "donor": "gcc_coreutils_64_O1_sum"}
  ],
  "clean_fit": ["gcc_coreutils_64_O1_yes", "gcc_coreutils_64_O1_env"],
  "gt_source": "injector-record + gen-gt insn_max (DWARF/symtab)",
  "detection_null": {"S_glob": 1.01, "S_spat": 0.105},
  "routing_gate": {"S_glob": 2.5147, "S_spat": 0.1052},
  "rows": $(( $(grep -c "" "$CSV") - 1 ))
}
JSON

echo
echo "wrote $CSV ($(( $(grep -c "" "$CSV") - 1 )) rows) and run_manifest.json"
echo "next: ./assemble_results.sh"
