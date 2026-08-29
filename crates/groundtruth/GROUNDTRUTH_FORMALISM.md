# Ground Truth for Disassembly — Mathematical Grounding

This is the formal spec the generator implements. The point of writing it down is to
make every truth set a *provable* object with a stated guarantee, so "ground truth"
stops being a judgment call and becomes something you can verify per binary.

## 1. Objects

- A binary image is a byte string `B = b_0 … b_{n-1}`. Addresses are virtual addresses.
- `X ⊆ addrs(B)` is the **executable region**: the union of sections with the
  `SHF_EXECINSTR` flag (`.init`, `.plt`, `.plt.sec`, `.text`, `.fini`, …).
- `decode : X ⇀ ℕ` is the (partial) ISA decode function. `decode(a) = ℓ` means the
  bytes at `a` form a valid instruction of length `ℓ`; otherwise `decode(a) = ⊥`.
  This is total knowledge — decoding is decidable at a fixed address.
- `I* ⊆ X` is the **true instruction stream**: the set of addresses that are actual
  instruction starts in the program as the toolchain emitted it. For clean,
  non-self-modifying, non-overlapping code, `I*` is the linear tiling of the code
  regions (each instruction's bytes are disjoint; the next starts where the last ends).
  `I*` is the latent object we are bounding. Two facts hold by definition:
  - `a ∈ I* ⟹ a ∈ dom(decode)` (real instructions decode), and
  - within a basic block, `I*` is gap-free and non-overlapping.

We cannot observe `I*` directly for an arbitrary binary (undecidable in general). The
framework brackets it:

> **`G_min ⊆ I* ⊆ G_max`**

with both bounds generated automatically and each containment carrying a proof.

## 2. The lower bound `G_min` (provably code)

`G_min` is a union of *evidence sets*, each of which is independently a subset of `I*`:

- `E_line = { a ∈ X : a is the address of a DWARF .debug_line row with ¬end_sequence }`.
  A line-program row is emitted only at a real instruction boundary, so `E_line ⊆ I*`.
- `E_fn = { a ∈ X : a = value of an STT_FUNC symbol } ∪ { DW_AT_low_pc of a concrete DW_TAG_subprogram }`.
  Function entries are real instruction starts, so `E_fn ⊆ I*`.
- `E_rel = { a ∈ X : a is the target of a code relocation }` *(optional; requires
  `-Wl,--emit-relocs`)*. Relocation code targets are real instruction starts, `E_rel ⊆ I*`.

**Definition.** `G_min := E_line ∪ E_fn (∪ E_rel)`.

**Proposition 1 (Soundness, zero FP).** `G_min ⊆ I*`.
*Proof.* A finite union of subsets of `I*` is a subset of `I*`. ∎

Consequence: a tool that fails to predict some `a ∈ G_min` has a *genuine* false
negative — there is no judgment involved. `G_min` is the recall target.

## 3. The upper bound `G_max` (all code, under one stated assumption)

Let `L = linearDecode(X)` be the set of instruction starts produced by decoding forward
from each executable-section start through `X` (equivalently: the instruction starts in
`objdump -d` over `X`, or a Capstone linear sweep for fixed-width ISAs).

**Assumption C (clean toolchain output).** Executable sections contain no embedded data
(jump tables live in `.rodata`, etc.), no hand-written overlapping code, and no
self-modification. This holds for compiler-generated, unobfuscated ELFs — i.e. our corpus.

**Definition.** `G_max := L`.

**Proposition 2 (Completeness under C, zero FN).** Under Assumption C, `I* ⊆ G_max`;
in fact `G_max = I*`.
*Proof sketch.* Under C the code regions are a clean linear tiling, so linear decode
anchored at section starts never desynchronizes and recovers exactly the emitted stream.
∎

Consequence: under C, an address `a ∉ G_max` is *provably not* a real instruction start
— it is either data or interior to a real instruction (an overlapping/punned decode).
So a tool predicting `a ∉ G_max` has a *genuine* false positive. `G_max` is the
precision ceiling.

**Verification (per binary).** `G_min ⊆ G_max` is checked directly. Any `a ∈ G_min`
with `a ∉ G_max` is a witness that Assumption C failed for this binary (data-in-code,
linear-sweep desync) — a useful red flag, not a silent error.

## 4. The interval and the neutral zone

Define the **neutral zone** `N := G_max \ G_min`. Under C, `G_max = I*` and `G_min ⊆ I*`,
so `N = I* \ G_min`: real instructions for which we have no *independent* metadata proof
(CRT/linker bodies without debug lines, alignment/padding NOPs). For `a ∈ N` we can
neither prove `a ∈ G_min` (no evidence) nor `a ∉ G_max` (it decodes in-stream), so a tool
is neither credited nor penalized for it.

This is the resolution of the qualitative/quantitative question: the GT only asserts what
it can *prove* (`G_min`, `G_max`). The "does this instruction make sense" judgment is
deliberately **not** encoded here — it lives in the probabilistic model (§7).

## 5. Range scoring

For a tool's prediction set `P ⊆ X`:

```
TP      = |P ∩ G_min|        Precision = TP / (TP + FP)
FN      = |G_min \ P|        Recall    = TP / (TP + FN)
FP      = |P \ G_max|
Neutral = |P ∩ N|            (excluded from both)
```

**Theorem 3 (Tier-invariance).** Let an evaluator instead pick any single-tier truth
`T` with `G_min ⊆ T ⊆ G_max` and score normally. The range-scored `FP` and `FN` above
are invariant to the choice of `T`.
*Proof.* `FN = G_min \ P` and `FP = P \ G_max` depend only on the endpoints `G_min` and
`G_max`, not on `T`. ∎

This is the whole point: published evaluations differ only in *where in `[G_min, G_max]`*
they implicitly placed `T` (DWARF-only ≈ `G_min`; symbol-table ≈ a middle tier; objdump
≈ `G_max`). Range scoring removes that free parameter.

## 6. Generation map (definition → mechanism)

| Set | Mechanism | Guarantee |
|---|---|---|
| `E_line` | DWARF `.debug_line` rows (`¬end_sequence`) ∩ `X` | `⊆ I*` |
| `E_fn` | `STT_FUNC` symbols ∪ `DW_TAG_subprogram` low_pc, ∩ `X` | `⊆ I*` |
| `E_rel` | code relocation targets (`--emit-relocs`), ∩ `X` | `⊆ I*` |
| `G_max` | `objdump -d` / Capstone linear sweep over `X` | `= I*` under C |
| `X` | sections with `SHF_EXECINSTR` | exact |

Everything is recoverable from a single debug ELF (DWARF + symtab + section table) plus
one `objdump` pass. No heuristics, no manual labeling, no tool cross-referencing.

## 7. The probabilistic bridge (why this touches UPD)

Let `y*(a) = 1[a ∈ I*]` be the true indicator. A probabilistic disassembler outputs
`p(a) ≈ E[y*(a)]`. The bounds give a *measurable* calibration target without ever
observing `I*` fully:

- For `a ∈ G_min`: `y*(a) = 1` (proven).
- For `a ∉ G_max`: `y*(a) = 0` (proven, under C).
- For `a ∈ N`: `y*(a)` is **unknown** → excluded from the calibration measurement.

So calibration/ECE is computed on `G_min ∪ (X \ G_max)` and the neutral zone is dropped —
which is exactly range scoring expressed over probabilities. The decode support
`dom(decode)` is *where* `p` is defined; `[G_min, G_max]` is *what* `p` is uncertain about.

When Assumption C fails (obfuscation, packing, hand-asm) `G_max` is no longer provably
`I*`, the interval widens into genuine uncertainty, and the calibrated `p(a)` is the
honest output in place of a hard label. **Min/max is the build-provable special case;
the calibrated posterior is the general case.**
