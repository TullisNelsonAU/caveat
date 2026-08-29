# `tool` — self-calibrating analysis core + CLI

A clean, callable, serde-serializable pass over the probabilistic-disassembly stack, plus the result
type every front-end reuses. `tool` is a **read-only consumer** of the engines — `probdisasm` (Soft
decode + cavity surprise), `probcfg` (Layer-2 function confirmation `F_h` + reachedness `R_a`), and
the Paper-2 consistency statistics — and the **serialization boundary**: it depends on them, adds
`serde` itself, and defines its own result types populated from their outputs. No engine crate is
modified. This is the foundation the CLI *and* a future TUI/GUI all consume, and the runnable artifact
for the papers. The honesty wall is inherited: Layer-2 confirmation/reachedness are decode weights,
the cavity surprise is a post-hoc read — neither overwrites the per-byte posterior π.

**Self-calibrating (Paper 2's payoff, live).** The tool runs the consistency detector, reads the
GT-free `(S_glob, S_spat)` signature, and classifies the regime with the trained benign-default
threshold rule from the switching experiment (the switching experiment, n=80 held-out, 1.00 selection accuracy): `S_glob > 2.566 ⇒ obfuscated`, else `S_spat > 0.108 ⇒ packed`, else benign. It
then applies that regime's bank config in two halves: the **engine config** (packed → entropy-prior,
obfuscated → chainfwd-prior — always applied), and — with `--bank bank.json` — the **fitted isotonic
calibration map** applied to the marginals. The map is the half that reproduces Paper 2's *restoration*
rather than just the routing, and it has two flavors: the **packed** map is the degenerate all-data
fit, so it drives packed posteriors to ≈0 (abstain / suppress, the paper's packed → ≈0); the
**obfuscated/desync** map is a **non-degenerate recalibration** fit on labeled desync code, which moves
a drifted binary's ECE back toward the oracle (the paper's lead result, ECE ≈0.074 → ≈0.025). The
choice (regime, engine, whether the map applied) is recorded in `calibration`.

*Honest scope.* This routes **structural** obfuscation — packing and anti-disassembly desync, where
the surprise statistic is diagnostic. Semantic / virtualized obfuscation (Tigress) preserves clean
decoding, so `S` is blind to it. The rule is benign-default so an unrecognized signature is never
mis-routed into a calibration-destroying map; when there is residual mid-band drift the rule can't
place, the tool falls back to benign **and flags `regime_uncertain` — "calibration may be stale"**
rather than asserting clean. Without `--bank` the tool applies the engine config only (the GT-free
half); with `--bank` it additionally applies the regime's GT-fit isotonic map.

## The two schemas

**`AnalysisResult`** — the JSON contract every frontend reuses (`src/lib.rs`):

| block | what it carries |
|-------|-----------------|
| `binary` | path, format, arch, entry, `.text` byte count |
| `regime` | `detected` (benign/packed/obfuscated), `source` (auto/override), `confidence` |
| `functions[]` | candidate heads: `addr`, `name?`, `confidence` (`F_h`), `reached_prob` (`R`), `in_body_of?`, `flagged_decoy` |
| `edges[]` | call graph: `from`, `to`, `kind` (direct/indirect), `confidence` (noisy-OR `C`) |
| `instructions` | `n`, `mean_pi`, `low_confidence[]` (+count); full per-insn is opt-in |
| `trust` | `overall`, `s_glob` (mean surprise), `s_spat` (Moran's I), `distrust_regions[]` (localized) |
| `suggestions[]` | active-mode: `addr`, `expected_info_gain`, `why` — what to confirm next |
| `calibration` | `selected_regime`, `classifier`, `map_applied`, `engine` `[entropy,chainfwd]`, `isotonic_applied`, `regime_uncertain`, `note` — the self-calibration record |

**`KnownFacts`** — everything the user knows, each field mapped to a clamp or the entry anchor
(`--facts f.json`). Addresses are JSON numbers (decimal):

```json
{
  "entry":       4213048,
  "symbols":     { "4249120": "recovered_secret_fn" },
  "annotations": { "4212330": "data" },
  "trace":       [ 4249120 ],
  "regime":      "benign"
}
```

Each fact enters where the corresponding evidence would: a known-real head (symbol / `"function"`
annotation / traced address) becomes a synthetic `probcfg::ResolvedEdge` from the entry root — exactly
the slot a recovered code pointer occupies — so the confirmation fixpoint lifts it from the residual
tail to the core with earned structure, never by overwriting a posterior. `"data"` annotations
suppress an address's confidence to 0; `entry`/`regime` set the anchor and override the regime read.
An absent file ⇒ analyze with just the binary.

## Usage

```
tool <binary> [--facts f.json] [--bank bank.json] [--out result.json] [--report] [--full-insns]
tool export-bank [--from bank.json] [--packed <elf> <upxgt>] [--desync <bins> <gt>]... \
                 [--desync-limit N] --out bank.json
```

- `--out FILE` writes the `AnalysisResult` JSON (the frontend contract).
- `--facts FILE` folds `KnownFacts` in as clamps/anchors.
- `--bank FILE` loads the fitted calibration-map bank; the selected regime's isotonic map is applied
  to the marginals (the map-switch). A prebuilt bank ships at `banks/upd_bank.json`.
- `export-bank` fits the regime isotonic maps the way the switching experiment does — `--packed`
  from a UPX binary + `.upxgt` window (all-data → PAVA → suppress), `--desync` from a labeled desync
  corpus (`(π, instr-start GT)` under the obfuscated engine → PAVA → non-degenerate recalibration).
  `--from` seeds unchanged maps; `--desync-limit N` caps bins (smallest first) for bounded runtime.
- `--report` prints the human-readable trust/confidence summary (the default when no flags are given).
- `--full-insns` includes the full per-address posterior list in the JSON.

Example — a benign binary is recognized and left on the benign config:

```
$ tool corpus_packed/ls_unpacked
regime   : benign (auto, conf 0.90)
calib    : selected benign via signature-rule → benign:identity (engine [0.0, 0.0])
— trust (trustworthy) —
  S_glob (mean surprise) : 0.8029   S_spat (Moran's I) : +0.0895
— functions (358 candidates · 351 confirmed · 7 flagged suspect) — …
```

Example — a UPX-packed binary trips the packed signature and self-calibrates. Without a bank the
engine config alone leaves the marginals mid-range; with the bank the packed isotonic map suppresses
them to ≈0 (the paper's packed → ≈0), i.e. the tool *abstains* on the packed payload:

```
$ tool corpus_packed/ls_packed                              # engine-switch only
regime   : packed (auto, conf 0.85)
calib    : selected packed → packed:entropy-prior   applied: engine only
— instructions — candidates 45567 · mean π 0.368

$ tool corpus_packed/ls_packed --bank banks/upd_bank.json   # + map-switch
calib    : selected packed → packed:entropy-prior + isotonic-packed   applied: engine + isotonic map (map-switch)
— instructions — candidates 45567 · mean π 0.000   ← restoration: packed → ≈0 (abstain)
```

Example — a **desync** (anti-disassembly) binary self-calibrates with a *non-degenerate* map. On a
held-out desync bin (`d1_med/echo`, 53,186 candidates) the map moves the calibration error toward the
oracle — a real recalibration, not an abstain. Post-hoc ECE (needs GT; measured offline, the tool
itself stays GT-free):

| arm | ECE |
|-----|-----|
| always-benign (benign engine, identity map) | 0.069 |
| engine-only (obfuscated engine, no map)      | 0.070 |
| **engine + isotonic** (obf engine + desync map) | **0.030** |

The engine config alone barely moves ECE (0.069 → 0.070); the **isotonic map** does the restoration
(0.070 → 0.030), toward the switching experiment's oracle (~0.017 on this level). This is Paper 2's
lead result live — the applied map recalibrates a drifted binary, it doesn't just route it.

(Header-stripped / packed ELFs have no `.text`; the tool falls back to the executable `PT_LOAD`.)

Adding `--facts` with a symbol at a flagged-suspect head lifts it from `F=0.10` (decoy) to `F≈0.99`
(confirmed, named); a `"data"` annotation zeroes a head; `regime` `"..."` overrides the classifier
(`source`/`classifier` → `override`).

## Scope

TUI (ratatui) and GUI (web/Tauri) are **out of scope** — they consume this JSON later; the whole point
is that the result format is designed so they're thin. The shipped bank (`banks/upd_bank.json`) fits
the **packed** map (degenerate suppress, from `ls_packed`'s data window) and the **obfuscated/desync**
map (non-degenerate recalibration, pooled over desync-corpus bins under the obfuscated engine) for
real; the **benign** map is identity (clean code needs none). The desync map here is fit on a bounded
subset for a shippable artifact — regenerate a fuller one over the whole desync corpus with
`export-bank --desync … --desync-limit 0` (or the switching harness) for maximum fidelity. Honest
scope is unchanged: structural obfuscation only (semantic/virtualized is the `regime_uncertain` blind
spot), benign-default routing, and the isotonic map calibrates the π **marginals**, not the Layer-2
`F_h` axis. One cleanup tracked for later: `S_glob`/`S_spat` are recomputed here because the
aggregation is private to the `consistency` binary — factor it into a shared lib so the tool and the
binary can't drift. `tool` never reads ground truth — it is a tool, not an eval.
