# gauntlet

Ground-truth-bearing adversarial binary corpus generator for disassembler evaluation.

`gauntlet` produces binaries that are *structurally* as hard to disassemble as real malware —
packing, anti-disassembly, obfuscation, headerless layout — but wrap **benign payloads** and ship
the exact instruction-level ground truth. It's the infrastructure behind the adversarial paper's
accuracy and calibration measurements: you can only have unimpeachable ground truth for something
you produced, so we produce the corpus and bake the labels in.

See `../../docs/adversarial_corpus_design.md` for the full design.

## Principles

- **GT is perfect or it isn't GT.** Labels come from the production process (compiler/assembler
  output, format metadata, transform logs) — never a disassembler. Every artifact records *why* its
  labels are true and `gauntlet validate` re-checks integrity.
- **Benign payload, adversarial structure.** No functional malware, ever. The hardness is purely
  structural — which is also *required* for ground truth (real malware has none).
- **The adversary is the literature's, not ours.** Real tools and published techniques (desync-cc,
  Tigress, OLLVM, UPX), so the corpus can't be dismissed as "obfuscations you built to beat."

## Generators (current)

| id | bucket | status |
|---|---|---|
| `native-code-in-data` | A — layout/encoding | working (native) |
| `native-headerless` | A — layout/encoding | working (native) |
| `desync-cc` | B — decode-confusion | availability wired; log→GT integration point open |

## Usage

```sh
gauntlet list
gauntlet generate --seed-elf path/to.elf --seed-gt path/to.gt --out corpus/
gauntlet validate corpus/
```

Each artifact emits four sidecars: `<name>.elf`, `<name>.gt` (true-instruction log),
`<name>.regions` (typed two-sided spans), `<name>.manifest.json` (provenance).

## Status

Experimental (`experimental/gauntlet`). Built to `crates/` quality so promotion is a `git mv`; the
promotion gate swaps `anyhow` for `thiserror` typed errors and FNV-1a hashes for SHA-256.
