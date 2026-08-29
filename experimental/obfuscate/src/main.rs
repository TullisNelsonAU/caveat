//! obfuscate — a GT-emitting binary transformer for the calibration-transfer corpus.
//!
//! The whole point of this tool: produce realistic *adversarial* binaries while keeping
//! exact ground truth, because we built them. desync-cc already proved the model works
//! (it emits a junk-insertion log and the true stream is everything else); this is the
//! same idea, ours, with more techniques, so the corpus is releasable + reproducible.
//!
//! Contract: every transform writes two things side by side —
//!   <out>.elf      the transformed binary
//!   <out>.gt       one hex offset per line: the true instruction-START addresses of the
//!                  stream WE emitted (the exact ground truth, no recovery step to drift)
//!
//! Two GT regimes, and the tool labels which it's producing:
//!   * instruction-preserving transforms (junk insertion, opaque predicates, control-flow
//!     flattening, instruction substitution) keep a real static instruction stream -> full
//!     positive GT -> these carry the *calibration* metrics (ECE/Brier).
//!   * packing / encryption: the payload isn't statically code, so there's no positive GT
//!     for it — but because WE packed a known thing, we DO know it's not-code (negative GT)
//!     and we know the stub's instructions. That two-sided GT is what makes the honest-
//!     uncertainty claim measurable instead of hand-wavy.
//!
//! Planned transforms (each is its own module under src/transforms/ as it lands):
//!   - junk        insert valid-but-dead bytes between real instructions (anti-linear-sweep)
//!   - opaque      opaque-predicate branches around always-taken paths
//!   - flatten     control-flow flattening (dispatcher + state machine)
//!   - subst       instruction substitution (semantically-equal re-encodings)
//!   - pack        compress .text behind a small decoder stub (no-GT-payload regime)
//!   - headerless  strip the ELF container, emit raw .text + a .gt of offsets
//!
//! Status: scaffold. Nothing transforms yet — this pins down the CLI shape and the GT
//! contract so the rest can be filled in transform by transform.

use anyhow::{bail, Result};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "obfuscate (scaffold)\n\
             usage: obfuscate <transform> <in.elf> <out>\n\
             transforms (planned): junk | opaque | flatten | subst | pack | headerless\n\
             writes <out>.elf and <out>.gt (true instruction-start offsets)."
        );
        return Ok(());
    }
    let transform = args[1].as_str();
    match transform {
        "junk" | "opaque" | "flatten" | "subst" | "pack" | "headerless" => {
            bail!("transform '{transform}' not implemented yet — scaffold only");
        }
        other => bail!("unknown transform '{other}'"),
    }
}
