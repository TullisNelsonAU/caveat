//! Probabilistic disassembly for x86-64 binaries.
//!
//! Implements Algorithm 1 from "Probabilistic Disassembly" (Miller et al.,
//! PLDI 2019). Computes a posterior probability for each byte address in
//! `.text` representing the likelihood that it begins a real instruction.

#![warn(missing_docs)] // every pub item needs ///
#![warn(rustdoc::broken_intra_doc_links)] // [Type] links must resolve
#![warn(rustdoc::missing_crate_level_docs)] // forces a //! crate doc
#![warn(rustdoc::invalid_codeblock_attributes)] // typos in ```rust,no_run flags

pub mod analysis;
pub mod header;
pub mod hints;
pub mod superset;

#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::wrap_pyfunction;

pub use analysis::{Analysis, AnalysisConfig, AnalysisMode, CavityStat};
pub use header::extract_text_section;
pub use hints::{
    HintKey, HintLabel, HintPair, extract_all_hints, extract_all_hints_legacy, extract_hint_pairs,
    extract_hint_pairs_with,
};
pub use superset::{Instruction, Superset};

/// Run probabilistic disassembly on the `.text` section of a binary.
pub fn disassemble(bytes: &[u8]) -> anyhow::Result<Vec<(u64, String, f64)>> {
    let (base, code) = extract_text_section(bytes)?;
    let superset = Superset::new(base, code)?;
    let hint_priors = extract_all_hints(&superset);
    let hint_pairs = extract_hint_pairs(&superset);
    let mut analysis = Analysis::new(&superset);
    analysis.run_with_config(&hint_priors, &hint_pairs, &AnalysisConfig::default());

    Ok(analysis
        .sorted_posteriors()
        .into_iter()
        .map(|(addr, posterior)| {
            let instruction = match superset.at(addr) {
                Some(i) if i.op_str.is_empty() => i.mnemonic.clone(),
                Some(i) => format!("{} {}", i.mnemonic, i.op_str),
                None => String::new(),
            };
            (addr, instruction, posterior)
        })
        .collect())
}

// ---- Python bindings ----

/// Python-facing wrapper: same shape as the Rust `disassemble`, with
/// anyhow errors converted to `ValueError`.
///
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "disassemble")]
fn disassemble_py(bytes: &[u8]) -> PyResult<Vec<(u64, String, f64)>> {
    disassemble(bytes).map_err(|e| PyValueError::new_err(format!("{e}")))
}

#[cfg(feature = "python")]
#[pymodule]
fn probdisasm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(disassemble_py, m)?)?;
    Ok(())
}
