# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.2] - 2026-05-22

### Added
- PE/PE32+ binary support in `header.rs`.

### Changed
- **Breaking:** `extract_text_section` signature dropped the `path` argument.

## [0.1.1] - 2026-05-21

### Fixed
- *(superset)* `successors_of` now handles `call` correctly.

### Tests
- *(superset)* Added missing unit tests.

## [0.1.0] - 2026-05-18

### Added
- Initial release: Algorithm 1 from Miller et al. (PLDI 2019).
- Hint extractors: control-flow convergence, crossing, weak CF, register def-use.
- Python bindings via pyo3 behind the `python` feature.
