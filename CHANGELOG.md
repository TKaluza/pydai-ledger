# Changelog

## 0.2.0 (2026-11-17)

### Breaking changes

- Ledgers now use format version 2. Open ledgers written by 0.1.x with pydai-ledger 0.1.1.
- In `RunPlan`, replace `attempts_per_case=n` with `repeats=n` to repeat a case under the same conditions. Use `seeds=[...]` to run it with different seeds. Each target runs at one concurrency. Move concurrency sweeps to `benchmark()`.
- Replace `autotune()` and `execute(..., autotune=True)` with `benchmark()`. Pass `result.tuning` to `RunPlan(concurrency=...)`, or read it later with `Ledger.tuning()`.
- Pass report titles, authors, numbers and dates through `header=ReportHeader(...)` in `write_report()` and `write_pdf()`. PDF tables now use a single banded style and the `style` argument has been removed.

### Runs and reports

- Draw a reproducible subset of a casebook with `sample` and `sample_seed`. A run records the seed and repeat of each attempt. Load named targets from TOML with `load_targets()` and use `sequential_targets=True` when targets share hardware.
- Benchmarks write a separate ledger for one target and recommend a concurrency from measured throughput. An interrupted stage can be measured again on resume. Benchmark reports show how latency changes with throughput.
- Validation reports follow the terms of ISO 5725 (setup, precision, trueness, accuracy); they are oriented on the standard and do not implement it. They distinguish same-seed repeatability, agreement across seeds, agreement across targets serving the same model, and matches against expected outputs. Comparisons show the setup of each run and include the cases from all runs. Use `sections` to choose what appears in HTML and PDF reports.
- On resume, a detected change in server version or served models causes attempts from the old server to run again. A complete ledger requires consistent recorded server identities per target. Sealing selects the latest attempt by completion time and includes only binary data referenced by retained attempts.
- HTML reports open light and have a switch to dark; print is always light. Lato replaces Jost as the text font, and Ubuntu Mono comes with its italic.
- `open_bundle()` now returns a context manager. Existing calls that unpack the casebook and ledgers still work.

## 0.1.1 (2026-11-15)

- Documented installation from PyPI with pip and uv.

## 0.1.0 (2026-11-15)

- Initial public release with casebooks, resumable batch runs, signed ledgers and bundles, and HTML and PDF reports.
