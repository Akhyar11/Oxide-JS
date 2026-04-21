# Copilot Instructions for ML-V2

## Build, test, and lint commands

```bash
# install deps
npm install

# build native addon (release/debug)
npm run build:rust
npm run build:rust:debug

# full test/benchmark entrypoint configured in package.json
npm test

# run the benchmark file directly (single test file in this repo)
node -r ts-node/register test/benchmark/synthetic_baseline_benchmark.ts
```

There is currently no dedicated lint script in `package.json`.

## High-level architecture

- This repository is a TypeScript ML library with a Rust N-API acceleration layer.
- `src/matrix` is the core data container (`Matrix`) using flat `Float32Array` storage; most higher-level code passes `Matrix` objects end-to-end.
- `src/math` provides numeric primitives (`dotProduct`, `add`, `sumAxis`, etc.) and decides between JS vs native execution via `src/math/rust_backend.ts` (adaptive thresholds + fallback behavior).
- `src/layers` and `src/models` build training/inference abstractions on top of `Matrix` + math primitives. `Sequential` is the base trainer; `Transformers` composes embedding/PE/MHA/FFN and uses `Sequential.fit`.
- Root `index.js` is the auto-generated N-API platform loader/exporter, not the high-level ML API entrypoint. In-repo usage imports from `src/*` (for example `src/math`, `src/models`, `src/tokenizer`).
- The current test pipeline is benchmark-oriented: `npm test` runs `test/benchmark/synthetic_baseline_benchmark.ts`, which trains a small transformer on synthetic samples built from files under `dataset/`.

## Key conventions in this codebase

- **Data layout convention:** tensors are generally `[rows, cols]`, with batch on columns; transformer token batches are `[seqLen, batchSize]`.
- **Target convention for token/class tasks:** sparse targets are represented as shape `[1, batch]`, and output layers commonly use `softmaxCrossEntropy`.
- **Matrix performance convention:** prefer `_data` (flat typed array) for hot paths; `_value` exists for backward compatibility and allocates.
- **Native dispatch convention:** native ops are optional and fallback to JS when unavailable; use `ML_DISABLE_NATIVE=1` to force JS execution for debugging/regression checks.
- **Output-buffer safety:** for `addInto`/`subInto`-style APIs, output buffers must match expected shape and must not alias input buffers.
- **Benchmark documentation convention:** per-version benchmark snapshots in `docs/benchmark-sintetis/` are append-only and should follow `docs/benchmark-sintetis/TEMPLATE.md`.

## Versioning
- `MAJOR`: significant changes, typically including breaking changes or major architectural shifts.
- `MINOR`: new features or improvements that remain backward-compatible.
- `PATCH`: bug fixes, small optimizations, cleanup, or other non-breaking internal improvements.

Examples:
- `1.1.6`: major release `1`, second feature set (`1`), with 6 internal patches/improvements (`6`).
- `1.1.4`: still major `1` and minor `1`, with 4 patch-level improvements from baseline `1.1.0`.

## Project directives

- Every feature update must include the appropriate versioning update based on change scope (MAJOR/MINOR/PATCH).
- Every feature update must review and update `docs/GUIDE-LINE/*` when behavior, API, or workflow is affected.
- Primary contribution goal: evolve this library into a production-ready system (stable, reliable, and maintainable).
