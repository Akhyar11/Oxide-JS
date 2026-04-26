# Synthetic Benchmark

This documentation stores synthetic benchmark records per version so that every performance improvement can be compared against a clear baseline.

## Goals

- Provide performance baselines for every version.
- Store measurement context so that results across versions can be compared fairly.
- Record successful benchmarks, failed benchmarks, and interpretation notes.

## Structure

- `README.md`: main index and filing rules.
- `TEMPLATE.md`: template for new version entries.
- `v<version>.md`: benchmark snapshot for a specific version.

## Correctness Companion

Every synthetic benchmark snapshot for recurrent models should be accompanied by minimum correctness proof, not just throughput numbers.

Current repository baselines:
- Combined entry suite: `test/index.ts`
- Recurrent correctness suite: `test/correctness/index.ts`
- Benchmark suite: `test/benchmark/index.ts`

Recommended command for official snapshots:

```bash
npm test
```

If only running the benchmark directly, also document the correctness command used, at minimum:

```bash
node -r ts-node/register test/correctness/index.ts
```

## Filing Rules

1. Create a new file for every version.
2. Use the format from `TEMPLATE.md` for consistency.
3. Fill in the minimum metadata:
   - Benchmark date
   - Application version
   - Reference commit
   - Brief environment info
   - Training data size / corpus used
   - CPU
   - RAM
   - OS / kernel
   - Node.js version
4. Record the command used to run the benchmark.
5. Also record the correctness companion status:
   - Correctness command
   - Pass/fail status
   - Brief relevant correctness coverage
6. If a benchmark fails, still record the result as `failed` along with a brief error message.
7. Do not overwrite old version files. History should be append-only.

## Current Reference Environment

The following environment is the machine used for the `v1.0.0` baseline:

| Component | Value |
| --- | --- |
| OS | `CachyOS` |
| Kernel | `Linux 6.19.10-1-cachyos` |
| Architecture | `x86_64` |
| CPU | `11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz` |
| Cores / Threads | `4 core / 8 thread` |
| RAM | `15 GiB` |
| Swap | `15 GiB` |
| Node.js | `v25.8.2` |
| npm | `11.12.1` |
| Rust | `rustc 1.94.1 (2026-03-25)` |

Notes:

- If a new version benchmark is run on a different machine, the environment metadata must be updated in that version file.
- Differences in CPU governor, thermal state, and native backend can significantly affect benchmark numbers.

## Version List

| Version | Date | Commit | Summary |
| --- | --- | --- | --- |
| [v1.0.0](./v1.0.0.md) | 2026-04-17 | `47e7734` | Initial synthetic benchmark baseline for the current version. |
| [v1.1.0](./v1.1.0.md) | 2026-04-18 | `b2ff012` | Benchmark snapshot using the `dataset/math_vocab.json` vocab asset. |
| [v1.1.6](./v1.1.6.md) | 2026-04-21 | `78bd441` | Test consolidation into a single synthetic benchmark baseline and latest results snapshot. |
| [v1.2.0](./v1.2.0.md) | 2026-04-21 | `78bd441` | Addition of benchmarks for Recurrent models (RNN, LSTM, GRU). |
| [v1.2.1](./v1.2.1.md) | 2026-04-21 | `78bd441` | Single entry test `test/index.ts` with correctness suite + benchmark suite. |
| [v1.2.2](./v1.2.2.md) | 2026-04-22 | `5a606f9` + local patch | Hardening of `RNN`/`LSTM`/`GRU` contracts, recurrent stateful guards, and recurrent benchmarks validly processing sample by sample. |
| [v1.2.3](./v1.2.3.md) | 2026-04-22 | `develop-mode` local patch | Benchmark refresh after recurrent models used valid time-major batch paths instead of sample-by-sample loops. |
| [v1.2.4](./v1.2.4.md) | 2026-04-22 | `537905a` + local patch | Refactored transformer to full-sequence causal LM training and benchmarked transformer workload again. |
| [v1.3.0](./v1.3.0.md) | 2026-04-22 | `537905a` + local patch | Major release for transformer training architecture shift to full-sequence causal LM. |
| [v1.3.1](./v1.3.1.md) | 2026-04-22 | `d58d71b` + local patch | Transformer bottleneck audit, apple-to-apple benchmarks, and internal loss-gradient path optimization. |
| [v1.3.2](./v1.3.2.md) | 2026-04-22 | `d58d71b` + local patch | Native masked sparse softmax-cross-entropy kernel, LM-specific inference projector, and inference-only benchmark. |
| [v2.0.0](./v2.0.0.md) | 2026-04-22 | `d58d71b` + local patch | Major update to transformer and native backend for full-sequence training and LM-specific inference. |
| [v2.0.1](./v2.0.1.md) | 2026-04-22 | `18134d6` + local patch | Further optimization of native masked sparse loss kernel with per-token parallelization. |
| [v2.0.2](./v2.0.2.md) | 2026-04-22 | `18134d6` + local patch | Transformer projector optimization by removing linear output copy and accelerating native broadcast bias. |
| [v2.0.3](./v2.0.3.md) | 2026-04-23 | `61dc7d4` + local patch | Blocked native loss kernel optimization and reduced copy overhead in `MHA.backward`. |
| [v2.1.0](./v2.1.0.md) | 2026-04-23 | `eea34f5` + local patch | Additional scaling benchmarks for `numBlocks=2/4/6` and minor release for transformer architecture now supporting multi-block. |
| [v2.2.0](./v2.2.0.md) | 2026-04-24 | `fa33aa0` + local patch | Dynamic padding feature (`trimPadding`), `math-reasoning-ai` project, and v2.2.0 baseline benchmark. |
| [v2.2.1](./v2.2.1.md) | 2026-04-24 | `24f4d55` + local patch | Buffer reuse optimization in the recurrent family and micro benchmark snapshot for `rnn`/`transformers`. |
| [v2.2.2](./v2.2.2.md) | 2026-04-24 | `7a0728f` + local patch | Root combined suite, recurrent/transformer benchmark family, and latest learning correctness snapshot. |
| [v2.2.3](./v2.2.3.md) | 2026-04-25 | `ac0806c` + local patch | Training/inference hot path optimization and model family benchmark refresh after latest performance patches. |
| [v2.2.4](./v2.2.4.md) | 2026-04-25 | `397ed48` + local patch | `predictMode` API ergonomics, docs sync, correctness suite refactor, and benchmark snapshot refresh. |
| [v2.2.5](./v2.2.5.md) | 2026-04-25 | `ffb55ff` + local patch | Training/validation hot path optimization, embedding lookup, and BPE tokenizer training/update optimization. |
| [v2.2.6](#) | 2026-04-25 | `local-docs-trans` | Documentation version: Full English translation of `GUIDE-LINE` modules. No architectural changes. |
| [v2.2.7](#) | 2026-04-25 | `local-tokenizer-unicode` | Unicode-aware tokenizer architecture and multilingual tests. |
| [v2.2.8](./v2.2.8.md) | 2026-04-27 | `local-sparse-native` | Native Sparse Embedding and Full Native Optimizer support for all families. |

## How to Add a New Version

1. Copy `TEMPLATE.md` into a new version file, e.g., `v1.1.0.md`.
2. Run the synthetic benchmark you want to use as a baseline.
3. Fill in the output, status, and interpretation notes.
4. Add the new entry to the table in this file.

## Reading Notes

- Benchmark results are only valid if the model configuration and test environment are clearly recorded.
- Training data size should be written at least in terms of record count and effective corpus size actually used by the benchmark.
- Comparisons between versions should focus on the same benchmarks, same commands, and same backend conditions.
- Failed benchmarks are still important as they can indicate regression, configuration mismatches, or harness validity issues.
- For the recurrent family, benchmark interpretations must always be read alongside the correctness companion so that throughput optimizations do not hide regressions in shape/state/save-load.
- Old recurrent snapshots before `v1.2.3` are still useful as historical references but are no longer fair for comparing recurrent throughput because the primary benchmark path still processed samples one by one within the effective batch.

## Versioning
The current active version of the project is `2.2.8`.

This project uses the `MAJOR.MINOR.PATCH` version format, such as `2.2.7`.

- The first number (`MAJOR`): major changes that usually bring breaking changes or major architectural shifts.
- The middle number (`MINOR`): addition of new features or improvements that remain compatible with previous versions.
- The last number (`PATCH`): bug fixes, minor optimizations, cleanup, or minor changes that do not alter the main API.

Example:
- `2.2.0`: minor release `2` for dynamic padding feature (`trimPadding`) and the `math-reasoning-ai` project.
- `2.2.2`: patch for the root combined suite, model family benchmarks, and learning correctness snapshot.
- `2.2.3`: patch for training/inference hot path optimization and latest benchmark/correctness snapshots.
- `2.2.4`: patch for `predictMode` API ergonomics, docs sync, and correctness suite refactor.
- `2.2.5`: patch for training/validation hot path, embedding, and BPE tokenizer optimization.
- `2.2.6`: documentation patch for full English translation of the core guides.
- `2.2.7`: tokenizer patch for Unicode-aware pre-tokenizers and multilingual BPE integration tests.
- `2.2.8`: optimization patch for Full Native Optimizer support and Sparse Embedding native backend.
