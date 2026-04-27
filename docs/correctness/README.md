# Correctness Snapshots

This documentation stores correctness snapshots per version so that benchmark changes are always read alongside evidence that the model is still learning and core contracts are not broken.

## Goals

- Store correctness suite results run on specific versions.
- Create a pass/fail history for training paths and sensitive benchmarks.
- Serve as a companion to `docs/benchmark-sintetis`.

## Structure

- `README.md`: main index for correctness snapshots.
- `v<version>.md`: correctness snapshot for a specific version.

## Reference Command

The official repository correctness command is currently:

```bash
node -r ts-node/register test/correctness/index.ts
```

Combined suite command:

```bash
npm test
```

## Filing Rules

1. Create a new version file for each correctness snapshot to be frozen.
2. Record the command used.
3. Record the pass/fail status.
4. Record the relevant suite coverage.
5. If there is a failure, include a brief reason and the affected area.

## Version List

| Version | Date | Commit | Summary |
| --- | --- | --- | --- |
| [v2.2.2](./v2.2.2.md) | 2026-04-24 | `7a0728f` + local patch | Correctness learning suite snapshot for recurrent and transformer, including `trimPad`. |
| [v2.2.3](./v2.2.3.md) | 2026-04-25 | `ac0806c` + local patch | Correctness snapshot after training hot path optimizations for batching/loss/buffer. |
| [v2.2.4](./v2.2.4.md) | 2026-04-25 | `397ed48` + local patch | Correctness snapshot after adding `predictMode` and splitting API vs learning suites. |
| [v2.2.5](./v2.2.5.md) | 2026-04-25 | `ffb55ff` + local patch | Correctness snapshot after optimizations for training/validation hot paths, embedding, and BPE tokenizer. |
| [v2.2.6](#) | 2026-04-25 | `local-docs-trans` | Documented version bump after full English translation of documentation guides and API reference pages. |
| [v2.2.7](#) | 2026-04-25 | `local-tokenizer-unicode` | Unicode-aware BPE pre-tokenizers and multilingual tokenizer correctness coverage. |
| [v2.2.8](./v2.2.8.md) | 2026-04-27 | `local-sparse-native` | Correctness snapshot for Full Native Optimizer support and Sparse Embedding integration. |
| [v2.3.0-alpha.0](./v2.3.0-alpha.0.md) | 2026-04-27 | `develop-mode` | Correctness snapshot for Native Backend and Batching support in RNN, LSTM, and GRU families. |

## Versioning

The current active version of the project is `2.3.0-alpha.0`.

This project uses the `MAJOR.MINOR.PATCH` version format, such as `2.2.7`.

- The first number (`MAJOR`): significant changes that usually bring breaking changes or major architectural shifts.
- The middle number (`MINOR`): addition of new features or improvements that remain compatible with previous versions.
- The last number (`PATCH`): bug fixes, minor optimizations, cleanup, or minor changes that do not alter the main API.
