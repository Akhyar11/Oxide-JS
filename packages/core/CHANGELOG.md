# @oxide-js/core

## 1.1.0

### Minor Changes

- 7db4802: Spiking Neural Networks (SNN) & JSON Compression

  - Added new `@oxide-js/spiking` package for Biomimetic AI, featuring LIF dynamics and Hebbian Contrastive Learning.
  - Optimized `BaseModel.serialize()` and `setWeights()` in `@oxide-js/models` to compress Float32Array into regular arrays, achieving up to 75% JSON size reductions.
  - Added `mj.argmax` and `mj.threshold` primitives in `@oxide-js/core`.
  - Included `"threshold"` activation typing inside `ActivationType`.

## 1.0.0

### Major Changes

- e54932b: Stabilized the Oxide-JS monorepo API, completely modernized the layers architecture, replaced legacy model wrappers (Transformers, RecurrentModel) with modular `Sequential` and `BaseModel` abstractions, and optimized core native/JS math primitives parity.

## 0.4.2

### Patch Changes

- Add scalar Matrix reducer wrappers with autodiff support: `dotSumScalar`, `dotSubScalar`, `dotMulScalar`, `dotDivScalar`, and `normScalar`.
- Preserve existing numeric reducer APIs to avoid breaking runtime behavior.
- Add autodiff regression tests for the new scalar wrappers.

## 0.4.1

### Patch Changes

- Migrate all math/activation ops to engine-side `engine.record(...)` with return-grad backward callbacks.
- Add `Tape.backward(loss, upstreamGrad?)` to support explicit upstream gradients.
- Refine autodiff stability handling by removing unnecessary live-input dependency in backward paths where snapshots are not required.

## 0.4.0

### Minor Changes

- b1d05cb: Fix native MemoryBank function names and add correctness tests

## 0.3.0

### Minor Changes

- 51b34eb: feat: add modular trainer and module abstractions with layer compatibility updates

### Patch Changes

- 9d4fa62: fix: transition to tsx runtime for ESM compatibility and improve AdaptiveMemoryRNN test convergence reliability.

## 0.2.0

### Minor Changes

- 7f5e0a4: feat: add external attention inputs, memory bank access hooks, and autodiff tape fixes

  This release adds external query/key/value injection for multi-head attention,
  external read/write access hooks for memory bank operations, and autodiff fixes
  for scalar math recording, tape snapshot restore, multi-output gradients, and
  loss/result access through `engine.grad()`.

## 0.1.1

### Patch Changes

- 143482d: fix: point package exports to compiled JS files to avoid type-stripping errors in Node.js v25+

## 0.1.0

### Minor Changes

- ef8dd3d: feat: add Keras-style model serialization and Gradient Tape auto-diff support

## 1.0.0

### Major Changes

- aac3b3a: Initial release of Oxide-JS modular monorepo. Transitioned from ML-V1 to Oxide-JS with modular architecture and Rust native acceleration.
