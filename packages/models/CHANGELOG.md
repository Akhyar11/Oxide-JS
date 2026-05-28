# @oxide-js/models

## 1.1.0

### Minor Changes

- 7db4802: Spiking Neural Networks (SNN) & JSON Compression

  - Added new `@oxide-js/spiking` package for Biomimetic AI, featuring LIF dynamics and Hebbian Contrastive Learning.
  - Optimized `BaseModel.serialize()` and `setWeights()` in `@oxide-js/models` to compress Float32Array into regular arrays, achieving up to 75% JSON size reductions.
  - Added `mj.argmax` and `mj.threshold` primitives in `@oxide-js/core`.
  - Included `"threshold"` activation typing inside `ActivationType`.

### Patch Changes

- Updated dependencies [7db4802]
  - @oxide-js/core@1.1.0
  - @oxide-js/layers@1.1.0

## 1.0.0

### Major Changes

- e54932b: Stabilized the Oxide-JS monorepo API, completely modernized the layers architecture, replaced legacy model wrappers (Transformers, RecurrentModel) with modular `Sequential` and `BaseModel` abstractions, and optimized core native/JS math primitives parity.

### Patch Changes

- Updated dependencies [e54932b]
  - @oxide-js/core@1.0.0
  - @oxide-js/layers@1.0.0
