# @oxide-js/spiking

## 1.3.0

### Minor Changes

- 1f6489c: - Added `SpikingDenseBPTT` temporal pooler with Sequence-as-Time dynamics.
  - Implemented Spike-Count Accumulation with L2 Normalization to solve membrane saturation in long sequences.
  - Updated documentation for SpikingDenseBPTT in API layers.

## 1.2.0

### Minor Changes

- f0b623b: Refactored Native Rust Backend to implement Rayon parallelism and added Heterogeneous Neuron Dynamics

  - **Parallel Computing**: Integrated `rayon` into the native Rust backend to parallelize operations (`lifStep`, `surrogateMask`, `dotProduct`, `deltaUpdates`).
  - **Heterogeneous Neuron Dynamics**: Decoupled `beta` (Leakage) and `threshold` scalars into random per-neuron array buffers for model stochasticity.
  - **Improved Mathematical Stability**: Enforced membrane potential clamping at `1.0` and strict weight/bias clipping within the `[-1.0, 1.0]` range to prevent gradient explosions during Hebbian Updates.
  - **Native Data Races Handled**: Safely extracted slice pointers before processing `Float32Array` buffers with Rayon to guarantee thread-safe (Send + Sync) execution across parallel units.

## 1.1.0

### Minor Changes

- 7db4802: Spiking Neural Networks (SNN) & JSON Compression

  - Added new `@oxide-js/spiking` package for Biomimetic AI, featuring LIF dynamics and Hebbian Contrastive Learning.
  - Optimized `BaseModel.serialize()` and `setWeights()` in `@oxide-js/models` to compress Float32Array into regular arrays, achieving up to 75% JSON size reductions.
  - Added `mj.argmax` and `mj.threshold` primitives in `@oxide-js/core`.
  - Included `"threshold"` activation typing inside `ActivationType`.
