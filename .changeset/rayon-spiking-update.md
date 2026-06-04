---
"@oxide-js/spiking": minor
---

Refactored Native Rust Backend to implement Rayon parallelism and added Heterogeneous Neuron Dynamics

- **Parallel Computing**: Integrated `rayon` into the native Rust backend to parallelize operations (`lifStep`, `surrogateMask`, `dotProduct`, `deltaUpdates`).
- **Heterogeneous Neuron Dynamics**: Decoupled `beta` (Leakage) and `threshold` scalars into random per-neuron array buffers for model stochasticity.
- **Improved Mathematical Stability**: Enforced membrane potential clamping at `1.0` and strict weight/bias clipping within the `[-1.0, 1.0]` range to prevent gradient explosions during Hebbian Updates.
- **Native Data Races Handled**: Safely extracted slice pointers before processing `Float32Array` buffers with Rayon to guarantee thread-safe (Send + Sync) execution across parallel units.
