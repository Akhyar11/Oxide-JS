---
"@oxide-js/core": minor
"@oxide-js/layers": minor
"@oxide-js/models": minor
"@oxide-js/spiking": minor
---

Spiking Neural Networks (SNN) & JSON Compression

- Added new `@oxide-js/spiking` package for Biomimetic AI, featuring LIF dynamics and Hebbian Contrastive Learning.
- Optimized `BaseModel.serialize()` and `setWeights()` in `@oxide-js/models` to compress Float32Array into regular arrays, achieving up to 75% JSON size reductions.
- Added `mj.argmax` and `mj.threshold` primitives in `@oxide-js/core`.
- Included `"threshold"` activation typing inside `ActivationType`.
