---
"@oxide-js/monorepo": patch
"@oxide-js/layers": patch
---

Fix ESM module resolution errors in CI by migrating test runtime from `ts-node` to `tsx`. 
Stabilize `AdaptiveMemoryRNN` correctness tests by increasing training epochs and learning rate for better convergence in CI environments.
