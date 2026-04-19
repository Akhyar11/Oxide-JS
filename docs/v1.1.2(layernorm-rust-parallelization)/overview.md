# Overview

## Bottleneck awal
Kernel Rust LayerNorm (`layer_norm_native_into` dan `layer_norm_backward_native_into`) sebelumnya berjalan serial walau komputasinya besar saat `seqLen * batch` meningkat.

## Tujuan improvement
- Meningkatkan pemakaian multi-core CPU untuk forward/backward LayerNorm.
- Menjaga correctness matematis output forward dan gradien backward (`dGamma`, `dBeta`, `dx`).
- Menjaga safety paralelisasi tanpa data race.
- Meminimalkan alokasi tambahan dan menjaga maintainability.

## Ruang lingkup perubahan
- `src-rust/src/lib.rs` (paralelisasi kernel LayerNorm forward/backward).
- `src/layers/layerNormalization.ts` (hapus zero-fill redundant di backward path).
- `test/layernorm_rust_correctness.ts`
- `test/layernorm_rust_regression.ts`
- `test/layernorm_rust_perf.ts`
- `package.json` (script test LayerNorm baru)
