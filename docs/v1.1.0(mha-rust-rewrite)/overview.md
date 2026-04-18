# Overview

## Ringkasan masalah awal
Kernel native Rust untuk Multi-Head Attention (MHA) sebelumnya benar secara matematis, tetapi memakai pola alokasi temporer besar per block (`Vec<f32>` output/attention/grad) lalu dikumpulkan (`collect`) dan disalin lagi ke output akhir. Pola ini meningkatkan tekanan allocator dan cache CPU.

## Tujuan improvement
- Menjaga ekivalensi matematis forward/backward MHA.
- Mengurangi alokasi temporer besar dan copy berulang.
- Menjaga keamanan paralelisasi (tanpa data race).
- Menjaga kompatibilitas API native TypeScript yang sudah ada.

## Ruang lingkup
Perubahan difokuskan pada:
- `src-rust/src/lib.rs` (kernel MHA native forward/backward)
- `src/layers/multiHeadAttention.ts` (penghapusan zero-fill ganda pada path native)
- `test/mha_rust_regression.ts` dan `test/mha_rust_perf.ts`
- baseline snapshot lama di `src-rust/src/baseline/mha_kernel_v1.rs`
