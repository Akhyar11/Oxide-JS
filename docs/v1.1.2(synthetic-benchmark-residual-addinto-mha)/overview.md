# Overview — Synthetic Benchmark for Residual + addInto/subInto + Native MHA Zero-Fill

## Tujuan
Dokumen ini menjelaskan benchmark sintetis untuk memvalidasi patch optimasi pada:

- `Transformers` residual path buffer-reuse.
- API `addInto` / `subInto` dan `add(..., out?)` / `sub(..., out?)`.
- Pengurangan alokasi matrix baru.
- Penghilangan zero-fill redundant pada jalur native MHA.

Patch difokuskan pada file:

- `src/models/transformers.ts`
- `src/math/add.ts`
- `src/math/sub.ts`
- `src/math/index.ts`
- `src/layers/multiHeadAttention.ts`
- `src/math/rust_backend.ts`
- `src-rust/src/lib.rs`

## Ringkasan benchmark yang dibuat
Benchmark baru ada di:

- `test/benchmark/synthetic_patch_benchmark.ts`

Benchmark ini mencakup:

1. **add/sub microbench**: `add` vs `addInto`, `sub` vs `subInto`.
2. **residual path bench**: allocation-heavy vs reusable-buffer.
3. **Transformers forward bench** pada beberapa ukuran `seqLen`/`batch`.
4. **Transformers backward bench** pada konfigurasi yang sama.
5. **Full synthetic training step** (forward + backward).
6. **Native MHA impact**: current behavior vs emulasi old behavior (pre zero-fill).

## Output benchmark
- Console output dengan tabel markdown.
- JSON hasil benchmark: `benchmark-results/synthetic_patch_benchmark.latest.json`.

## Catatan baseline
Repo tidak menyimpan binary baseline sebelum patch sebagai executable benchmark target yang langsung dapat dipanggil. Karena itu, pembanding baseline dibuat dengan pendekatan:

- **Path lama sintetis** (allocation-heavy / emulated old zero-fill).
- **Path baru** (buffer reuse / no redundant zero-fill).

