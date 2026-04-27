# Changelog

## 2.3.0

### Minor Changes

- 18d9870: fix: resolve frozen gradients in RNN, LSTM, and GRU batched training.
  feat: implement hyper-speed native kernels with parallel pre-projection and optimized memory layout.
  fix: synchronize GRU mathematical architecture with ML-V1's specific reset-gate scaling.

Semua perubahan penting pada proyek **ML-V1** akan didokumentasikan di file ini.

Format ini didasarkan pada [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), dan proyek ini mengikuti [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.8] - 2026-04-27

### Added

- **Full Native Optimizer Support**: Implementasi native Rust untuk semua optimizer (`Adam`, `SGD`, `AdaGrad`, `Momentum`, `NAG`).
- **Sparse Native Updates**: Dukungan update sparse di backend Rust untuk `Embedding` layer dan optimizer, meningkatkan efisiensi memori dan kecepatan pada vocabulary besar.
- **Rayon Parallelism**: Menggunakan `Rayon` untuk paralelisasi update densitas pada optimizer native.

### Changed

- Refactor `Embedding.backward` untuk menggunakan akumulasi gradien sparse secara native.
- Generalisasi deteksi backend native di semua kelas optimizer.
- Update benchmark dan correctness snapshots untuk versi 2.2.8.

## [2.2.7] - 2026-04-25

### Added

- **Unicode-aware Pre-tokenizers**: Menambahkan mode `unicode-grapheme`, `unicode-word`, `whitespace`, dan `script-aware`.
- **Multilingual Support**: Integrasi `Intl.Segmenter` untuk segmentasi teks yang akurat pada berbagai sistem penulisan.
- Tes integrasi multibahasa untuk BPE Tokenizer.

## [2.2.6] - 2026-04-25

### Documentation

- Terjemahan penuh panduan teknis (`GUIDE-LINE`) dari Bahasa Indonesia ke Bahasa Inggris.
- Update API reference documentation.

## [2.2.5] - 2026-04-25

### Added

- Optimasi hot-path training dan validation pada generic `Sequential`.
- Dukungan batching untuk model non-recurrent dalam `validationGeneric`.
- Native masked sparse loss untuk Transformer.

### Changed

- Optimasi lookup `Embedding` menggunakan typed token buffers.
- BPE Tokenizer menggunakan cache encode dan in-place merge.

## [2.2.4] - 2026-04-25

### Added

- API `predictMode` pada model Transformers (`next-token` vs `full-sequence`).

### Changed

- Refactor suite pengujian untuk memisahkan API tests dan learning tests.
- Sinkronisasi dokumentasi offline.

## [2.2.3] - 2026-04-25

### Fixed

- Optimasi training/inference hot path.
- Update snapshot pembelajaran untuk model recurrent dan transformer.

## [2.2.2] - 2026-04-24

### Added

- Root combined suite (`test/index.ts`).
- Benchmark snapshot terbaru untuk semua keluarga model.

## [2.2.0] - 2026-04-24

### Added

- Fitur **Dynamic Padding Trim** (`trimPadding`) untuk efisiensi sequence panjang.
- Dukungan `positionOffset` pada `PositionalEncoding` untuk left-padding.
- Proyek `math-reasoning-ai`.

---

_Daftar ini mencakup rilis terbaru. Untuk histori penuh, silakan cek catatan commit Git._
