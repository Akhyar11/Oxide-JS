# Overview

ML-V1 adalah library machine learning custom berbasis TypeScript dengan akselerasi Rust native.

## Tujuan library
- Menyediakan blok dasar ML yang bisa dipahami dan dimodifikasi end-to-end.
- Mendukung eksperimen model feedforward, attention, dan transformer kecil.
- Menjaga fallback JS agar tetap berjalan meskipun native tidak aktif.

## Filosofi desain
- Transparan terhadap shape dan alur data.
- Operasi numerik dioptimasi lewat `Float32Array` dan buffer reuse.
- Native acceleration dipakai saat tersedia, tanpa mengunci pengguna pada native-only mode.

## Target user
- Developer yang ingin belajar detail implementasi training loop.
- Maintainer yang mengembangkan layer/model/optimizer custom.
- Pengguna yang butuh library ML ringan untuk eksperimen lokal.

## Ruang lingkup
Tercakup:
- matrix & math ops
- activation / loss / optimizer
- layers (dense, attention, embedding, normalization, dll)
- models (`Sequential`, `Transformers`, `DimentionalityReduction`)
- tokenizer BPE
- pipeline worker-thread untuk transformer

Tidak tercakup sebagai API stabil saat ini:
- package entry point high-level dari root package (pemakaian utama via `src/*`)
- framework deployment/serving lengkap
