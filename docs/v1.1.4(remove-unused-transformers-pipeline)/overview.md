# Overview

Dokumentasi ini mencatat cleanup area `transformers pipeline` yang tidak dipakai pada jalur aktif library.

## Ringkasan audit
- Ditemukan implementasi pipeline di `src/pipeline/transformer-pipeline.ts`, `src/pipeline/training-worker.ts`, dan `src/pipeline/pipeline-worker.ts`.
- Tidak ditemukan import/usage aktif dari area pipeline di jalur utama (`src/*` entry aktif dan `test/test.ts`).
- Tidak ada export/barrel resmi yang mengekspos pipeline sebagai API utama library.
- Ditemukan dokumentasi root yang memberi kesan pipeline aktif (`README.md`, `PIPELINE_GUIDE.md`).
- Tidak ada test utama yang bergantung pada pipeline.

## Keputusan
Dipilih **hapus penuh** (bukan placeholder), karena pipeline tidak dipakai, tidak terhubung ke jalur aktif, dan tidak memiliki kontrak API publik yang harus dipertahankan.
