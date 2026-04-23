# Benchmark Sintetis `vX.Y.Z`

## Metadata

- Tanggal:
- Versi:
- Commit:
- Tujuan snapshot:

## Environment

- OS:
- Kernel:
- Arsitektur:
- CPU:
- Core / Thread:
- RAM:
- Swap:
- npm:
- Rust:
- Runtime Node.js:
- Backend native aktif:
- Catatan hardware:

## Training Data

- Dataset path:
- Sumber data:
- Jumlah record raw:
- Jumlah record valid:
- Ukuran file dataset:
- Ukuran corpus efektif:
- Ukuran tokenized corpus:
- Subset yang dipakai benchmark:
- Catatan preprocessing:

## Daftar Benchmark

| Nama | File | Command | Status |
| --- | --- | --- | --- |
| training_step | `test/benchmark_training_step.ts` | `node -r ts-node/register/transpile-only test/benchmark_training_step.ts` | pending |
| transformer_perf_breakdown | `test/benchmark/transformer_perf_breakdown.ts` | `node -r ts-node/register test/benchmark/transformer_perf_breakdown.ts` | pending |

## Hasil

### 1. `training_step`

- Status:
- Ukuran data training:
- Metrik utama:
- Output ringkas:

### 2. `nama_benchmark_lain`

- Status:
- Ukuran data training:
- Metrik utama:
- Output ringkas:

### 3. `stage_profile` atau `perf_breakdown`

- Status:
- Konfigurasi:
- Metrik utama:
  - `inferenceOnlyMsPerIter` bila ada benchmark generation/inference
  - `forwardOnlyMsPerIter`
  - `backwardOnlyMsPerIter`
  - `trainingStepMsPerIter`
  - stage profile rata-rata
- Output ringkas:

## Benchmark Gagal

- Nama:
- Penyebab singkat:
- Error ringkas:

## Perubahan Penting Dibanding Versi Sebelumnya

- Belum diisi.

## Interpretasi

- Ringkas apa arti angka-angka di atas.
- Catat benchmark mana yang paling cocok dijadikan acuan utama.
- Catat jika ada benchmark yang belum stabil atau masih perlu diperbaiki.

## Tindak Lanjut

- Tambahkan benchmark baru jika ada area performa baru yang penting.
- Pertahankan command dan konfigurasi benchmark agar histori tetap comparable.

## Versioning
Versi aktif proyek saat ini adalah `2.1.0`.

Proyek ini memakai format versi `MAJOR.MINOR.PATCH` seperti `2.1.0`.

- Angka paling depan (`MAJOR`): perubahan besar yang biasanya membawa breaking change atau perubahan arsitektur utama.
- Angka tengah (`MINOR`): penambahan fitur baru atau peningkatan yang tetap kompatibel dengan versi sebelumnya.
- Angka paling belakang (`PATCH`): perbaikan bug, optimasi kecil, cleanup, atau perubahan minor yang tidak mengubah API utama.

Contoh:
- `2.1.0`: rilis mayor `2`, minor `1`, patch `0` untuk fitur/kapabilitas baru yang tetap kompatibel.
- `2.0.2`: masih di mayor `2` dan minor `0`, dengan patch kedua (`2`) untuk optimasi internal tanpa perubahan API.
