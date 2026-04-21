# Benchmark Sintetis

Dokumentasi ini menyimpan riwayat benchmark sintetis per versi agar setiap peningkatan performa bisa dibandingkan terhadap baseline yang jelas.

## Tujuan

- Menyediakan baseline performa untuk setiap versi.
- Menyimpan konteks pengukuran agar hasil antar versi bisa dibandingkan dengan adil.
- Mencatat benchmark yang berhasil, benchmark yang gagal, dan catatan interpretasinya.

## Struktur

- `README.md`: index utama dan aturan pengisian.
- `TEMPLATE.md`: template untuk entri versi baru.
- `v<version>.md`: snapshot benchmark untuk satu versi tertentu.

## Aturan Pengisian

1. Buat satu file baru untuk setiap versi.
2. Gunakan format dari `TEMPLATE.md` agar konsisten.
3. Isi metadata minimum:
   - tanggal benchmark
   - versi aplikasi
   - commit acuan
   - environment singkat
   - ukuran training data / corpus yang dipakai
   - CPU
   - RAM
   - OS / kernel
   - versi Node.js
4. Catat command yang dipakai untuk menjalankan benchmark.
5. Jika ada benchmark gagal, tetap tulis hasilnya sebagai `failed` beserta error ringkas.
6. Jangan menimpa file versi lama. Riwayat harus append-only.

## Environment Referensi Saat Ini

Environment berikut adalah mesin yang dipakai untuk baseline `v1.0.0`:

| Komponen | Nilai |
| --- | --- |
| OS | `CachyOS` |
| Kernel | `Linux 6.19.10-1-cachyos` |
| Arsitektur | `x86_64` |
| CPU | `11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz` |
| Core / Thread | `4 core / 8 thread` |
| RAM | `15 GiB` |
| Swap | `15 GiB` |
| Node.js | `v25.8.2` |
| npm | `11.12.1` |
| Rust | `rustc 1.94.1 (2026-03-25)` |

Catatan:

- Jika benchmark versi baru dijalankan di mesin berbeda, metadata environment wajib diperbarui di file versi tersebut.
- Perbedaan CPU governor, thermal state, dan backend native dapat memengaruhi angka benchmark secara signifikan.

## Daftar Versi

| Versi | Tanggal | Commit | Ringkasan |
| --- | --- | --- | --- |
| [v1.0.0](./v1.0.0.md) | 2026-04-17 | `47e7734` | Baseline awal synthetic benchmark untuk versi sekarang |
| [v1.1.0](./v1.1.0.md) | 2026-04-18 | `b2ff012` | Snapshot benchmark versi sekarang dengan aset vocab `dataset/math_vocab.json` |
| [v1.1.6](./v1.1.6.md) | 2026-04-21 | `78bd441` | Konsolidasi test menjadi satu baseline synthetic benchmark dan snapshot hasil terbaru |

## Cara Menambah Versi Baru

1. Salin `TEMPLATE.md` menjadi file versi baru, misalnya `v1.1.0.md`.
2. Jalankan benchmark sintetis yang ingin dijadikan acuan.
3. Isi hasil, status, dan catatan interpretasi.
4. Tambahkan entri baru ke tabel pada file ini.

## Catatan Pembacaan

- Hasil benchmark hanya valid jika konfigurasi model dan environment pengujian dicatat dengan jelas.
- Ukuran data training sebaiknya ditulis minimal dalam bentuk jumlah record dan ukuran korpus efektif yang benar-benar dipakai benchmark.
- Perbandingan antar versi sebaiknya fokus pada benchmark yang sama, command yang sama, dan kondisi backend yang sama.
- Benchmark yang gagal tetap penting karena bisa menunjukkan regresi, mismatch konfigurasi, atau masalah validitas harness.

## Versioning
Versi aktif proyek saat ini adalah `1.1.6`.

Proyek ini memakai format versi `MAJOR.MINOR.PATCH` seperti `1.1.6`.

- Angka paling depan (`MAJOR`): perubahan besar yang biasanya membawa breaking change atau perubahan arsitektur utama.
- Angka tengah (`MINOR`): penambahan fitur baru atau peningkatan yang tetap kompatibel dengan versi sebelumnya.
- Angka paling belakang (`PATCH`): perbaikan bug, optimasi kecil, cleanup, atau perubahan minor yang tidak mengubah API utama.

Contoh:
- `1.1.6`: rilis mayor `1`, fitur set kedua (`1`), dengan 6 patch/perbaikan internal (`6`).
- `1.1.4`: masih di mayor `1` dan minor `1`, tetapi sudah ada 4 patch/perbaikan kecil dari baseline `1.1.0`.
