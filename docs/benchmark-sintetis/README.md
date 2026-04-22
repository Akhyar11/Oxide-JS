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

## Correctness Companion

Setiap snapshot benchmark sintetis untuk recurrent sebaiknya didampingi bukti correctness minimum, bukan hanya angka throughput.

Baseline repo saat ini:
- entry suite gabungan: `test/index.ts`
- correctness suite recurrent: `test/correctness/index.ts`
- benchmark suite: `test/benchmark/index.ts`

Command yang disarankan untuk snapshot resmi:

```bash
npm test
```

Jika yang dijalankan hanya benchmark langsung, dokumentasikan juga command correctness yang dipakai, minimal:

```bash
node -r ts-node/register test/correctness/index.ts
```

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
5. Catat juga status correctness companion:
   - command correctness
   - status pass/fail
   - cakupan singkat correctness yang relevan
6. Jika ada benchmark gagal, tetap tulis hasilnya sebagai `failed` beserta error ringkas.
7. Jangan menimpa file versi lama. Riwayat harus append-only.

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
| [v1.2.0](./v1.2.0.md) | 2026-04-21 | `78bd441` | Penambahan benchmark untuk model Recurrent (RNN, LSTM, GRU) |
| [v1.2.1](./v1.2.1.md) | 2026-04-21 | `78bd441` | Entry test tunggal `test/index.ts` dengan correctness suite + benchmark suite |
| [v1.2.2](./v1.2.2.md) | 2026-04-22 | `5a606f9` + local patch | Hardening kontrak `RNN`/`LSTM`/`GRU`, guard recurrent stateful, dan benchmark recurrent yang memproses sample per sample secara valid |
| [v1.2.3](./v1.2.3.md) | 2026-04-22 | `develop-mode` local patch | Refresh benchmark setelah recurrent memakai jalur batch time-major yang valid, bukan loop sample-per-sample |
| [v1.2.4](./v1.2.4.md) | 2026-04-22 | `537905a` + local patch | Refactor transformer ke full-sequence causal LM training dan benchmark ulang workload transformer |
| [v1.3.0](./v1.3.0.md) | 2026-04-22 | `537905a` + local patch | Major release untuk perubahan arsitektur training transformer ke full-sequence causal LM |

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
- Untuk recurrent family, interpretasi benchmark harus selalu dibaca bersama correctness companion agar optimasi throughput tidak menutupi regresi shape/state/save-load.
- Snapshot recurrent lama sebelum `v1.2.3` masih berguna sebagai referensi historis, tetapi tidak lagi fair untuk membandingkan throughput recurrent karena jalur benchmark utamanya masih memproses sample satu per satu di dalam batch efektif.

## Versioning
Versi aktif proyek saat ini adalah `1.3.0`.

Proyek ini memakai format versi `MAJOR.MINOR.PATCH` seperti `1.3.0`.

- Angka paling depan (`MAJOR`): perubahan besar yang biasanya membawa breaking change atau perubahan arsitektur utama.
- Angka tengah (`MINOR`): penambahan fitur baru atau peningkatan yang tetap kompatibel dengan versi sebelumnya.
- Angka paling belakang (`PATCH`): perbaikan bug, optimasi kecil, cleanup, atau perubahan minor yang tidak mengubah API utama.

Contoh:
- `1.3.0`: rilis mayor `1`, minor `3`, patch `0` untuk perubahan arsitektur training transformer.
- `1.1.4`: masih di mayor `1` dan minor `1`, tetapi sudah ada 4 patch/perbaikan kecil dari baseline `1.1.0`.
