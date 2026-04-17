# CPU Training Optimization Design

**Date:** 2026-04-17  
**Target:** Turunkan training transformer CPU-only dari kisaran jam ke sedekat mungkin dengan `< 10 menit / epoch` pada dataset Wikipedia yang digunakan user.

## Context

Framework saat ini adalah custom ML stack berbasis TypeScript + Rust via N-API. Training utama berjalan lewat [project/generative-bot/main.ts](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/project/generative-bot/main.ts:1) dengan konfigurasi default:

- `seqLen = 128`
- `units = 64`
- `heads = 8`
- `batchSize = 64`
- `optimizer = adam`
- `loss = softmaxCrossEntropy`
- `vocab ≈ 20k`

Data Wikipedia lokal yang diperiksa ada di [dataset/wikipedia_belum_normalisasi.txt](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/dataset/wikipedia_belum_normalisasi.txt:1) dengan ukuran sekitar:

- `7,639` baris non-kosong
- `873,837` karakter non-kosong
- estimasi kasar upper bound next-token pairs berbasis karakter: `~866k`

Benchmark sintetis runtime saat ini untuk `forward + backward` pada konfigurasi di atas menunjukkan throughput sekitar:

- `~506 ms / batch`
- `~1.98 batch / detik`
- `~126 sample / detik`

Dengan estimasi pair Wikipedia yang besar, durasi `~2 jam / epoch` konsisten dengan throughput engine saat ini. Artinya masalah utamanya bukan ekspektasi user yang salah, tetapi implementasi compute path yang memang belum cukup efisien untuk target CPU-only agresif.

## Goal

Capai peningkatan throughput training CPU secara drastis dengan prioritas:

1. menurunkan `ms / batch`
2. menaikkan utilisasi multicore CPU secara nyata
3. mengurangi overhead JS ↔ Rust
4. mengurangi alokasi dan GC
5. menjaga hasil training tetap numerik stabil

## Non-Goals

- Tidak mendesain backend GPU/CUDA dalam fase ini
- Tidak mengganti arsitektur model secara fundamental
- Tidak memprioritaskan pipeline/worker JS sampai hot kernel CPU efisien
- Tidak melakukan refactor kosmetik di luar jalur panas training

## Findings

### 1. Hot path masih campur `Float32` dan `Float64`

Beberapa operasi dan buffer runtime masih mengalokasikan `Float32Array` walaupun storage `Matrix` sudah `Float32Array`. Ini menambah bandwidth memori, cache pressure, dan risiko konversi implisit.

Contoh lokasi:

- [src/layers/layerNormalization.ts](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/src/layers/layerNormalization.ts:1)
- [src/math/add.ts](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/src/math/add.ts:1)
- [src/math/dotProduct.ts](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/src/math/dotProduct.ts:1)
- [src/math/sumAxis.ts](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/src/math/sumAxis.ts:1)
- [src/cost/softmaxCrossEntropy.ts](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/src/cost/softmaxCrossEntropy.ts:1)

### 2. Attention menghasilkan object churn besar

[src/layers/multiHeadAttention.ts](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/src/layers/multiHeadAttention.ts:1) membuat banyak `Matrix.fromFlat(...)` dalam loop `head × batch × forward/backward`. Ini mahal di V8, mengganggu locality, dan mengurangi manfaat native backend karena crossing tetap dipecah menjadi operasi kecil.

### 3. Buffer reuse belum menyeluruh

[src/models/transformers.ts](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/src/models/transformers.ts:1) masih mengalokasikan buffer error sequence baru per backward. Beberapa layer lain juga resize/alokasi pada jalur training reguler.

### 4. Profiling terlalu kasar

[src/utils/profiler.ts](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/src/utils/profiler.ts:1) hanya cukup untuk `Forward` vs `Backward`. Belum ada pembacaan per layer, per kernel, `p95`, counter resize buffer, atau statistik binding native.

### 5. JS ↔ Rust crossing masih terlalu sering untuk operasi kecil

Backend Rust sudah dipakai untuk beberapa primitive, tetapi belum cukup “gemuk”. Banyak operasi kecil masih dilakukan terpisah sehingga overhead N-API tetap terasa.

## Optimization Strategy

### Phase 1: Instrumentation and Quick Wins

Tujuan fase ini adalah menghilangkan overhead yang paling murah dibersihkan dan menghasilkan baseline yang bisa dipercaya.

Perubahan:

- pasang profiler scoped per batch, per layer, per kernel
- samakan hot path ke `Float32`
- ubah seluruh buffer training yang masih membuat array baru per batch menjadi reusable
- tambahkan benchmark tetap untuk:
  - synthetic training step
  - subset Wikipedia yang fixed
  - full epoch estimate

Target hasil:

- angka `ms / batch`, `samples / sec`, `tokens / sec` yang repeatable
- penurunan GC dan alokasi transient
- baseline hotspot yang jelas sebelum rewrite native

### Phase 2: Native CPU Kernel Consolidation

Tujuan fase ini adalah memindahkan biaya terbesar ke Rust dengan jumlah crossing minimum.

Perubahan:

- fused `softmax + crossentropy` forward/backward native
- fused `layernorm` forward/backward native
- `adam` native update yang full in-place dan reusable
- attention score/value path native batched
- native instrumentation untuk hit count dan total durasi kernel

Target hasil:

- `ms / batch` turun besar tanpa mengubah API model tingkat atas
- utilisasi CPU naik karena kerja berat pindah ke Rust dan `rayon`

### Phase 3: Attention and Parallelism Rewrite

Tujuan fase ini adalah menyasar bottleneck compute utama dan memaksimalkan multicore CPU.

Perubahan:

- rewrite `MultiHeadAttention` supaya compute berjalan per batch/head dalam kernel batched, bukan view-object per sample
- parallelism `rayon` untuk softmax, layernorm, embedding backward, elementwise ops besar, dan attention loops
- evaluasi blocking/tiling pada jalur GEMM-sensitive
- micro-batching atau pipeline JS hanya dipakai jika setelah ini CPU masih underutilized

Target hasil:

- utilisasi multicore stabil tinggi
- throughput cukup untuk mendekati atau menembus `< 10 menit / epoch`, tergantung jumlah pair final dataset dan spesifikasi CPU user

## File Responsibilities

### TypeScript

- [project/generative-bot/main.ts](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/project/generative-bot/main.ts:1)
  - benchmark harness, epoch logging, profiler integration, dataset accounting
- [src/utils/profiler.ts](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/src/utils/profiler.ts:1)
  - scoped timing, summary, percentiles, resize/allocation counters
- [src/models/transformers.ts](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/src/models/transformers.ts:1)
  - full buffer reuse, last-token extraction mapping, training step instrumentation
- [src/layers/multiHeadAttention.ts](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/src/layers/multiHeadAttention.ts:1)
  - remove object churn, route batched attention ke native
- [src/layers/layerNormalization.ts](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/src/layers/layerNormalization.ts:1)
  - full `Float32`, native fused path, reusable buffers
- [src/optimizer/adam.ts](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/src/optimizer/adam.ts:1)
  - native-first in-place updates
- [src/math/*.ts](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/src/math)
  - dtype cleanup, out-buffer consistency
- [src/math/rust_backend.ts](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/src/math/rust_backend.ts:1)
  - expose new fused kernels and counters

### Rust

- [src-rust/src/lib.rs](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/src-rust/src/lib.rs:1)
  - fused kernels
  - batch/head attention kernels
  - `rayon` parallel loops
  - optional native perf counters

## Benchmark Plan

### Baseline

Jalankan sebelum perubahan:

1. synthetic benchmark:
   - config: `seqLen=128`, `units=64`, `heads=8`, `batch=64`
   - metric: `ms/batch`, `batches/sec`, `samples/sec`
2. dataset benchmark:
   - subset tetap dari Wikipedia
   - metric: `ms/batch`, `tokens/sec`, `epoch estimate`
3. profiler snapshot:
   - 20 batch pertama
   - top 10 operasi berdasarkan total waktu

### After Each Phase

Ulangi benchmark yang sama dan simpan:

- `ms/batch`
- `samples/sec`
- `tokens/sec`
- `CPU utilization`
- top hotspot
- resize/allocation counts

### Success Thresholds

- Phase 1: `>= 1.5x` speedup
- Phase 2: `>= 2x` additional speedup pada hot path attention/loss/norm
- Phase 3: utilisasi multicore tinggi dan total epoch time mendekati target operasional user

## Risks

### Numeric Stability

Migrasi penuh ke `Float32` bisa mengubah hasil training sedikit. Risiko ini diterima karena targetnya CPU throughput, tetapi perlu validasi loss curve dan tidak boleh memecah training sepenuhnya.

### Indexing Bugs in Attention

Rewrite batched attention adalah bagian paling rawan salah indeks. Ini harus dibungkus benchmark dan regression checks kecil sebelum dipakai untuk full epoch.

### False Parallelism

Menambah worker JS terlalu dini berisiko menambah overhead sinkronisasi tanpa memperbaiki kernel inti. Karena itu paralelisme utama ditempatkan di Rust `rayon`.

### Dirty Worktree

Repo saat ini sudah memiliki perubahan lokal yang tidak saya buat. Implementasi harus berhati-hati agar tidak menimpa perubahan user yang sudah ada.

## Testing Strategy

- synthetic step benchmark untuk membandingkan `forward + backward`
- regression benchmark untuk subset Wikipedia tetap
- validation check:
  - loss tidak `NaN`
  - shape tetap benar
  - training tetap berjalan beberapa batch dan beberapa epoch
- optional comparison:
  - loss awal vs loss setelah beberapa batch pada build lama dan baru

## Recommendation

Mulai dari Phase 1 dan langsung lanjut ke Phase 2 tanpa jeda panjang. Quick wins saja tidak akan cukup untuk target `< 10 menit / epoch`, tetapi quick wins wajib agar rewrite native berikutnya diukur dengan benar dan tidak tertutup noise GC/alokasi.

Prioritas implementasi pertama:

1. profiler granular
2. full `Float32` hot path
3. buffer reuse di transformer/layernorm/dense
4. rewrite `MultiHeadAttention` untuk menghilangkan `Matrix.fromFlat` churn
5. fused native kernels dengan `rayon`
