# Benchmark Methodology

## Prinsip
Agar benchmark nyata, terukur, dan repeatable:

- Menggunakan input sintetis deterministik (seeded RNG).
- Setiap skenario memiliki **warmup** dan **measure iterations**.
- Pengukuran tidak one-shot; diambil min/max/avg/total.
- Skenario dibandingkan dengan shape dan beban yang sama.
- Mengumpulkan metrik memory dari `process.memoryUsage()`.
- Jika tersedia, menjalankan `global.gc()` (Node dengan `--expose-gc`) untuk menurunkan noise heap.

## Konfigurasi umum benchmark harness
- Runner: `node --expose-gc -r ts-node/register test/benchmark/synthetic_patch_benchmark.ts`
- Hasil disimpan ke: `benchmark-results/synthetic_patch_benchmark.latest.json`
- Metrik utama:
  - wall-clock total ms
  - average ms/iter
  - min/max ms
  - iter/sec
  - delta heapUsed dan RSS (MB)

## Skenario detail

### 1) Elementwise add/sub
- Bentuk: small, medium, large, transformer-like.
- Perbandingan:
  - `mj.add(a,b)` vs `mj.addInto(a,b,out)`
  - `mj.sub(a,b)` vs `mj.subInto(a,b,out)`
- Tujuan: mendeteksi dampak alokasi object/typed-array baru.

### 2) Residual path
Mensimulasikan pola residual transformer:
- `res1 = h + attn`
- `res2 = res1 + ffn`
- `res1Err = res2Err + errLn2`
- `peErr = res1Err + errLn1`

Dibandingkan:
- allocation-heavy (setiap operasi menghasilkan matrix baru)
- reusable-buffer (semua menggunakan `addInto` + buffer pre-allocated)

### 3) Transformer forward/backward
Konfigurasi representatif:
- `(seqLen=128, batch=8, units=128, heads=8)`
- `(seqLen=256, batch=16, units=128, heads=8)`
- `(seqLen=512, batch=8, units=128, heads=8)`

Dropout diset 0 untuk meminimalkan noise stochastic runtime.

### 4) Full synthetic training step
- Satu iterasi = `forward` + `backward` pada input/label sintetis.
- Mengukur throughput level step.

### 5) Native MHA impact
- Jika native backend tersedia:
  - **current path**: panggil native forward/backward tanpa pre-zero fill.
  - **emulated old path**: `fill(0)` pada semua output buffer sebelum panggilan native.
- Ini memberi estimasi dampak penghapusan zero-fill redundant.
- Tambahan pembanding: JS fallback MHA (native dipaksa off).

## Batasan metodologi
- Tidak mengukur allocation count per-object secara profiler-level (misalnya via V8 allocation timeline).
- Delta memory bersifat indikatif karena ada pengaruh GC dan allocator OS.
- Baseline “sebelum patch” diestimasi lewat emulasi path lama, bukan checkout commit lama secara otomatis.

