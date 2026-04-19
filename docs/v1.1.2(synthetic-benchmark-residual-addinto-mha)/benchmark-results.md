# Benchmark Results

Run date (UTC): **2026-04-19**

Source data:
- `benchmark-results/synthetic_patch_benchmark.latest.json`

## Hasil utama (ringkas)

- **addInto/subInto** konsisten lebih cepat pada shape medium/large/transformer-like.
- **Residual reusable-buffer** lebih cepat dibanding allocation-heavy.
- **Native MHA no-pre-zero** lebih cepat dibanding emulasi old zero-fill.
- Transformer forward/backward pada ukuran besar tetap mahal secara komputasi; patch lebih terasa pada hotpath elementwise/residual dibanding total wall-time transformer end-to-end.

## Cuplikan hasil numerik

### add/sub (contoh representatif)
- add alloc medium `[128x1024]`: **1.291 ms**
- addInto medium `[128x1024]`: **0.580 ms**
- sub alloc large `[256x4096]`: **2.595 ms**
- subInto large `[256x4096]`: **1.563 ms**

### residual pattern
- allocation-heavy: **7.131 ms/iter**
- reusable-buffer: **4.165 ms/iter**

### native MHA zero-fill impact
- current no pre-zero: **144.863 ms/iter**
- emulated old zero-fill: **152.560 ms/iter**

## Tabel lengkap

> Lihat tabel lengkap pada output console benchmark atau di JSON `markdownTable` field.

## Interpretasi kehati-hatian

- Angka improvement pada microbench tidak otomatis linear pada end-to-end training step.
- Untuk klaim final, jalankan benchmark ini beberapa kali (mis. 3-5 run) dan ambil median.
- Jalankan juga pada mesin target produksi untuk validasi eksternal.

