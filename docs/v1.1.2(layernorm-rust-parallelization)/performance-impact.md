# Performance Impact

## Kenapa LayerNorm layak diparalelkan
- LayerNorm dipanggil berulang di blok Transformer.
- Kompleksitas dominan ada pada loop `rows * cols` untuk forward dan backward.
- Saat `cols = seqLen * batch` besar, pemrosesan serial jadi bottleneck CPU.

## Dampak teknis dari perubahan
- Forward:
  - statistik kolom berjalan paralel.
  - tahap normalize+affine berjalan paralel per-row.
- Backward:
  - grad parameter (`dGamma`, `dBeta`) dihitung paralel per-row.
  - reduksi kolom untuk `dx` dijalankan paralel, lalu `dx` ditulis paralel per-row.

## Hipotesis dampak performa
- Utilisasi multi-core meningkat pada workload LayerNorm menengah-besar.
- Overhead sinkronisasi rendah karena partisi output eksklusif per worker.
- Alokasi tambahan terbatas pada dua buffer kolom (`sum1_cols`, `sum2_cols`) di backward.

## Benchmark awal
- Disediakan benchmark sederhana: `test/layernorm_rust_perf.ts`.
- Benchmark ini membandingkan runtime native vs fallback JS pada ukuran tensor tetap.
