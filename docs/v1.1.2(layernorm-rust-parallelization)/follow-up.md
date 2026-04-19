# Follow-up

## Risiko
- Urutan operasi floating-point berubah karena paralelisasi, sehingga deviasi kecil numerik vs versi serial tetap mungkin.
- Speedup bergantung pada ukuran tensor; ukuran kecil bisa tidak mendapat manfaat berarti.

## Tradeoff
- Kode kernel menjadi sedikit lebih kompleks karena pemisahan tahap reduksi dan tahap tulis output.
- Backward menambah dua buffer reduksi kolom (`sum1_cols`, `sum2_cols`) untuk menjaga safety dan maintainability.

## Rekomendasi benchmark lanjutan
- Jalankan sweep ukuran `(rows, cols)` dengan beberapa kombinasi `seqLen`, `batch`, `units`.
- Ukur median/p95 dengan warmup untuk mengurangi noise.
- Bandingkan scaling terhadap jumlah core CPU.

## Peluang optimasi berikutnya
- Eksplor vectorization/manual unroll pada loop inner LayerNorm.
- Evaluasi threshold dinamis untuk fallback serial saat ukuran tensor sangat kecil.
