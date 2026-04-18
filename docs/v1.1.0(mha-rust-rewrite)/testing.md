# Testing

## Test baru
- `test/mha_rust_regression.ts`
  - forward shape kecil
  - backward shape kecil
  - validasi causal masking
  - validasi pad masking
  - output finite (tanpa NaN/Inf)
  - gradient finite (tanpa NaN/Inf)
  - konsistensi native (rewrite) vs fallback baseline pada input kecil
- `test/mha_rust_perf.ts`
  - benchmark sederhana native vs fallback

## Cara menjalankan
- `npm run test:mha-rust`
- `npm run test:mha-rust:perf`

## Verifikasi tambahan yang disarankan
- Jalankan suite test existing project untuk memastikan tidak ada regresi di luar MHA.
- Jalankan benchmark dengan variasi `(units, heads, seqLen, batchSize)` untuk profil performa yang lebih representatif.
