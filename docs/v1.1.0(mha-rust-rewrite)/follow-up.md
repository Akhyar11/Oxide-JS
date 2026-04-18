# Follow-up

## Risiko
- Paralelisasi per-head mengasumsikan layout tensor `[units, totalCols]` row-major yang konsisten.
- Perbedaan urutan operasi floating-point bisa menimbulkan deviasi numerik kecil vs fallback.

## Tradeoff
- Struktur kode lebih kompleks dari versi block-return karena fokus direct-write.
- Menyimpan baseline snapshot menambah file maintenance, namun memudahkan audit dan rollback.

## Rekomendasi lanjutan
- Tambahkan benchmark otomatis terstandar (median/p95 over multiple runs).
- Evaluasi vectorization lanjut pada loop dot-product kecil (head dimension).
- Pertimbangkan validasi dimensi input/output lebih ketat di boundary NAPI.

## Lokasi baseline lama
- `src-rust/src/baseline/mha_kernel_v1.rs`

## Lokasi test baru
- `test/mha_rust_regression.ts`
- `test/mha_rust_perf.ts`
