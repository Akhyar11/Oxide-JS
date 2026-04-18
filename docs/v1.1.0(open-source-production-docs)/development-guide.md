# Development Guide (Maintainer)

## Menambah layer baru
1. Implement class layer di `src/layers/` dengan method standar: `forward`, `backward`, `save`, `load`, `compile`, `resetLoss`.
2. Tambahkan export di `src/layers/index.ts`.
3. Jika layer harus bisa di-load dari model file, tambahkan mapping di `src/utils/setLayers.ts`.
4. Tambahkan test minimal (forward/backward/shape).

## Menambah model baru
1. Buat file di `src/models/` (umumnya turunan `Sequential` atau komposer layer).
2. Export di `src/models/index.ts`.
3. Definisikan format `save/load` yang jelas.

## Menambah optimizer
1. Tambahkan class optimizer di `src/optimizer/` dengan method `calculate(grad, alpha)`.
2. Daftarkan di `src/utils/setOptimizer.ts`.
3. Tambahkan tipe optimizer di `src/@types/type.ts`.

## Menambah native op
1. Implement fungsi Rust di `src-rust/src/lib.rs` dengan annotation `#[napi]`.
2. Build ulang `npm run build:rust`.
3. Tambahkan wrapper TS di `src/math/rust_backend.ts`.
4. Integrasikan ke fungsi math/layer dengan fallback JS.
5. Tambahkan regression test native vs fallback.

## Menulis test
- Gunakan `test/*.ts` dengan `npx ts-node`.
- Untuk parity native vs JS, gunakan pola seperti `test/native_training_hotpaths.ts`.

## Menjaga kompatibilitas
- Pertahankan signature method model/layer yang sudah dipakai test.
- Saat ubah format `save/load`, sediakan backward compatibility jika memungkinkan.
- Hindari perubahan shape convention tanpa migrasi dan docs update.
