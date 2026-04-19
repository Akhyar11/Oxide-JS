# Testing

## Test baru/diupdate
- `test/layernorm_rust_correctness.ts`
  - forward shape kecil
  - backward shape kecil
  - validasi finite (tidak NaN/Inf)
  - compare native vs fallback (forward, `dx`, `dGamma`, `dBeta`, update parameter)
- `test/layernorm_rust_regression.ts`
  - regression nilai numerik deterministik untuk input kecil tetap
- `test/layernorm_rust_perf.ts`
  - benchmark sederhana native vs fallback

## Cara menjalankan
- `npx napi build --platform --release --cargo-cwd src-rust`
- `npm run test:layernorm-rust`
- `npm run test:layernorm-rust:perf`

## Verifikasi tambahan
- Jalankan `npm test` untuk cek regresi umum project.
- Catat bahwa ada satu kegagalan baseline lama yang tidak terkait perubahan LayerNorm (`log(e)=1` toleransi ketat).
