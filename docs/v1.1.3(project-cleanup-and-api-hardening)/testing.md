# v1.1.3 — Testing

## Baseline Sebelum Perubahan
1. `npm test`
   - Hasil: suite berjalan, terdapat 1 kegagalan existing/non-regresi pada toleransi `log(e)=1` (`0.99999994` vs `1.0`).
2. `npx ts-node test/test_addinto_buffer_reuse.ts`
   - Hasil: lulus.

## Update Test pada Rilis Ini
1. `test/test.ts`
   - Tambah guard test:
     - `addInto` reject aliasing `out===input`
     - `subInto` reject aliasing `out===input`

2. `test/test_addinto_buffer_reuse.ts`
   - Tambah section khusus aliasing guard untuk `addInto/subInto`.

## Verifikasi Pasca Perubahan
Jalankan:
- `npm test`
- `npx ts-node test/test_addinto_buffer_reuse.ts`

Ekspektasi:
- Guard aliasing baru lulus (throw sesuai kontrak).
- Semua regression test lama tetap konsisten, kecuali kegagalan baseline yang memang sudah ada sebelumnya.
