# Testing

## File test yang diperbarui
- `test/test.ts`

## Cakupan test tambahan
1. Correctness `addInto` / `subInto`
2. Correctness `add(..., out)` / `sub(..., out)`
3. Verifikasi buffer reuse (`result === out` dan referensi `_data` tetap)
4. Perbandingan hasil legacy path vs path with `out`
5. Transformer forward output shape benar
6. Transformer forward output finite
7. Transformer backward loss finite

## Catatan baseline repo
- Sebelum patch, `npm test` sudah punya 1 kegagalan existing pada test presisi `log(e)=1` (nilai aktual `0.99999994`).
- `tsc --noEmit` baseline juga gagal karena dua import test ke `project/math-bot/main` yang tidak tersedia di environment ini.
- Kegagalan baseline tersebut tidak diperkenalkan oleh patch ini.
