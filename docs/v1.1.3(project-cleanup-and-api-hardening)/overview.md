# v1.1.3 — Project Cleanup and API Hardening (Overview)

## Tujuan
Rilis ini fokus pada cleanup engineering tanpa menambah fitur baru:
- hardening API math rawan misuse,
- pengurangan technical debt state/helper yang tidak dipakai,
- penyederhanaan path buffer internal,
- menjaga perilaku training tetap stabil.

## Ringkasan Perubahan
1. **API hardening `addInto/subInto`**
   - Menambahkan guard eksplisit untuk melarang aliasing buffer `out` dengan input matrix (`a`/`b`) pada operasi matrix-vs-matrix.
   - Error message diperjelas agar kontrak pemakaian jelas.

2. **Cleanup state/helper tidak terpakai**
   - `Transformers`: menghapus field internal warisan yang tidak pernah dibaca.
   - `MultiHeadAttention`: menghapus head-view cache/helper statis yang sudah obsolete dan tidak dipakai alur eksekusi.

3. **Perapihan maintainability**
   - `LayerNormalization`: menghapus duplikasi alokasi buffer forward melalui helper internal tunggal.

4. **Testing & regression**
   - Menambah test guard aliasing untuk `addInto/subInto`.
   - Menjaga test regresi existing untuk residual buffer dan correctness tetap berjalan.

## File Utama yang Diubah
- `src/math/add.ts`
- `src/math/sub.ts`
- `src/models/transformers.ts`
- `src/layers/multiHeadAttention.ts`
- `src/layers/layerNormalization.ts`
- `test/test.ts`
- `test/test_addinto_buffer_reuse.ts`
