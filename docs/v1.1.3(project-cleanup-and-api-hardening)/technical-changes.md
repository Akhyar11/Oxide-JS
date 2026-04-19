# v1.1.3 — Technical Changes

## Prioritas Tinggi

### 1) API safety `addInto/subInto` terhadap aliasing `out`
**Masalah**
- Pada path native matrix-vs-matrix, aliasing `out` ke `a` atau `b` berisiko terhadap correctness (terutama karena kernel native berjalan paralel).

**Perubahan**
- Menambahkan guard aliasing di:
  - `src/math/add.ts`
  - `src/math/sub.ts`
- Guard aktif untuk operasi matrix-vs-matrix dengan `out`.
- Menambahkan helper validasi shape output agar kontrak lebih konsisten.
- Menambahkan dokumentasi kontrak singkat pada `addInto`/`subInto`.

**Dampak**
- Mencegah silent corruption akibat misuse API.
- Perilaku valid tetap kompatibel.

---

### 2) Audit reuse buffer correctness (Transformers/MHA/LayerNorm/math out)
**Temuan audit**
- Reuse residual/error buffer pada `Transformers` tetap valid (tidak ada overwrite sebelum data selesai dipakai).
- `LayerNormalization` buffer reuse aman, namun terdapat duplikasi alokasi code path.

**Perubahan**
- `src/layers/layerNormalization.ts`
  - Menyatukan alokasi forward buffers ke `ensureForwardBuffers(rows, cols)`.
  - Tidak mengubah rumus/urutan komputasi.

**Dampak**
- Maintainability naik, risiko drift antar-path turun.

---

### 3) Audit state internal membingungkan/tidak terpakai
**Temuan**
- `Transformers` menyimpan beberapa state warisan yang tidak dibaca ulang.
- `MultiHeadAttention` menyimpan head views/helper masking yang tidak dipakai alur saat ini.

**Perubahan**
- `src/models/transformers.ts`
  - Hapus field tak terpakai: `xInput`, `xEmb`, `xPe`, `xLn1`, `xLn2`, `lastTokenIndex`.
- `src/layers/multiHeadAttention.ts`
  - Hapus cache/head views tak terpakai.
  - Hapus helper `createHeadViews`, `applyMasks`, `zeroMaskedColumnsInPlace` yang obsolete.

**Dampak**
- Mengurangi technical debt dan kebingungan maintainer.

## Prioritas Sedang

### 4) Konsistensi API math (`add/sub/addInto/subInto`)
- Menyatukan validasi shape output via helper internal pada `add.ts`/`sub.ts`.
- Memperjelas kontrak penggunaan `addInto/subInto` pada komentar fungsi.

### 5) Pola reset/alokasi buffer
- `LayerNormalization` kini memakai satu titik alokasi buffer forward untuk native + fallback.

## Prioritas Kecil

### 6) Perapihan minor tanpa churn besar
- Penyederhanaan `let` → `const` pada variabel temporer yang tidak dimutasi di `Transformers.forward`.

## Area yang Sengaja Tidak Diubah
- Tidak ada perubahan arsitektur besar pada Transformer.
- Tidak menghapus fallback native/non-native yang masih dipakai.
- Tidak mengubah API publik lain secara breaking.
