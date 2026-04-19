# Technical Changes

## 1) API math untuk output buffer reuse

### File
- `src/math/add.ts`
- `src/math/sub.ts`
- `src/math/index.ts`

### Perubahan
- `add` dan `sub` sekarang menerima parameter opsional `out?: Matrix`.
- Ditambahkan helper API eksplisit:
  - `addInto(a: Matrix, b: Matrix, out: Matrix)`
  - `subInto(a: Matrix, b: Matrix, out: Matrix)`
- Tetap ada validasi shape input dan validasi shape `out`.
- Native backend tetap digunakan untuk jalur matrix-matrix dan sekarang menulis langsung ke `out` bila disediakan.

## 2) Optimasi residual path Transformer

### File
- `src/models/transformers.ts`

### Perubahan
- Forward residual:
  - `res1`: dari `mj.add(h, xDrop1Out)` menjadi `mj.addInto(h, xDrop1Out, this.xRes1)`
  - `res2`: dari `mj.add(res1, xDrop2Out)` menjadi `mj.addInto(res1, xDrop2Out, this.xRes2)`
- Backward error merge:
  - `res1Err`: dari alokasi baru menjadi `mj.addInto(res2Err, errLn2, this.errRes1Buf)`
  - `peErr`: dari alokasi baru menjadi `mj.addInto(res1Err, errLn1, this.errRes2Buf)`
- Ditambahkan resize guard buffer agar shape tetap benar untuk dynamic batch (`totalTokens`).

## 3) Audit zero-fill dan eliminasi redundant fill

### File
- `src/layers/embedding.ts`

### Perubahan
- `outputData.fill(0)` dipindah agar hanya dilakukan pada fallback JS.
- Pada jalur native, zero-fill TS dihapus karena `embedding_forward_native_into` Rust sudah menginisialisasi output ke nol.

## Catatan kompatibilitas
- API lama tidak dihapus.
- Kode existing yang memanggil `mj.add(a, b)` dan `mj.sub(a, b)` tetap kompatibel.
