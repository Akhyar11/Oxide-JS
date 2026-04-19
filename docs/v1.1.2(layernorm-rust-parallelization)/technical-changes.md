# Technical Changes

## Audit kernel lama
- Forward serial di loop kolom: hitung mean/var/std lalu normalize + affine.
- Backward serial:
  - `dGamma`/`dBeta` dihitung per-row.
  - `dx` dihitung per-kolom dengan reduksi `sum1/sum2`.

## Perubahan implementasi Rust

### 1) `layer_norm_native_into` diparalelkan
- Tahap statistik kolom (`mean`, `std`) diparalelkan memakai:
  - `out_means.par_iter_mut().zip(out_stds.par_iter_mut()).enumerate()`
- Tahap tulis output (`out_norm`, `out_res`) diparalelkan per-row memakai:
  - `out_norm.par_chunks_mut(cols).zip(out_res.par_chunks_mut(cols)).enumerate()`
- Strategi ini menghindari write overlap dan tetap menjaga formula LayerNorm yang sama.

### 2) `layer_norm_backward_native_into` diparalelkan
- `dGamma`/`dBeta` diparalelkan aman per-row:
  - tiap worker menulis indeks row eksklusif (`d_gamma_out[i]`, `d_beta_out[i]`), tanpa shared mutable write.
- Reduksi untuk komponen `dx` diparalelkan per-kolom:
  - menghitung `sum1_cols[j]` dan `sum2_cols[j]` pada buffer kolom.
- Perhitungan akhir `dx` diparalelkan per-row dengan membaca buffer reduksi kolom.

### 3) Zero-fill redundant
- Di `LayerNormalization.backward`, zero-fill `dGamma`, `dBeta`, `dx` dihapus.
- Aman dihapus karena kernel native dan fallback JS sama-sama overwrite seluruh elemen buffer output.

## API compatibility
- API native NAPI tetap sama (`layerNormNativeInto`, `layerNormBackwardNativeInto`).
- Wrapper TypeScript tetap kompatibel, tanpa perubahan signature publik.
