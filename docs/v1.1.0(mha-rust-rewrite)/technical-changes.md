# Technical Changes

## File yang diubah
- `src-rust/src/lib.rs`
- `src/layers/multiHeadAttention.ts`
- `src-rust/src/baseline/mha_kernel_v1.rs` (baru, baseline)
- `test/mha_rust_regression.ts` (baru)
- `test/mha_rust_perf.ts` (baru)
- `package.json`

## Detail implementasi

### 1) Rewrite forward kernel native
- Mengganti pola `mha_forward_block -> Vec block -> collect -> copy` dengan `mha_forward_head_into`.
- Tiap worker Rayon memproses satu head dan menulis langsung ke:
  - slice unik `out_head`
  - slice unik `attention_head`
- Tidak ada lagi fan-in vector hasil block ke thread utama.

### 2) Rewrite backward kernel native
- Mengganti `mha_backward_block` yang mengembalikan tiga `Vec<f32>` grad block dengan `mha_backward_head_into` yang menulis langsung ke `d_q_head`, `d_k_head`, `d_v_head`.
- Menghapus temporer matriks besar `err_attention[seq_len*seq_len]` dan `err_score[seq_len*seq_len]` per block.
- Diganti scratch lokal kecil `err_attention[seq_len]` yang di-reuse per query.

### 3) Paralelisasi dan safety
- Paralelisasi sekarang berbasis slice per-head memakai `par_chunks_mut(...).zip(...)`.
- Tiap thread menulis ke region output eksklusif (rows milik head tersebut), sehingga tidak ada data race.

### 4) Zero-fill path native
- Di TypeScript (`MultiHeadAttention`), zero-fill buffer output dipindahkan agar hanya terjadi pada fallback path.
- Path native tidak lagi zero-fill di TS karena Rust kernel sudah melakukan zeroing internal.

## Keputusan desain
- API NAPI publik tetap dipertahankan (`multiHeadAttentionForwardNativeInto`, `multiHeadAttentionBackwardNativeInto`).
- Baseline lama disimpan sebagai snapshot terpisah agar mudah dibandingkan/rollback tanpa mempengaruhi jalur runtime.
