# Best Practices

## Pemilihan model
- `Sequential`: tabular, regresi, klasifikasi sederhana.
- `Transformers`: task sequence/token dan next-token prediction.

## Pemilihan seqLen
- Mulai dari panjang konteks yang realistis terhadap data.
- seqLen terlalu besar menaikkan biaya MHA kuadratik (`O(seqLen^2)`).

## Pemilihan batch size
- Gunakan batch kecil jika RAM terbatas.
- Untuk pipeline worker-thread, batch lebih besar memberi speedup lebih baik.

## Pad token handling
- Gunakan pad token konsisten di tokenizer (`getPadId`) dan model (`padTokenId`).
- Jangan gunakan nilai pad sebagai target token valid kecuali memang diinginkan.

## Native backend
- Build native untuk training serius.
- Saat debug mismatch numerik, bandingkan mode native vs fallback.

## Hindari shape mismatch
- Catat shape setiap layer boundary.
- Validasi dimensi sebelum `dotProduct`.
- Gunakan `summary()` untuk model sequential.

## Debug loss tidak turun
- Kecilkan `alpha`.
- Pastikan loss dan target cocok (sparse vs dense).
- Cek token index out-of-range pada embedding.
- Cek data preprocessing (padding/truncation).

## Save/load
- Simpan model dan tokenizer bersama.
- Setelah load, jalankan satu forward smoke test.

## Struktur project pengguna
Direkomendasikan:
```text
my-app/
  data/
  checkpoints/
  scripts/
    train.ts
    infer.ts
  src/
```
