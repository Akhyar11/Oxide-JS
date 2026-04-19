# Follow-up Documentation

## Area yang perlu dikembangkan berikutnya
1. Dokumentasi entry point publik package (setelah arsitektur ekspor distabilkan).
2. Cookbook dataset preprocessing lebih banyak (classification, seq2seq, ranking).
3. Dokumen benchmark resmi berbasis metrik reproducible lintas mesin.
4. Panduan migrasi format model jika struktur save/load berubah.

## Area library yang membingungkan saat audit
1. Script `project/math-bot/*` masih terdaftar di `package.json` tetapi folder tidak ada di snapshot repo.
2. `PIPELINE_GUIDE.md` menyebut file benchmark/training tertentu yang tidak ditemukan.
3. `setLayers` belum memuat semua jenis layer (misalnya `MultiHeadAttention`, `Dropout`, `LayerNormalization` hanya parsial tergantung format model).
4. Entry point root package berfokus pada native binding, bukan ekspor library TS high-level.

## Rekomendasi improvement dokumentasi berikutnya
- Tambahkan "API stability policy" (experimental vs stable).
- Tambahkan diagram save/load format per model.
- Tambahkan section troubleshooting khusus `worker_threads` dan environment CI.
- Tambahkan tabel kompatibilitas OS/arch untuk binary native.
