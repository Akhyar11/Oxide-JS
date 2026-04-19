# Technical Changes

## File yang dihapus
- `src/pipeline/transformer-pipeline.ts`
- `src/pipeline/training-worker.ts`
- `src/pipeline/pipeline-worker.ts`
- `PIPELINE_GUIDE.md`

## File yang diubah
- `README.md`
  - Menghapus klaim bahwa pipeline worker-thread adalah fitur aktif.
  - Menghapus referensi struktur folder `src/pipeline`.
  - Menghapus catatan performa pipeline yang tidak lagi relevan.
  - Menyesuaikan roadmap agar tidak misleading.

## Dampak teknis
- Tidak ada perubahan API utama yang diekspor oleh jalur aktif (`src/models`, `src/layers`, `src/math`, dll).
- Tidak ada perubahan pada perilaku model `Transformers` utama.
- Cleanup ini menurunkan risiko misinterpretasi bahwa pipeline worker-thread siap pakai di produksi.
