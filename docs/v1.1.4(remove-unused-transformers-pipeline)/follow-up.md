# Follow-up

## Jika pipeline ingin dihidupkan lagi
1. Tentukan dulu kontrak API publik yang jelas (entry point, lifecycle, dan perilaku error).
2. Implementasikan ulang pipeline di jalur yang benar-benar terintegrasi dengan flow training/inference aktif.
3. Tambahkan test khusus pipeline (unit + integrasi worker-thread) dan jadikan bagian dari CI utama.
4. Dokumentasikan status fitur secara eksplisit (experimental vs stable) agar tidak misleading.

## Rekomendasi tambahan
- Jika ada rencana jangka pendek mengembalikan fitur ini, gunakan branch/fitur terisolasi dengan dokumen desain sebelum merge ke mainline.
