# v1.1.3 — Follow-up Recommendation

## Rekomendasi Lanjutan (Non-blocking)
1. **Perjelas kontrak aliasing untuk semua op `out` lain**
   - Audit operasi math lain yang menerima `out` agar kontrak aliasing konsisten lintas API.

2. **Pertimbangkan helper assert internal bersama**
   - Jika jumlah op math `out` bertambah, satukan helper validasi shape/aliasing agar lebih DRY.

3. **Stabilisasi toleransi test numerik existing**
   - Failure baseline `log(e)=1` bisa dipindahkan ke assertion berbasis toleransi float32 untuk mengurangi false negative.

4. **Lanjut audit legacy path MHA/Transformer**
   - Fokus pada dead-path yang tersisa dari kompatibilitas model lama, tanpa menghapus fallback penting.

## Tradeoff yang Diambil
- Guard aliasing pada `addInto/subInto` dipilih sebagai hard fail.
- Konsekuensi: caller yang sebelumnya mengandalkan aliasing harus migrasi ke buffer terpisah.
- Benefit: mencegah perilaku tidak deterministik/silent corruption pada path native.
