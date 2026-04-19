# FAQ

## Q: Kenapa import utama dari `src/*`, bukan package root?
A: Snapshot repo ini tidak menyediakan entry point TypeScript publik yang stabil; pemakaian aktual di test mengimpor langsung dari `src/*`.

## Q: Apa arti error `Native backend not available`?
A: Binding Rust tidak berhasil dimuat. Build ulang dengan `npm run build:rust` atau gunakan fallback JS.

## Q: Kenapa `npm test` bisa gagal satu assert kecil?
A: Ada assert presisi ketat pada floating point (`log(e)`), beda kecil float32 bisa memicu fail.

## Q: Bagaimana mencegah shape mismatch?
A: Catat shape per layer, validasi `units/outputUnits`, dan cek `seqLen` konsisten.

## Q: Kapan pakai `softmaxCrossEntropy`?
A: Saat output berupa logits multi-kelas dan target berupa index kelas (sparse).

## Q: Bagaimana menonaktifkan native saat debugging?
A: Jalankan proses dengan `ML_DISABLE_NATIVE=1`.

## Q: Apakah ada generator helper built-in?
A: Tidak ada helper generation tingkat tinggi khusus; lakukan loop inferensi manual memakai `predict/forward` + decoding tokenizer.
