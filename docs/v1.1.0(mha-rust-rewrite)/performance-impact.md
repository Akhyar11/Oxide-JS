# Performance Impact

## Bottleneck lama
- Alokasi `Vec<f32>` besar berulang per (head, sample) block.
- `collect()` menghasilkan container besar tambahan.
- Data block dicopy ulang ke output final setelah parallel map selesai.

## Perubahan yang menarget bottleneck
- Direct-write ke output final per-head slice.
- Menghapus fan-in collect block vectors.
- Backward scratch diperkecil dari O(seq_len^2) menjadi O(seq_len) untuk jalur error softmax.

## Dampak performa (hipotesis teknis)
- Mengurangi pressure allocator dan overhead copy memory.
- Meningkatkan locality karena worker menulis langsung ke region output yang kontigu per-head.
- Mengurangi biaya sinkronisasi implicit akibat pengumpulan hasil block.

> Catatan: tanpa benchmark terkontrol lintas beberapa ukuran tensor, angka speedup absolut belum diklaim final.

## Benchmark yang tersedia
- `test/mha_rust_perf.ts` menyediakan perbandingan runtime native vs fallback (JS) untuk workload kecil-menengah sebagai sanity benchmark awal.
