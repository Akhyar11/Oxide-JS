# Follow-up

## Rekomendasi lanjutan
1. Tambahkan pola `into` untuk operasi elementwise lain yang masih allocation-heavy (`div`, sebagian path `mul`).
2. Audit `optimizer/*` untuk peluang reuse buffer pada operasi chaining add/sub.
3. Tambahkan micro-benchmark terisolasi untuk:
   - `add/sub` legacy vs `out` mode
   - training step Transformer sebelum/sesudah patch pada beberapa ukuran batch/seq.
4. Pertimbangkan instrumentasi alokasi (heap snapshots / GC stats) agar dampak bisa diukur kuantitatif.

## Risiko / trade-off
- API menjadi lebih kaya (ada mode `out`) sehingga validasi shape harus disiplin.
- Reuse buffer memerlukan perhatian aliasing saat dipakai lintas operasi berantai.
- Perubahan saat ini menarget bottleneck utama; belum mengoptimasi seluruh jalur math secara menyeluruh.
