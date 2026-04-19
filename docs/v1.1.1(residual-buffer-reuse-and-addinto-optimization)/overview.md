# Overview

Perubahan ini fokus pada bottleneck traffic memori di jalur residual Transformer dan operasi elementwise add/sub.

## Ringkasan utama
- Menambahkan API buffer-reuse untuk operasi add/sub:
  - `mj.add(a, b, out?)`
  - `mj.sub(a, b, out?)`
  - `mj.addInto(a, b, out)`
  - `mj.subInto(a, b, out)`
- Mengubah residual path di `Transformers` agar tidak membuat matrix baru setiap langkah residual/error-merge.
- Menghapus zero-fill yang redundant pada `Embedding.forward` di sisi TypeScript ketika native backend dipakai.

## Dampak target
- Mengurangi alokasi `Float32Array` baru pada hot path training Transformer.
- Menurunkan memory traffic di residual add dan error merge.
- Menjaga kompatibilitas API lama (pemakaian `mj.add(a, b)` / `mj.sub(a, b)` tetap berjalan).
