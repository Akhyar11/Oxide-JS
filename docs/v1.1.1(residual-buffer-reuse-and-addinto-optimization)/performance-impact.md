# Performance Impact

## Sumber bottleneck sebelum patch
- Residual add di Transformer forward/backward membuat matrix baru berulang.
- Error merge (`res1Err`, `peErr`) membuat alokasi tambahan pada tiap backward step.
- Redundant zero-fill pada embedding forward (TS + Rust) menambah memory write yang tidak perlu.

## Dampak yang diharapkan
- Penurunan frekuensi alokasi objek matrix dan `Float32Array` pada hot path.
- Penurunan garbage collection pressure saat training sequence panjang/batch lebih besar.
- Memory bandwidth lebih efisien karena write langsung ke buffer reusable.

## Area yang tetap sengaja dipertahankan
- Zero-fill yang diperlukan untuk correctness (mis. buffer grad yang diakumulasi dengan `+=`) tetap dipertahankan.
- Perubahan dibatasi ke bottleneck utama agar risiko regresi fungsional rendah.
