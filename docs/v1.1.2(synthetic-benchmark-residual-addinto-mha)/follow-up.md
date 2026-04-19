# Follow-up Recommendations

## 1) Tambahkan baseline antar-commit
Buat script pembanding dua commit:
- checkout baseline commit (pre-patch)
- jalankan benchmark yang sama
- simpan hasil berdampingan
- hitung delta % otomatis

## 2) Perkuat memory profiling
Tambahkan mode opsional dengan:
- `--trace-gc` log parser,
- heap snapshot sebelum/sesudah skenario,
- (opsional) allocation profiling dengan inspector/clinic.

## 3) Uji skala batch besar
Tambahkan konfigurasi:
- seqLen 128/256 dengan batch 32 dan 64 (jika memori cukup),
- untuk melihat kapan bottleneck berpindah dari compute ke memory bandwidth.

## 4) Isolasi MHA lebih granular
Pisahkan benchmark:
- MHA forward-only,
- MHA backward-only,
- ukuran sequence/head berbeda,
untuk melihat sensitivitas patch zero-fill pada workload spesifik.

## 5) Integrasi CI perf guardrail
Simpan hasil benchmark median (rolling) dan pasang ambang alert regresi
misalnya >10% slowdown pada skenario prioritas:
- addInto/subInto transformer-like
- residual reusable-buffer
- transformer forward/backward config utama

