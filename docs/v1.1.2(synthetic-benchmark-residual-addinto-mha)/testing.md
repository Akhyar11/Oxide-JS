# Testing & Verification

## Cara menjalankan benchmark

```bash
npm run benchmark:synthetic-patch
```

Perintah di atas akan:
- menjalankan benchmark sintetis dengan `--expose-gc`
- mencetak tabel hasil ke console
- menyimpan hasil JSON ke `benchmark-results/synthetic_patch_benchmark.latest.json`

## Sanity checks yang dilakukan harness

- Validasi shape output pada benchmark `Transformers.forward()`.
- Menjalankan `forward` sebelum `backward` saat benchmark backward/training-step.
- Menggunakan checksum ringan / guard agar kerja tidak dihapus oleh optimisasi runtime.

## Verifikasi environment

- `nativeAvailable` dicetak di output.
- Jika native backend tidak tersedia, benchmark MHA native ditandai skip.
- `gcExposed` dicetak untuk transparansi kualitas metrik memory.

## Catatan stabilitas hasil

Untuk konsistensi:
- gunakan mode performa tinggi (CPU governor performance jika memungkinkan),
- hentikan beban proses lain,
- ulang run minimal 3 kali dan gunakan median.

