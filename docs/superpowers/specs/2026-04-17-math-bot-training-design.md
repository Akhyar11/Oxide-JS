# Math Bot Training Design

**Date:** 2026-04-17  
**Target:** Buat folder training baru yang terpisah untuk melatih model matematika dari nol menggunakan `dataset/mtk.json`, tanpa menyentuh artefak `generative-bot` yang sudah ada.

## Context

Repo saat ini sudah memiliki alur training generatif di [project/generative-bot/main.ts](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/project/generative-bot/main.ts:1). Alur itu:

- membaca dataset teks bebas, saat ini `dataset/cerita_rakyat.txt`
- menyimpan checkpoint ke `project/generative-bot/dataset/`
- mendukung mode lanjut training atau reset

User ingin kebutuhan yang lebih sempit dan lebih aman:

- model lama `generative-bot` tidak diutak-atik
- training matematika disimpan di folder baru yang terpisah
- training dimulai benar-benar dari nol
- sumber data utama adalah [dataset/mtk.json](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/dataset/mtk.json:1)

Format `dataset/mtk.json` adalah array objek `instruction`, `input`, dan `output`. Ini berbeda dari alur `generative-bot` saat ini yang membaca corpus teks mentah per baris.

## Goal

Bangun jalur training baru yang:

1. memiliki folder kerja sendiri, terpisah dari `project/generative-bot/`
2. membaca `dataset/mtk.json`
3. mengubah data instruction menjadi sample training yang cocok untuk next-token training
4. selalu membangun tokenizer, vocab, dan model baru dari nol
5. menyimpan semua artefak ke folder dataset milik `math-bot`

## Non-Goals

- Tidak mengubah perilaku `project/generative-bot/*`
- Tidak melakukan fine-tuning dari checkpoint lama
- Tidak menambah UI chat baru dalam fase ini
- Tidak menggabungkan semua pipeline training repo menjadi framework umum

## Recommended Approach

Pendekatan yang dipilih adalah membuat folder baru `project/math-bot/` dengan script training independen.

Alasannya:

- paling aman untuk artefak lama
- paling mudah di-reset total
- paling kecil risiko regresinya
- tidak perlu mempersulit `generative-bot` dengan banyak branch konfigurasi baru

## Proposed Structure

Tambahan struktur file:

- `project/math-bot/main.ts`
  - entry point training matematika dari nol
- `project/math-bot/chat.ts`
  - wrapper inferensi sederhana jika pola existing project perlu dipertahankan
- `project/math-bot/dataset/`
  - lokasi artefak model/vocab/tokenizer khusus matematika

Pada fase implementasi minimum, file yang wajib hanya `project/math-bot/main.ts` dan `project/math-bot/dataset/`. `chat.ts` opsional dan hanya dibuat jika diperlukan untuk menjaga pola folder project tetap konsisten.

## Data Flow

### Input Source

Sumber data utama adalah [dataset/mtk.json](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/dataset/mtk.json:1).

Setiap record:

- `instruction`
- `input`
- `output`

### Normalization

Setiap record akan diubah menjadi satu string training terstruktur, misalnya:

`Instruksi: <instruction>\nInput: <input>\nJawaban: <output>`

Aturan normalisasi:

- teks diubah ke lowercase agar konsisten dengan pipeline yang ada
- `input` kosong tetap diizinkan
- field kosong tidak menyebabkan sample dibuang selama `output` valid
- record yang tidak punya `output` string di-skip

Tujuan format ini adalah mempertahankan konteks instruction-following sambil tetap cocok dengan next-token prediction yang sudah didukung engine saat ini.

### Tokenizer

`math-bot` akan selalu:

- membuat tokenizer baru
- melatih vocab baru dari corpus matematika hasil normalisasi
- menyimpan vocab ke folder dataset milik `math-bot`

Tidak ada reuse vocab lama dari `generative-bot`.

### Training Samples

Setelah teks tiap record dinormalisasi, sample training dibuat dengan pola yang sama seperti `generative-bot`:

- tokenize satu string training
- bentuk context window sepanjang `seqLen`
- target adalah token berikutnya

Ini menjaga implementasi tetap dekat dengan engine transformer yang sudah ada, sehingga perubahan terbatas pada input loader dan lokasi artefak.

## Reset Behavior

`math-bot` bersifat reset-only pada fase ini.

Artinya saat training dimulai:

- artefak lama di `project/math-bot/dataset/` boleh dihapus atau ditimpa
- script tidak menawarkan mode resume
- tokenizer dan model selalu dibuat dari nol

Perilaku yang disarankan adalah menghapus file artefak target yang dikenal sebelum training dimulai agar hasil reset eksplisit dan mudah dipahami user.

## Artifact Layout

Artefak baru disimpan di:

- `project/math-bot/dataset/math_model.json`
- `project/math-bot/dataset/math_vocab.json`

Nama file dibuat spesifik agar tidak tertukar dengan `generative_model.json` milik folder lain.

Jika nanti diperlukan file tambahan, nama turunannya mengikuti namespace `math_*`.

## Error Handling

Kasus yang harus ditangani:

- `dataset/mtk.json` tidak ditemukan
- isi file bukan array JSON
- semua record invalid setelah normalisasi
- training samples kosong setelah tokenisasi

Respons yang diharapkan:

- tampilkan pesan error yang jelas
- hentikan proses dengan exit code non-zero
- jangan membuat klaim bahwa model sudah siap bila training belum berjalan

## Testing Strategy

Fokus test fase ini adalah memastikan loader dan normalisasi benar, bukan kualitas model.

Test minimum:

1. loader `mtk.json` membaca array instruction dataset dengan benar
2. normalizer menghasilkan string training yang benar untuk kasus `input` terisi
3. normalizer tetap valid untuk kasus `input` kosong
4. record invalid dilewati tanpa meledakkan proses

Jika test repo saat ini belum punya harness khusus untuk file baru, test dapat ditempatkan dalam file `test/` yang hanya memverifikasi transformasi data.

## Implementation Boundaries

File yang kemungkinan disentuh:

- [project/math-bot/main.ts](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/project/math-bot/main.ts:1)
- [test/](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/test)
- opsional `package.json` bila perlu script baru untuk menjalankan training matematika

File yang tidak boleh diubah perilakunya:

- [project/generative-bot/main.ts](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/project/generative-bot/main.ts:1)
- artefak lama di `project/generative-bot/dataset/`

## Risks

### Dataset Size and Quality

`mtk.json` berisi instruction dataset yang relatif kecil. Training from scratch mungkin menghasilkan model yang sangat terbatas dan cenderung menghafal pola. Risiko ini diterima karena target fase ini adalah memisahkan pipeline dan memastikan dataset matematika dapat dilatih dengan benar.

### Prompt Format Lock-In

Memilih format `Instruksi/Input/Jawaban` berarti perilaku inferensi nantinya akan paling stabil bila prompt mengikuti format serupa. Ini tradeoff yang diterima karena dataset memang berbentuk instruction tuning sederhana.

### Artifact Confusion

Jika user menjalankan folder yang salah, bisa terjadi kebingungan model mana yang dipakai. Karena itu penamaan file dan folder harus eksplisit bertema `math-bot`.

## Recommendation

Implementasi pertama sebaiknya dibatasi pada:

1. buat `project/math-bot/main.ts`
2. tambahkan loader + normalizer untuk `dataset/mtk.json`
3. simpan artefak ke `project/math-bot/dataset/`
4. tambahkan test kecil untuk transformasi data

Itu sudah cukup untuk memenuhi tujuan user tanpa memperluas scope ke runtime chat atau fine-tuning lanjutan.
