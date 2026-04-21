# Dokumentasi Sistem: Overview

Selamat datang di dokumentasi resmi **ML-V1**, sebuah framework machine learning kustom yang dirancang untuk memberikan kontrol penuh, fleksibilitas, dan performa tinggi bagi para peneliti dan pengembang AI.

## Apa itu ML-V1?

**ML-V1** adalah library machine learning low-level hingga mid-level yang dibangun menggunakan **TypeScript** dengan akselerasi backend berbasis **Rust (N-API)**. Proyek ini lahir dari kebutuhan akan sebuah ekosistem ML yang transparan, di mana setiap operasi matematika dan logika training loop dapat diinspeksi dan dimodifikasi secara manual tanpa ketergantungan pada framework komersial yang kompleks.

## Visi dan Tujuan

Proyek ini dirancang dengan beberapa tujuan utama:
- **Kontrol Penuh**: Memberikan kemampuan untuk mengatur setiap detail teknis, mulai dari *shape* matriks hingga mekanisme pembaruan parameter.
- **Efisiensi Hybrid**: Menggabungkan kenyamanan penulisan kode di TypeScript dengan kecepatan eksekusi operasi numerik kritikal menggunakan Rust.
- **Riset Arsitektur**: Menjadi wadah (playground) untuk bereksperimen dengan arsitektur model kustom seperti Transformers, Dimensionality Reduction, dan lain-lain.

---

## Arsitektur Inti

Sistem ini dibagi menjadi beberapa modul utama yang saling terintegrasi:

### 1. Struktur Data & Matematika (`src/matrix` & `src/math`)
Jantung dari framework ini adalah kelas `Matrix` yang berbasis `Float32Array`. 
- **Flat Memory**: Menggunakan memori kontigu untuk efisiensi cache.
- **Math Primtive**: Menyediakan operasi dasar seperti `dotProduct`, `add`, `sumAxis`, dan `clipGradients`.

### 2. Backend Hybrid (`src-rust`)
Untuk operasi yang memakan waktu lama (hot paths), ML-V1 secara otomatis melakukan delegasi ke backend Rust jika tersedia.
- **Akselerasi Native**: Mempercepat operasi berat seperti *Multi-Head Attention* dan *Layer Normalization*.
- **Fallback Mechanism**: Jika binary native tidak ditemukan, sistem akan secara otomatis beralih ke implementasi JavaScript murni tanpa menghentikan proses.

### 3. Komponen Jaringan Saraf (`src/layers`)
Modul ini menyediakan blok bangunan untuk menyusun model:
- **Linear/Dense**: Full-connected layers dengan dukungan optimizer.
- **Attention**: Implementasi *Self-Attention* dan *Multi-Head Attention* dengan skema causal masking.
- **Normalization**: *Layer Normalization* untuk stabilitas training.
- **Specialized**: *Embedding*, *Dropout*, *Positional Encoding*, dan *Flatten*.

### 4. Komposisi Model (`src/models`)
Abstraksi tingkat tinggi untuk mengelola alur data:
- **Sequential**: Penumpukan layer secara linier.
- **Transformers**: Arsitektur lengkap yang siap digunakan untuk tugas-tugas NLP.
- **Dimentionality Reduction**: Model khusus untuk reduksi dimensi data.

### 5. Preprocessing Teks (`src/tokenizer`)
Implementasi **Byte Pair Encoding (BPE) Tokenizer** yang mendukung:
- Training kosakata dari dataset mentah.
- Encoding/Decoding teks ke token ID.
- Manajemen special tokens dan padding.

---

## Fitur Utama

- **Matrix-Driven**: Semua operasi berpusat pada manipulasi matriks yang efisien.
- **BPE-Native**: Dukungan built-in untuk tokenisasi tingkat lanjut.
- **Optimizer & Loss Functions**: Berbagai pilihan seperti Adam optimizer, MSE, dan Softmax Cross-Entropy.
- **Training Workflow**: API yang intuitif dengan metode `.forward()`, `.backward()`, dan `.fit()`.

---

## Filosofi Performa

ML-V1 memprioritaskan performa melalui:
1. **Pre-allocated Buffers**: Mengurangi frekuensi Garbage Collection (GC) selama training loop yang intens.
2. **Native Dispatching**: Menggunakan `napi-rs` untuk meminimalkan *overhead* antara layer JavaScript dan Rust.
3. **Optimized Hot-Paths**: Implementasi manual untuk operasi-operasi kritis guna memastikan latensi terendah.

---

> [!NOTE]
> Proyek ini sedang dalam pengembangan aktif (v1.1.6). Pastikan untuk selalu memeriksa kompatibilitas antara versi library dan backend native yang digunakan.

**Langkah Berikutnya:**
Lanjutkan ke bagian [Instalasi](02-installation.md) untuk mulai menyiapkan lingkungan pengembangan Anda.
