# Panduan Instalasi dan Persiapan

Ikuti langkah-langkah di bawah ini untuk menyiapkan lingkungan pengembangan ML-V1 di mesin lokal Anda.

## Prasyarat Utama

Sebelum memulai, pastikan Anda telah menginstal perangkat lunak berikut:
1. **Node.js**: Versi LTS (direkomendasikan v20.x atau lebih baru).
2. **Rust Toolchain**: Diperlukan untuk melakukan build backend native (`cargo`, `rustc`). Instal melalui [rustup.rs](https://rustup.rs/).
3. **TypeScript**: Framework ini menggunakan TypeScript untuk pengembangan inti.

---

## Langkah-langkah Instalasi

### 1. Kloning Repositori
Jika Anda belum melakukannya, klon repositori ini ke mesin lokal Anda:
```bash
git clone <url-repositori>
cd ML-V1
```

### 2. Instal Dependency Node.js
Gunakan npm untuk menginstal semua package yang diperlukan:
```bash
npm install
```

### 3. Build Backend Native (Opsional namun Direkomendasikan)
Untuk mendapatkan performa maksimal, Anda perlu mem-build modul Rust menggunakan `napi-rs`.

**Untuk Build Produksi (Release):**
```bash
npm run build:rust
```

**Untuk Build Debug (Lebih Cepat, Performa Lebih Rendah):**
```bash
npm run build:rust:debug
```

Hasil build akan menghasilkan file binary `.node` di root proyek (misalnya `ml-native.linux-x64-gnu.node`).

---

## Verifikasi Instalasi

Setelah instalasi selesai, Anda dapat memverifikasi apakah backend native aktif dengan menjalankan test sederhana:

```ts
import { isNativeAvailable } from "./src/math/rust_backend";

console.log("Status Native Backend:", isNativeAvailable());
```

Atau jalankan test suite lengkap:
```bash
npm test
```

---

## Konfigurasi Lingkungan

### Menonaktifkan Native Backend Secara Paksa
Jika Anda mengalami masalah dengan binary native atau ingin melakukan debugging pada implementasi JavaScript murni, Anda dapat menggunakan environment variable `ML_DISABLE_NATIVE`:

```bash
ML_DISABLE_NATIVE=1 node script-anda.js
```

### Penggunaan TypeScript
Karena proyek ini menggunakan TypeScript secara intensif, Anda mungkin perlu melakukan kompilasi sebelum menjalankan script dengan `node` murni:
```bash
npm run build # Menjalankan tsc
```
Atau gunakan `ts-node` untuk eksekusi langsung (sudah termasuk dalam `devDependencies`).

---

> [!WARNING]
> Jika Anda mengganti sistem operasi atau arsitektur CPU (misalnya dari Linux ke macOS), Anda **wajib** menjalankan kembali `npm run build:rust` agar file binary sesuai dengan platform yang baru.

**Langkah Berikutnya:**
Pelajari cara menggunakan library ini di bagian [Tutorial Singkat](03-tutorial.md).
