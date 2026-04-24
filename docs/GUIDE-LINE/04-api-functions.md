# Referensi API Lengkap: ML-V1

Halaman ini berisi dokumentasi teknis menyeluruh untuk seluruh komponen di dalam library **ML-V1**. Dokumentasi ini disusun secara bertahap, diawali dengan komponen inti seperti **`Matrix`**.

---

## 1. Struktur Data Core: Kelas `Matrix` (`src/matrix`)

Kelas `Matrix` adalah tulang punggung dari seluruh operasi numerik. Menggunakan **`Float32Array`** untuk penyimpanan data guna memastikan efisiensi memori dan kecepatan akses maksimal.

### A. Properti Utama
- **`_data: Float32Array`**: Buffer data mendatar (flat). Akses elemen $(i, j)$ dihitung secara internal dengan indeks `i * cols + j`.
- **`_shape: [rows, cols]`**: Dimensi matriks (misal: `[2, 3]` untuk 2 baris dan 3 kolom).

### B. Inisialisasi & Kreasi

#### `constructor({ array: number[][] })`
Membuat matriks dari array 2D standar.
```ts
const m = new Matrix({ 
  array: [
    [1, 2], 
    [3, 4]
  ] 
});
// Hasil Internal: _data = [1, 2, 3, 4], _shape = [2, 2]
```

#### `static fromFlat(data, shape)`
Membuat matriks langsung dari data datar. Lebih cepat karena tidak ada proses konversi (looping) dari array bertingkat.
```ts
const rawData = new Float32Array([10, 20, 30, 40]);
const m = Matrix.fromFlat(rawData, [2, 2]);
// Matriks:
// [[10, 20],
//  [30, 40]]
```

### C. Akses & Modifikasi Elemen

#### `get(i, j)` & `set(i, j, val)`
Akses dan ubah elemen pada posisi spesifik secara cepat.
```ts
const m = new Matrix({ array: [[1, 2], [3, 4]] });

console.log(m.get(0, 1)); // Output: 2 (Baris 0, Kolom 1)

m.set(1, 0, 99); 
// Matriks sekarang:
// [[1,  2],
//  [99, 4]]
```

#### `getCol(index)` & `setCol(index, data)`
Manipulasi seluruh kolom data menggunakan typed array.
```ts
const m = new Matrix({ array: [[1, 2], [3, 4]] });

const col1 = m.getCol(1); 
// col1 = Float32Array([2, 4])

m.setCol(0, new Float32Array([10, 30]));
// Matriks sekarang:
// [[10, 2],
//  [30, 4]]
```

### D. Operasi Element-wise (Berbasis Salinan Baru)

#### `add(a)`, `sub(a)`, `mul(a)`, `div(a)`
Operasi aritmatika dasar yang menghasilkan **matriks baru** (tidak mengubah matriks asli).
```ts
const a = new Matrix({ array: [[1, 2], [3, 4]] });

// 1. Dengan Skalar
const b = a.add(10); 
// b: [[11, 12], [13, 14]]

// 2. Dengan Matriks (Element-wise)
const c = a.mul(a); 
// c: [[1, 4], [9, 16]] (Hadamard product)
```

### E. Operasi In-Place (Optimasi Maksimal)

Mengubah data langsung pada buffer asli `_data`. Sangat hemat memori karena tidak mengalokasikan matriks baru. Diakselerasi otomatis jika backend Rust tersedia.

#### `addInPlace(other)`, `subInPlace(other)`, `mulInPlace(other)`
```ts
const m = new Matrix({ array: [[1, 2], [3, 4]] });
m.addInPlace(5); 

// m BERUBAH menjadi:
// [[6, 7],
//  [8, 9]]
```

### F. Transformasi & Utilitas

#### `reshape(shape)` & `flatten()`
Mengubah interpretasi dimensi tanpa mengubah urutan data di memori.
```ts
const m = new Matrix({ array: [[1, 2], [3, 4]] }); // [2, 2]

m.reshape([1, 4]); 
// m sekarang: [[1, 2, 3, 4]] (1 Baris, 4 Kolom)

m.flatten(); 
// m sekarang: [[1, 2, 3, 4]] (Vektor mendatar)
```

#### `clone()` & `map(func)`
```ts
const original = new Matrix({ array: [[1, 2], [3, 4]] });

// Clone literal
const copy = original.clone(); 

// Transformasi kustom per elemen
original.map(v => v * 2);
// original: [[2, 4], [6, 8]]
```

#### `print()`
Menampilkan struktur matriks dalam format tabel di konsol untuk mempermudah debugging.
```ts
m.print(); 
// ┌─────────┬────┬────┐
// │ (index) │ 0  │ 1  │
// ├─────────┼────┼────┤
// │    0    │ 2  │ 4  │
// │    1    │ 6  │ 8  │
// └─────────┴────┴────┘
```

---

## 2. Modul Matematika (`src/math`)

Modul `math` (sering dialiaskan sebagai `mj`) adalah kumpulan fungsi murni untuk pemrosesan tensor/matriks. Hampir semua fungsi mendukung operandi berupa `Matrix` atau `number` (skalar).

---

### A. Operasi Utama & Aljabar Linier

#### `mj.dotProduct(a, b, out?, transA?, transB?)`
Fungsi paling kritikal dalam machine learning untuk perkalian matriks (Matrix Multiplication). Operasi ini diakselerasi secara otomatis oleh backend Rust untuk beban kerja besar.

> [!IMPORTANT]
> **Aturan Dimensi**: Untuk operasi `(M x K) * (K x N)`, hasil akhirnya selalu berukuran `(M x N)`. Jumlah kolom matriks pertama harus sama dengan jumlah baris matriks kedua.

##### 1. Penggunaan Dasar
```ts
const a = mj.matrix([
  [1, 2], 
  [3, 4]
]); // [2x2]

const b = mj.matrix([
  [5, 6], 
  [7, 8]
]); // [2x2]

const res = mj.dotProduct(a, b);
// a * b = 
// [[(1*5 + 2*7), (1*6 + 2*8)],
//  [(3*5 + 4*7), (3*6 + 4*8)]]
// res: [[19, 22], [43, 50]]
```

##### 2. On-the-fly Transposition (Fitur Lanjutan)
Anda dapat melakukan perkalian terhadap matriks transpose tanpa perlu melakukan operasi transpose fisik yang memakan memori.
- `transA = true`: Menganggap matriks `a` sebagai `a.transpose()`.
- `transB = true`: Menganggap matriks `b` sebagai `b.transpose()`.

```ts
// Contoh: A * B^T
// A [2x3], B [2x3] -> B dimanipulasi jadi [3x2] secara on-the-fly
const res = mj.dotProduct(a, b, undefined, false, true); 
```

##### 3. Optimasi dengan Parameter `out`
Untuk performa tinggi di dalam loop training, gunakan matriks yang sudah dialokasikan sebelumnya.
```ts
const output = mj.zeros([2, 2]);
mj.dotProduct(a, b, output); // Hasil langsung ditulis ke 'output'
```

#### `mj.transpose(a: Matrix)`
Menghasilkan matriks baru dengan menukar baris dan kolom.
```ts
const t = mj.transpose(mj.matrix([[1, 2], [3, 4]]));
// t: [[1, 3], [2, 4]]
```

#### `mj.concat(a, b): Matrix`
Menggabungkan dua matriks/vektor. Saat ini dioptimasi untuk penggabungan vektor baris (shape `[1, N]`).
```ts
const a = mj.matrix([[1, 2]]);
const b = mj.matrix([[3, 4]]);
const res = mj.concat(a, b); // res: [[1, 2, 3, 4]]
```

#### `mj.addBias(a, bias): void`
Menambahkan vektor bias secara in-place ke matriks melalui teknik *broadcasting* (menambahkan vektor yang sama ke setiap kolom).
```ts
const input = mj.matrix([[1, 2], [3, 4]]); // [2x2]
const bias = mj.matrix([[10], [20]]);      // [2x1] vektor kolom
mj.addBias(input, bias);
// input: [[11, 12], [23, 24]]
```

#### `mj.norm(a): number`
Menghitung L2 Norm atau panjang Euclidean dari matriks (akar dari jumlah kuadrat seluruh elemen).
```ts
const length = mj.norm(weights);
```

---

### B. Aritmatika & Fungsi Element-wise

Fungsi ini memproses setiap elemen secara independen dan mendukung parameter `out` opsional untuk optimasi memori.

| Fungsi | Deskripsi | Contoh |
| :--- | :--- | :--- |
| `add`, `sub` | Penjumlahan & Pengurangan | `mj.add(m, 10)` |
| `mul`, `div` | Perkalian & Pembagian | `mj.mul(m1, m2)` |
| `absm(a)` | Nilai absolut per elemen | `mj.absm(m)` |
| `expm(a)` | Eksponensial (`e^x`) | `mj.expm(m)` |
| `logm(a)` | Logaritma natural (`ln`) | `mj.logm(m)` |
| `map(a, f)` | Fungsi kustom per elemen | `mj.map(m, x => x * x)` |

```ts
const res = mj.add(a, b, outputBuffer); // Menghindari alokasi baru
```

---

### C. Operasi Reduksi (Menghasilkan Satu Angka)

Fungsi-fungsi ini merangkum seluruh isi matriks menjadi satu nilai tunggal. Sangat berguna untuk kalkulasi loss, metrik performa, atau agregasi data.

#### `mj.mean(a)`
Menghitung nilai rata-rata dari seluruh elemen di dalam matriks.
```ts
const a = mj.matrix([[2, 4], [6, 8]]);
const avg = mj.mean(a); // (2+4+6+8) / 4 = 5
```

#### `mj.dotSum(a)`
Menghitung total penjumlahan ($\sum$) dari seluruh elemen.
```ts
const a = mj.matrix([[1, 2], [3, 4]]);
const total = mj.dotSum(a); // 1+2+3+4 = 10
```

#### `mj.dotMul(a)`
Menghitung total perkalian ($\prod$) dari seluruh elemen.
```ts
const a = mj.matrix([[1, 2], [3, 4]]);
const total = mj.dotMul(a); // 1*2*3*4 = 24
```

#### `mj.dotSub(a)`
Menghitung hasil pengurangan beruntun seluruh elemen (dimulai dari 0).
```ts
const a = mj.matrix([[1, 2], [3, 4]]);
const res = mj.dotSub(a); // 0 - 1 - 2 - 3 - 4 = -10
```

#### `mj.dotDiv(a)`
Menghitung hasil pembagian beruntun seluruh elemen (dimulai dari 1).
```ts
const a = mj.matrix([[2, 5]]);
const res = mj.dotDiv(a); // 1 / 2 / 5 = 0.1
```

---

### D. Statistik & Reduksi Axis

#### `mj.sumAxis(a, axis, out?)`
Menjumlahkan nilai sepanjang arah tertentu.
- **`axis 1`**: Menjumlahkan baris (Hasil: Matriks Kolom `[rows x 1]`).
- **`axis 0`**: Menjumlahkan kolom (Hasil: Matriks Baris `[1 x cols]`).

```ts
const a = mj.matrix([[1, 2], [3, 4]]);
const sumRows = mj.sumAxis(a, 1); // [[3], [7]]
```

---

### E. Generator & Inisialisasi Matriks

Fungsi-fungsi ini digunakan untuk menciptakan matriks baru dengan nilai awal tertentu sesuai kebutuhan arsitektur model. Seluruh fungsi dalam kategori ini menerima satu parameter utama berupa **`shape: [rows, cols]`**.

#### `mj.zeros([rows, cols])`
Membuat matriks yang seluruh elemennya bernilai **0**. Sangat efisien karena menggunakan inisialisasi default `Float32Array`.
```ts
const z = mj.zeros([2, 3]);
// Result:
// [[0, 0, 0],
//  [0, 0, 0]]
```

#### `mj.ones([rows, cols])`
Membuat matriks yang seluruh elemennya bernilai **1**.
```ts
const o = mj.ones([2, 2]);
// Result:
// [[1, 1],
//  [1, 1]]
```

#### `mj.random([rows, cols])`
Membuat matriks dengan nilai acak seragam antara **0** hingga **1**.
```ts
const r = mj.random([3, 1]); // Vektor kolom acak
```

#### `mj.xavier([rows, cols])`
Inisialisasi Xavier (Glorot) yang menjaga varians aktivasi tetap konstan di seluruh layer. Sangat direkomendasikan untuk layer yang menggunakan fungsi aktivasi **Sigmoid** atau **Tanh**.
```ts
const w = mj.xavier([128, 64]);
```

#### `mj.he([rows, cols])`
Inisialisasi He (Kaiming) yang dioptimalkan untuk layer dengan fungsi aktivasi **ReLU**. Inisialisasi ini mencegah masalah *vanishing gradients* pada jaringan yang sangat dalam.
```ts
const w = mj.he([64, 10]);
```

---

### F. Operasi Khusus (Deep Learning)

#### `mj.convolution(a, kernel)`
Operasi konvolusi 2D mendasar untuk ekstraksi fitur spasial. Filter (kernel) akan bergeser di atas input untuk menghasilkan jalur aktivasi baru.
```ts
const input = mj.matrix([
  [1, 1, 1, 0, 0],
  [0, 1, 1, 1, 0],
  [0, 0, 1, 1, 1],
  [0, 0, 1, 1, 0],
  [0, 1, 1, 0, 0]
]);

const kernel = mj.matrix([
  [1, 0, 1],
  [0, 1, 0],
  [1, 0, 1]
]);

const features = mj.convolution(input, kernel);
// Hasil Calculation (3x3):
// [[4, 3, 4],
//  [2, 4, 3],
//  [2, 3, 4]]
```

#### `mj.clipGradients(a, limit)`
Membatasi nilai absolut dalam matriks secara **in-place** agar berada dalam rentang `[-limit, limit]`. Ini sangat penting untuk mencegah masalah *Exploding Gradients* di mana nilai gradient menjadi terlalu besar dan merusak stabilitas training.
```ts
const grads = mj.matrix([
  [0.5,  5.0], // 5.0 melebihi limit
  [-10.2, 0.1] // -10.2 lebih kecil dari -limit
]);

mj.clipGradients(grads, 1.0);

// grads sekarang BERUBAH menjadi:
// [[0.5,  1.0],
//  [-1.0, 0.1]]
```

#### `mj.reshape(a, shape)` & `mj.flatten(a)`
Standalone function untuk manipulasi bentuk matriks tanpa alokasi baru.
```ts
const a = mj.matrix([[1, 2], [3, 4]]); // [2, 2]

const reshaped = mj.reshape(a, [1, 4]);
// reshaped: [[1, 2, 3, 4]]

const flat = mj.flatten(a);
// flat: [[1, 2, 3, 4]]
```

---

## 3. Fungsi Aktivasi (`src/activation`)

Fungsi aktivasi memperkenalkan non-linearitas ke dalam jaringan saraf, memungkinkan model untuk mempelajari pola yang kompleks. Dalam ML-V1, hampir semua fungsi aktivasi mengembalikan tuple **`[Matrix, Matrix]`**:
1.  **Hasil Aktivasi (Forward)**: Output yang diteruskan ke layer berikutnya.
2.  **Gradien/Turunan (Backward)**: Digunakan untuk menghitung koreksi error saat backpropagation.

---

### A. Sigmoid
Mengubah nilai input menjadi rentang **(0, 1)**. Sangat umum digunakan pada layer output untuk klasifikasi biner.

```ts
import { sigmoid } from "./src/activation";

const input = mj.matrix([[-1, 0, 2]]);
const [out, grad] = sigmoid(input);

// out (Hasil Aktivasi):
// [[0.268, 0.5, 0.880]]

// grad (Turunan/Derivative):
// [[0.196, 0.25, 0.105]]
```

---

### B. ReLU (Rectified Linear Unit)
Fungsi aktivasi paling populer. Mengubah semua nilai negatif menjadi **0** dan membiarkan nilai positif tetap.

```ts
import { relu } from "./src/activation";

const input = mj.matrix([[ -1.5, 0.5, 2.0 ]]);
const [out, grad] = relu(input);

// out (Hanya nilai positif yang lolos):
// [[ 0, 0.5, 2.0 ]]

// grad (1 jika input > 0, selain itu 0):
// [[ 0, 1, 1 ]]
```

---

### C. Tanh (Hyperbolic Tangent)
Mirip dengan sigmoid tetapi rentang outputnya adalah **(-1, 1)**. Seringkali memberikan performa lebih baik pada hidden layers dibanding sigmoid.

```ts
import { tanh } from "./src/activation";

const input = mj.matrix([[ -1, 0, 1 ]]);
const [out, grad] = tanh(input);

// out (Dipetakan ke rentang -1 s/d 1):
// [[ -0.761, 0, 0.761 ]]

// grad (1 - out^2):
// [[ 0.419, 1, 0.419 ]]
```

---

### D. Softmax (Multi-Class Output)
Menghasilkan distribusi probabilitas di mana total seluruh elemen adalah **1.0**.

#### `softmax(a, row = false)`
- **`row = true`**: Menghitung probabilitas per baris (Standard batch).
- **`row = false`**: Menghitung probabilitas per kolom.

```ts
import { softmax } from "./src/activation";

const logits = mj.matrix([[ 1, 2, 3 ]]);
const [probs, dSoftmax] = softmax(logits, true);

// probs (Total elemen adalah 1.0):
// [[ 0.09, 0.24, 0.66 ]]
```

---

### E. Leaky ReLU (lRelu)
Varian ReLU yang memberikan sedikit nilai (leak) pada input negatif (multiplier $10^{-5}$) untuk mencegah masalah "neuron mati".

```ts
import { lRelu } from "./src/activation";

const input = mj.matrix([[ -1, 1 ]]);
const [out, grad] = lRelu(input);

// out:
// [[ -0.00001, 1 ]]
```

---

### F. Linear (Identity)
Biasanya digunakan pada layer output untuk tugas **Regresi**.

```ts
import linear from "./src/activation"; // Default export

const [out, grad] = linear(inputMatrix);
// out: identik dengan input
// grad: berisi angka 1 (karena turunan x adalah 1)
```

---

> [!TIP]
> Saat melakukan *Manual Training Loop*, pastikan Anda menyimpan matriks `grad` (elemen kedua dari tuple) untuk digunakan saat menghitung pembaruan bobot (Update Weights).

---

## 4. Layers & Models (`src/layers` & `src/models`)

Bagian ini mendokumentasikan blok pembangun utama untuk menyusun jaringan saraf tiruan, mulai dari model kontainer hingga berbagai tipe layer spesifik.

---

### A. Model Kontainer: `Sequential`

`Sequential` adalah model pembungkus yang memungkinkan Anda menumpuk layer secara berurutan.

#### `constructor()`
Membuat instance model baru.
```ts
import { Sequential } from "./src/models";
const model = new Sequential();
```

#### `add(layer)`
Menambahkan layer ke dalam urutan eksekusi.
```ts
model.add(new Dense({ units: 4, outputUnits: 2 }));
```

#### `forward(input)` & `predict(input)`
Menjalankan data melalui seluruh layer. `predict` secara otomatis menonaktifkan mode pelatihan (seperti Dropout).
```ts
const output = model.predict(inputMatrix);
```

#### `compile({ alpha, optimizer, clipGradient })`
Mengonfigurasi parameter pembelajaran secara global untuk seluruh layer dalam model.
- **`alpha`**: Learning rate.
- **`optimizer`**: Nama optimizer (misal: `"adam"`).
- **`clipGradient`**: Batas clipping gradien kustom (number atau boolean).

#### `fit(X, y, epochs, config?): FitResult`
Melatih model secara otomatis menggunakan pasangan data input dan target. Mendukung batching, validation split, early stopping, shuffle, verbose logging, dan callback per epoch.

##### Signature yang didukung

```ts
// 1. API baru berbasis config (recommended)
const result = model.fit(X, y, epochs, config?: FitConfig): FitResult;

// 2. Legacy callback (backward compatible)
model.fit(X, y, epochs, (loss: number) => void): FitResult;
```

##### Parameter `FitConfig`

| Opsi | Tipe | Default | Deskripsi |
| :--- | :--- | :--- | :--- |
| `batchSize` | `number` | `max(1, floor(N/10))` | Jumlah sample per mini-batch |
| `validationSplit` | `number` | `0` | Proporsi data untuk validasi (0–1, eksklusif) |
| `earlyStoppingPatience` | `number` | `Infinity` | Epoch tanpa improvement sebelum training berhenti |
| `shuffle` | `boolean` | `true` | Acak urutan training setiap epoch |
| `verbose` | `boolean` | `false` | Cetak progress loss ke konsol tiap epoch |
| `onEpochEnd` | `(epoch, loss, valLoss?) => void` | `() => {}` | Callback setelah setiap epoch selesai |
| `monitorMetric` | `"loss" \| "valLoss"` | `"valLoss"` jika ada validasi, else `"loss"` | Metrik yang dipantau untuk early stopping |
| `minDelta` | `number` | `0` | Minimum perubahan yang dianggap sebagai improvement |
| `mode` | `"min" \| "max"` | `"min"` | `"min"` = berhenti jika tidak turun, `"max"` = berhenti jika tidak naik |
| `trimPadding` | `boolean` | `true` | Secara dinamis memotong PAD dari setiap batch sebelum forward/backward. Hanya aktif untuk full-sequence target (Y.shape[0] === X.shape[0]) dan model yang mendukung `getPadTokenId()` / `setPositionOffset()` (mis. Transformers). Untuk model lain atau legacy target Y=[1,batch], training berlanjut normal tanpa trimming. |
| `paddingSide` | `"left" \| "right"` | `"right"` | Sisi padding pada data input. `"right"` memotong trailing PAD (direkomendasikan untuk full-sequence causal LM). `"left"` memotong leading PAD dan menyesuaikan positional encoding offset. |

##### Return Value `FitResult`

```ts
interface FitResult {
  history: {
    loss: number[];      // Training loss per epoch
    valLoss?: number[];  // Validation loss per epoch (ada jika validationSplit > 0)
  };
  bestEpoch: number;       // Indeks epoch dengan loss terbaik (0-indexed)
  bestLoss: number;        // Nilai loss terbaik yang tercatat
  stoppedEarly: boolean;   // true jika early stopping aktif
  stoppingEpoch?: number;  // Epoch tempat early stopping terjadi
}
```

##### Contoh Penggunaan

###### API Baru (Recommended)
```ts
const result = model.fit(trainData, labels, 100, {
  batchSize: 16,
  validationSplit: 0.2,
  earlyStoppingPatience: 10,
  verbose: true,
  onEpochEnd: (epoch, loss, valLoss) => {
    console.log(`Epoch ${epoch}: loss=${loss.toFixed(4)}, valLoss=${valLoss?.toFixed(4)}`);
  },
});

console.log(`Best epoch: ${result.bestEpoch}, Best loss: ${result.bestLoss}`);
console.log("Training history:", result.history.loss);
```

###### Legacy Callback (Tetap Didukung)
```ts
model.fit(trainData, labels, 100, (loss) => {
  console.log(`Current Loss: ${loss}`);
});
```

> ```ts
> const result = autoencoderModel.fit(X, epochs, { batchSize: 8 });
> ```

---

### B. Transformers Model

Model arsitektur Transformer yang lengkap (berbasis arsitektur `Sequential`) untuk causal language modeling.

Perubahan penting pada versi ini:
- training path sekarang memakai **full-sequence causal LM**
- inference path tetap memakai **last-token logits**
- arsitektur sekarang mendukung **multi-block depth** lewat `numBlocks`
- target training yang benar adalah **shifted next-token targets** dengan shape `[seqLen, batch]`
- kontrak lama `backward(y)` dengan target `[1, batch]` masih diterima sebagai compatibility path terbatas, tetapi bukan lagi path training yang direkomendasikan

#### Kontrak Shape

- **Input token IDs**: `Matrix` dengan shape `[seqLen, batch]`
- **Training logits** (`model.train(); model.forward(x)` atau `model.forwardFullSequence(x)`): `[vocabSize, seqLen * batch]`
- **Inference logits** (`model.eval(); model.forward(x)`, `model.forwardNextToken(x)`, atau `model.predict(x)`): `[vocabSize, batch]`
- **Training target**: `Matrix` sparse index `[seqLen, batch]`
- **Legacy target**: `Matrix` sparse index `[1, batch]`

Urutan kolom logits training bersifat **sample-major**:
- sample 0 posisi `0..seqLen-1`
- sample 1 posisi `0..seqLen-1`
- dan seterusnya

Posisi valid untuk loss full-sequence:
- causal shift harus valid
- token input saat ini bukan `padTokenId`
- token target shifted juga bukan `padTokenId`

#### `constructor(config)`
- **`units`**: Dimensi model (`d_model`).
- **`seqLen`**: Panjang urutan input.
- **`vocabSize`**: Ukuran kosakata.
- **`heads`**: Jumlah attention heads (default: 8).
- **`numBlocks`**: Jumlah block Transformer bertingkat (default: 1).
- **`dropoutRate`**: Tingkat dropout (default: 0.1).
- **`alpha`**: Learning rate (default: 0.01).
- **`padTokenId`**: Token padding yang harus diabaikan di embedding, attention pad mask, dan loss full-sequence.
- **`clipGradient`**: Batas clipping gradien global untuk seluruh sub-layer (default: 5.0).

```ts
import { Transformers } from "./src/models";

const model = new Transformers({
  units: 128,
  seqLen: 50,
  vocabSize: 5000,
  heads: 8,
  numBlocks: 4,
  padTokenId: 0,
  clipGradient: 1.5
});
```

#### Arsitektur Internal

Setiap block Transformer berisi:
- `LayerNormalization -> MultiHeadAttention -> Dropout -> Residual`
- `LayerNormalization -> Dense(4x units, relu) -> Dropout -> Dense(units, linear) -> Dropout -> Residual`

Jika `numBlocks = 1`, perilaku dan topology dasarnya setara dengan versi lama single-block.

Jika `numBlocks > 1`, seluruh block dijalankan berurutan saat forward dan di-unroll terbalik saat backward.

#### `forward(input)`

Behavior `forward()` sekarang tergantung mode model:
- saat `model.train()`: mengembalikan logits full-sequence `[vocabSize, seqLen * batch]`
- saat `model.eval()`: mengembalikan logits last-token `[vocabSize, batch]`

Ini sengaja memisahkan default training path dan inference path tanpa menghapus ergonomi generation yang sudah ada.

#### `forwardFullSequence(input)`

Memaksa jalur training/full-sequence tanpa bergantung pada mode saat ini.

Gunakan jika Anda ingin eksplisit bahwa output yang diinginkan adalah logits seluruh sequence.

#### `forwardNextToken(input)`

Memaksa jalur inference/last-token tanpa bergantung pada mode saat ini.

Gunakan untuk sampling token berikutnya pada loop generation.

#### `predict(input)`

`predict()` sekarang memakai jalur inference last-token dan mengembalikan shape `[vocabSize, batch]`.

#### `backward(target)`

Kontrak target yang direkomendasikan:
- **shape**: `[seqLen, batch]`
- **isi**: target next-token yang sudah di-shift satu posisi ke kiri
- **last row**: umumnya diisi `padTokenId` karena tidak ada token berikutnya

Loss dihitung untuk seluruh posisi valid non-pad, bukan hanya satu posisi terakhir.

Compatibility path:
- target `[1, batch]` masih diterima untuk legacy loop yang hanya melatih token terakhir
- path ini dipertahankan hanya untuk meminimalkan breaking change, bukan best practice baru

#### `save(path)` / `load(path)`

`Transformers` tetap memakai format serialisasi flat-array seperti model `Sequential`, tetapi sekarang dapat menyimpan banyak block.

Aturan penting:
- model single-block lama tetap bisa di-load
- model multi-block baru juga bisa di-save/load
- instance yang memanggil `load()` harus dibuat dengan `numBlocks` yang sama dengan artefak model

Contoh aman:

```ts
const model = new Transformers({
  units: 128,
  seqLen: 50,
  vocabSize: 5000,
  heads: 8,
  numBlocks: 4,
  padTokenId: 0,
});

model.load("transformer_model.json");
```

Jika artefak model memiliki jumlah block berbeda dengan instance saat ini, `load()` akan melempar error eksplisit.

#### Contoh Training Full-Sequence

```ts
import mj from "./src/math";
import { Transformers } from "./src/models";

const padTokenId = 0;
const model = new Transformers({
  units: 64,
  seqLen: 6,
  vocabSize: 2000,
  heads: 8,
  numBlocks: 2,
  alpha: 0.001,
  padTokenId,
});

const x = mj.matrix([
  [0, 0],
  [11, 21],
  [12, 22],
  [13, 23],
  [14, 24],
  [15, 25],
]);

const y = mj.matrix([
  [0, 0],
  [12, 22],
  [13, 23],
  [14, 24],
  [15, 25],
  [0, 0],
]);

model.train();
const logits = model.forward(x); // [vocabSize, seqLen * batch]
model.backward(y);
console.log(logits._shape, model.loss);
```

#### Contoh Inference / Generation

```ts
import mj from "./src/math";
import { Transformers } from "./src/models";

const model = new Transformers({
  units: 64,
  seqLen: 6,
  vocabSize: 2000,
  heads: 8,
  numBlocks: 2,
  alpha: 0.001,
  padTokenId: 0,
});

model.eval();
const x = mj.matrix([
  [0],
  [11],
  [12],
  [13],
  [14],
  [15],
]);

const nextTokenLogits = model.predict(x); // [vocabSize, 1]
```

#### Best Practices

- Gunakan target shifted `[seqLen, batch]`, bukan target tunggal `[1, batch]`, untuk training LM yang benar.
- Konsistenkan `seqLen`, `vocabSize`, dan `padTokenId` antara tokenizer, preprocessing, dan model.
- Mulai dari `numBlocks=2` atau `numBlocks=4` jika ingin model lebih dalam, lalu benchmark karena biaya runtime akan naik signifikan.
- Pastikan posisi pad di target tetap `padTokenId` agar loss tidak menghitung area padding.
- Gunakan `model.predict()` atau `model.forwardNextToken()` pada generation loop agar sampling tetap memakai last-token logits.
- Gunakan `model.forwardFullSequence()` bila Anda butuh inspeksi logits semua posisi saat eval/debug.
- Aktifkan `trimPadding: true` (default) di `fit()` untuk performa optimal saat data memiliki banyak padding.

#### Dynamic Padding Trim

Mulai versi 2.2.0, `Transformers` mendukung pemotongan PAD dinamis per batch selama training melalui opsi `trimPadding` dan `paddingSide` di `FitConfig`.

##### Kapan memakai `paddingSide="right"` (default)

Gunakan saat dataset sudah dalam format **right-padded**:
```
[token0, token1, ..., tokenN, PAD, PAD]
```
- Positional encoding token asli tetap mulai dari posisi 0.
- Dynamic trimming memotong trailing PAD di ujung.
- `positionOffset = 0`.

Contoh penggunaan:
```ts
model.fit(trainX, trainY, epochs, {
  batchSize: 8,
  trimPadding: true,
  paddingSide: "right",
  shuffle: true
});
```

##### Kapan memakai `paddingSide="left"`

Gunakan saat dataset lama masih dalam format **left-padded**:
```
[PAD, PAD, token0, token1, ..., tokenN]
```
- Library memotong leading PAD dan mengeset `positionOffset = firstUsefulPos`.
- Absolute positional encoding token asli tetap sama setelah trim.

Contoh penggunaan:
```ts
model.fit(trainX, trainY, epochs, {
  batchSize: 8,
  trimPadding: true,
  paddingSide: "left",
  shuffle: true
});
```

##### Catatan correctness

- `trimPadding` hanya aktif untuk full-sequence target dengan shape `Y=[seqLen, batch]`.
- Legacy last-token target `Y=[1, batch]` tidak di-trim.
- PAD tetap di-ignore pada loss/gradient (via `buildShiftedLossGradient`).
- Trimming tidak mengubah token non-PAD.
- Untuk left-padding, `positionOffset` menjaga positional encoding tetap konsisten.

##### Catatan performa

- `trimPadding` tidak mengubah `maxSeqLen` model; `seqLen`/`contextLen` tetap bisa 1024.
- Yang berubah adalah `effectiveSeqLen` per batch (biasanya jauh lebih kecil).
- Attention cost turun dari O(seqLen²) menjadi O(effectiveSeqLen²).
- Dense output cost turun dari `vocabSize × seqLen × batch` menjadi `vocabSize × effectiveSeqLen × batch`.
- Untuk hasil terbaik, gunakan data right-padded dan set `paddingSide: "right"`.

##### Contoh konfigurasi untuk context panjang

```ts
const model = new Transformers({
  units: 64,
  seqLen: 1024,
  vocabSize,
  heads: 8,
  numBlocks: 2,
  padTokenId: 0
});

model.fit(trainX, trainY, 80, {
  batchSize: 8,
  trimPadding: true,
  paddingSide: "right",
  shuffle: true
});
```

##### API bridge methods

Transformers mengekspos tiga method untuk kebutuhan manual atau advanced use:

| Method | Deskripsi |
| :--- | :--- |
| `getPadTokenId(): number \| null` | Mengembalikan `padTokenId` dari embedding layer. |
| `setPositionOffset(n: number): this` | Set offset posisi PE untuk batch berikutnya (digunakan saat left-padding trim). |
| `resetPositionOffset(): this` | Reset offset posisi kembali ke 0. Otomatis dipanggil oleh `fit()` setelah setiap batch. |

#### Migration Note

Sebelum refactor ini, training transformer hanya memakai representasi token terakhir untuk memprediksi token berikutnya.

Sesudah refactor:
- training default memakai objective full-sequence causal LM
- shape output `forward()` saat mode train berubah dari `[vocabSize, batch]` menjadi `[vocabSize, seqLen * batch]`
- `backward()` idealnya menerima target shifted `[seqLen, batch]`
- jalur inference last-token dipertahankan lewat `predict()` dan `forwardNextToken()`
- arsitektur kini dapat ditingkatkan kedalamannya dengan `numBlocks` tanpa mengubah API training/inference utama
- `trimPadding: true` (default) aktif untuk Transformers dan mengurangi biaya komputasi pada batch dengan banyak padding

Jika training data sebelumnya left-padded, set `paddingSide: "left"`.
Jika membuat dataset baru untuk full-sequence causal LM, gunakan right-padding dan set `paddingSide: "right"`.
Jika ingin behavior lama tanpa trimming, set `trimPadding: false`.

---

### C. Dense Layer (Fully Connected)

Layer standar di mana setiap input terhubung ke setiap output.

#### `constructor(config)`
- **`units`**: Jumlah neuron input.
- **`outputUnits`**: Jumlah neuron output.
- **`activation`**: Nama fungsi aktivasi (misal: `"relu"`, `"sigmoid"`).
- **`optimizer`**: Algoritma optimasi (misal: `"sgd"`, `"adam"`).
- **`clipGradient`**: Batas clipping gradien khusus untuk layer ini (default: 5.0).

```ts
import { Dense } from "./src/layers";

const layer = new Dense({
  units: 128,
  outputUnits: 64,
  activation: "relu",
  optimizer: "adam"
});
```

---

### D. Embedding Layer

Digunakan untuk mengubah indeks kata (integer) menjadi vektor padat (*dense vector*). Sangat penting untuk tugas NLP.

#### `constructor(config)`
- **`vocabSize`**: Ukuran total kamus kata.
- **`embeddingDim`**: Dimensi vektor untuk setiap kata.

```ts
import { Embedding } from "./src/layers";

const embed = new Embedding({
  vocabSize: 5000,
  embeddingDim: 128
});
```

---

### E. Multi-Head Attention

Inti dari arsitektur Transformer yang memungkinkan model fokus pada bagian input yang berbeda secara bersamaan.

#### `constructor(config)`
- **`units`**: Dimensi internal (harus habis dibagi jumlah `heads`).
- **`heads`**: Jumlah mekanisme atensi paralel.
- **`seqLen`**: Panjang urutan input maksimal.
- **`clipGradient`**: Batas clipping gradien (default: 5.0).

```ts
import { MultiHeadAttention } from "./src/layers";

const attention = new MultiHeadAttention({
  units: 512,
  heads: 8,
  seqLen: 128
});
```

---

### F. Keluarga Recurrent Layer (`RNN`, `LSTM`, `GRU`)

Keluarga recurrent layer dipakai untuk data berurutan dengan format input **`[features, seqLen]`** untuk satu sample sequence.

#### Konvensi Umum
- **Input**: `Matrix` dengan shape `[units, seqLen]`.
- **`returnSequences: false`**: output shape `[hiddenUnits, 1]` untuk `RNN`/`LSTM`, atau `[hiddenUnits * 2, 1]` untuk `GRU` bidirectional.
- **`returnSequences: true`**: output shape `[hiddenUnits, seqLen]` untuk `RNN`/`LSTM`, atau `[hiddenUnits * 2, seqLen]` untuk `GRU` bidirectional.
- **`stateful: true`**: hidden state dibawa ke pemanggilan `forward()` berikutnya sampai `resetState()` dipanggil.
- **`returnState`**: saat ini **belum didukung** untuk seluruh keluarga recurrent dan akan melempar error eksplisit ketika `forward()` dipanggil.
- **`Sequential.fit()`**: untuk recurrent generic path saat ini gunakan **`batchSize=1`**. Jika `stateful=true`, hindari `shuffle=true` dan `validationSplit > 0`.

#### `RNN(config)`
Recurrent layer dasar dengan satu hidden state dan Backpropagation Through Time (BPTT).

##### `constructor(config)`
- **`units`**: Jumlah fitur input per time step.
- **`hiddenUnits`**: Jumlah unit hidden state.
- **`activation`**: Aktivasi recurrent (`"tanh"` default, atau `"relu"`).
- **`returnSequences`**: Jika `true`, kembalikan output untuk setiap time step.
- **`returnState`**: Disimpan di konfigurasi, tetapi belum didukung saat inferensi/training.
- **`stateful`**: Jika `true`, hidden state terakhir dipertahankan antar pemanggilan.
- **`optimizer`**: Optimizer parameter recurrent.
- **`clipGradient`**: Batas gradient clipping (default: `5.0`).

```ts
import mj from "./src/math";
import { RNN } from "./src/layers";

const layer = new RNN({
  units: 8,
  hiddenUnits: 16,
  activation: "tanh",
  returnSequences: true,
  stateful: false,
});

const x = mj.matrix([
  [1, 2, 3],
  [0, 1, 0],
  [1, 0, 1],
  [0, 0, 1],
  [1, 1, 1],
  [0, 1, 1],
  [1, 0, 0],
  [0, 0, 0],
]); // [8, 3]

const out = layer.forward(x); // [16, 3] karena returnSequences=true
```

#### `LSTM(config)`
Recurrent layer dengan **cell state** dan gate input/forget/output untuk menangani dependency sequence yang lebih panjang.

##### `constructor(config)`
- **`units`**: Jumlah fitur input per time step.
- **`hiddenUnits`**: Jumlah unit hidden state dan cell state.
- **`returnSequences`**: Jika `true`, kembalikan output untuk setiap time step.
- **`returnState`**: Belum didukung dan akan throw eksplisit saat `forward()`.
- **`stateful`**: Jika `true`, hidden state dan cell state dipertahankan antar pemanggilan.
- **`optimizer`**: Optimizer parameter gate LSTM.
- **`clipGradient`**: Batas gradient clipping (default: `5.0`).

```ts
import mj from "./src/math";
import { LSTM } from "./src/layers";

const layer = new LSTM({
  units: 8,
  hiddenUnits: 32,
  returnSequences: false,
  stateful: true,
});

const out = layer.forward(
  mj.matrix([
    [1, 2, 3, 4],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 0, 1, 1],
    [1, 1, 1, 1],
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [0, 0, 0, 1],
  ])
); // [32, 1]

layer.resetState(); // kosongkan hidden/cell state stateful
```

#### `GRU(config)`
Recurrent layer dengan gate **update/reset**. Implementasi ini juga mendukung mode **`bidirectional`**.

##### `constructor(config)`
- **`units`**: Jumlah fitur input per time step.
- **`hiddenUnits`**: Jumlah unit hidden state per arah.
- **`bidirectional`**: Jika `true`, jalankan GRU maju dan mundur lalu gabungkan output keduanya.
- **`returnSequences`**: Jika `true`, kembalikan output untuk setiap time step.
- **`returnState`**: Belum didukung dan akan throw eksplisit saat `forward()`.
- **`stateful`**: Jika `true`, hidden state per arah dipertahankan antar pemanggilan.
- **`optimizer`**: Optimizer parameter gate GRU.
- **`clipGradient`**: Batas gradient clipping (default: `5.0`).

```ts
import mj from "./src/math";
import { GRU } from "./src/layers";

const layer = new GRU({
  units: 8,
  hiddenUnits: 16,
  bidirectional: true,
  returnSequences: true,
});

const out = layer.forward(
  mj.matrix([
    [1, 2, 3],
    [0, 1, 0],
    [1, 0, 1],
    [0, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [1, 0, 0],
    [0, 0, 0],
  ])
); // [32, 3] = 16 forward + 16 backward
```

#### Catatan Praktis
- Untuk sequence modeling di dalam `Sequential`, recurrent layer biasanya diikuti `Dense` output layer.
- Jika `returnSequences=false`, layer akan mengembalikan representasi time step terakhir.
- Method `save()`/`load()` sudah menyimpan bobot recurrent dan state internal stateful.
- Method `getState()` tersedia untuk inspeksi state saat debugging.

---

### G. Layer Utilitas Lainnya

- **`Flatten`**: Meratakan matriks menjadi satu dimensi (biasanya sebelum Dense layer).
- **`Dropout({ rate })`**: Menonaktifkan neuron secara acak untuk mencegah overfitting.
- **`LayerNormalization({ units, clipGradient })`**: Menstabilkan distribusi nilai di dalam jaringan.
- **`Convolution({ kernelSize, inputShape, activation, clipGradient })`**: Operasi filter 2D untuk data spasial (Gambar).
- **`SelfAttention({ units, alpha, clipGradient })`**: Mekanisme atensi dasar untuk satu input.

---

## 5. Preprocessing & Tokenizer (`src/tokenizer`)

Sebelum teks dapat diproses oleh model machine learning, ia harus diubah menjadi angka. ML-V1 menggunakan algoritma **BPE (Byte Pair Encoding)** yang sangat efisien dalam menangani kosakata besar dan kata-kata baru (*Out-of-Vocabulary*).

---

### A. BPETokenizer

`BPETokenizer` bekerja dengan memecah kata-kata langka menjadi subword (potongan kata) dan mempertahankan kata-kata populer sebagai satu token utuh.

#### `constructor(config)`
- **`vocabSize`**: Ukuran kosakata target (misal: `5000`).
- **`minFrequency`**: Jumlah minimal kemunculan pasangan karakter untuk digabung (default: `2`).

```ts
import { BPETokenizer } from "./src/tokenizer";

const tokenizer = new BPETokenizer({
  vocabSize: 1000,
  minFrequency: 2
});
```

#### `train(texts: string[])`
Melatih tokenizer agar mengenali pola kata dari sekumpulan teks (corpus).
```ts
const corpus = ["saya makan nasi", "kamu makan roti"];
tokenizer.train(corpus);
```

#### `encode(text)` & `decode(ids)`
Mengonversi teks ke angka dan sebaliknya.
```ts
const ids = tokenizer.encode("saya makan"); 
// ids: [12, 45, 67]

const text = tokenizer.decode(ids);
// text: "saya makan"
```

#### `encodeWithSpecial(text)`
Menambahkan token otomatis **BOS** (Beginning of Sequence) di awal dan **EOS** (End of Sequence) di akhir. Sangat berguna untuk model generatif/Transformers.

#### `padSequence(ids, maxLength)`
Menambahkan token **PAD** agar semua urutan memiliki panjang yang sama untuk diproses dalam batch.
```ts
const padded = tokenizer.padSequence([1, 2], 5);
// padded: [1, 2, 0, 0, 0] (asumsikan PAD_ID = 0)
```

---

### B. Penyimpanan & Pemuatan

Tokenizer dapat disimpan ke dalam file `.json` agar tidak perlu dilatih ulang setiap kali aplikasi dijalankan.

```ts
// Simpan ke file
tokenizer.save("./model/vocab.json");

// Muat kembali
const loadedTokenizer = BPETokenizer.load("./model/vocab.json");
```

---

> [!CAUTION]
> Pastikan ukuran `vocabSize` pada layer **Embedding** identik dengan `vocabSize` yang dikembalikan oleh `tokenizer.getVocabSize()`. Ketidakcocokan akan menyebabkan error *index out of bounds*.

---

## 6. Algoritma Optimasi (`src/optimizer`)

Optimizer bertanggung jawab untuk memperbarui bobot (weights) dan bias model berdasarkan gradien yang dihitung selama backpropagation.

---

### A. Tipe Optimizer Tersedia

Anda dapat memilih tipe optimizer melalui string literal saat menginisialisasi layer atau model.

| Nama | Deskripsi | Rekomendasi |
| :--- | :--- | :--- |
| **`"sgd"`** | Stochastic Gradient Descent sederhana. | Model sangat sederhana atau debugging. |
| **`"momentum"`**| SGD dengan memori arah (*velocity*). | Konvergensi lebih halus dibanding SGD. |
| **`"adam"`** | Adaptive Moment Estimation. | **Default Terbaik** untuk hampir semua kasus. |
| **`"adaGrad"`** | Melakukan adaptasi learning rate per parameter. | Bagus untuk data yang jarang (*sparse*). |
| **`"nag"`** | Nesterov Accelerated Gradient. | Lebih akurat dibanding momentum standar. |

---

### B. Adam (Adaptive Moment Estimation)

Optimizer paling populer karena menggabungkan keuntungan dari Momentum dan RMSProp.

#### Karakteristik:
- **Akselerasi Native**: Dioptimasi menggunakan backend Rust untuk pembaruan parameter yang sangat cepat.
- **Kestabilan**: Memiliki mekanisme *bias correction* untuk langkah-langkah awal training.

```ts
// Contoh konfigurasi dalam Dense layer
const layer = new Dense({
  optimizer: "adam",
  alpha: 0.001 // Learning Rate
});
```

---

### C. Mekanisme Kerja

Setiap optimizer dalam ML-V1 mengimplementasikan metode `calculate(grad, alpha)` yang mengembalikan matriks **Update** yang kemudian akan dikurangi dari bobot asli secara *in-place*.

```ts
// Logika update internal yang dilakukan layer
const update = optimizer.calculate(gradWeight, alpha);
weight.subInPlace(update);
```

---

### Penutup
Anda sekarang memiliki referensi lengkap untuk seluruh API inti **ML-V1**. Gunakan panduan ini bersama dengan [01-overview.md](./01-overview.md) dan [03-tutorial.md](./03-tutorial.md) untuk membangun aplikasi AI yang kuat dan efisien.

---

> [!TIP]
> Untuk operasi intensif dalam training loop, selalu utamakan penggunaan **`get()`**, **`set()`**, dan metode **`InPlace`** untuk menghindari *bottleneck* performa.
