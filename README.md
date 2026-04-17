# ML_V2

`ML_V2` adalah playground sekaligus library machine learning berbasis TypeScript yang membangun sendiri komponen inti deep learning: `Matrix`, operasi numerik, activation, loss, optimizer, layer, model `Sequential`, model `Transformers`, tokenizer BPE, dan pipeline training berbasis worker thread.

Project ini berfokus pada framework dan blok bangunan ML. Anda bisa menambahkan use case, dataset, dan eksperimen lokal sendiri di luar API inti tanpa harus menjadikannya bagian dari repo.

README ini dibuat berdasarkan struktur dan implementasi project saat ini, jadi isinya mengikuti kode yang benar-benar ada di repo.

## 1. Gambaran Besar Arsitektur

Secara mental, project ini punya 6 lapisan:

1. `src/matrix` dan `src/math`
   Menyediakan struktur `Matrix` berbasis `Float32Array` dan operasi numerik seperti `dotProduct`, `add`, `mul`, `reshape`, `sumAxis`, `clipGradients`, dan lain-lain.
2. `src/activation`, `src/cost`, `src/optimizer`
   Menyediakan activation function, loss function, dan optimizer.
3. `src/layers`
   Blok bangunan model: `Dense`, `Embedding`, `SelfAttention`, `MultiHeadAttention`, `LayerNormalization`, `Dropout`, `Flatten`, `Convolution`, `PositionalEncoding`.
4. `src/models`
   Model tingkat tinggi: `Sequential`, `Transformers`, `DimentionalityReduction`.
5. `src/tokenizer`
   Tokenizer BPE untuk text-to-token dan token-to-text.
6. `build/` dan area kerja lokal Anda
   Dipakai untuk hasil build, eksperimen, dan integrasi end-to-end di mesin lokal.

Catatan penting:

- `index.js` dan `index.d.ts` di root bukan entry point library model, melainkan binding N-API untuk backend native Rust.
- Script contoh di repo mengimpor library dari `src/*`, bukan dari root package.
- Backend native dipakai sebagai akselerasi numerik. Jika native tidak tersedia, kode TypeScript/JavaScript tetap punya fallback.

## 2. Struktur Folder

```text
src/
  activation/        Fungsi aktivasi dan softmax backward
  cost/              MSE, cross entropy, softmax cross entropy
  layers/            Dense, Embedding, Attention, Dropout, dst
  math/              Operasi numerik inti + wrapper native Rust
  matrix/            Class Matrix
  models/            Sequential, Transformers, DimentionalityReduction
  optimizer/         SGD, AdaGrad, Momentum, NAG, Adam
  pipeline/          Worker-thread training/pipeline helper
  tokenizer/         BPE tokenizer
  utils/             Selector helper, profiler, cosine similarity

src-rust/
  src/lib.rs         Implementasi native ops via napi-rs

test/
  Test dan benchmark untuk operasi matriks, tokenizer, transformer, dan training
```

## 3. Cara Menjalankan Project

### Install dependency

```bash
npm install
```

### Build backend native Rust

Direkomendasikan agar operasi matrix, activation, layer norm, embedding, convolution, attention, dan Adam lebih cepat:

```bash
npm run build:rust
```

Untuk mode debug:

```bash
npm run build:rust:debug
```

### Menjalankan test bawaan

```bash
npm test
```

Untuk training/inference aplikasi Anda sendiri, buat script lokal yang mengimpor modul dari `src/*`.

## 4. Alur Membuat Model ML Dengan Project Ini

Secara umum, workflow di repo ini seperti berikut:

1. Siapkan dataset.
2. Jika data berupa teks, latih atau muat `BPETokenizer`.
3. Ubah data menjadi `Matrix`.
4. Bangun model dengan `Sequential` atau `Transformers`.
5. `compile()` model untuk menentukan optimizer, learning rate, dan loss.
6. Jalankan `forward()` lalu `backward()` berulang di loop training.
7. Simpan model dengan `save()` dan tokenizer dengan `tokenizer.save()`.
8. Saat inferensi, muat kembali model dan tokenizer, lalu panggil `forward()` atau helper generate/chat.

### Kapan memakai `Sequential`

Pakai `Sequential` saat:

- input sudah fixed-size,
- arsitektur cukup berupa tumpukan layer,
- tugas Anda klasifikasi, regresi, atau autoencoder sederhana.

### Kapan memakai `Transformers`

Pakai `Transformers` saat:

- input berupa token sequence,
- target adalah next-token prediction,
- Anda ingin language model kecil atau generator teks domain-spesifik.

## 5. Tutorial Cepat: Membuat Model Klasifikasi

Contoh ini menunjukkan pola umum untuk klasifikasi teks.

### Langkah 1. Siapkan teks dan tokenizer

```ts
import { BPETokenizer } from "./src/tokenizer";

const tokenizer = new BPETokenizer({ vocabSize: 200, minFrequency: 2 });
tokenizer.train([
  "halo",
  "siapa nama kamu",
  "terima kasih",
]);
```

### Langkah 2. Ubah kalimat menjadi input matrix

```ts
import mj from "./src/math";

const tokens = tokenizer.encode("halo");
const padded = tokenizer.padSequence(tokens, 15);
const x = mj.matrix(padded.map((t) => [t])); // shape [seqLen, 1]
```

### Langkah 3. Bangun model

```ts
import { Sequential } from "./src/models";
import { Embedding, PositionalEncoding, SelfAttention, Flatten, Dense } from "./src/layers";

const model = new Sequential();
model.add(new Embedding({ vocabSize: tokenizer.getVocabSize(), embeddingDim: 16 }));
model.add(new PositionalEncoding({ dModel: 16, maxSeqLen: 15 }));
model.add(new SelfAttention({ units: 16, seqLen: 15, alpha: 0.005 }));
model.add(new Flatten());
model.add(new Dense({
  units: 16 * 15,
  outputUnits: 5,
  activation: "linear",
  status: "output",
  loss: "softmaxCrossEntropy",
}));

model.compile({
  alpha: 0.005,
  optimizer: "adam",
  error: "softmaxCrossEntropy",
});
```

### Langkah 4. Training

```ts
for (let epoch = 0; epoch < 300; epoch++) {
  model.forward(xTrain);
  model.backward(yTrain);
  console.log(model.loss);
}
```

### Langkah 5. Simpan model

```ts
model.save("model.json");
tokenizer.save("vocab.json");
```

## 6. Tutorial Cepat: Membuat Model Generatif / Next Token

Pola ini cocok untuk training next-token prediction secara umum.

### Ide training yang dipakai repo ini

Untuk setiap teks:

- tokenizer mengubah teks menjadi token ID,
- dibuat jendela konteks sepanjang `seqLen`,
- target adalah token berikutnya,
- model `Transformers` memprediksi token selanjutnya dari token terakhir pada jendela konteks.

### Bentuk data training

Jika sequence token adalah:

```text
[10, 20, 30, 40]
```

maka pasangan training menjadi:

- context `[PAD, PAD, 10]` target `20`
- context `[PAD, 10, 20]` target `30`
- context `[10, 20, 30]` target `40`

### Bangun model transformer

```ts
import { Transformers } from "./src/models";

const model = new Transformers({
  units: 64,
  seqLen: 64,
  vocabSize: tokenizer.getVocabSize(),
  heads: 8,
  alpha: 5e-9,
  padTokenId: tokenizer.getPadId(),
});

model.compile({
  alpha: 5e-9,
  optimizer: "adam",
  error: "softmaxCrossEntropy",
});
```

### Training loop minimal

```ts
for (const batch of batches) {
  model.forward(batch.x);
  model.backward(batch.y);
}
```

### Inference / generate

Alur inferensi yang dipakai repo:

1. encode prompt,
2. potong ke `seqLen`,
3. pad di kiri dengan `padId`,
4. `model.forward(context)`,
5. ambil logit output,
6. sampling token berikutnya,
7. ulangi sampai token cukup atau bertemu stop condition.

## 7. Referensi Model

## `src/models/sequential.ts`

`Sequential` adalah container model berurutan.

| Fungsi | Kegunaan |
| --- | --- |
| `constructor({ layers })` | Inisialisasi model dengan daftar layer opsional. |
| `summary()` | Menampilkan nama layer, shape input/output, dan total parameter. |
| `add(layer)` | Menambahkan layer ke model. |
| `save(path)` | Menyimpan semua layer ke JSON. |
| `load(path)` | Memuat layer dari JSON memakai `setLayers()`. |
| `compile(config)` | Meneruskan config optimizer/loss/alpha ke layer yang punya `compile()`. |
| `forward(x)` | Menjalankan inferensi maju dari layer pertama sampai terakhir. |
| `backward(y)` | Menjalankan backprop dari belakang ke depan; mengisi `model.loss` dari layer output. |
| `predict(x)` | Alias inference yang secara praktik sama dengan `forward(x)`. |
| `fit(X, y, epochs, cb)` | Loop training sederhana berbasis sampel. |

Kapan cocok:

- classifier sederhana,
- regressor,
- autoencoder atau reduction model,
- eksperimen layer stack manual.

## `src/models/transformers.ts`

`Transformers` adalah turunan `Sequential` yang merangkai:

`Embedding -> PositionalEncoding -> LayerNorm -> MultiHeadAttention -> Dropout -> residual -> LayerNorm -> FFN -> Dropout -> residual -> Dense output`

Perilaku penting:

- input berupa token index matrix,
- output hanya memakai representasi token terakhir untuk prediksi next-token,
- loss default output dipaksa ke `softmaxCrossEntropy`,
- mendukung `resizeVocab()` saat vocabulary tokenizer bertambah.

| Fungsi | Kegunaan |
| --- | --- |
| `constructor(config)` | Membangun transformer lengkap dari `units`, `seqLen`, `vocabSize`, `heads`, `dropoutRate`, `alpha`, `padTokenId`. |
| `forward(x)` | Embedding, positional encoding, attention block, FFN block, ambil state token terakhir, lalu proyeksi ke vocab logits. |
| `backward(y)` | Backprop dari dense output kembali ke FFN, attention, PE, lalu embedding. |
| `load(path)` | Memuat bobot layer transformer dari JSON. |
| `resizeVocab(newVocabSize)` | Memperbesar embedding table dan output dense jika tokenizer tumbuh. |
| `fit(X, y, epochs, cb)` | Training loop sederhana khusus transformer. |

## `src/models/dimentionalityReduction.ts`

Model ini memecah layer menjadi encoder dan decoder berdasarkan `status === "outputReduction"`.

| Fungsi | Kegunaan |
| --- | --- |
| `constructor({ layers })` | Mengelompokkan layer menjadi encoder/decode stack. |
| `load(path)` | Load layer dari file lalu hitung ulang encoder/decode split. |
| `encode(x)` | Menjalankan forward hanya sampai bottleneck. |
| `decode(enc)` | Menjalankan forward dari bottleneck ke keluaran akhir. |

## 8. Referensi Layer

## `Dense`

File: `src/layers/dense.ts`

| Fungsi | Kegunaan |
| --- | --- |
| `constructor({...})` | Membuat fully connected layer dengan activation, optimizer, status, alpha, dan loss. |
| `save()` | Serialize weight, bias, dan metadata. |
| `load(weight, bias)` | Load ulang parameter. |
| `compile({ alpha, optimizer, error })` | Ubah learning rate, optimizer, atau loss. |
| `forward(x)` | `weight * x + bias`, lalu activation. |
| `backward(y, err)` | Hitung error, grad weight/bias, clipping, update parameter, dan grad ke layer sebelumnya. |
| `resize(newOutputUnits)` | Memperbesar jumlah output neuron; penting untuk ekspansi vocab. |
| `resetLoss()` | Reset rata-rata loss layer output per epoch. |

Catatan:

- menggunakan Xavier initialization,
- bias ditambahkan dengan `addBias`,
- mendukung sparse target untuk klasifikasi.

## `Embedding`

File: `src/layers/embedding.ts`

| Fungsi | Kegunaan |
| --- | --- |
| `constructor({...})` | Membuat embedding table `[embeddingDim, vocabSize]`. |
| `save()` | Serialize embedding table. |
| `load(weight)` | Muat embedding lama. |
| `compile({ alpha, optimizer })` | Set ulang optimizer/lr. |
| `forward(x)` | Ubah token ID menjadi vektor embedding. |
| `backward(y, err)` | Akumulasi gradien ke token yang muncul lalu update embedding. |
| `resize(newVocabSize)` | Perbesar vocab embedding sambil mempertahankan bobot lama. |
| `resetLoss()` | Reset loss internal. |

## `PositionalEncoding`

File: `src/layers/positionalEncoding.ts`

| Fungsi | Kegunaan |
| --- | --- |
| `constructor({...})` | Precompute sinusoidal positional encoding. |
| `save()` | Simpan metadata PE. |
| `forward(x)` | Menambahkan encoding posisi ke embedding input. |
| `backward(_y, err)` | Karena PE konstan, gradien diteruskan apa adanya. |
| `resetLoss()` | Reset loss placeholder. |

## `LayerNormalization`

File: `src/layers/layerNormalization.ts`

| Fungsi | Kegunaan |
| --- | --- |
| `constructor({...})` | Membuat gamma dan beta trainable. |
| `save()` | Serialize gamma dan beta. |
| `load(gamma, beta)` | Muat parameter LN. |
| `forward(x)` | Normalisasi per kolom/token. |
| `backward(_y, err)` | Hitung grad gamma/beta dan grad terhadap input. |
| `compile({ alpha, optimizer })` | Set ulang optimizer dan lr. |
| `resetLoss()` | Reset loss placeholder. |

## `Dropout`

File: `src/layers/dropout.ts`

| Fungsi | Kegunaan |
| --- | --- |
| `constructor({ rate, status })` | Membuat dropout layer. |
| `save()` | Simpan rate dan mode. |
| `load({ rate, status })` | Load konfigurasi dropout. |
| `forward(x)` | Jika `status === "train"`, nol-kan sebagian aktivasi dan scale sisanya. |
| `backward(y, err)` | Kalikan gradien dengan mask dropout. |

Mode penting:

- `train`: dropout aktif,
- `test`: dropout dimatikan.

## `SelfAttention`

File: `src/layers/selfAttention.ts`

| Fungsi | Kegunaan |
| --- | --- |
| `constructor({...})` | Membuat Q/K/V tunggal untuk self-attention sederhana. |
| `compile({ alpha, optimizer })` | Set optimizer dan lr. |
| `save()` | Serialize Q/K/V. |
| `load(q, k, v)` | Muat parameter attention. |
| `forward(x)` | Hitung attention score, masking, softmax, lalu output context. |
| `backward(y, err)` | Hitung grad Q/K/V, update bobot, lalu teruskan grad ke input. |

## `MultiHeadAttention`

File: `src/layers/multiHeadAttention.ts`

| Fungsi | Kegunaan |
| --- | --- |
| `constructor({...})` | Membuat fused multi-head Q/K/V dan output projection `wo`. |
| `compile({ alpha, optimizer, error })` | Set ulang optimizer/lr. |
| `setPadMask(padMask)` | Menyimpan mask pad untuk forward attention. |
| `forward(x)` | Hitung Q/K/V, split per head, attention, gabungkan, lalu proyeksi output. |
| `backward(y, err)` | Backprop seluruh head dan update parameter attention. |
| `save()` | Serialize semua bobot MHA. |
| `load(data)` | Muat bobot MHA modern atau format legacy. |

## `Flatten`

File: `src/layers/flatten.ts`

| Fungsi | Kegunaan |
| --- | --- |
| `constructor(status)` | Inisialisasi layer flatten. |
| `forward(x)` | Ubah shape menjadi vector `[n, 1]`. |
| `backward(y, err)` | Kembalikan error ke shape input semula. |
| `resetLoss()` | Reset loss placeholder. |
| `save()` | Simpan metadata layer. |
| `load()` | Tidak melakukan apa-apa; hanya untuk kompatibilitas. |
| `compile()` | Tidak melakukan apa-apa karena tidak punya bobot. |

## `Activation`

File: `src/layers/activation.ts`

| Fungsi | Kegunaan |
| --- | --- |
| `constructor({...})` | Layer aktivasi generik. |
| `save()` | Simpan jenis activation/loss. |
| `load({...})` | Muat ulang konfigurasi activation layer. |
| `forward(x)` | Jalankan activation. |
| `backward(y, err)` | Hitung gradien aktivasi; bila output layer, sekaligus hitung loss. |

## `Convolution`

File: `src/layers/convolution.ts`

| Fungsi | Kegunaan |
| --- | --- |
| `constructor({...})` | Membuat kernel, bias, dan metadata layer konvolusi 2D sederhana. |
| `save()` | Serialize kernel dan bias. |
| `load(kernel, bias)` | Muat ulang parameter. |
| `compile({...})` | Set ulang lr, optimizer, dan loss. |
| `forward(x)` | Jalankan konvolusi, tambah bias, lalu activation. |
| `backward(y, err)` | Hitung grad kernel/bias dan grad ke input. |
| `resetLoss()` | Reset akumulasi loss. |

## 9. Referensi Tokenizer

File: `src/tokenizer/bpe.ts`

`BPETokenizer` adalah salah satu komponen paling penting di repo ini karena hampir semua proyek text bergantung padanya.

| Fungsi | Kegunaan |
| --- | --- |
| `constructor(config)` | Membuat tokenizer dengan `vocabSize`, `minFrequency`, dan special token tambahan. |
| `train(texts)` | Melatih vocab dan merge rule dari nol. |
| `update(texts, newVocabSize?)` | Incremental update tokenizer tanpa mengganti ID lama. |
| `encode(text)` | Encode teks menjadi token ID. |
| `encodeWithSpecial(text)` | Encode teks dengan `BOS` dan `EOS`. |
| `decode(ids)` | Decode ID kembali menjadi teks. |
| `getVocabSize()` | Ambil ukuran vocab saat ini. |
| `getTokenId(token)` | Ambil ID token. |
| `getToken(id)` | Ambil token dari ID. |
| `getPadId()` | Ambil ID untuk token `<PAD>`. |
| `padSequence(ids, maxLength)` | Potong atau pad sequence ke panjang tertentu. |
| `save(filepath)` | Simpan vocab dan merge ke JSON. |
| `static load(filepath)` | Muat tokenizer dari file JSON. |
| `summary()` | Tampilkan ringkasan tokenizer. |

Karakteristik implementasi tokenizer ini:

- mulai dari level karakter,
- memakai word boundary `▁`,
- mendukung reserved token untuk ekspansi vocab,
- punya `sanitize()` internal agar token tercemar bisa dibersihkan saat load/update,
- incremental update tidak mengubah token ID lama.

## 10. Referensi Matrix dan Math

## `src/matrix/index.ts`

`Matrix` adalah inti representasi tensor 2D di repo ini.

| Fungsi | Kegunaan |
| --- | --- |
| `constructor({ array })` | Membuat `Matrix` dari `number[][]`. |
| `static fromFlat(data, shape)` | Membuat `Matrix` langsung dari flat typed array. |
| `get(i, j)` | Ambil elemen. |
| `set(i, j, val)` | Set elemen. |
| `print()` | Tampilkan isi matrix ke console. |
| `getCol(colIndex)` | Ambil satu kolom sebagai `Float32Array`. |
| `setCol(colIndex, data)` | Tulis satu kolom. |
| `map(func)` | Modifikasi semua elemen in-place. |
| `add(a)` | Tambah scalar atau matrix in-place. |
| `sub(a)` | Kurang scalar atau matrix in-place. |
| `mul(a)` | Kali scalar atau matrix in-place. |
| `div(a)` | Bagi scalar atau matrix in-place. |
| `flatten()` | Ubah shape menjadi `[1, n]`. |
| `reshape(shape)` | Ubah interpretasi shape tanpa menyalin data. |
| `copyFrom(other)` | Salin data matrix lain ke buffer saat ini. |
| `clone()` | Membuat salinan matrix baru. |
| `addInPlace(other)` | Penjumlahan in-place teroptimasi. |
| `subInPlace(other)` | Pengurangan in-place teroptimasi. |
| `mulInPlace(other)` | Perkalian elemen-per-elemen in-place teroptimasi. |

## `src/math/index.ts`

Objek `mj` mengumpulkan helper numerik berikut:

| Fungsi | Kegunaan |
| --- | --- |
| `matrix()` | Membuat `Matrix` dari `number[][]`. |
| `zeros(shape)` | Matrix nol. |
| `ones(shape)` | Matrix satu. |
| `random(shape)` | Matrix acak. |
| `xavier(shape)` | Xavier init. |
| `he(shape)` | He init. |
| `add(a, b)` | Penjumlahan matrix/scalar. |
| `sub(a, b)` | Pengurangan matrix/scalar. |
| `mul(a, b)` | Perkalian elemen-per-elemen. |
| `div(a, b)` | Pembagian elemen-per-elemen. |
| `dotProduct(a, b, out?, transA?, transB?)` | Perkalian matriks inti. |
| `transpose(a)` | Transpose matrix. |
| `reshape(a, shape)` | Ubah shape. |
| `flatten(a)` | Flatten matrix. |
| `concat(a, b)` | Gabung matrix. |
| `convolution(a, kernel)` | Konvolusi 2D sederhana. |
| `addBias(a, bias)` | Tambah bias broadcast per kolom. |
| `sumAxis(a, axis, out?)` | Jumlah sepanjang axis tertentu. |
| `clipGradients(a, limit)` | Batasi magnitude gradien. |
| `mean(a)` | Rata-rata semua elemen. |
| `norm(a)` | Norma vector/matrix. |
| `map(a, func)` | Map elemen. |
| `absm(a)` | Nilai absolut. |
| `expm(a)` | Eksponensial per elemen. |
| `logm(a)` | Log per elemen. |
| `dotSum(a)` | Jumlah seluruh elemen flat. |
| `dotSub(a)` | Operasi bantu reduksi. |
| `dotMul(a)` | Operasi bantu reduksi. |
| `dotDiv(a)` | Operasi bantu reduksi. |

## 11. Referensi Activation, Loss, Optimizer, Utils

## Activation

File: `src/activation/index.ts`

Fungsi penting:

- `sigmoid(a)`
- `tanh(a)`
- `relu(a)`
- `lRelu(a)`
- `linear(a)`
- `softmaxInto(a, out, row?)`
- `softmaxOnly(a, row?)`
- `softmax(a, row?)`
- `softmaxBackwardInto(s, g, out, row?)`
- `softmaxBackward(s, g, row?)`
- `softmaxGradient(a)`

Setiap activation mengembalikan pasangan `[output, derivative]`, kecuali helper softmax tertentu yang mengembalikan matrix langsung.

## Loss

File: `src/cost/*`

| Fungsi | Kegunaan |
| --- | --- |
| `MeanSquerError(yTrue, yPred)` | Loss MSE. |
| `CategoricalCrossEntropy(yTrue, yPred)` | Cross entropy kategorikal. |
| `BinaryCrossEntropy(yTrue, yPred)` | Cross entropy biner. |
| `SoftmaxCrossEntropy(yTrue, yPred)` | Softmax + cross entropy terintegrasi. |

## Optimizer

File: `src/optimizer/*`

Tersedia:

- `SGD`
- `AdaGrad`
- `Momentum`
- `NAG`
- `Adam`

Semua dipilih via `setOptimizer()` di `src/utils/setOptimizer.ts`.

## Utils

| File | Fungsi |
| --- | --- |
| `setActivation.ts` | Mengubah string activation menjadi fungsi aktivasi yang benar. |
| `setLoss.ts` | Mengubah nama loss menjadi implementasi loss. |
| `setOptimizer.ts` | Mengubah nama optimizer menjadi object optimizer. |
| `setLayers.ts` | Membangun ulang layer saat load JSON. |
| `cosineSimilarity.ts` | Menghitung cosine similarity antar matrix/vector. |
| `profiler.ts` | Profiling waktu operasi forward/backward. |
| `state.ts` | Utility state sederhana. |

## 12. Backend Native Rust

Root file `index.js` akan memuat addon native yang dibangun dari `src-rust`.

Fungsi native yang diekspos di `index.d.ts` mencakup:

- `dotProduct`, `dotProductInto`
- `addMatricesInto`, `subMatricesInto`, `mulMatricesInto`, `divMatricesInto`
- `softmaxNativeInto`, `softmaxBackwardNativeInto`
- `layerNormNativeInto`, `layerNormBackwardNativeInto`
- `reluNativeInto`, `sigmoidNativeInto`, `tanhNativeInto`
- `embeddingForwardNativeInto`, `embeddingBackwardNative`
- `convolutionNativeInto`, `convBackwardInputNativeInto`
- `applyAttentionMaskNative`
- `multiHeadAttentionForwardNativeInto`, `multiHeadAttentionBackwardNativeInto`
- `adamUpdateNative`
- `addInPlace`, `subInPlace`, `mulInPlace`
- `mseNative`
- `addBiasNative`, `sumAxisNative`, `clipGradientsNative`

Praktiknya, Anda jarang memanggil fungsi ini langsung. Layer dan helper di `src/math/rust_backend.ts` yang akan memilih native path atau fallback JS.

## 13. Pola Integrasi Yang Disarankan

Jika Anda ingin memakai library ini tanpa memasukkan script aplikasi atau dataset ke repo, pola kerja yang paling aman adalah memisahkan:

- `src/*` untuk framework inti,
- folder lokal ter-ignore untuk dataset, checkpoint, dan script eksperimen,
- file model/vocab hasil training di luar area yang dilacak git.

Contoh struktur lokal yang aman:

```text
src/
test/
.gitignore

local/
  datasets/
  models/
  scripts/
```

Contoh `.gitignore`:

```gitignore
/node_modules
/build
/local
```

### Script training minimal

Buat script lokal yang:

1. memuat corpus Anda sendiri,
2. melatih atau memuat `BPETokenizer`,
3. membangun `Sequential` atau `Transformers`,
4. menjalankan loop `forward()` dan `backward()`,
5. menyimpan model ke folder lokal non-tracked.

Contoh arah implementasi:

```ts
const tokenizer = new BPETokenizer({ vocabSize: 8000, minFrequency: 2 });
tokenizer.train(corpus);
tokenizer.save("local/models/vocab.json");

const model = new Transformers({
  units: 64,
  seqLen: 128,
  vocabSize: tokenizer.getVocabSize(),
  heads: 8,
  alpha: 1e-4,
  padTokenId: tokenizer.getPadId(),
});

model.compile({ alpha: 1e-4, optimizer: "adam", error: "softmaxCrossEntropy" });
```

### Script inferensi minimal

Untuk inferensi, pola umumnya:

1. load vocab,
2. load model,
3. ubah dropout ke mode `test` bila dipakai,
4. encode prompt,
5. lakukan `forward()` berulang sambil sampling token.

### Script fine-tune minimal

Untuk fine-tuning model lama:

1. muat tokenizer lama,
2. `tokenizer.update()` dengan data baru,
3. muat bobot model lama,
4. `resizeVocab()` bila vocab bertambah,
5. compile ulang dengan learning rate lebih kecil,
6. training pada data tambahan saja.

## 14. Referensi Pipeline dan Worker

## `src/pipeline/transformer-pipeline.ts`

`TransformerPipeline` di implementasi saat ini lebih tepat disebut helper data parallel berbasis worker thread.

| Fungsi | Kegunaan |
| --- | --- |
| `constructor(model, numWorkers, microBatchSize)` | Inisialisasi helper worker. |
| `init(modelPath, modelConfig)` | Spawn worker dan tunggu semua siap. |
| `forwardPipeline(input)` | Fallback single-sample forward di main thread. |
| `forwardMicroBatches(inputs)` | Sebar input ke banyak worker untuk inferensi paralel. |
| `trainBatch(samples)` | Sebar micro-batch ke banyak worker untuk train paralel. |
| `shutdown()` | Terminasi semua worker. |
| `initialized` | Status apakah worker sudah siap. |

## `src/pipeline/training-worker.ts`

Fungsi file ini:

- memuat model terpisah di worker thread,
- menonaktifkan native addon di worker agar stabil,
- menerima pesan `forward` dan `train`,
- mengirim hasil kembali ke parent thread.

## 15. Cara Menambah Use Case ML Baru

Jika Anda ingin memakai repo ini untuk project lain, pola paling aman adalah:

### A. Untuk klasifikasi teks

1. Siapkan dataset `[{ text, label }]`.
2. Latih `BPETokenizer`.
3. Encode dan `padSequence()` setiap teks.
4. Bentuk `x` sebagai `Matrix [seqLen, 1]`.
5. Bentuk `y` one-hot atau class target sesuai loss.
6. Bangun `Sequential`.
7. Training loop manual dengan `forward()` dan `backward()`.

### B. Untuk generator teks domain khusus

1. Gabungkan semua teks domain menjadi corpus.
2. Latih `BPETokenizer`.
3. Buat pasangan `(context, next token)`.
4. Bangun `Transformers`.
5. Training batch demi batch.
6. Saat inference, pakai loop sampling token dari logits output.

### C. Untuk fine-tune model lama

1. Load tokenizer lama.
2. `tokenizer.update()` dengan data baru.
3. Load model lama.
4. Jika ukuran vocab bertambah, `model.resizeVocab(newSize)`.
5. Compile ulang dengan learning rate kecil.
6. Training pada data baru.

## 16. Catatan Desain dan Temuan Penting

Berdasarkan analisis kode, ada beberapa hal yang penting dipahami saat memakai repo ini:

- Library utama praktis hidup di `src/*`, bukan dari root package export umum.
- `Transformers` di repo ini adalah decoder-style next-token predictor kecil, bukan implementasi transformer encoder-decoder penuh.
- Output transformer hanya memproyeksikan hidden state token terakhir, sehingga cocok untuk next-token prediction.
- Tokenizer BPE mendukung pertumbuhan vocabulary secara incremental dan model juga sudah mendukung ekspansi vocab.
- Banyak operasi performance-sensitive sudah punya optimasi buffer reuse dan native Rust path.
- Worker pipeline saat ini memakai model copy per worker, jadi pendekatannya lebih dekat ke asynchronous data parallelism daripada pipeline parallelism murni per layer.

## 17. Saran Pemakaian Nyata

Kalau tujuan Anda adalah memakai repo ini untuk eksperimen ML pribadi, urutan belajar terbaik adalah:

1. Mulai dari `Matrix`, `mj`, dan `Dense` untuk memahami alur numeriknya.
2. Lanjut ke `Sequential` untuk klasifikasi/regresi sederhana.
3. Setelah itu pelajari `BPETokenizer`, `Embedding`, dan `Transformers` untuk tugas text generation.
4. Gunakan modul `pipeline` bila Anda ingin eksplorasi worker-thread training.
5. Simpan dataset dan checkpoint di folder lokal yang di-ignore git agar repo tetap bersih.

## 18. Ringkasan Singkat

Kalau diringkas, repo ini bisa dipakai untuk:

- belajar cara kerja library ML dari bawah,
- membuat classifier teks sederhana,
- membuat next-token predictor,
- membuat generator teks domain-spesifik,
- fine-tune model text kecil,
- eksperimen optimasi lewat native Rust dan worker thread.
