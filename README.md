# ML-V1 (TypeScript + Rust Native)

Library machine learning custom berbasis TypeScript dengan backend Rust (N-API) untuk akselerasi operasi numerik kritikal.

## What this project is
ML-V1 adalah library low-level sampai mid-level untuk eksperimen dan pengembangan model ML secara manual: `Matrix`, math ops, layer, model, dan tokenizer BPE.

## Why this project exists
- Menyediakan kontrol penuh atas detail training loop, shape, dan update parameter.
- Menjadi playground riset arsitektur custom tanpa dependency framework ML besar.
- Menggabungkan kemudahan TypeScript dengan performa Rust untuk hot paths.

## Versioning
Versi aktif proyek saat ini adalah `2.2.5`.

Proyek ini memakai format versi `MAJOR.MINOR.PATCH` seperti `2.2.5`.

- Angka paling depan (`MAJOR`): perubahan besar yang biasanya membawa breaking change atau perubahan arsitektur utama.
- Angka tengah (`MINOR`): penambahan fitur baru atau peningkatan yang tetap kompatibel dengan versi sebelumnya.
- Angka paling belakang (`PATCH`): perbaikan bug, optimasi kecil, cleanup, atau perubahan minor yang tidak mengubah API utama.

Contoh:
- `2.2.0`: rilis mayor `2`, minor `2`, dynamic padding trim + positional encoding offset.
- `2.2.5`: patch untuk optimasi hot path training/validation, lookup embedding, dan training/update tokenizer BPE.
- `2.2.4`: patch untuk ergonomi API `Transformers.predictMode`, sinkronisasi docs, dan refactor correctness suite.
- `2.2.3`: patch untuk optimasi hot path training/inference, refresh benchmark family model, dan correctness learning snapshot terbaru.
- `2.2.2`: patch untuk suite gabungan root, benchmark family model, dan correctness learning snapshot.
- `2.0.2`: rilis mayor `2`, minor `0`, patch `2` untuk optimasi projector transformer tanpa perubahan API.

## Key features
- `Matrix` berbasis `Float32Array` (flat contiguous memory).
- Operasi math inti (`dotProduct`, `add`, `sumAxis`, `clipGradients`, dst).
- Layer: `Dense`, `Embedding`, `RNN`, `LSTM`, `GRU`, `SelfAttention`, `MultiHeadAttention`, `LayerNormalization`, `Dropout`, `PositionalEncoding`, `Flatten`, `Convolution`.
- Model: `Sequential`, `Transformers`, `DimentionalityReduction`.
- Tokenizer BPE (`train`, `update`, `encode`, `decode`, `padSequence`, `save/load`).
- Native Rust fallback-aware (otomatis ke JS jika native tidak tersedia).
- **Dynamic padding trim** (`trimPadding`) untuk training Transformer full-sequence yang lebih efisien.

## Architecture overview
1. `src/matrix`: struktur data matrix.
2. `src/math`: primitive numerik + jembatan ke native backend.
3. `src/activation`, `src/cost`, `src/optimizer`: blok training.
4. `src/layers`: komponen jaringan saraf.
5. `src/models`: komposisi layer tingkat model.
6. `src/tokenizer`: text preprocessing.
7. `src-rust`: implementasi native ops via `napi-rs`.

## Project structure
```text
src/
  activation/  cost/  optimizer/
  matrix/      math/
  layers/      models/
  tokenizer/
  utils/
src-rust/
  src/lib.rs

test/
dataset/
docs/
```

## Installation

Instal library menggunakan npm:

```bash
npm install @akhyar11/ml-v1
```

### Prerequisites (Penting)
Untuk menggunakan fitur **Native Rust Acceleration** (yang mempercepat operasi Matrix hingga 10x lipat), sistem Anda **WAJIB** memiliki:
1.  **Rust Toolchain**: Instal via [rustup.rs](https://rustup.rs/).
2.  **C/C++ Build Tools**: Diperlukan untuk kompilasi native binding.

*Catatan: Jika Rust tidak terinstal, library akan tetap berjalan menggunakan **Pure JavaScript fallback**, namun performa akan jauh lebih lambat untuk model besar.*

## Build and setup
Jika Anda melakukan cloning repo ini atau ingin melakukan kompilasi manual:

```bash
# Install dependencies
npm install

# Build native Rust (release)
npm run build:rust

# Build TypeScript
npm run build:publish
```

## Rust native backend
Native loader berada di `src/math/rust_backend.ts` dan mencoba memuat binding dari root `index.js`.

Verifikasi runtime:
```ts
import { isNativeAvailable } from "./src/math/rust_backend";
console.log("Native active:", isNativeAvailable());
```

Nonaktifkan native secara paksa:
```bash
ML_DISABLE_NATIVE=1 node your-script.js
```

## Quick start
```ts
import mj from "./src/math";
import { Sequential } from "./src/models";
import { Dense } from "./src/layers";

const model = new Sequential({
  layers: [
    new Dense({ units: 2, outputUnits: 4, activation: "relu", status: "input" }),
    new Dense({ units: 4, outputUnits: 1, activation: "sigmoid", status: "output", loss: "mse" }),
  ],
});

model.compile({ alpha: 0.01, optimizer: "adam", error: "mse" });

const X = [mj.matrix([[0], [0]]), mj.matrix([[0], [1]]), mj.matrix([[1], [0]]), mj.matrix([[1], [1]])];
const Y = [mj.matrix([[0]]), mj.matrix([[1]]), mj.matrix([[1]]), mj.matrix([[0]])];

const result = model.fit(X, Y, 200, {
  batchSize: 4,
  validationSplit: 0.25,
  earlyStoppingPatience: 10,
  verbose: true,
  onEpochEnd: (epoch, loss, valLoss) => {
    console.log(`epoch=${epoch} loss=${loss} valLoss=${valLoss}`);
  },
});
console.log("best", result.bestEpoch, result.bestLoss);
const pred = model.predict(mj.matrix([[1], [0]]));
pred.print();
```

Backward compatibility tetap tersedia:
```ts
model.fit(X, Y, 200, (loss) => console.log("loss", loss));
```

## Core concepts
- **Shape convention**: mayoritas layer menggunakan `[rows, cols]`; sample batched untuk transformer direpresentasikan dalam layout kolom sequence.
- **Recurrent convention**: recurrent layer menerima satu sample sequence dengan shape `[features, seqLen]`. `Sequential.fit()` generic belum mendukung batching sequence recurrent, jadi gunakan `batchSize=1`.
- **Sparse target untuk klasifikasi**: gunakan `softmaxCrossEntropy` (dense output + target indeks `[1, batch]`).
- **Mode training/eval**: `model.train()` dan `model.eval()` memengaruhi layer seperti `Dropout`.

## Example usage
### Matrix + math
```ts
import mj from "./src/math";

const a = mj.matrix([[1, 2], [3, 4]]);
const b = mj.matrix([[5, 6], [7, 8]]);
const c = mj.dotProduct(a, b);
const d = mj.add(c, 1);
console.log(c._shape, d._shape);
```

### Tokenizer BPE
```ts
import { BPETokenizer } from "./src/tokenizer";

const tokenizer = new BPETokenizer({ vocabSize: 120, minFrequency: 2 });
tokenizer.train(["saya makan nasi", "saya makan roti"]);
const ids = tokenizer.encodeWithSpecial("saya makan nasi");
const padded = tokenizer.padSequence(ids, 12);
console.log(ids, padded, tokenizer.decode(ids));
```

### Transformer causal LM training
```ts
import mj from "./src/math";
import { Transformers } from "./src/models";

const model = new Transformers({ units: 64, seqLen: 8, vocabSize: 500, heads: 8, alpha: 0.001, padTokenId: 0 });
model.compile({ alpha: 0.001, optimizer: "adam", error: "softmaxCrossEntropy" });
model.train();

const x = mj.matrix([[0], [0], [10], [20], [30], [40], [50], [60]]); // [seqLen, 1]
const y = mj.matrix([[0], [10], [20], [30], [40], [50], [60], [0]]); // shifted targets [seqLen, 1]

const logits = model.forward(x); // [vocabSize, seqLen * batch]
model.backward(y);
console.log("shape", logits._shape, "loss", model.loss);
```

### Transformer generation / inference
```ts
import mj from "./src/math";
import { Transformers } from "./src/models";

const model = new Transformers({
  units: 64,
  seqLen: 8,
  vocabSize: 500,
  heads: 8,
  alpha: 0.001,
  padTokenId: 0,
  predictMode: "next-token",
});
model.eval();

const x = mj.matrix([[0], [0], [10], [20], [30], [40], [50], [60]]);
const nextTokenLogits = model.predict(x); // [vocabSize, batch]
model.setPredictMode("full-sequence");
const fullSequenceLogits = model.predict(x); // [vocabSize, seqLen * batch]
```

## Models overview
- `Sequential`: stack layer umum (dense/embedding/attention/cnn).
- `Transformers`: model transformer bertingkat dengan `numBlocks >= 1`, training full-sequence causal LM, dan inference configurable via `predictMode`.
- `DimentionalityReduction`: turunan `Sequential` dengan pemisahan encoder/decoder via status layer `outputReduction`.

## Layers overview
- `Dense`: FC + activation + optimizer/loss handling.
- `Embedding`: lookup token ID ke embedding vector + dukungan `resize()`.
- `LayerNormalization`: normalisasi per kolom/token.
- `Dropout`: aktif di mode train.
- `PositionalEncoding`: sinusoidal fixed encoding.
- `MultiHeadAttention`/`SelfAttention`: attention mask causal + pad handling.
- `RNN`/`LSTM`/`GRU`: recurrent sequence modeling dengan BPTT, gradient clipping, save/load, dan mode stateful. `returnSequences` didukung; `returnState` saat ini belum didukung dan akan throw eksplisit.

## Tokenizer overview
`BPETokenizer` mendukung:
- training awal (`train`)
- incremental update (`update`)
- encode/decode + special token
- padding sequence
- persist ke file JSON (`save/load`)

## Training workflow
1. Siapkan data (`Matrix` input + target).
2. Bangun model + layer.
3. `compile({ alpha, optimizer, error })`.
4. Iterasi `forward()` + `backward()` atau `fit()`.
5. Simpan model/tokenizer (`save`).

## Inference workflow
1. Muat model/tokenizer.
2. Ubah input ke token/matrix.
3. `predict()` atau `forward()`.
   Untuk `Transformers`, `predict()` mengikuti `predictMode`.
4. Ambil argmax/logit sesuai kebutuhan task.
5. Decode token ke teks (jika NLP).

## Performance notes
- Native Rust mempercepat dot-product, activation, layernorm, embedding, attention, optimizer hotpath.
- `Matrix` menggunakan `Float32Array` untuk mengurangi overhead alokasi.
- Beberapa layer menggunakan pre-allocated buffer untuk menekan GC.
- **Dynamic padding trim** (`trimPadding: true`, default): mengurangi `effectiveSeqLen` per batch, sehingga attention cost turun dari O(seqLen²) ke O(effectiveSeqLen²) dan dense output cost turun dari `vocabSize × seqLen × batch` ke `vocabSize × effectiveSeqLen × batch`.

## Dynamic Padding Trim (v2.2.0+)

Untuk training Transformer full-sequence causal LM dengan context panjang (mis. seqLen=1024), aktifkan `trimPadding`:

```ts
import { Transformers } from "./src/models";

const model = new Transformers({
  units: 64,
  seqLen: 1024,
  vocabSize: 5000,
  heads: 8,
  numBlocks: 2,
  padTokenId: 0
});

// Right-padding (direkomendasikan untuk dataset baru)
model.fit(trainX, trainY, 80, {
  batchSize: 8,
  trimPadding: true,
  paddingSide: "right",
  shuffle: true
});

// Left-padding (untuk dataset lama)
model.fit(trainX, trainY, 80, {
  batchSize: 8,
  trimPadding: true,
  paddingSide: "left",
  shuffle: true
});
```

**Catatan:**
- `trimPadding = true` (default) – aktif secara otomatis.
- `paddingSide = "right"` (default) – trailing PAD dipotong; positionOffset = 0.
- `paddingSide = "left"` – leading PAD dipotong; positionOffset disesuaikan agar positional encoding token asli tidak berubah.
- Untuk menonaktifkan: `trimPadding: false`.
- Hanya aktif untuk full-sequence target `Y=[seqLen, batch]`; legacy `Y=[1, batch]` tidak di-trim.

## Benchmark workflow
- Entry point benchmark dan correctness sekarang ada di [test/index.ts](./test/index.ts).
- Suite correctness ada di [test/correctness](./test/correctness/index.ts).
- Suite benchmark sintetis ada di [test/benchmark](./test/benchmark/index.ts).
- Jalankan seluruh suite dengan `npm test`.
- Benchmark model recurrent ada di [test/benchmark/testFamilyRnn.test.ts](./test/benchmark/testFamilyRnn.test.ts).
- Benchmark transformer mode ada di [test/benchmark/testFamilyTransformers.test.ts](./test/benchmark/testFamilyTransformers.test.ts).
- Histori benchmark dibekukan di [docs/benchmark-sintetis/README.md](./docs/benchmark-sintetis/README.md) dan correctness companion di [docs/correctness/README.md](./docs/correctness/README.md).

## Best practices
- Gunakan `softmaxCrossEntropy` untuk klasifikasi sparse token.
- Konsistenkan `seqLen` antara preprocessing dan model constructor.
- Tetapkan `padTokenId` di tokenizer + model embedding.
- Untuk `Transformers`, siapkan target shifted next-token dengan shape `[seqLen, batch]` dan isi posisi yang tidak valid dengan `padTokenId`.
- Gunakan `model.train()` untuk training full-sequence.
- Untuk inferensi transformer, gunakan `model.predict()` sebagai entry point utama dan atur `predictMode` ke `"next-token"` atau `"full-sequence"` sesuai kebutuhan.
- Untuk recurrent `stateful`, hindari `shuffle=true` dan `validationSplit > 0` di loop `Sequential.fit()` generic saat ini.
- Awali debug dengan `ML_DISABLE_NATIVE=1` saat membandingkan perilaku JS vs native.
- Cek shape di setiap boundary layer bila loss tidak turun.

## Troubleshooting
- **`Native backend not available`**: jalankan `npm run build:rust` atau pastikan `.node` binary cocok platform.
- **Shape mismatch dot product**: validasi dimensi `[aRows x aCols] * [bRows x bCols]` (harus `aCols === bRows`).
- **Loss NaN/Inf**: kecilkan `alpha`, cek target format, cek token out-of-range pada embedding.

## 📖 Panduan Lengkap (Guide-Line)

Untuk memahami library ini secara mendalam, silakan baca panduan resmi kami:

1.  **[Overview & Filosofi](docs/GUIDE-LINE/01-overview.md)**: Pengenalan dasar dan arsitektur sistem.
2.  **[Instalasi & Setup](docs/GUIDE-LINE/02-installation.md)**: Cara menginstal dan mengaktifkan akselerasi Rust Native.
3.  **[Tutorial Praktis](docs/GUIDE-LINE/03-tutorial.md)**: Langkah demi langkah membangun bot logika dan bot generatif (GPT-style).
4.  **[Referensi API Lengkap](docs/GUIDE-LINE/04-api-functions.md)**: Dokumentasi teknis Matrix, Math, Layers, dan Tokenizer.

## Development / contribution
```bash
npm install
npm test
npm run build:rust
```

Catatan status saat audit dokumentasi ini:
- `npm test` sekarang menjalankan correctness suite lalu synthetic benchmark dari satu entry `test/index.ts`.
- `npx tsc --noEmit` lulus pada snapshot dokumentasi ini.

## Roadmap / future improvements
- Stabilkan API entry point publik (saat ini impor utama melalui `src/*`).
- Tambah test deterministic untuk numerik floating.
- Rapikan script yang merujuk folder proyek yang belum ada.
- Tambah dokumentasi dataset recipe dan workflow benchmark.

## License / support / credits
- License: `ISC` (mengacu `package.json`).
- Backend native: `napi-rs`, `matrixmultiply`, `rayon`.
- Dukungan: gunakan issue tracker repository untuk bug/fitur.
- Donasi:
  [![Saweria](https://img.shields.io/badge/Saweria-Donasi-orange?style=for-the-badge&logo=saweria)](https://saweria.co/akhyaruhui)
