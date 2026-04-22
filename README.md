# ML-V1 (TypeScript + Rust Native)

Library machine learning custom berbasis TypeScript dengan backend Rust (N-API) untuk akselerasi operasi numerik kritikal.

## What this project is
ML-V1 adalah library low-level sampai mid-level untuk eksperimen dan pengembangan model ML secara manual: `Matrix`, math ops, layer, model, dan tokenizer BPE.

## Why this project exists
- Menyediakan kontrol penuh atas detail training loop, shape, dan update parameter.
- Menjadi playground riset arsitektur custom tanpa dependency framework ML besar.
- Menggabungkan kemudahan TypeScript dengan performa Rust untuk hot paths.

## Versioning
Versi aktif proyek saat ini adalah `2.0.0`.

Proyek ini memakai format versi `MAJOR.MINOR.PATCH` seperti `2.0.0`.

- Angka paling depan (`MAJOR`): perubahan besar yang biasanya membawa breaking change atau perubahan arsitektur utama.
- Angka tengah (`MINOR`): penambahan fitur baru atau peningkatan yang tetap kompatibel dengan versi sebelumnya.
- Angka paling belakang (`PATCH`): perbaikan bug, optimasi kecil, cleanup, atau perubahan minor yang tidak mengubah API utama.

Contoh:
- `2.0.0`: rilis mayor `2`, minor `0`, patch `0` untuk major update transformer: full-sequence causal LM, kernel native masked sparse softmax-cross-entropy, dan jalur inference khusus LM.
- `1.1.4`: masih di mayor `1` dan minor `1`, tetapi sudah ada 4 patch/perbaikan kecil dari baseline `1.1.0`.

## Key features
- `Matrix` berbasis `Float32Array` (flat contiguous memory).
- Operasi math inti (`dotProduct`, `add`, `sumAxis`, `clipGradients`, dst).
- Layer: `Dense`, `Embedding`, `RNN`, `LSTM`, `GRU`, `SelfAttention`, `MultiHeadAttention`, `LayerNormalization`, `Dropout`, `PositionalEncoding`, `Flatten`, `Convolution`.
- Model: `Sequential`, `Transformers`, `DimentionalityReduction`.
- Tokenizer BPE (`train`, `update`, `encode`, `decode`, `padSequence`, `save/load`).
- Native Rust fallback-aware (otomatis ke JS jika native tidak tersedia).

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
```bash
npm install
```

## Build and setup
```bash
# Build native Rust (release)
npm run build:rust

# Debug native build
npm run build:rust:debug
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

const model = new Transformers({ units: 64, seqLen: 8, vocabSize: 500, heads: 8, alpha: 0.001, padTokenId: 0 });
model.eval();

const x = mj.matrix([[0], [0], [10], [20], [30], [40], [50], [60]]);
const nextTokenLogits = model.predict(x); // [vocabSize, batch]
// setara dengan model.forwardNextToken(x)
```

## Models overview
- `Sequential`: stack layer umum (dense/embedding/attention/cnn).
- `Transformers`: satu blok transformer (embedding + PE + pre-norm MHA + FFN + output projector) dengan training full-sequence causal LM dan inference last-token.
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
4. Ambil argmax/logit sesuai kebutuhan task.
5. Decode token ke teks (jika NLP).

## Performance notes
- Native Rust mempercepat dot-product, activation, layernorm, embedding, attention, optimizer hotpath.
- `Matrix` menggunakan `Float32Array` untuk mengurangi overhead alokasi.
- Beberapa layer menggunakan pre-allocated buffer untuk menekan GC.

## Benchmark workflow
- Entry point benchmark dan correctness sekarang ada di [test/index.ts](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/test/index.ts:1).
- Suite correctness ada di [test/correctness](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/test/correctness/index.ts:1).
- Suite benchmark sintetis ada di [test/benchmark](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/test/benchmark/index.ts:1).
- Jalankan seluruh suite dengan `npm test`.
- Benchmark utama memakai harness [test/benchmark/synthetic_baseline_benchmark.ts](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/test/benchmark/synthetic_baseline_benchmark.ts:1) dan histori hasilnya dicatat di [docs/benchmark-sintetis/README.md](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/docs/benchmark-sintetis/README.md:1).
- Benchmark pembanding transformer yang lebih detail tersedia di [test/benchmark/transformer_perf_breakdown.ts](/home/akhyar/Dokumen/Code/NODE%20JS/ML_V2/test/benchmark/transformer_perf_breakdown.ts:1) untuk inference-only, forward-only, backward-only, training-step, dan stage profile apple-to-apple.

## Best practices
- Gunakan `softmaxCrossEntropy` untuk klasifikasi sparse token.
- Konsistenkan `seqLen` antara preprocessing dan model constructor.
- Tetapkan `padTokenId` di tokenizer + model embedding.
- Untuk `Transformers`, siapkan target shifted next-token dengan shape `[seqLen, batch]` dan isi posisi yang tidak valid dengan `padTokenId`.
- Gunakan `model.train()` untuk training full-sequence dan `model.predict()` / `model.forwardNextToken()` untuk inferensi generatif berbasis last-token.
- Untuk recurrent `stateful`, hindari `shuffle=true` dan `validationSplit > 0` di loop `Sequential.fit()` generic saat ini.
- Awali debug dengan `ML_DISABLE_NATIVE=1` saat membandingkan perilaku JS vs native.
- Cek shape di setiap boundary layer bila loss tidak turun.

## Troubleshooting
- **`Native backend not available`**: jalankan `npm run build:rust` atau pastikan `.node` binary cocok platform.
- **Shape mismatch dot product**: validasi dimensi `[aRows x aCols] * [bRows x bCols]` (harus `aCols === bRows`).
- **Loss NaN/Inf**: kecilkan `alpha`, cek target format, cek token out-of-range pada embedding.
- **Script `project/math-bot/*` gagal**: folder tersebut tidak ada di snapshot repo saat ini.

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
