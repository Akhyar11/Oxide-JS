# ML-V1 (TypeScript + Rust Native)

Library machine learning custom berbasis TypeScript dengan backend Rust (N-API) untuk akselerasi operasi numerik kritikal.

## What this project is
ML-V1 adalah library low-level sampai mid-level untuk eksperimen dan pengembangan model ML secara manual: `Matrix`, math ops, layer, model, tokenizer BPE, dan pipeline worker-thread untuk workload Transformer.

## Why this project exists
- Menyediakan kontrol penuh atas detail training loop, shape, dan update parameter.
- Menjadi playground riset arsitektur custom tanpa dependency framework ML besar.
- Menggabungkan kemudahan TypeScript dengan performa Rust untuk hot paths.

## Key features
- `Matrix` berbasis `Float32Array` (flat contiguous memory).
- Operasi math inti (`dotProduct`, `add`, `sumAxis`, `clipGradients`, dst).
- Layer: `Dense`, `Embedding`, `SelfAttention`, `MultiHeadAttention`, `LayerNormalization`, `Dropout`, `PositionalEncoding`, `Flatten`, `Convolution`.
- Model: `Sequential`, `Transformers`, `DimentionalityReduction`.
- Tokenizer BPE (`train`, `update`, `encode`, `decode`, `padSequence`, `save/load`).
- Native Rust fallback-aware (otomatis ke JS jika native tidak tersedia).
- Pipeline paralel berbasis worker threads (`src/pipeline/transformer-pipeline.ts`).

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
  tokenizer/   pipeline/
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

model.fit(X, Y, 200, (loss) => console.log("loss", loss));
const pred = model.predict(mj.matrix([[1], [0]]));
pred.print();
```

## Core concepts
- **Shape convention**: mayoritas layer menggunakan `[rows, cols]`; sample batched untuk transformer direpresentasikan dalam layout kolom sequence.
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

### Transformer next-token
```ts
import mj from "./src/math";
import { Transformers } from "./src/models";

const model = new Transformers({ units: 64, seqLen: 8, vocabSize: 500, heads: 8, alpha: 0.001, padTokenId: 0 });
model.compile({ alpha: 0.001, optimizer: "adam", error: "softmaxCrossEntropy" });

const x = mj.matrix([[0], [0], [10], [20], [30], [40], [50], [60]]); // [seqLen, 1]
const y = mj.matrix([[70]]); // next token index

model.forward(x);
model.backward(y);
console.log("loss", model.loss);
```

## Models overview
- `Sequential`: stack layer umum (dense/embedding/attention/cnn).
- `Transformers`: satu blok transformer (embedding + PE + pre-norm MHA + FFN + output projector).
- `DimentionalityReduction`: turunan `Sequential` dengan pemisahan encoder/decoder via status layer `outputReduction`.

## Layers overview
- `Dense`: FC + activation + optimizer/loss handling.
- `Embedding`: lookup token ID ke embedding vector + dukungan `resize()`.
- `LayerNormalization`: normalisasi per kolom/token.
- `Dropout`: aktif di mode train.
- `PositionalEncoding`: sinusoidal fixed encoding.
- `MultiHeadAttention`/`SelfAttention`: attention mask causal + pad handling.

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
- Pipeline worker-thread tersedia untuk workload transformer batch.

## Best practices
- Gunakan `softmaxCrossEntropy` untuk klasifikasi sparse token.
- Konsistenkan `seqLen` antara preprocessing dan model constructor.
- Tetapkan `padTokenId` di tokenizer + model embedding.
- Awali debug dengan `ML_DISABLE_NATIVE=1` saat membandingkan perilaku JS vs native.
- Cek shape di setiap boundary layer bila loss tidak turun.

## Troubleshooting
- **`Native backend not available`**: jalankan `npm run build:rust` atau pastikan `.node` binary cocok platform.
- **Shape mismatch dot product**: validasi dimensi `[aRows x aCols] * [bRows x bCols]` (harus `aCols === bRows`).
- **Loss NaN/Inf**: kecilkan `alpha`, cek target format, cek token out-of-range pada embedding.
- **Script `project/math-bot/*` gagal**: folder tersebut tidak ada di snapshot repo saat ini.

## Documentation pointer
Dokumentasi lengkap ada di:
- `docs/v1.1.0(open-source-production-docs)/overview.md`
- `docs/v1.1.0(open-source-production-docs)/quickstart.md`
- `docs/v1.1.0(open-source-production-docs)/api-reference.md`

## Development / contribution
```bash
npm install
npm test
npm run build:rust
```

Catatan status saat audit dokumentasi ini:
- `npm test` punya 1 kegagalan presisi floating (`log(e)=1` ~ `0.99999994`).
- `npx tsc --noEmit` gagal karena import `project/math-bot/main` tidak ditemukan di test tertentu.

## Roadmap / future improvements
- Stabilkan API entry point publik (saat ini impor utama melalui `src/*`).
- Tambah test deterministic untuk numerik floating.
- Rapikan script yang merujuk folder proyek yang belum ada.
- Tambah dokumentasi formal untuk pipeline worker dan dataset recipe.

## License / support / credits
- License: `ISC` (mengacu `package.json`).
- Backend native: `napi-rs`, `matrixmultiply`, `rayon`.
- Dukungan: gunakan issue tracker repository untuk bug/fitur.
