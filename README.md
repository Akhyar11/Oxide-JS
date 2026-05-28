# Oxide-JS (formerly ML-V1)

<p align="center">
  <img src="./OxideJS-logo-benner.png" width="100%" alt="Oxide-JS Banner" />
</p>

> **Historical Note:** This project was originally published as **ML-V1**. Version [v2.3.0](https://github.com/Akhyar11/ML-V1/releases/tag/v2.3.0) represents the stable research artifact for recurrent network stability evaluations.

> A TypeScript + Rust Native machine learning library — Matrix operations, neural network layers, Transformer models, and a BPE tokenizer, all in one package.

[![npm version](https://img.shields.io/npm/v/@akhyar11/oxide-js?style=flat-square)](https://www.npmjs.com/package/@akhyar11/oxide-js)
[![License: ISC](https://img.shields.io/badge/License-ISC-blue?style=flat-square)](https://opensource.org/licenses/ISC)
[![Node.js >=18](https://img.shields.io/badge/node-%3E%3D18-brightgreen?style=flat-square)](https://nodejs.org)

---

## What is Oxide-JS?

**Oxide-JS** is a low-to-mid-level machine learning library built with TypeScript and accelerated by a Rust native backend (via [napi-rs](https://napi.rs/)). It gives you full control over every detail of the training loop — shapes, parameter updates, and custom architectures — without depending on a large ML framework.

**Why Oxide-JS?**
- Full manual control over training loops, tensor shapes, and parameter updates.
- A research playground for custom model architectures.
- The productivity of TypeScript combined with Rust performance on hot paths.
- Graceful fallback to pure JavaScript when the native backend is unavailable.

---

## Features

- **`Matrix`** — Flat `Float32Array`-backed tensor with zero-copy hot-path access via `_data` and dynamic autodiff tracking.
- **Math primitives** — `dotProduct`, `add`, `sub`, `sum`, `clipGradients`, and more; automatically dispatched to Rust native kernels or pure JS fallback.
- **Layers Catalog** — `Dense`, `Dropout`, `Embedding`, `LayerNormalization`, `BatchNormalization`, `Conv1D`, `Conv2D`, `MaxPooling1D`, `MaxPooling2D`, `AveragePooling1D`, `AveragePooling2D`, `Reshape`, `Flatten`, `SimpleRNN`, `GRU`, `LSTM`, `Attention`, `MultiHeadAttention`, and `Residual` skip connections.
- **Auto-Diff Engine (Tape)** — Dynamic Gradient Tape to record mathematical operations and run reverse-mode automatic differentiation.
- **Model Containers** — Abstract `BaseModel` for custom neural network architectures and `Sequential` for stacked feed-forward graphs.
- **Callback Observers** — Hook into training epochs and batches with built-in `HistoryCallback`, `ProgressLogger`, and `EarlyStopping` observers.
- **BPE Tokenizer** — Train BPE vocabularies, update incrementally, perform multilingual pre-tokenization (script-aware, unicode-word, etc.), and encode/decode sequences.
- **Keras Interoperability** — Native serialization (`serialize()`, `setWeights()`) to load and save parameters into standard JSON formats.
- **Rust-accelerated ops** — Dot-product, activations, LayerNorm, embedding lookup, attention, and optimizer updates; auto-fallback to JS when unavailable.

---

## Installation

```bash
npm install @oxide-js/core @oxide-js/layers @oxide-js/models @oxide-js/spiking
```

### Prerequisites for Native Acceleration

The library works out of the box with a pure JavaScript fallback. For up to **10× faster** matrix operations, install the Rust toolchain so the native addon can be compiled automatically on `npm install`:

1. **Rust Toolchain** — install via [rustup.rs](https://rustup.rs/).
2. **C/C++ Build Tools** — required by the native binding compiler (e.g. `build-essential` on Linux, Xcode CLI on macOS, MSVC on Windows).

> **Note:** If Rust is not installed, a warning is printed and the library falls back to pure JavaScript automatically. Performance will be noticeably slower for large models.

---

## Building from Source

Oxide-JS uses a monorepo structure with NPM workspaces. To build everything:

```bash
# Install dependencies for all packages
npm install

# Build the native Rust kernels for @oxide-js/core
npm run build:rust

# Build all TypeScript packages (core, layers, models)
npm run build
```

---

## Rust Native Backend

The native backend is loaded by `src/math/rust_backend.ts`. You can check whether it is active at runtime:

```ts
import { isNativeAvailable } from "@oxide-js/core";
console.log("Native active:", isNativeAvailable());
```

To force JavaScript-only execution (useful for debugging or regression comparisons):

```bash
ML_DISABLE_NATIVE=1 node your-script.js
```

---

## Quick Start

### 1. Build a Stacked Sequential Classifier
Compose, compile, and train an XOR classifier in a few lines of code:

```ts
import { Sequential } from "@oxide-js/models";
import { Dense } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";

// 1. Compose stacked feed-forward layers
const model = new Sequential([
  new Dense({ units: 2, outputUnits: 4, activation: "relu" }),
  new Dense({ units: 4, outputUnits: 1, activation: "linear" })
]);

// 2. Compile model settings
model.compile({
  optimizer: "sgd",
  loss: "mse",
  learningRate: 0.1
});

// 3. Prepare XOR datasets
const X = Matrix.fromFlat(new Float32Array([
  0.0, 0.0,
  0.0, 1.0,
  1.0, 0.0,
  1.0, 1.0
]), [4, 2]);

const Y = Matrix.fromFlat(new Float32Array([
  0.0,
  1.0,
  1.0,
  0.0
]), [4, 1]);

// 4. Fit the model stack
console.log("Fitting XOR model...");
model.fit(X, Y, {
  epochs: 200,
  batchSize: 4,
  verbose: 1
});

// 5. Predict outcomes
const predictions = model.predict(X);
console.log("\nPredictions:");
predictions.print();
```

### 2. Build Custom Non-Linear Architectures
To build complex structures (like multi-branch, skip, or residual connections), subclass `BaseModel` directly:

```ts
import { BaseModel } from "@oxide-js/models";
import { Dense } from "@oxide-js/layers";
import { Matrix, mj } from "@oxide-js/core";

class CustomResidualModel extends BaseModel {
  private dense1: Dense;
  private dense2: Dense;

  constructor() {
    super({ name: "custom_residual_model" });
    this.dense1 = new Dense({ units: 2, outputUnits: 4, activation: "relu" });
    this.dense2 = new Dense({ units: 4, outputUnits: 4, activation: "relu" });

    // Register layers to make parameters tracked by the model
    this.add(this.dense1);
    this.add(this.dense2);
  }

  // Define forward routing with skip connection
  public forward(inputs: Matrix, optionsOrTraining?: any): Matrix {
    const h1 = this.dense1.forward(inputs, optionsOrTraining);
    const h2 = this.dense2.forward(h1, optionsOrTraining);
    // Bypass h1 and add it element-wise to h2
    return mj.add(h1, h2);
  }
}
```

---

## Examples

### Matrix & Math Operations

```ts
import { mj } from "@oxide-js/core";

const a = mj.matrix([[1, 2], [3, 4]]);
const b = mj.matrix([[5, 6], [7, 8]]);
const c = mj.dotProduct(a, b);
const d = mj.add(c, 1);
console.log(c._shape, d._shape);
```

### BPE Tokenizer

```ts
import { BPETokenizer } from "@oxide-js/core";

const tokenizer = new BPETokenizer({ vocabSize: 120, minFrequency: 2 });
tokenizer.train(["hello world", "hello there"]);
const ids = tokenizer.encodeWithSpecial("hello world");
const padded = tokenizer.padSequence(ids, 12);
console.log(ids, padded, tokenizer.decode(ids));
```

### Unicode and Multilingual Tokenization

Oxide-JS supports custom and built-in pre-tokenizers for non-Latin text. The default is still `"char"` for backward compatibility; use `"unicode-grapheme"` or `"script-aware"` for multilingual corpora.

Supported modes:
- `char`
- `unicode-grapheme`
- `unicode-word`
- `whitespace`
- `script-aware`

```ts
import { BPETokenizer } from "@oxide-js/core";

const tokenizer = new BPETokenizer({
  vocabSize: 1000,
  preTokenizer: "script-aware"
});

tokenizer.train([
  "hello world",
  "مرحبا بالعالم",
  "こんにちは世界",
  "你好世界",
  "ภาษาไทย",
  "한국어테스트",
  "ꦱꦺꦴꦥꦺꦴ",
  "x² + y² = z²",
  "hello ꦱꦺꦴꦥꦺꦴ 😊 你好"
]);
```

BPE alone is not enough for every writing system. Pre-tokenization is important for scripts without spaces, combining marks, emoji sequences, and mixed text. `script-aware` is a general built-in mode; for language-specific behavior, pass a custom `(text: string) => string[]` pre-tokenizer. `Intl.Segmenter` improves grapheme and word segmentation when the runtime supports it. Fallback behavior is deterministic but may be less linguistically accurate.

### Recurrent Sequence Modeling (LSTM)

Oxide-JS supports sequence processing using modular recurrent layers (`SimpleRNN`, `GRU`, `LSTM`). The following example demonstrates composing a sequence classifier:

```ts
import { Sequential } from "@oxide-js/models";
import { Embedding, LSTM, Dense } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";

// 1. Setup a sequence classifier
const model = new Sequential([
  new Embedding({ vocabSize: 1000, units: 32 }),
  new LSTM({ units: 64, returnSequences: false }),
  new Dense({ units: 64, outputUnits: 5, activation: "softmax" })
]);

// 2. Compile model settings
model.compile({
  optimizer: "adam",
  loss: "softmaxCrossEntropy",
  learningRate: 0.001,
  metrics: ["accuracy"]
});

// inputs: [batch=2, seqLen=4]
const inputs = Matrix.fromFlat(new Float32Array([
  10, 20, 30, 40,
  50, 60, 70, 80
]), [2, 4]);

// targets: one-hot encoded classes [batch=2, classes=5]
const targets = Matrix.fromFlat(new Float32Array([
  0, 0, 1, 0, 0,
  1, 0, 0, 0, 0
]), [2, 5]);

// 3. Train
model.fit(inputs, targets, { epochs: 10, batchSize: 2 });
```

---

## API Overview

### Models

| Model | Description |
|---|---|
| `BaseModel` | Abstract base class that manages compilation, fit steps, state modes, callbacks, parameter counting, and model weights registry. |
| `Sequential` | Streamlined subclass container designed for stacked feed-forward layers sequentially, handling unique layer name generation and shape propagation automatically. |

### Layers

| Layer | Description |
|---|---|
| `Dense` | Fully-connected layer with activation, optimizer, and loss handling. |
| `Embedding` | Token-ID-to-vector lookup with `resize()`, `trainable`, and pretrained `fillWeight()` support. |
| `LayerNormalization` | Per-column/token normalization. |
| `Dropout` | Active only during training mode. |
| `PositionalEncoding` | Fixed sinusoidal positional encoding. |
| `MultiHeadAttention` / `SelfAttention` | Causal attention mask with padding support. |
| `RNN` / `LSTM` / `GRU` | Recurrent sequence modeling with BPTT, gradient clipping, save/load, and stateful mode. `returnSequences` is supported; `returnState` is not yet supported and will throw explicitly. |
| `Flatten` / `Convolution` | Standard CNN building blocks. |

### Spiking Neural Networks (SNN)

| Layer / Model | Description |
|---|---|
| `SpikingDense` | Dense layer dengan fungsi *Leaky Integrate-and-Fire* (LIF), *surrogate gradient*, dan `dotProductAddOnly` native. |
| `SpikingEmbedding` | Lookup spasial untuk NLP spiking, mendukung *Word2Vec CBOW-style Hebbian Contrastive Learning* melalui `learnHebbian()`. |
| `SpikingSentenceEmbedder` | Container model untuk memproses kalimat secara sekuensial menjadi *Semantic Vector* tunggal dengan *Native Contrastive Learning* bawaan. |

### Tokenizer

`BPETokenizer` supports:

| Method | Description |
|---|---|
| `train(corpus)` | Initial BPE training on a string array. |
| `update(corpus)` | Incremental vocabulary update without retraining from scratch. |
| `encode(text)` / `encodeWithSpecial(text)` | Encode text to token IDs, with or without special tokens. |
| `decode(ids)` | Convert token IDs back to text. |
| `padSequence(ids, length)` | Pad or truncate a sequence to a fixed length. |
| `save(path)` / `load(path)` | Persist and restore the tokenizer as a JSON file. |

Tokenizer options:

```ts
type PreTokenizer = (text: string) => string[];

type BuiltInPreTokenizer =
  | "char"
  | "unicode-grapheme"
  | "unicode-word"
  | "whitespace"
  | "script-aware";

type BPETokenizerOptions = {
  vocabSize?: number;
  minFrequency?: number;
  preTokenizer?: BuiltInPreTokenizer | PreTokenizer;
};
```

Built-in pre-tokenizer names are saved in tokenizer JSON files. Custom pre-tokenizer functions are not serialized; saved metadata records `"custom"`, and the same function must be passed again to `BPETokenizer.load(path, { preTokenizer })`.

---

## Core Concepts

- **Shape convention:** Most layers use `[rows, cols]`; sequential layers propagate shapes automatically.
- **Sparse classification targets:** Use `softmaxCrossEntropy` with a dense output layer and a target of shape `[1, batch]` containing class indices.
- **Training / eval mode:** Call `model.train()` before training and `model.eval()` before inference. Layers like `Dropout` and `BatchNormalization` respect this flag.

---

## Training Workflow

1. Prepare data as `Matrix` inputs and targets.
2. Build your model stack (`Sequential`) or custom architecture subclassing `BaseModel`.
3. Call `model.compile({ optimizer, loss, learningRate, metrics })`.
4. Run `model.fit()` (high-level) or loop `forward()` → tape backprop manually.
5. Save the model and tokenizer weights.

## Inference Workflow

1. Load the model and tokenizer.
2. Convert input text to token IDs and pad/trim as required.
3. Call `model.predict()` or `model.forward()`.
4. Extract the argmax class index or raw logits as required by your task.
5. Decode token IDs back to text for NLP tasks.

---

## Performance Notes

- The Rust backend accelerates dot-product, activations, LayerNorm, embedding lookup, attention, and optimizer hot paths.
- `Matrix` uses `Float32Array` to minimize allocation overhead. Use `_data` directly in hot paths.
- Several layers use pre-allocated output buffers to reduce garbage collection pressure.

---

## Dynamic Padding Trim (v2.2.0+)

When processing batches of sequences (e.g., shapes `[seqLen, batchSize]`) with a variable length of padding tokens (e.g. `padTokenId`), you can use the `trimPaddingBatch` utility to trim redundant padding tokens, reducing attention and recurrent projection complexity from O(seqLen²) to O(effectiveSeqLen²):

```ts
import { trimPaddingBatch, Matrix } from "@oxide-js/core";

// inputs: shape [seqLen=10, batch=2]
// targets: shape [seqLen=10, batch=2]
const padId = 0;

const result = trimPaddingBatch(inputs, targets, padId, "right");
if (result.trimmed) {
  const trimmedInputs = result.x;           // shape [effectiveSeqLen, batch]
  const trimmedTargets = result.y;          // shape [effectiveSeqLen, batch]
  const offset = result.positionOffset;     // e.g. 0 for right padding
  console.log(`Trimmed sequence to ${result.effectiveSeqLen}`);
}
```

**Supported Options:**
- `"right"` — Trim trailing padding tokens. `positionOffset` is 0.
- `"left"` — Trim leading padding tokens. `positionOffset` indicates the index of the first useful token, which can be used to shift positional encodings correctly.

---

## Best Practices

- Use `softmaxCrossEntropy` for sparse token classification tasks.
- Keep sequence lengths consistent between your preprocessing pipeline and the model constructor.
- Call `model.train()` before training and `model.eval()` before inference.
- Start debugging with `ML_DISABLE_NATIVE=1` when comparing JS vs. native behavior.
- If loss does not decrease, verify tensor shapes at every layer boundary.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Native backend not available` | Run `npm run build:rust`, or verify that the `.node` binary matches your current platform. |
| Shape mismatch in dot product | Check that dimensions satisfy `[aRows × aCols] · [bRows × bCols]` where `aCols === bRows`. |
| Loss is `NaN` or `Inf` | Reduce the learning rate `alpha`, verify target format, and check for out-of-range token IDs in the embedding. |

---

## Project Structure

```text
packages/
  core/              ← Matrix operations & Rust native acceleration
    src/             ← TypeScript wrappers
    src-rust/        ← Rust native ops (napi-rs)
  layers/            ← Neural network layer implementations
    src/
  models/            ← High-level model compositions
    src/
test/                ← E2E Correctness & Benchmark suite
dataset/             ← Pre-processed datasets for research
docs/                ← In-depth guides and API reference
```

---

## Architecture Overview

| Package | Role |
|---|---|
| `@oxide-js/core` | Core `Matrix` data structure, numeric primitives, and modular Rust backend. |
| `@oxide-js/layers` | Reusable NN layers (Dense, Attention, RNN, etc.) with native acceleration support. |
| `@oxide-js/models` | High-level compositions like `Sequential` and `Transformers`. |
| `@oxide-js/spiking` | Biomimetic Spiking Neural Networks (SNN) dengan LIF, Surrogate Gradient, dan Hebbian Contrastive Learning. |
| `oxide-native` | NAPI-RS binding linking TypeScript to the optimized Rust kernels. |

---

## Benchmark & Testing

- Full test + benchmark entry point: [`test/index.ts`](./test/index.ts) — run with `npm test`.
- Correctness suite: [`test/correctness/index.ts`](./test/correctness/index.ts).
- Synthetic benchmark suite: [`test/benchmark/index.ts`](./test/benchmark/index.ts).
- Recurrent model benchmarks: [`test/benchmark/testFamilyRnn.test.ts`](./test/benchmark/testFamilyRnn.test.ts).
- Transformer mode benchmarks: [`test/benchmark/testFamilyTransformers.test.ts`](./test/benchmark/testFamilyTransformers.test.ts).
- Benchmark history: [`docs/benchmark-sintetis/README.md`](./docs/benchmark-sintetis/README.md).
- Correctness history: [`docs/correctness/README.md`](./docs/correctness/README.md).

---

## 📖 Documentation & API Reference

For in-depth guides and complete technical specifications, see the official documentation workspace:

1. **[Master API Navigation Hub](docs/README.md)** — The central entrypoint to all API specifications.

### 🗺️ API Documentation Map

The API documentation is fully decoupled and detailed with complete mathematical explanations, config typings, lifecycle triggers, and standalone copy-pasteable TypeScript code examples for every symbol:

* **[📥 @oxide-js/core](docs/README.md#🧭-oxide-jscore-api-navigation-directory)**:
  - **[Matrix Operations](docs/api/core/matrix.md)** — Flat Float32Array arrays, shape modifications, and zero-copy bindings.
  - **[Math Dispatcher](docs/api/core/math.md)** — JS & Rust native math primitives, zero-skipping logic, and shape calculations.
  - **[Autodiff Tape](docs/api/core/autodiff.md)** — Reverse-mode automatic differentiation tape and dynamic gradients recording.
  - **[BPETokenizer](docs/api/core/tokenizer.md)** — Byte-Pair Encoding text encoding, grapheme-segmentation, and special tokens padding.
  - **[Optimizers Registry](docs/api/core/optimizer.md)** — Monolithic parameters optimization kernels (Adam, SGD, Momentum, NAG, AdaGrad).
  - **[Cost / Loss Functions](docs/api/core/cost.md)** — Mathematical classification and regression loss boundaries (CrossEntropy, Huber, MAE, MSE, hinge).
  - **[Type Definitions](docs/api/core/types.md)** — Monorepo schemas, execution settings, and status descriptors.
* **[🧱 @oxide-js/layers](docs/README.md#🧭-oxide-jslayers-api-navigation-directory)**:
  - **[Base Specification](docs/api/layers/base.md)** — Lifecycle hooks, properties, weights management, and Keras interfaces.
  - **[Core Layers](docs/api/layers/core.md)** — Modular feed-forward elements (Dense, Dropout gates, Activations, Flatten, Reshape).
  - **[Normalization](docs/api/layers/normalization.md)** — Dynamic scaling stabilizers (LayerNormalization, BatchNormalization).
  - **[Sequence Embedding](docs/api/layers/embedding.md)** — Sparse integer token to continuous vector representations lookup tables.
  - **[Convolution & Pooling](docs/api/layers/convolution.md)** — Spatial feature downsizers (Conv1D/2D, MaxPooling1D/2D, AveragePooling1D/2D).
  - **[Recurrent Networks](docs/api/layers/recurrent.md)** — Temporal state transition kernels (SimpleRNN, GRU, LSTM).
  - **[Attention Mechanisms](docs/api/layers/attention.md)** — Scaled dot-product self-attention and causal MultiHeadAttention (MHA) cross-attention.
  - **[Residual Skip Connections](docs/api/layers/residual.md)** — Element-wise additive bypass paths.
* **[📈 @oxide-js/models](docs/README.md#🧭-oxide-jsmodels-api-navigation-directory)**:
  - **[BaseModel Specification](docs/api/models/base.md)** — Base model compiler settings, fitting loops, and Keras binary serialization.
  - **[Sequential Stack](docs/api/models/sequential.md)** — Linear layer compose stack and unique layer naming allocations.
  - **[Callback Observers](docs/api/models/callbacks.md)** — Epoch-level and batch-level listeners (ProgressLogger, EarlyStopping, custom observer hooks).
  - **[Metrics & Data Helpers](docs/api/models/metrics.md)** — Accuracy calculators, training-validation dividers, and mini-batch iterators.
* **[⚡ @oxide-js/spiking](docs/README.md)**:
  - **Spiking Layers** — Biomimetic `SpikingDense`, `SpikingEmbedding` with LIF dynamics, Add-Only ops, and Native Rust kernels.
  - **Sentence Embedder** — `SpikingSentenceEmbedder` for high-performance zero-floating-point Semantic representations and built-in Contrastive CBOW learning.

---

## Versioning

This project follows `MAJOR.MINOR.PATCH` semantic versioning. The current version is **`2.4.0`**.

- **MAJOR** — breaking changes or major architectural shifts.
- **MINOR** — new backward-compatible features or improvements.
- **PATCH** — bug fixes, small optimizations, or minor internal changes.

**Recent changelog:**

| Version | Summary |
|---|---|
| `2.5.0` | **Spiking Neural Networks & Compression**: Penambahan paket `@oxide-js/spiking` untuk AI Biomimetik (LIF, Hebbian Learning), dukungan `learnContrastive` pada SNN Embedder, serta kompresi serialisasi matriks JSON menjadi 1/4 ukuran asli. |
| `2.4.0` | **Interoperability & Auto-Diff Update**: Introduced Keras-style model serialization (`model.json` + `weights.bin`) and Gradient Tape for dynamic automatic differentiation. |
| `2.3.1` | **Modularization Milestone**: Monorepo split (`@oxide-js/core`, `@oxide-js/layers`, `@oxide-js/models`), Modular Rust kernels, and ESM-first test suite. |
| `2.3.0` | Initial Monorepo structure and decoupled Layer Registry. |
| `2.2.8` | Full Native Optimizer support (Adam, SGD, AdaGrad, Momentum, NAG) and Sparse Embedding native backend. |
| `2.2.7` | Unicode-aware BPE pre-tokenizers and multilingual tokenizer documentation. |
| `2.2.5` | Hot-path optimizations for training/validation, embedding lookup, and BPE tokenizer. |
| `2.2.4` | `Transformers.predictMode` API ergonomics, docs sync, and correctness suite refactor. |
| `2.2.3` | Training/inference hot-path optimizations and updated correctness learning snapshots. |
| `2.2.2` | Combined root suite, family model benchmarks, and correctness learning snapshots. |
| `2.2.0` | Dynamic padding trim + positional encoding offset. |
| `2.0.2` | Transformer projector optimizations with no API changes. |

---

## Development & Contributing

```bash
npm install          # install dependencies
npm run build:rust   # compile the Rust native addon
npm test             # run the correctness suite + synthetic benchmark
```

Type-check only (no emit):

```bash
npx tsc --noEmit
```

---

## Roadmap

- Stabilize public API entry points (currently imported directly from `src/*`).
- Add deterministic floating-point tests.
- Clean up scripts that reference non-existent project folders.
- Add dataset recipe documentation and benchmark workflow guides.

---

## License & Credits

- **License:** ISC — see [`package.json`](./package.json).
- **Native backend:** [`napi-rs`](https://napi.rs/), [`matrixmultiply`](https://crates.io/crates/matrixmultiply), [`rayon`](https://crates.io/crates/rayon).
- **Issues & feature requests:** use the [GitHub issue tracker](https://github.com/Akhyar11/Oxide-JS/issues).
- **Support the project:**
  [![Saweria](https://img.shields.io/badge/Saweria-Support-orange?style=for-the-badge&logo=saweria)](https://saweria.co/akhyaruhui)
