# Layers

Neural network layer implementations in ML-V1.

## Import

```ts
import {
  Dense,
  Convolution,
  Activation,
  CompileDenseLayers,
  SelfAttention,
  MultiHeadAttention,
  Embedding,
  Flatten,
  PositionalEncoding,
  LayerNormalization,
  Dropout,
  RNN,
  LSTM,
  GRU,
  AdaptiveMemoryRNN
} from "@akhyar11/ml-v1"
```

## Overview

All layers share a common interface: `forward(input)` for the forward pass and `backward(grad)` for the backward pass. Layers are composable inside `Sequential`, `Transformers`, or `DimentionalityReduction` models, or they can be used standalone in a manual training loop.

---

## API Reference

### `Dense`

Fully-connected layer where every input is connected to every output.

#### `constructor(config)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `units` | `number` | — | Number of input neurons |
| `outputUnits` | `number` | — | Number of output neurons |
| `activation` | `ActivationType` | `"linear"` | Activation function name |
| `alpha` | `number` | — | Layer-specific learning rate |
| `loss` | `Cost` | — | Loss function name for the output layer |
| `optimizer` | `Optimzier` | — | Optimization algorithm name |
| `status` | `StatusLayer` | — | Layer role in the model graph |
| `clipGradient` | `number \| boolean` | `5.0` | Gradient clipping limit for this layer |

```ts
import { Dense } from "@akhyar11/ml-v1"

const layer = new Dense({
  units: 128,
  outputUnits: 64,
  activation: "relu",
  optimizer: "adam"
});
```

> [!IMPORTANT]
> Combining `activation: "softmax"` with `loss: "softmaxCrossEntropy"` will throw an error — softmax would be applied twice.

---

### `Embedding`

Transforms integer token IDs into dense vectors. Required for NLP tasks.

#### `constructor(config)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vocabSize` | `number` | — | Total vocabulary size |
| `embeddingDim` | `number` | — | Vector dimension per token |
| `alpha` | `number` | `0.01` | Layer-specific learning rate |
| `status` | `StatusLayer` | `"input"` | Layer role in the model graph |
| `optimizer` | `Optimzier` | `"adam"` | Optimizer for the embedding table |
| `padTokenId` | `number \| null` | `null` | PAD token ID to skip during backward pass |
| `trainable` | `boolean` | `true` | Freeze weight updates when set to `false` |

```ts
import { Embedding } from "@akhyar11/ml-v1"

const embed = new Embedding({
  vocabSize: 5000,
  embeddingDim: 128,
  trainable: false,
});

embed.fillWeight("./pretrained-embedding.json");
```

Notes:
- `trainable: false` keeps `forward()` active but prevents `backward()` from updating the embedding table.
- `fillWeight()` only replaces the weight matrix. It does not change `trainable`, `alpha`, `optimizer`, `status`, `padTokenId`, `vocabSize`, or `embeddingDim`.
- The incoming weight shape must exactly match `[embeddingDim, vocabSize]`.
- JSON sources for `fillWeight()` must be either an Embedding-layer artifact or a model artifact whose first layer is the Embedding layer.

---

### `MultiHeadAttention`

Causal multi-head self-attention — the core of the Transformer architecture. Allows the model to simultaneously focus on different parts of the input.

#### `constructor(config)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `units` | `number` | — | Internal dimension (must be divisible by `heads`) |
| `heads` | `number` | — | Number of parallel attention heads |
| `seqLen` | `number` | — | Maximum input sequence length |
| `alpha` | `number` | — | Layer-specific learning rate |
| `status` | `StatusLayer` | — | Layer role in the model graph |
| `clipGradient` | `number` | `5.0` | Gradient clipping limit |

```ts
import { MultiHeadAttention } from "@akhyar11/ml-v1"

const attention = new MultiHeadAttention({
  units: 512,
  heads: 8,
  seqLen: 128
});
```

---

### `SelfAttention`

Basic single-head self-attention mechanism. For most use cases prefer `MultiHeadAttention`.

#### `constructor(config)`

| Parameter | Type | Description |
|---|---|---|
| `units` | `number` | Internal dimension |
| `alpha` | `number` | Learning rate |
| `clipGradient` | `number` | Gradient clipping limit |

---

### `LayerNormalization`

Per-column normalization that stabilizes the distribution of values within the network.

#### `constructor(config)`

| Parameter | Type | Description |
|---|---|---|
| `units` | `number` | Number of features to normalize |
| `clipGradient` | `number` | Gradient clipping limit |

```ts
import { LayerNormalization } from "@akhyar11/ml-v1"

const norm = new LayerNormalization({ units: 128 });
```

---

### `Dropout`

Randomly deactivates a fraction of neurons during training to prevent overfitting. Inactive during evaluation mode.

#### `constructor(config)`

| Parameter | Type | Description |
|---|---|---|
| `rate` | `number` | Fraction of neurons to drop (0–1) |

```ts
import { Dropout } from "@akhyar11/ml-v1"

const drop = new Dropout({ rate: 0.1 });
```

---

### `PositionalEncoding`

Injects fixed sinusoidal positional information into the token embeddings. Used in Transformer architectures.

---

### `Flatten`

Flattens a matrix to a single row. Typically placed before a `Dense` layer in a CNN pipeline.

---

### `Convolution`

2D convolution layer for spatial feature extraction.

#### `constructor(config)`

| Parameter | Type | Description |
|---|---|---|
| `kernelSize` | `number` | Size of the convolution kernel |
| `inputShape` | `[number, number]` | Shape of the input matrix |
| `activation` | `ActivationType` | Activation function name |
| `clipGradient` | `number` | Gradient clipping limit |

---

### `Activation`

Standalone activation layer. Wraps any named activation function as a layer.

---

### `CompileDenseLayers`

Utility helper that compiles a list of dense layers with shared optimizer and learning-rate settings. Used internally by `Sequential.compile()`.

---

### Recurrent Layers: `RNN`, `LSTM`, `GRU`
 
Recurrent layers process sequential data. They support both **single-sample (features × seqLen)** and **batched (features × (seqLen * batchSize))** input layouts.

#### General Conventions

- **`returnSequences: false`** — output shape `[hiddenUnits, 1]` (last time step).
- **`returnSequences: true`** — output shape `[hiddenUnits, seqLen]` (all time steps).
- **`stateful: true`** — hidden state is carried over across `forward()` calls until `resetState()` is called.
- **`returnState`** — currently **not supported** across the entire recurrent family; will throw an explicit error.
- `getState()` / `resetState()` — available on all recurrent layers.
- `save()` / `load()` — persist weights, configuration, and stateful hidden states.

#### Batched Training Support (v2.3.0+)

Starting from version **2.3.0**, the entire recurrent family supports high-performance batched training via the native Rust backend. 
- Use **`batchSize > 1`** in `Sequential.fit()` to benefit from parallel sequence processing and "Hyper-Speed" native kernels.
- **RESTRICTION**: If **`stateful: true`**, the layer is still restricted to **`batchSize: 1`** due to persistent hidden state dependencies across batches.

---

#### `RNN`

Basic recurrent layer with one hidden state and Backpropagation Through Time (BPTT).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `units` | `number` | — | Input features per time step |
| `hiddenUnits` | `number` | — | Hidden state size |
| `activation` | `string` | `"tanh"` | Recurrent activation |
| `returnSequences` | `boolean` | `false` | Return output for every time step |
| `stateful` | `boolean` | `false` | Maintain hidden state across calls |
| `optimizer` | `Optimzier` | — | Optimizer name |
| `clipGradient` | `number` | `5.0` | Gradient clipping limit |

```ts
import { RNN, mj } from "@akhyar11/ml-v1"

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

const out = layer.forward(x); // [16, 3] with returnSequences=true
```

---

#### `LSTM`

Recurrent layer with cell state and input/forget/output gates for longer-range dependencies.

| Parameter | Type | Description |
|---|---|---|
| `units` | `number` | Input features per time step |
| `hiddenUnits` | `number` | Hidden and cell state size |
| `forgetBias` | `number` | Initial forget-gate bias for new layers. Default `1` |
| `returnSequences` | `boolean` | Return all time steps |
| `stateful` | `boolean` | Carry state across calls |
| `optimizer` | `Optimzier` | Optimizer name |
| `clipGradient` | `number` | Gradient clipping limit |

`getState()` returns `{ h, c }`.

Notes:
- New `LSTM` layers initialize `bf` to `1` by default for better long-range retention.
- `forgetBias` only affects fresh initialization. `load()` always preserves serialized `bf` from saved weights.

```ts
import { LSTM, mj } from "@akhyar11/ml-v1"

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

layer.resetState();
```

---

#### `GRU`

Recurrent layer with update/reset gates. Also supports **bidirectional** mode.

| Parameter | Type | Description |
|---|---|---|
| `units` | `number` | Input features per time step |
| `hiddenUnits` | `number` | Hidden state size per direction |
| `bidirectional` | `boolean` | Run forward and backward and concatenate outputs |
| `returnSequences` | `boolean` | Return all time steps |
| `stateful` | `boolean` | Carry state across calls |
| `optimizer` | `Optimzier` | Optimizer name |
| `clipGradient` | `number` | Gradient clipping limit |

`getState()` returns `{ forward, backward? }`.

```ts
import { GRU, mj } from "@akhyar11/ml-v1"

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
); // [32, 3]  — 16 forward + 16 backward
```

---

#### `AdaptiveMemoryRNN`

Vanilla RNN core with an external memory bank. This layer is not GRU or LSTM: the recurrent cell remains a simple RNN, then the cell reads from and writes to memory slots with a learned vector gate.

At each time step it builds a query from `[x_t; h_{t-1}]`, retrieves a memory read vector using softmax over memory slots, feeds `concat(x_t, memory_read_t)` into the RNN core, then updates the selected memory slot with gated retention:

```ts
k_i <- (1 - g_t) * k_i + g_t * q_t
v_i <- (1 - g_t) * v_i + g_t * c_t
```

The retrieval is attention-like, but it attends over the layer's memory slots rather than over all input tokens.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `units` | `number` | — | Input features per time step |
| `hiddenUnits` | `number` | — | Vanilla RNN hidden state size |
| `activation` | `"tanh" \| "relu"` | `"tanh"` | RNN core activation |
| `memorySlots` | `number` | `32` | Number of external memory slots |
| `memoryDim` | `number` | `hiddenUnits` | Key/value dimension for each memory slot |
| `returnSequences` | `boolean` | `false` | Return all time steps or only the last hidden state |
| `stateful` | `boolean` | `false` | Carry hidden state and memory bank across calls |
| `optimizer` | `Optimzier` | `"adam"` | Optimizer for trainable recurrent/query/write projections |
| `clipGradient` | `number \| boolean` | `5.0` | Gradient clipping limit |

```ts
import { AdaptiveMemoryRNN, Dense, Embedding, Sequential } from "@akhyar11/ml-v1"

const model = new Sequential({
  layers: [
    new Embedding({ vocabSize: 5000, embeddingDim: 128, padTokenId: 0 }),
    new AdaptiveMemoryRNN({
      units: 128,
      hiddenUnits: 256,
      memorySlots: 32,
      memoryDim: 256,
      returnSequences: false,
    }),
    new Dense({
      units: 256,
      outputUnits: 10,
      activation: "softmax",
      status: "output",
      loss: "crossEntropy",
    }),
  ],
});
```

Gradient support:
- Trainable via optimizer updates: `Wxh`, `Whh`, `bh`, `Wq`, `Wm`, `Wg`, `bg`.
- `memoryKeys`, `memoryValues`, and `memoryUsage` are treated as dynamic recurrent state, not optimizer-trained global parameters.
- Slot selection (`selectWriteSlot`) is discrete. Gradients flow through the selected slot's read/write computations, but not through the slot-choice decision itself.
- The JavaScript path remains the correctness reference, and the native backward path is implemented to match it using the same forward caches.

---

## Notes

- For sequence modeling in `Sequential`, recurrent layers are typically followed by a `Dense` output layer.
- `RNN.getState()` returns a `Matrix`, `LSTM.getState()` returns `{ h, c }`, `GRU.getState()` returns `{ forward, backward? }`.
- `AdaptiveMemoryRNN.getState()` returns `{ h, memoryKeys, memoryValues, memoryUsage }`.
- `save()` / `load()` preserve recurrent weights and stateful hidden states.

---

### `MemoryBank`

Generic, reusable runtime memory layer suitable for insertion after any layer. `MemoryBank` provides a short-term, session-persistent key/value memory that is updated at runtime during `forward()` and can be saved/loaded separately from model weights.

Key properties:
- Auto-infers `units` from the first `forward()` if omitted in constructor.
- Memory state (`memoryKeys`, `memoryValues`, `memoryFilled`, `memoryUsage`, `memoryAge`) is runtime state — it is NOT trained by the model optimizer and is saved via `saveMemory()` / `loadMemory()`.
- Trainable read/output weights (`queryKernel`, `needKernel`, `outputKernel`, `outputBias`) are stored in model `save()` / `load()` and are trained via `backward()`.
- Optional write-side weights (`writeValueKernel`, `writeGateKernel`, `writeKeyKernel`) are only present in the matching modes. They are trainable parameters, but they are still distinct from runtime memory state.
- Memory persists across `forward()` calls until `resetMemory()` is called.

How `MemoryBank` works conceptually:
- Input `x` first becomes a query vector through `queryKernel`.
- That query is compared against stored `memoryKeys`.
- The best matching slots are read and combined into one `read` vector.
- `needKernel` decides how much of that read vector should matter.
- The layer then combines `x` and memory context according to `mode`.
- If writes are enabled, the layer may also write a new key/value pair into one slot.

Constructor config:
- `units?: number` — input rows; if omitted, inferred on first forward.
- `memorySlots: number` — number of memory slots (required).
- `memoryDim?: number` — key/value dim; defaults to `units` when omitted.
- `outputUnits?: number` — output projection size for `mode='project'` or `mode='read-project'`.
- `mode?: "project" | "concat" | "add" | "read-project"` — how memory read result combines with input.
- `similarity?: "cosine" | "dot"` — how query vectors compare against memory keys.
- `readTopK?: number` — top-K memory reads (default `min(4,memorySlots)`).
- `updateMode?: "replace" | "merge" | "gated-merge"` — how an occupied slot is updated.
- `writePolicy?: "empty-first" | "least-used" | "oldest" | "least-relevant"` — how a write target slot is chosen when memory is full.
- `writeThreshold?: number` — gate threshold to trigger writes (default `0.5`).
- `persistence?: "session" | "manual"` — persistence strategy marker for the layer config.
- `writeEnabled?: boolean` — allow writes during forward (default `true`).
- `forceNeedGate?: number` — override learned read-need gate with a fixed value in `[0,1]`.
- `valueMode?: "identity" | "project"` — how new memory values are generated during writes.
- `writeKeyMode?: "shared-query" | "separate-project"` — how new memory keys are generated during writes.
- `writeGateMode?: "always" | "threshold" | "learned"` — how the write decision is made.
- `trainablePolicy?: boolean` — whether policy projection weights are trainable (default `true`).

API (important methods):
- `forward(x: Matrix): Matrix` — read & optionally write memory per column; shapes:
  - `project` => `[outputUnits, cols]`
  - `read-project` => `[outputUnits, cols]`
  - `concat`  => `[units + memoryDim, cols]`
  - `add`     => `[units, cols]` (requires `memoryDim===units`)
- `backward(y: Matrix, err: Matrix): Matrix` — returns `dx` with gradients to inputs and updates differentiable read/output weights.
- `beginSequence({ maxHistorySteps? })` — start explicit sequence-history mode so several `forward()` calls become one unrolled BPTT timeline.
- `backwardSequence(err: Matrix)` — run backward over the currently active sequence history across multiple `forward()` calls.
- `detachSequence()` — keep numeric memory state, but cut gradient/history to earlier sequence steps.
- `endSequence()` — stop sequence mode and clear stored history.
- `getSequenceLength()` / `isSequenceActive()` — inspect active sequence-history state.
- `resetMemory()` / `clearMemory()` — clear runtime memory to empty state.
- `hasMemory()` — returns `boolean` whether any slot is filled.
- `getMemoryState()` / `setMemoryState(state)` — get/set runtime memory snapshot (useful for saving/restoring session memory programmatically).
- `saveMemory(path)` / `loadMemory(path)` — persist runtime memory to a file (JSON).
- `freezeWrites()` / `unfreezeWrites()` / `setWriteFrozen(boolean)` — disable/enable runtime writes while still allowing reads.
- `trainLastWriteKey(targetKey)` — optional supervised helper for directly steering `writeKeyKernel` in `writeKeyMode='separate-project'`.
- `trainLastWriteValue(targetValue)` — optional supervised helper for directly steering `writeValueKernel` in `valueMode='project'`.
- `trainLastWriteGate(targetGate)` — optional supervised helper for directly steering `writeGateKernel` in `writeGateMode='learned'`.

Recommended correctness-first config:

```ts
new MemoryBank({
  memorySlots: 64,
  memoryDim: 32,
  mode: "read-project",
  similarity: "cosine",
  writeKeyMode: "shared-query",
  valueMode: "identity", // only valid if memoryDim === units
  writeGateMode: "always",
  updateMode: "replace",
  writePolicy: "empty-first",
  forceNeedGate: 1,
});
```

This config is the easiest to reason about:
- read path is isolated
- write always happens when enabled
- keys use the same projection space for write and read
- values are stored directly from input
- no hidden learned write gate is blocking memory writes

#### Cara Memilih Nilai Config

Kalau pembaca belum paham teori memory layer, cara paling aman adalah membaca opsi-opsi ini sebagai keputusan praktis:
- output akhir mau dipaksa datang dari memory, atau boleh campur dengan input?
- write mau selalu terjadi, atau hanya saat kondisi tertentu?
- slot lama mau ditimpa penuh, atau dicampur perlahan?
- key saat write mau pakai ruang fitur yang sama dengan query read, atau ruang terpisah?
- value yang disimpan mau raw input, atau hasil proyeksi?

Kalau masih bingung, mulai dari kombinasi ini:
- `mode="read-project"`
- `similarity="cosine"`
- `updateMode="replace"`
- `writePolicy="empty-first"`
- `writeKeyMode="shared-query"`
- `valueMode="identity"` jika `memoryDim === units`
- `writeGateMode="always"`

Alasannya sederhana:
- behavior paling mudah dipahami
- retrieval paling mudah diverifikasi
- tidak ada gate learned yang diam-diam memblok write
- write key dan read query otomatis selaras

#### Meaning of Each Enum

##### `MemoryBankMode`

Ini menentukan bagaimana output akhir dibentuk setelah memory dibaca.

`"project"`
- Arti praktis:
  output dibentuk dari input `x` dan memory context sekaligus.
- Rumus:
  `output = outputKernel * [x; context] + outputBias`
- Kapan dipakai:
  saat kamu ingin `MemoryBank` menjadi bagian normal dari model dan memberi model kebebasan memakai input maupun memory.
- Kapan jangan dipakai:
  saat kamu sedang menguji "apakah model benar-benar memakai memory atau tidak".
- Risiko:
  model bisa terlihat bagus karena jalur input langsung masih kuat, bukan karena memory bekerja.

`"read-project"`
- Arti praktis:
  output dipaksa berasal dari hasil baca memory.
- Rumus:
  `output = outputKernel * read + outputBias`
- Kapan dipakai:
  untuk correctness test, episodic memory test, causal check, dan debugging retrieval.
- Kapan jangan dipakai:
  saat kamu butuh model tetap stabil walau memory belum terisi baik.
- Risiko:
  kalau memory kosong atau salah, performa langsung jatuh, dan itu memang expected.

Perbedaan inti `project` vs `read-project`:
- `project` = output boleh bergantung pada input dan memory
- `read-project` = output dipaksa bergantung pada memory read

`"concat"`
- Arti praktis:
  layer tidak mencampur input dan memory, hanya menempelkan keduanya.
- Rumus:
  `output = [x; context]`
- Kapan dipakai:
  kalau kamu ingin layer berikutnya yang memutuskan cara mencampur input dan memory.
- Risiko:
  output size membesar menjadi `units + memoryDim`.

`"add"`
- Arti praktis:
  memory context langsung ditambahkan ke input seperti residual correction.
- Rumus:
  `output = x + context`
- Kapan dipakai:
  kalau input dan memory context memang hidup di ruang fitur yang sama.
- Syarat:
  `memoryDim === units`
- Risiko:
  kurang fleksibel dibanding `project`.

##### `MemorySimilarity`

Ini menentukan cara query dibandingkan dengan `memoryKeys`.

`"cosine"`
- Arti praktis:
  yang dinilai adalah arah vektor, bukan besar kecilnya.
- Kapan dipakai:
  hampir selalu jadi pilihan awal terbaik untuk retrieval.
- Kenapa aman:
  lebih tahan terhadap perubahan skala magnitude.

`"dot"`
- Arti praktis:
  magnitude ikut mempengaruhi similarity.
- Kapan dipakai:
  kalau magnitude memang ingin kamu jadikan sinyal penting.
- Risiko:
  vektor besar bisa mendominasi walau arahnya tidak paling cocok.

Cara pilih cepat:
- default aman: `"cosine"`
- pakai `"dot"` hanya kalau kamu memang paham kenapa magnitude perlu berpengaruh

##### `MemoryUpdateMode`

Ini menentukan apa yang terjadi jika slot target sudah terisi.

`"replace"`
- Arti praktis:
  isi slot lama dibuang, isi baru masuk penuh.
- Kapan dipakai:
  saat kamu ingin behavior deterministic dan mudah diuji.
- Default terbaik untuk correctness.

`"merge"`
- Arti praktis:
  isi lama dan baru dicampur rata.
- Rumus kasar:
  `0.5 * old + 0.5 * new`
- Kapan dipakai:
  saat kamu ingin memory berubah lebih halus, bukan overwrite penuh.
- Risiko:
  informasi lama dan baru bisa bercampur sehingga interpretasinya makin kabur.

`"gated-merge"`
- Arti praktis:
  isi lama vs baru dicampur berdasarkan gate.
- Rumus kasar:
  `(1-gate) * old + gate * new`
- Kapan dipakai:
  saat kamu ingin partial overwrite pada slot yang sudah terisi.
- Detail penting:
  slot kosong tetap diisi full replace, bukan partial merge.

Cara pilih cepat:
- mau paling jelas: `"replace"`
- mau transisi halus: `"merge"`
- mau transisi halus tapi tergantung gate: `"gated-merge"`

##### `MemoryWritePolicy`

Ini menentukan slot mana yang dipilih saat mau write.

`"empty-first"`
- Arti praktis:
  isi slot kosong dulu; kalau penuh semua, ganti slot yang paling jarang dipakai.
- Kapan dipakai:
  hampir selalu sebagai default awal.
- Kenapa aman:
  mudah diprediksi dan stabil.

`"least-used"`
- Arti praktis:
  slot yang paling sedikit dipakai akan diganti dulu.
- Kapan dipakai:
  kalau kamu ingin recycle slot yang paling jarang berguna.

`"oldest"`
- Arti praktis:
  slot paling lama akan dibuang lebih dulu.
- Kapan dipakai:
  kalau kamu ingin perilaku mirip FIFO atau recency buffer.

`"least-relevant"`
- Arti praktis:
  slot yang paling tidak relevan dengan query sekarang akan diganti.
- Kapan dipakai:
  kalau kamu ingin replacement policy yang context-aware.
- Risiko:
  lebih sulit dipahami dan dianalisis dibanding policy lain.

##### `MemoryPersistence`

Ini bukan backend storage terpisah, tetapi label semantik tentang bagaimana memory diharapkan dikelola.

`"session"`
- Arti praktis:
  memory diharapkan hidup selama sesi runtime sekarang.
- Kapan dipakai:
  untuk pemakaian normal.

`"manual"`
- Arti praktis:
  kamu sendiri yang diharapkan mengatur snapshot memory via `getMemoryState()`, `setMemoryState()`, `saveMemory()`, `loadMemory()`.
- Kapan dipakai:
  kalau kamu ingin kontrol eksplisit atas serialize/restore memory.

Catatan penting:
- secara implementasi, keduanya tetap memakai runtime memory state yang sama
- ini lebih ke penanda intent/config daripada mode penyimpanan yang benar-benar berbeda

##### `MemoryValueMode`

Ini menentukan bagaimana `memoryValues` baru dibuat saat write.

`"identity"`
- Arti praktis:
  yang disimpan adalah input itu sendiri.
- Rumus:
  `newValue = x`
- Kapan dipakai:
  saat input memang sudah merupakan representasi yang ingin disimpan.
- Sangat cocok untuk:
  deterministic write-read test dan episodic toy task.
- Syarat:
  `memoryDim === units`

`"project"`
- Arti praktis:
  input diubah dulu sebelum disimpan sebagai value.
- Rumus:
  `newValue = writeValueKernel * x`
- Kapan dipakai:
  saat value memory memang perlu berada di ruang fitur berbeda dari input.
- Risiko:
  lebih sulit dipahami, karena sekarang ada projection tambahan di jalur write.

Perbedaan inti:
- `identity` = simpan apa adanya
- `project` = proyeksikan dulu baru simpan

##### `MemoryWriteKeyMode`

Ini menentukan bagaimana key baru dibuat saat write.

`"shared-query"`
- Arti praktis:
  write key memakai projection yang sama dengan query read.
- Rumus:
  `newKey = queryKernel * x`
- Kapan dipakai:
  hampir selalu sebagai default awal.
- Kenapa aman:
  key space write dan read otomatis selaras.

`"separate-project"`
- Arti praktis:
  write key punya projection sendiri.
- Rumus:
  `newKey = writeKeyKernel * x`
- Kapan dipakai:
  kalau kamu memang butuh kebijakan key write yang berbeda dari query read.
- Risiko:
  memory bisa terisi, tapi query nanti gagal menemukannya karena write key dan read query drift.

Perbedaan inti:
- `shared-query` = write key dan read query hidup di ruang yang sama
- `separate-project` = write key punya ruang projection sendiri

##### `MemoryWriteGateMode`

Ini menentukan kapan write benar-benar terjadi.

`"always"`
- Arti praktis:
  selama writes enabled dan tidak frozen, layer selalu menulis.
- Kapan dipakai:
  untuk correctness, debugging, dan causal verification.
- Kenapa aman:
  tidak ada learned gate yang diam-diam memblok write.

`"threshold"`
- Arti praktis:
  write terjadi kalau skor `need` cukup tinggi.
- Kapan dipakai:
  saat kamu ingin selective write yang tetap sederhana dan deterministic.
- Karakter:
  lebih selektif dari `always`, tapi masih mudah dijelaskan.

`"learned"`
- Arti praktis:
  ada projection gate khusus yang memutuskan kapan harus write.
- Rumus:
  `gate = sigmoid(writeGateKernel * [x; read])`
- Kapan dipakai:
  kalau deterministic policy sudah terbukti benar dan kamu memang butuh write policy yang lebih fleksibel.
- Risiko:
  debugging jadi lebih sulit karena kegagalan write bisa datang dari learned gate, bukan dari read path.

Perbedaan inti:
- `always` = selalu tulis
- `threshold` = tulis kalau need cukup tinggi
- `learned` = biarkan model memutuskan kapan menulis

#### Which Parts Are Trainable vs Runtime State?

Trainable by `backward()`:
- `queryKernel`
- `needKernel`
- `outputKernel`
- `outputBias`
- `writeValueKernel` when `valueMode="project"` and the current sequence contains future loss that depends on previous writes
- `writeGateKernel` when `writeGateMode="learned"` and the current sequence contains future loss that depends on previous writes
- `writeKeyKernel` when `writeKeyMode="separate-project"` and the current sequence contains future loss that depends on previous writes

Optionally trainable through explicit supervised helper APIs:
- `trainLastWriteValue(...)`
- `trainLastWriteGate(...)`
- `trainLastWriteKey(...)`

Runtime state only, not optimizer weights:
- `memoryKeys`
- `memoryValues`
- `memoryFilled`
- `memoryUsage`
- `memoryAge`
- `memoryStep`

This distinction is important:
- memory content changes because the layer writes to memory during `forward()`
- memory content does not change because Adam/SGD optimized it

#### Save/Load Behavior

- `save()` stores:
  config, dimensions, trainable parameters, optimizer settings, and current memory state.
- `load()` restores all of the above.
- `saveMemory()` / `loadMemory()` only store and restore runtime memory state.

#### Sequence-History BPTT Across Multiple `forward()` Calls

By default, one `forward(x)` call is one training sequence for `MemoryBank.backward(...)`.

If you want gradient to flow across several separate `forward()` calls, use sequence mode:

```ts
memory.beginSequence({ maxHistorySteps: 64 });

memory.forward(step1);
memory.forward(step2);
memory.forward(step3);

memory.backwardSequence(sequenceErr);
memory.endSequence();
```

What this means:
- each `forward()` appends its step caches to an active sequence-history buffer
- `backwardSequence(...)` walks that buffer backward, so a read/loss in a later call can train writes from earlier calls
- `detachSequence()` keeps current memory contents but clears gradient history, similar to truncated BPTT in recurrent training

When to use it:
- stateful episodic tasks split across multiple `forward()` calls
- training loops that naturally emit one step at a time

When you do not need it:
- if your whole training sequence already fits into one `forward(x)` call with multiple columns

#### Simple Usage Example

`MemoryBank` is a special-case layer and is not supported inside `Sequential`.
Use it in a manual/custom model loop where you control `forward()`, `backward()`, and optional sequence-history boundaries yourself.

For a first experiment, use this:

```ts
import { Dense, MemoryBank, mj } from "@akhyar11/ml-v1";

const encoder = new Dense({
  units: 10,
  outputUnits: 32,
  activation: "relu",
  status: "input",
});

const memory = new MemoryBank({
  memorySlots: 64,
  memoryDim: 32,
  mode: "read-project",
  similarity: "cosine",
  writeKeyMode: "shared-query",
  valueMode: "identity",
  writeGateMode: "always",
  updateMode: "replace",
  writePolicy: "empty-first",
  forceNeedGate: 1,
});

const head = new Dense({
  units: 32,
  outputUnits: 3,
  activation: "linear",
  status: "output",
  loss: "softmaxCrossEntropy",
});

const x = mj.matrix(new Array(10).fill(0).map(() => new Array(2).fill(Math.random())));
const z = encoder.forward(x);
const m = memory.forward(z);
const y = head.forward(m);
```

If `memoryDim !== units`, then `valueMode: "identity"` is invalid, so change it to:

```ts
valueMode: "project"
```

Manual backward sketch:

```ts
const errHead = head.backward(target, mj.matrix([[]]));
const errMemory = memory.backward(target, errHead);
encoder.backward(target, errMemory);
```

Notes & design constraints:
- `Sequential` intentionally rejects `MemoryBank`. Build a custom training loop instead.
- Memory content is runtime state, not optimizer state.
- `MemoryBank` is intentionally generic — it does not assume sequences, batches, or tokens. Inputs are handled as `[features, columns]`.
- Slot selection and replacement are non-differentiable runtime operations. The backward pass computes exact gradients through the read/output path using the memory snapshot captured during forward.
- The current implementation is correctness-first:
  read/output path is differentiable through `backward()`, and write-side learned policies also receive exact within-sequence gradient flow when later columns in the same `forward()` depend on earlier writes.
- Important limitation:
  this is sequence-level BPTT inside one `forward()` call. It is not an unlimited autograd graph across separate `forward()` calls from different training steps.
