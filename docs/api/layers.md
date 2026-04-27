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
  GRU
} from "@akhyar11/ml-v1"
```

## Overview

All layers share a common interface: `forward(input)` for the forward pass and `backward(grad)` for the backward pass. Layers are composable inside `Sequential`, `Transformers`, or `DimentionalityReduction` models, or they can be used standalone in a manual training loop.

### Common Methods

| Method | Description |
|---|---|
| `forward(input: Matrix, options?)` | Executes the forward pass. Returns the output `Matrix`. |
| `backward(y: Matrix, err: Matrix)` | Executes the backward pass (BPTT for recurrent). Updates gradients internally. |
| `dispose()` | **New in v2.3.0.** Explicitly releases memory for internal buffers and matrices. Use this to prevent memory growth in long-running processes. |
| `releaseWorkspace()` | Releases temporary workspace buffers while keeping persistent weights. |
| `resetLoss()` | Resets the accumulated loss counter for the layer. |

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

| Parameter | Type | Description |
|---|---|---|
| `vocabSize` | `number` | Total vocabulary size |
| `embeddingDim` | `number` | Vector dimension per token |
| `alpha` | `number` | Layer-specific learning rate |
| `status` | `StatusLayer` | Layer role in the model graph |
| `optimizer` | `Optimzier` | Optimizer for the embedding table |
| `padTokenId` | `number` | PAD token ID to skip during backward pass |

```ts
import { Embedding } from "@akhyar11/ml-v1"

const embed = new Embedding({
  vocabSize: 5000,
  embeddingDim: 128
});
```

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

Recurrent layers process sequential data. They expect a single-sample input with shape **`[features, seqLen]`**.

#### General Conventions

- **`returnSequences: false`** — output shape `[hiddenUnits, 1]` (last time step).
- **`returnSequences: true`** — output shape `[hiddenUnits, seqLen]` (all time steps).
- **`stateful: true`** — hidden state is carried over across `forward()` calls until `resetState()` is called.
- **`returnState`** — currently **not supported** across the entire recurrent family; will throw an explicit error.
- `getState()` / `resetState()` — available on all recurrent layers.
- `save()` / `load()` — persist weights, configuration, and stateful hidden states.

For use inside `Sequential.fit()`, set **`batchSize: 1`**. If `stateful: true`, avoid `shuffle: true` and `validationSplit > 0`.

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
| `returnSequences` | `boolean` | Return all time steps |
| `stateful` | `boolean` | Carry state across calls |
| `optimizer` | `Optimzier` | Optimizer name |
| `clipGradient` | `number` | Gradient clipping limit |

`getState()` returns `{ h, c }`.

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

## Notes

- For sequence modeling in `Sequential`, recurrent layers are typically followed by a `Dense` output layer.
- `RNN.getState()` returns a `Matrix`, `LSTM.getState()` returns `{ h, c }`, `GRU.getState()` returns `{ forward, backward? }`.
- `save()` / `load()` preserve recurrent weights and stateful hidden states.
