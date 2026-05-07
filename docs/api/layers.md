# Layers

Neural network layer implementations in Oxide-JS.

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
  AttentionPooling,
  Flatten,
  PositionalEncoding,
  LayerNormalization,
  Dropout,
  RNN,
  LSTM,
  GRU,
  AdaptiveMemoryRNN
} from "@oxide-js/layers"
```

## Overview

All layers share a common interface: `forward(input)` for the forward pass and `backward(grad)` for the backward pass. Layers are composable inside `Sequential`, `Transformers`, or `DimentionalityReduction` models, or they can be used standalone in a manual training loop.

### Keras Compatibility

All standard layers support **Keras-style serialization**. They implement:
- `toKerasConfig()`: Returns the JSON configuration.
- `getWeightsManifest()`: Returns the weight buffer metadata.
- `setWeightsFromBinary()`: Loads weights from a binary buffer.

This allows Oxide-JS models to be saved as `model.json` and `weights.bin` files.

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
| `optimizer` | `Optimizer` | — | Optimization algorithm name |
| `status` | `StatusLayer` | — | Layer role in the model graph |
| `clipGradient` | `number \| boolean` | `5.0` | Gradient clipping limit for this layer |

```ts
import { Dense } from "@oxide-js/layers"

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
| `optimizer` | `Optimizer` | `"adam"` | Optimizer for the embedding table |
| `padTokenId` | `number \| null` | `null` | PAD token ID to skip during backward pass |
| `trainable` | `boolean` | `true` | Freeze weight updates when set to `false` |

```ts
import { Embedding } from "@oxide-js/layers"

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

### `AttentionPooling`

Trainable masked pooling for token sequences. It learns one attention score per token, applies masked softmax over valid positions, then returns a weighted sum of token vectors.

Use this when you want to compress `[features, seqLen]` into a single `[features, 1]` vector without averaging all tokens equally.

#### `constructor(config)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `units` | `number` | — | Feature dimension per token |
| `maxTokens` | `number` | — | Fixed token columns expected by the layer |
| `alpha` | `number` | `0.01` | Layer learning rate |
| `optimizer` | `Optimizer` | `"adam"` | Optimizer for the internal scorer |
| `status` | `StatusLayer` | `"train"` | Layer role in the graph |
| `clipGradient` | `number \| boolean` | `5.0` | Gradient clipping limit |

```ts
import { AttentionPooling } from "@oxide-js/layers"

const pooling = new AttentionPooling({
  units: 128,
  maxTokens: 16,
});

pooling.setValidLength(7);
```

Important notes:
- Input shape must be exactly `[units, maxTokens]`.
- Call `setValidLength(n)` before `forward()` when padded tokens are present.
- Only the first `validLength` columns participate in attention and pooling.
- Output shape is always `[units, 1]`.

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
import { MultiHeadAttention } from "@oxide-js/layers"

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
import { LayerNormalization } from "@oxide-js/layers"

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
import { Dropout } from "@oxide-js/layers"

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
| `optimizer` | `Optimizer` | — | Optimizer name |
| `clipGradient` | `number` | `5.0` | Gradient clipping limit |

```ts
import { mj } from "@oxide-js/core"
import { RNN } from "@oxide-js/layers"

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
| `optimizer` | `Optimizer` | Optimizer name |
| `clipGradient` | `number` | Gradient clipping limit |

`getState()` returns `{ h, c }`.

Notes:
- New `LSTM` layers initialize `bf` to `1` by default for better long-range retention.
- `forgetBias` only affects fresh initialization. `load()` always preserves serialized `bf` from saved weights.

```ts
import { mj } from "@oxide-js/core"
import { LSTM } from "@oxide-js/layers"

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
| `optimizer` | `Optimizer` | Optimizer name |
| `clipGradient` | `number` | Gradient clipping limit |

`getState()` returns `{ forward, backward? }`.

```ts
import { mj } from "@oxide-js/core"
import { GRU } from "@oxide-js/layers"

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
| `optimizer` | `Optimizer` | `"adam"` | Optimizer for trainable recurrent/query/write projections |
| `clipGradient` | `number \| boolean` | `5.0` | Gradient clipping limit |

```ts
import { AdaptiveMemoryRNN, Dense, Embedding, Sequential } from "@oxide-js/layers"

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

Generic runtime memory layer for custom model loops. `MemoryBank` stores raw input vectors as memory values and uses a trainable `queryKernel` to address memory slots. The simplified implementation only supports `mode="project"` and `mode="concat"`.

Key properties:
- Auto-infers `units` from the first `forward()` if omitted.
- `memoryDim` is no longer configurable; it is always equal to `units`.
- Runtime memory state (`memoryKeys`, `memoryValues`, `memoryFilled`, `memoryUsage`, `memoryAge`) is mutable session state, not optimizer-trained weights.
- Trainable weights are now limited to:
  - `queryKernel`
  - `writeGateKernel` / `writeGateBias`
  - `writeQueryKernel`
  - `needKernel` in `mode="project"`
  - `outputKernel` / `outputBias` in `mode="project"`
- Every write computes:
  - `newKey = queryKernel * x`
  - `newValue = x`
  - `memorySummary = AttentionPooling(activeMemoryValues)`
  - `writeContext = [x; memorySummary]`
  - `writeGate = sigmoid(writeGateKernel * writeContext + writeGateBias)`
  - `writeQuery = writeQueryKernel * writeContext`
- The selected slot is updated with a gated blend:
  - `postKey = (1 - writeGate) * oldKey + writeGate * newKey`
  - `postValue = (1 - writeGate) * oldValue + writeGate * newValue`
- Replacement policy is overwrite-aware:
  - if the best matching filled slot score is above `overwriteThreshold`, overwrite that slot
  - otherwise allocate an empty slot if available
  - otherwise replace the least-used slot

How it works:
- Input `x` is projected by `queryKernel` into a query vector `q`.
- `q` is compared against all filled `memoryKeys`.
- Top-`K` slots are read and combined with softmax attention.
- `AttentionPooling` summarizes only the active memory slots into one `memorySummary` vector.
- `writeGateKernel` decides whether the current input should write using `[x; memorySummary]`.
- `writeQueryKernel` scores candidate overwrite slots against the current memory keys.
- In `project` mode, `needKernel` computes how much of the read result should matter from `[read; x]`.
- In `concat` mode, the raw read result is concatenated directly with the input.
- If writes are enabled and not frozen, the current input is stored into memory after the read step.

Constructor config:
- `units?: number` — input rows; if omitted, inferred on first forward.
- `memorySlots: number` — number of memory slots.
- `outputUnits?: number` — output projection size for `mode="project"`.
- `mode?: "project" | "concat"` — how memory read result combines with input.
- `similarity?: "cosine" | "dot"` — query/key similarity.
- `readTopK?: number` — how many slots participate in the softmax read.
- `persistence?: "session" | "manual"` — semantic persistence label.
- `writeEnabled?: boolean` — allow runtime writes during `forward()`.
- `overwriteThreshold?: number` — minimum slot-match score required before the layer overwrites an existing slot.

API (important methods):
- `forward(x: Matrix): Matrix`
  - `project` => `[outputUnits, cols]`
  - `concat` => `[2 * units, cols]`
- `backward(y: Matrix, err: Matrix): Matrix`
- `beginSequence({ maxHistorySteps? })`
- `backwardSequence(err: Matrix)`
- `detachSequence()`
- `endSequence()`
- `getSequenceLength()` / `isSequenceActive()`
- `resetMemory()` / `clearMemory()`
- `getMemoryState()` / `setMemoryState(state)`
- `saveMemory(path)` / `loadMemory(path)`
- `freezeWrites()` / `unfreezeWrites()` / `setWriteFrozen(boolean)`

Recommended starting config:

```ts
new MemoryBank({
  memorySlots: 64,
  mode: "project",
  similarity: "cosine",
  readTopK: 1,
});
```

#### Mode Summary

`"project"`
- `need = sigmoid(needKernel * [read; x])`
- `projected = outputKernel * [x; context] + outputBias`
- Output uses a soft residual gate:
  `output = need * projected + (1 - need) * x`
- `context = need * read`
- `needKernel` is active only in this mode.
- Use this when you want the layer itself to learn how much memory should influence the output.

`"concat"`
- Output is `[x; read]`.
- No `needKernel` gate is applied.
- Use this when you want the next layer to decide how to mix input and memory.

#### Similarity

`"cosine"`
- compares direction only
- usually the safest default for retrieval

`"dot"`
- lets magnitude influence slot ranking
- use only when magnitude should matter

#### Trainable vs Runtime State

Trainable by `backward()`:
- `queryKernel`
- `writeGateKernel`
- `writeGateBias`
- `writeQueryKernel` with a surrogate slot-selection signal
- `needKernel` in `project`
- `outputKernel`
- `outputBias`

Runtime state only:
- `memoryKeys`
- `memoryValues`
- `memoryFilled`
- `memoryUsage`
- `memoryAge`
- `memoryStep`

#### Save/Load Behavior

- `save()` stores config, trainable weights, optimizer settings, and runtime memory state.
- `load()` restores all of the above.
- `saveMemory()` / `loadMemory()` only serialize runtime memory state.

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
- `backwardSequence(...)` walks that buffer backward, so a read/loss in a later call can train earlier writes and the `queryKernel` that created their keys
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
import { mj } from "@oxide-js/core";
import { Dense, MemoryBank } from "@oxide-js/layers";

const encoder = new Dense({
  units: 10,
  outputUnits: 32,
  activation: "relu",
  status: "input",
});

const memory = new MemoryBank({
  memorySlots: 64,
  mode: "project",
  similarity: "cosine",
  readTopK: 1,
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
- The current implementation is correctness-first and intentionally simpler than the previous version:
  raw inputs are always stored as memory values, while memory keys always come from `queryKernel * x`.
- Important limitation:
  this is sequence-level BPTT inside one `forward()` call. It is not an unlimited autograd graph across separate `forward()` calls from different training steps.
