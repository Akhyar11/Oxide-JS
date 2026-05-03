# Models

High-level model compositions in ML-V1.

## Import

```ts
import {
  Sequential,
  Transformers,
  DimentionalityReduction,
  RecurrentModel
} from "@akhyar11/ml-v1"
```

## Overview

ML-V1 provides four model classes:

| Model | Description |
|---|---|
| `Sequential` | Generic multi-layer stack for feed-forward / per-sample supervised learning. |
| `Transformers` | Multi-block causal language model with full-sequence training and configurable inference modes. |
| `RecurrentModel` | High-level stacked `RNN` / `LSTM` / `GRU` model for many-to-one and aligned many-to-many sequence training. |
| `DimentionalityReduction` | Extends `Sequential` with an encoder/decoder split for autoencoder scenarios. |

---

## API Reference

### `Sequential`

A wrapper model that stacks layers sequentially for **per-sample supervised learning**. `Sequential.fit()` handles batching, validation, early stopping, and shuffling for targets that are averaged per sample.

Use:
- `Sequential.fit()` for feed-forward classification/regression.
- `Transformers.fit()` for causal LM / token-level full-sequence loss.
- `RecurrentModel.fit()` for recurrent sequence training (`RNN` / `LSTM` / `GRU`).

#### `constructor(config?)`

Optionally accepts `{ layers: Layer[] }` to initialize with a predefined layer list.

```ts
import { Sequential } from "@akhyar11/ml-v1"

const model = new Sequential();
```

#### `add(layer): void`

Appends a layer to the execution sequence.

```ts
import { Sequential, Dense } from "@akhyar11/ml-v1"

const model = new Sequential();
model.add(new Dense({ units: 4, outputUnits: 2 }));
```

#### `forward(input: Matrix): Matrix`

Passes data through all layers in training mode.

#### `predict(input: Matrix): Matrix`

Passes data through all layers with training mode disabled (e.g. Dropout is inactive).

#### `fillEmbeddingWeight(source): this`

Loads pretrained embedding weight into the first `Embedding` layer in the model without changing that layer's configuration.

```ts
model.fillEmbeddingWeight("./pretrained-embedding.json");
```

Use this when the model contains an `Embedding` layer and you want model-level access instead of calling `layer.fillWeight(...)` manually.

#### `compile({ alpha, optimizer, error, clipGradient }): void`

Configures learning parameters for all layers.

| Parameter | Type | Description |
|---|---|---|
| `alpha` | `number` | Learning rate |
| `optimizer` | `Optimzier` | Optimizer name (e.g. `"adam"`) |
| `error` | `Cost` | Global loss function name (e.g. `"mse"`, `"softmaxCrossEntropy"`) |
| `clipGradient` | `number \| boolean` | Global gradient clipping limit |

#### `fit(X, y, epochs, config?): FitResult`

Trains the model on input/target pairs. Supports batching, validation split, early stopping, shuffle, verbose logging, and per-epoch callbacks.

> `Sequential.fit()` only supports per-sample supervised loss. If you need full-sequence causal LM loss or recurrent sequence training, use `Transformers.fit()` or `RecurrentModel.fit()`.

##### Supported Signatures

```ts
// Config-based API (recommended)
const result = model.fit(X, y, epochs, config?: FitConfig): FitResult;

// Legacy callback (backward compatible)
model.fit(X, y, epochs, (loss: number) => void): void;
```

##### `FitConfig` Parameters

| Option | Type | Default | Description |
|---|---|---|---|
| `batchSize` | `number` | `max(1, floor(N/10))` | Samples per mini-batch |
| `validationSplit` | `number` | `0` | Proportion of data for validation (0–1) |
| `earlyStoppingPatience` | `number` | `Infinity` | Epochs without improvement before stopping |
| `shuffle` | `boolean` | `true` | Shuffle training order each epoch |
| `verbose` | `boolean` | `false` | Print loss to console each epoch |
| `onEpochEnd` | `(epoch, loss, valLoss?) => void` | — | Callback after each epoch |
| `monitorMetric` | `"loss" \| "valLoss"` | `"valLoss"` if validation exists | Metric used for early stopping |
| `minDelta` | `number` | `0` | Minimum change counted as improvement |
| `mode` | `"min" \| "max"` | `"min"` | Direction for improvement |
##### `FitResult` Return Value

```ts
interface FitResult {
  history: {
    loss: number[];      // Training loss per epoch
    valLoss?: number[];  // Validation loss per epoch
  };
  bestEpoch: number;        // Index of the best epoch (0-indexed)
  bestLoss: number;         // Best recorded loss value
  stoppedEarly: boolean;    // true if early stopping was triggered
  stoppingEpoch?: number;   // Epoch where early stopping occurred
}
```

##### Example

```ts
import { Sequential, Dense, mj } from "@akhyar11/ml-v1"

const model = new Sequential({
  layers: [
    new Dense({ units: 2, outputUnits: 4, activation: "relu", status: "input" }),
    new Dense({ units: 4, outputUnits: 1, activation: "sigmoid", status: "output", loss: "mse" }),
  ],
});

model.compile({ alpha: 0.01, optimizer: "adam", error: "mse" });

const X = [
  mj.matrix([[0], [0]]),
  mj.matrix([[0], [1]]),
  mj.matrix([[1], [0]]),
  mj.matrix([[1], [1]])
];
const Y = [
  mj.matrix([[0]]),
  mj.matrix([[1]]),
  mj.matrix([[1]]),
  mj.matrix([[0]])
];

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

---

### `DimentionalityReduction`

Extends `Sequential` for autoencoder / encoder-decoder scenarios. Splits the layer list into `layersEncode` and `layersDecode` at the first layer with `status === "outputReduction"`.

#### `constructor({ layers })`

```ts
import { DimentionalityReduction, Dense } from "@akhyar11/ml-v1"

const model = new DimentionalityReduction({
  layers: [
    new Dense({ units: 8, outputUnits: 4, activation: "relu", status: "input" }),
    new Dense({ units: 4, outputUnits: 2, activation: "relu", status: "outputReduction" }),
    new Dense({ units: 2, outputUnits: 4, activation: "relu" }),
    new Dense({ units: 4, outputUnits: 8, activation: "linear", status: "output", loss: "mse" }),
  ],
});
```

#### `encode(x: Matrix): Matrix`

Runs only the encoder half (up to and including the `outputReduction` layer).

#### `decode(enc: Matrix): Matrix`

Runs only the decoder half.

```ts
const latent = model.encode(inputMatrix);
const reconstructed = model.decode(latent);
```

#### `fit(X, epochs, config?): FitResult`

Autoencoder training shortcut. Internally calls `super.fit(X, X, epochs, config)`.

```ts
const result = model.fit(trainX, 50, { batchSize: 8, verbose: true });
```

---

### `RecurrentModel`

High-level recurrent model wrapper built on top of the existing `RNN`, `LSTM`, and `GRU` layers. It supports stacked recurrent layers plus:
- `many-to-one`
- aligned `many-to-many` (same target length as input sequence)

#### `constructor(config)`

```ts
import { RecurrentModel } from "@akhyar11/ml-v1"

const model = new RecurrentModel({
  kind: "lstm",
  vocabSize: 1000,
  embeddingDim: 32,
  embeddingTrainable: false,
  hiddenSizes: [64, 64],
  outputSize: 5,
  seqLen: 20,
  mode: "many-to-one",
  loss: "softmaxCrossEntropy",
});
```

Key rules:
- `hiddenSizes` takes priority over `hiddenSize` + `numLayers`.
- If `Embedding` is not used, provide `inputSize`.
- Set `embeddingTrainable: false` to freeze the internal `Embedding` layer.
- `pooling` controls many-to-one sequence aggregation:
  - `"last"` keeps the legacy last-hidden behavior.
  - `"mean"` performs differentiable mean pooling over all valid timesteps.
  - `"max"` performs differentiable max pooling with argmax-routed backward.
- If `padTokenId` is set and the model uses `Embedding`, `"mean"` / `"max"` pooling ignore PAD positions. An all-PAD sample throws `RecurrentModel pooling: sample has no valid non-pad tokens.`
- `many-to-one` is supported.
- aligned `many-to-many` is supported with target shape `[1, seqLen]` (sparse) or `[outputSize, seqLen]` (dense/one-hot or regression).

#### Stateful helpers

- `resetState()`
- `resetStates()`

If `stateful=true`:
- `batchSize` must be `1`
- `shuffle` must be `false`
- `validationSplit` must be `0`

---

### `Transformers`

Full Transformer architecture model for causal language modeling. Built on top of `Sequential`, but it uses its **own** `fit()` loop so loss and validation are averaged per valid token rather than per sample.

#### `constructor(config)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `units` | `number` | — | Model dimension (`d_model`) |
| `seqLen` | `number` | — | Input sequence length |
| `vocabSize` | `number` | — | Vocabulary size |
| `heads` | `number` | `8` | Number of attention heads |
| `numBlocks` | `number` | `1` | Number of stacked Transformer blocks |
| `dropoutRate` | `number` | `0.1` | Dropout rate |
| `alpha` | `number` | `0.01` | Learning rate |
| `padTokenId` | `number` | — | PAD token ID (ignored in embedding, attention, and loss) |
| `embeddingTrainable` | `boolean` | `true` | Freeze the internal embedding table when set to `false` |
| `clipGradient` | `number` | `5.0` | Global gradient clipping limit |
| `predictMode` | `"next-token" \| "full-sequence"` | `"next-token"` | Default mode for `predict()` |

```ts
import { Transformers } from "@akhyar11/ml-v1"

const model = new Transformers({
  units: 128,
  seqLen: 50,
  vocabSize: 5000,
  heads: 8,
  numBlocks: 4,
  padTokenId: 0,
  embeddingTrainable: false,
  clipGradient: 1.5,
  predictMode: "next-token",
});
```

#### Internal Architecture (per block)

```
LayerNormalization → MultiHeadAttention → Dropout → Residual
LayerNormalization → Dense(4×units, relu) → Dropout → Dense(units, linear) → Dropout → Residual
```

#### Shape Contracts

- **Input token IDs:** `Matrix` shape `[seqLen, batch]`.
- **Training logits** (`model.train()` + `forward()` or `forwardFullSequence()`): `[vocabSize, seqLen * batch]`.
  - Column order: sample 0 positions `0..seqLen-1`, then sample 1, etc.
- **Inference logits** (`model.eval()` + `forward()` or `forwardNextToken()`): `[vocabSize, batch]`.
- **Training target:** `Matrix` sparse indices, shape `[seqLen, batch]` (shifted next-token).
- **Legacy target:** `Matrix` sparse indices, shape `[1, batch]` (last-token only).

#### `forward(input: Matrix): Matrix`

- In `model.train()` mode: full-sequence logits `[vocabSize, seqLen * batch]`.
- In `model.eval()` mode: last-token logits `[vocabSize, batch]`.

#### `forwardFullSequence(input: Matrix): Matrix`

Forces the full-sequence path regardless of current mode.

#### `forwardNextToken(input: Matrix): Matrix`

Forces the last-token path regardless of current mode. Use for generation loops or legacy next-token training.

#### `predict(input: Matrix): Matrix`

Inference entry point. Output shape follows `predictMode`:
- `"next-token"` → `[vocabSize, batch]`
- `"full-sequence"` → `[vocabSize, seqLen * batch]`

#### `setPredictMode(mode) / getPredictMode()`

Change `predictMode` without recreating the model.

```ts
model.setPredictMode("full-sequence");
const logitsAll = model.predict(x); // [vocabSize, seqLen * batch]

model.setPredictMode("next-token");
const nextLogits = model.predict(x); // [vocabSize, batch]
```

#### `backward(target: Matrix): void`

Recommended: target shape `[seqLen, batch]` — next-token shifted targets; last row is `padTokenId`.

Compatibility path: shape `[1, batch]` still accepted for legacy loops.

#### `train() / eval()`

Switch between training and evaluation modes.

#### `save(path) / load(path)`

Persist and restore model weights. The instance calling `load()` must be created with the same `numBlocks` as the saved artifact.

Embedding metadata such as `trainable` is also restored.

#### Advanced API Methods

| Method | Description |
|---|---|
| `getPadTokenId(): number \| null` | Returns the `padTokenId` from the embedding layer. |
| `fillEmbeddingWeight(source)` | Fills the model's embedding weight from matrix data or supported JSON files without changing embedding config. |
| `setPositionOffset(n): this` | Sets the PE position offset (used for left-padding trim). |
| `resetPositionOffset(): this` | Resets position offset to 0. Called automatically by `fit()`. |
| `resizeVocab(newVocabSize)` | Expands embedding vocabulary and output projector. |
| `enableProfiling(enabled?) / disableProfiling()` | Enable/disable internal stage profiler. |
| `resetProfiling()` | Clear profiler statistics. |
| `getProfilingReport(reset?)` | Returns `{ totalMs, avgMs, count }` per stage. |

#### Full-Sequence Training Example

```ts
import { Transformers, mj } from "@akhyar11/ml-v1"

const model = new Transformers({
  units: 64,
  seqLen: 6,
  vocabSize: 2000,
  heads: 8,
  numBlocks: 2,
  alpha: 0.001,
  padTokenId: 0,
});

const x = mj.matrix([
  [0,  0],
  [11, 21],
  [12, 22],
  [13, 23],
  [14, 24],
  [15, 25],
]);

const y = mj.matrix([
  [0,  0],
  [12, 22],
  [13, 23],
  [14, 24],
  [15, 25],
  [0,  0],
]);

model.train();
const logits = model.forward(x); // [vocabSize, seqLen * batch]
model.backward(y);
console.log(logits._shape, model.loss);
```

#### Inference / Generation Example

```ts
import { Transformers, mj } from "@akhyar11/ml-v1"

const model = new Transformers({
  units: 64,
  seqLen: 6,
  vocabSize: 2000,
  heads: 8,
  numBlocks: 2,
  alpha: 0.001,
  padTokenId: 0,
  predictMode: "next-token",
});

model.eval();

const x = mj.matrix([[0], [11], [12], [13], [14], [15]]);
const nextTokenLogits = model.predict(x); // [vocabSize, 1]
```

#### Dynamic Padding Trim (v2.2.0+)

`fit()` supports `trimPadding` and `paddingSide` in `FitConfig` to dynamically trim PAD tokens per batch, reducing attention cost from O(seqLen²) to O(effectiveSeqLen²).

| Option | Value | Behavior |
|---|---|---|
| `trimPadding` | `true` *(default)* | Trim PAD tokens from each batch |
| `trimPadding` | `false` | Disable trimming entirely |
| `paddingSide` | `"right"` *(default)* | Trim trailing PAD; `positionOffset = 0` |
| `paddingSide` | `"left"` | Trim leading PAD; `positionOffset` adjusted |

Only active for full-sequence targets with shape `[seqLen, batch]`. Legacy targets `[1, batch]` are not trimmed.

```ts
model.fit(trainX, trainY, 80, {
  batchSize: 8,
  trimPadding: true,
  paddingSide: "right",
  shuffle: true
});
```

#### Best Practices

- Use shifted next-token targets `[seqLen, batch]` — not `[1, batch]` — for correct LM training.
- Keep `seqLen`, `vocabSize`, and `padTokenId` consistent across tokenizer, preprocessing, and model.
- Call `model.train()` before training and `model.eval()` before inference.
- Use `model.predict()` as the primary inference entry point; set `predictMode` as needed.
- Enable `trimPadding: true` (default) for best performance when data contains significant padding.

---

## Notes

- `Sequential`, `DimentionalityReduction`, and `Transformers` all call `model.train()` / `model.eval()` to switch between training and inference modes. Layers like `Dropout` and `LayerNormalization` respect this flag.
- **Batched RNN Support**: As of **v2.3.0**, recurrent models (`RNN`, `LSTM`, `GRU`) in `Sequential` fully support and are optimized for **`batchSize > 1`** via the native backend.
- **Stateful Restriction**: For stateful recurrent models (`stateful: true`), avoid `shuffle: true` and keep **`batchSize: 1`** as hidden states are linked to specific sample indices.
- Optimizer type names available via `compile()`: `"sgd"`, `"momentum"`, `"nag"`, `"adaGrad"`, `"adam"`.
