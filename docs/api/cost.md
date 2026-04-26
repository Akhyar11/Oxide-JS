# Cost Functions

Cost (loss) functions measure how far the model's predictions are from the ground truth. They are used during training to compute the error signal for backpropagation.

## Import

```ts
import {
  MeanSquerError,
  CategoricalCrossEntropy,
  BinaryCrossEntropy,
  SoftmaxCrossEntropy
} from "@akhyar11/ml-v1"
```

## Overview

ML-V1 provides four built-in cost functions. You can select a cost function by name string when calling `model.compile({ error: "..." })`, or use the class directly in a manual training loop.

Available string identifiers for `compile()`:
- `"mse"` → `MeanSquerError`
- `"crossEntropy"` → `CategoricalCrossEntropy`
- `"binaryCrossEntropy"` → `BinaryCrossEntropy`
- `"softmaxCrossEntropy"` → `SoftmaxCrossEntropy`

---

## API Reference

### `MeanSquerError`

Mean Squared Error. Measures the average of the squared differences between predictions and targets.

**Best for:** Regression tasks.

```ts
import { MeanSquerError } from "@akhyar11/ml-v1"

const loss = new MeanSquerError();
// Used internally by layers when loss: "mse" is configured
```

> **Note:** The export name is `MeanSquerError` (matching the source code spelling). Do not rename it to `MeanSquareError`.

---

### `CategoricalCrossEntropy`

Cross-entropy loss for multi-class classification with one-hot encoded targets.

**Best for:** Multi-class classification where targets are probability distributions or one-hot vectors.

```ts
import { CategoricalCrossEntropy } from "@akhyar11/ml-v1"

const loss = new CategoricalCrossEntropy();
```

---

### `BinaryCrossEntropy`

Binary cross-entropy loss for two-class classification.

**Best for:** Binary classification (single sigmoid output unit).

```ts
import { BinaryCrossEntropy } from "@akhyar11/ml-v1"

const loss = new BinaryCrossEntropy();
```

---

### `SoftmaxCrossEntropy`

Combined softmax activation + cross-entropy loss. Numerically more stable than applying them separately. Works with **sparse integer class indices** as targets.

**Best for:** Multi-class and token classification tasks (including Transformer LM training).

```ts
import { SoftmaxCrossEntropy } from "@akhyar11/ml-v1"

const loss = new SoftmaxCrossEntropy();
```

Typical usage in a `Dense` output layer:

```ts
import { Dense } from "@akhyar11/ml-v1"

const outputLayer = new Dense({
  units: 64,
  outputUnits: 500,   // vocabSize
  activation: "linear",
  loss: "softmaxCrossEntropy"
});
```

> [!IMPORTANT]
> Do **not** combine `activation: "softmax"` with `loss: "softmaxCrossEntropy"` on the same layer — that would apply softmax twice and destabilize training.

---

## Notes

- Cost functions are normally selected by string name via `model.compile({ error: "..." })` or `new Dense({ loss: "..." })`.
- The optimizer's internal update loop calls `loss.calculate(prediction, target)` and `loss.gradient(prediction, target)` automatically.
- For Transformer training with sparse targets of shape `[seqLen, batch]`, use `"softmaxCrossEntropy"`.
