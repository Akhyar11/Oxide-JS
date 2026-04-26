# Utils

Utility functions for configuring models, preprocessing data, and formatting training output.

## Import

```ts
import {
  setActivation,
  setLayers,
  registerLayer,
  setLoss,
  setOptimizer,
  cosineSimilarity,
  shuffleInPlace,
  splitTrainValidation,
  formatLoss,
  formatProgressBar,
  formatTime
} from "@akhyar11/ml-v1"
```

## Overview

The `utils` module provides helpers used internally by the library's model and layer APIs, as well as standalone functions for data preprocessing and output formatting.

---

## API Reference

### Configuration Helpers

These functions are used internally to wire up activations, layers, loss functions, and optimizers from string names. They can also be used in custom training loops or when extending the library.

#### `setActivation(name: ActivationType): (input: Matrix) => [Matrix, Matrix]`

Returns the activation function corresponding to the given name string.

#### `setLoss(name: Cost): CostFunction`

Returns the loss/cost function corresponding to the given name string.

#### `setOptimizer(name: Optimzier): OptimizerInstance`

Returns a new optimizer instance corresponding to the given name string.

Valid names: `"sgd"`, `"momentum"`, `"nag"`, `"adaGrad"`, `"adam"`.

#### `setLayers(layers: Layers[], config): void`

Applies shared compile configuration (learning rate, optimizer, loss) to an array of layers.

#### `registerLayer(name: string, layerClass: any): void`

Registers a custom layer class under a string name so it can be referenced by name in model configurations.

---

### Data Utilities

#### `cosineSimilarity(a: Matrix, b: Matrix): number`

Computes the cosine similarity between two vectors.

**Returns:** a value in `[-1, 1]` where `1` is identical direction and `-1` is opposite.

```ts
import { cosineSimilarity, mj } from "@akhyar11/ml-v1"

const a = mj.matrix([[1, 0, 0]]);
const b = mj.matrix([[0, 1, 0]]);
const sim = cosineSimilarity(a, b);
// 0  (orthogonal vectors)
```

#### `shuffleInPlace<T>(arr: T[]): void`

Shuffles an array in-place using the Fisher-Yates algorithm.

```ts
import { shuffleInPlace } from "@akhyar11/ml-v1"

const samples = [1, 2, 3, 4, 5];
shuffleInPlace(samples);
```

#### `splitTrainValidation<T>(data: T[], validationSplit: number): { train: T[], val: T[] }`

Splits an array into training and validation sets without shuffling.

| Parameter | Type | Description |
|---|---|---|
| `data` | `T[]` | Array to split |
| `validationSplit` | `number` | Fraction reserved for validation (0–1) |

```ts
import { splitTrainValidation } from "@akhyar11/ml-v1"

const { train, val } = splitTrainValidation(samples, 0.2);
```

---

### Formatting Helpers

These functions are used by `Sequential.fit()` verbose output but can also be used in custom training loops.

#### `formatLoss(loss: number): string`

Formats a loss value to a fixed-precision string for display.

#### `formatProgressBar(current: number, total: number, width?: number): string`

Returns an ASCII progress bar string for the current epoch progress.

#### `formatTime(ms: number): string`

Formats a millisecond duration into a human-readable string (e.g. `"1.2s"`, `"3m 4s"`).

---

## Notes

- `setActivation`, `setLoss`, `setOptimizer`, and `setLayers` are primarily used internally by `compile()` and layer constructors. Direct usage is only needed when building custom model architectures outside the standard `Sequential` pattern.
- `registerLayer` enables plugin-style custom layers that can be referenced by name in model config objects.
