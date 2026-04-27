# Optimizers

Optimizers calculate parameter updates from gradients during backpropagation. In normal usage, ML-V1 creates optimizer instances internally when you configure a layer or call `model.compile(...)`.

## Import

```ts
import { setOptimizer } from "@akhyar11/ml-v1"
import type { Matrix, MatrixShape, Optimzier } from "@akhyar11/ml-v1"
```

> **Note:** The public type name is currently `Optimzier` (matching the source code spelling).

## Overview

Choose an optimizer with the `optimizer` string in `compile()` or layer constructors:

```ts
import { Sequential, Dense } from "@akhyar11/ml-v1"

const model = new Sequential([
  new Dense({ units: 2, outputUnits: 8, activation: "relu" }),
  new Dense({ units: 8, outputUnits: 1, activation: "linear" })
]);

model.compile({
  alpha: 0.001,
  optimizer: "adam",
  error: "mse"
});
```

Available optimizer names:

| Name | Type Literal | Best for |
|---|---|---|
| SGD | `"sgd"` | Simple models, baselines, and debugging |
| Momentum | `"momentum"` | Smoother convergence than plain SGD |
| NAG | `"nag"` | Momentum-style updates with a look-ahead correction |
| AdaGrad | `"adaGrad"` | Sparse or uneven feature gradients |
| Adam | `"adam"` | General-purpose default for most training runs |

---

## API Reference

### `setOptimizer(name, shape, alpha)`

Creates a new optimizer instance from a string name.

```ts
setOptimizer(
  name: Optimzier,
  shape: MatrixShape,
  alpha: number
): { calculate(grad: Matrix, alpha: number): Matrix }
```

| Parameter | Type | Description |
|---|---|---|
| `name` | `Optimzier` | One of `"sgd"`, `"momentum"`, `"nag"`, `"adaGrad"`, or `"adam"` |
| `shape` | `MatrixShape` | Shape of the parameter matrix that will be updated |
| `alpha` | `number` | Value passed to the AdaGrad constructor as `epsilon`; ignored by other optimizer constructors |

```ts
import { mj, setOptimizer } from "@akhyar11/ml-v1"

const grad = mj.matrix([[0.1, -0.2]]);
const optimizer = setOptimizer("adam", grad._shape, 0.001);
const update = optimizer.calculate(grad, 0.001);
```

`setOptimizer()` is mostly an internal helper. Use it directly only for custom training loops or custom layer implementations.

---

## Optimizer Behavior

Every optimizer implements:

```ts
calculate(grad: Matrix, alpha: number): Matrix
dispose(): void
```

- `calculate()` returns an update matrix.
- `dispose()` **(New in v2.3.0)** explicitly releases memory for internal moment and state buffers.

`calculate()` returns an update matrix. Layers subtract that update from their parameters:

```ts
const update = optimizer.calculate(gradWeight, alpha);
weight.subInPlace(update);
```

The returned value is the update, not the final parameter matrix.

### `SGD`

Plain stochastic gradient descent.

```ts
update = alpha * grad
```

`SGD` has no internal state and does not need the parameter shape.

### `Momentum`

Stores a velocity matrix initialized to zeros.

```ts
velocity = beta * previousVelocity + alpha * grad
update = velocity
```

Default internal value:

| Option | Value |
|---|---:|
| `beta` | `0.9` |

### `NAG`

Nesterov Accelerated Gradient variant with the same `beta` default as Momentum.

```ts
betaVelocity = beta * previousVelocity
velocity = betaVelocity + alpha * (grad - betaVelocity)
update = velocity
```

Default internal value:

| Option | Value |
|---|---:|
| `beta` | `0.9` |

### `AdaGrad`

Accumulates squared gradients and scales each parameter update by its accumulated magnitude.

```ts
sumGrad = sumGrad + grad * grad
update = alpha * grad / sqrt(sumGrad + epsilon)
```

In ML-V1, `setOptimizer("adaGrad", shape, alpha)` passes the third argument to the `AdaGrad` constructor as `epsilon`.

### `Adam`

Adaptive Moment Estimation with first and second moment tracking plus bias correction.

```ts
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad * grad
mHat = m / (1 - beta1^t)
vHat = v / (1 - beta2^t)
update = alpha * mHat / (sqrt(vHat) + epsilon)
```

Default internal values:

| Option | Value |
|---|---:|
| `beta1` | `0.9` |
| `beta2` | `0.999` |
| `epsilon` | `1e-8` |

Adam can use the Rust native backend for larger update arrays when the native module is available.

---

## Usage With Layers

Most layers that own trainable parameters accept an optimizer name directly:

```ts
import { Dense, Embedding, RNN } from "@akhyar11/ml-v1"

const dense = new Dense({
  units: 16,
  outputUnits: 4,
  activation: "relu",
  optimizer: "adam"
});

const embedding = new Embedding({
  vocabSize: 5000,
  embeddingDim: 128,
  optimizer: "adam"
});

const rnn = new RNN({
  inputSize: 32,
  hiddenSize: 64,
  outputSize: 10,
  optimizer: "adam"
});
```

`Sequential.compile()` applies a shared optimizer setting to compatible layers:

```ts
model.compile({
  alpha: 0.001,
  optimizer: "adam",
  error: "softmaxCrossEntropy",
  clipGradient: 1
});
```

---

## Notes

- Use `"adam"` as the default choice unless you have a reason to compare against another optimizer.
- Use `"sgd"` when debugging because it has no optimizer state.
- Use `"adaGrad"` for sparse-gradient cases, but be aware that accumulated squared gradients can reduce the effective learning rate over time.
- `rmsprop` is not currently available.
- Optimizer classes (`SGD`, `Adam`, `AdaGrad`, `Momentum`, `NAG`) live under `src/optimizer`, but they are not exported as top-level public API. Prefer string configuration or `setOptimizer()`.
