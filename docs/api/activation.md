# Activation Functions

Activation functions introduce non-linearity into neural networks, enabling the model to learn complex patterns.

## Import

```ts
import {
  linear,
  sigmoid,
  tanh,
  relu,
  lRelu,
  softmax,
  softmaxOnly,
  softmaxInto,
  softmaxBackward,
  softmaxBackwardInto,
  softmaxGradient
} from "@oxide-js/core"
```

## Overview

In Oxide-JS, most activation functions return a tuple **`[Matrix, Matrix]`**:

1. **Forward output** — the activation result passed to the next layer.
2. **Gradient/derivative** — used during backpropagation to compute weight updates.

> [!TIP]
> When writing a manual training loop, save the second element of the tuple (the gradient matrix) for use when computing weight updates.

---

## API Reference

### `linear(input: Matrix): [Matrix, Matrix]`

Identity activation. Output equals input. Derivative is `1` for all elements.

Typically used in output layers for **regression** tasks.

```ts
import { linear } from "@oxide-js/core"

const [out, grad] = linear(inputMatrix);
// out:  identical to input
// grad: matrix filled with 1s
```

---

### `sigmoid(input: Matrix): [Matrix, Matrix]`

Maps values to the range **(0, 1)**. Commonly used in output layers for **binary classification**.

```ts
import { sigmoid } from "@oxide-js/core"

const input = mj.matrix([[-1, 0, 2]]);
const [out, grad] = sigmoid(input);

// out  (activation): [[0.268, 0.5, 0.880]]
// grad (derivative):  [[0.196, 0.25, 0.105]]
```

---

### `tanh(input: Matrix): [Matrix, Matrix]`

Maps values to the range **(-1, 1)**. Often better than sigmoid for hidden layers.

```ts
import { tanh } from "@oxide-js/core"

const input = mj.matrix([[-1, 0, 1]]);
const [out, grad] = tanh(input);

// out:  [[-0.761, 0, 0.761]]
// grad: [[ 0.419, 1, 0.419]]  (1 - out²)
```

---

### `relu(input: Matrix): [Matrix, Matrix]`

Rectified Linear Unit. Sets negative values to `0`, leaves positive values unchanged. The most popular activation for hidden layers.

```ts
import { relu } from "@oxide-js/core"

const input = mj.matrix([[-1.5, 0.5, 2.0]]);
const [out, grad] = relu(input);

// out:  [[0, 0.5, 2.0]]   (negative values clipped to 0)
// grad: [[0, 1,   1  ]]   (1 where input > 0, else 0)
```

---

### `lRelu(input: Matrix): [Matrix, Matrix]`

Leaky ReLU. Like ReLU but gives a small value (`1e-5` multiplier) to negative inputs to prevent the "dying neuron" problem.

```ts
import { lRelu } from "@oxide-js/core"

const input = mj.matrix([[-1, 1]]);
const [out, grad] = lRelu(input);

// out: [[-0.00001, 1]]
```

---

### `softmax(input: Matrix, row?: boolean): [Matrix, Matrix]`

Produces a probability distribution where all elements sum to `1.0`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input` | `Matrix` | — | Input logits |
| `row` | `boolean` | `false` | `true` = probability per row; `false` = probability per column |

```ts
import { softmax } from "@oxide-js/core"

const logits = mj.matrix([[1, 2, 3]]);
const [probs, dSoftmax] = softmax(logits, true);

// probs: [[0.09, 0.24, 0.66]]  (sums to 1.0)
```

> **Note:** The second return value is a diagonal approximation `s * (1 - s)`, not the full Jacobian. For backpropagation through an incoming error gradient, use `softmaxBackward` or `softmaxBackwardInto`.

---

### `softmaxOnly(input: Matrix, row?: boolean): Matrix`

Computes only the forward softmax output without allocating a gradient matrix. Useful in inference-only paths.

---

### `softmaxInto(input: Matrix, out: Matrix, row?: boolean): void`

Computes the softmax forward pass and writes the result directly into the `out` buffer. Avoids allocation in hot paths.

---

### `softmaxBackward(softmaxOutput: Matrix, dOut: Matrix): Matrix`

Computes the full softmax backward pass given the softmax output and the incoming gradient.

---

### `softmaxBackwardInto(softmaxOutput: Matrix, dOut: Matrix, out: Matrix): void`

Same as `softmaxBackward` but writes the result into the provided `out` buffer.

---

### `softmaxGradient(softmaxOutput: Matrix): Matrix`

Returns the diagonal gradient approximation `s * (1 - s)` for the softmax output.

---

## Notes

- The second element of the `[out, grad]` tuple from `softmax(...)` is an approximation. Use `softmaxBackward` / `softmaxBackwardInto` for correct gradient computation in backpropagation.
- When using `Dense` with `loss: "softmaxCrossEntropy"`, do **not** also set `activation: "softmax"` — that would apply softmax twice.
- The `softmaxInto` / `softmaxBackwardInto` variants accept an `out` buffer that must not alias the input buffers.
