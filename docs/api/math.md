# Math Module

The `mj` object is a collection of pure functions for tensor and matrix processing. Almost all functions accept operands of type `Matrix` or `number` (scalar).

## Import

```ts
import { mj } from "@akhyar11/ml-v1"
```

## Overview

`mj` provides the numeric primitives used throughout the library — matrix multiplication, element-wise arithmetic, reductions, initializers, and specialized deep-learning ops. Computationally heavy operations are automatically dispatched to the Rust backend when available, with a transparent JavaScript fallback.

---

## API Reference

### Matrix Creation

#### `mj.matrix(array: number[][]): Matrix`

Creates a `Matrix` from a 2D array. Convenience wrapper around the `Matrix` constructor.

```ts
const a = mj.matrix([[1, 2], [3, 4]]);
```

---

### Linear Algebra

#### `mj.dotProduct(a, b, out?, transA?, transB?): Matrix`

Matrix multiplication. Automatically accelerated by the Rust backend for large workloads.

**Dimension rule:** `(M × K) · (K × N)` → result is `(M × N)`. The number of columns of `a` must equal the number of rows of `b`.

| Parameter | Type | Description |
|---|---|---|
| `a` | `Matrix` | Left matrix |
| `b` | `Matrix` | Right matrix |
| `out` | `Matrix \| undefined` | Optional pre-allocated output buffer |
| `transA` | `boolean` | If `true`, treats `a` as transposed |
| `transB` | `boolean` | If `true`, treats `b` as transposed |

```ts
const a = mj.matrix([[1, 2], [3, 4]]); // [2×2]
const b = mj.matrix([[5, 6], [7, 8]]); // [2×2]

const res = mj.dotProduct(a, b);
// [[19, 22], [43, 50]]

// On-the-fly transpose: A * B^T
const res2 = mj.dotProduct(a, b, undefined, false, true);

// Pre-allocated output buffer (avoids allocation in hot loops)
const out = mj.zeros([2, 2]);
mj.dotProduct(a, b, out);
```

#### `mj.transpose(a: Matrix): Matrix`

Returns a new matrix with rows and columns swapped.

```ts
const t = mj.transpose(mj.matrix([[1, 2], [3, 4]]));
// [[1, 3], [2, 4]]
```

#### `mj.concat(a: Matrix, b: Matrix): Matrix`

Concatenates two matrices/vectors. Optimised for row-vector concatenation (shape `[1, N]`).

```ts
const a = mj.matrix([[1, 2]]);
const b = mj.matrix([[3, 4]]);
const res = mj.concat(a, b);
// [[1, 2, 3, 4]]
```

#### `mj.addBias(a: Matrix, bias: Matrix): void`

Adds a bias column-vector **in-place** to a matrix using broadcasting (the same vector is added to every column).

```ts
const input = mj.matrix([[1, 2], [3, 4]]); // [2×2]
const bias  = mj.matrix([[10], [20]]);      // [2×1]
mj.addBias(input, bias);
// [[11, 12], [23, 24]]
```

#### `mj.norm(a: Matrix): number`

Calculates the L2 (Euclidean) norm — the square root of the sum of squares of all elements.

```ts
const length = mj.norm(weights);
```

---

### Arithmetic & Element-wise Functions

These functions process each element independently. Most accept an optional `out` parameter for memory reuse.

| Function | Description |
|---|---|
| `mj.add(a, b, out?)` | Element-wise addition |
| `mj.sub(a, b, out?)` | Element-wise subtraction |
| `mj.mul(a, b, out?)` | Element-wise multiplication |
| `mj.div(a, b)` | Element-wise division |
| `mj.absm(a)` | Absolute value per element |
| `mj.expm(a)` | Exponential (`e^x`) per element |
| `mj.logm(a)` | Natural logarithm per element |
| `mj.map(a, f)` | Custom function applied per element — **returns a new `Matrix`** |

```ts
const res = mj.add(a, b, outputBuffer); // Reuses outputBuffer, avoids new allocation
```

**Important contracts:**
- `mj.add`, `mj.sub`, and `mj.mul` accept an optional `out` buffer to reuse memory.
- `mj.div` does not accept an `out` parameter.
- `mj.map(a, f)` returns a **new** `Matrix`. This differs from `matrix.map(f)` (a `Matrix` instance method) which works **in-place**.
- For `mj.add(a, b, out)` and `mj.sub(a, b, out)`, the `out` buffer must not alias the `a` or `b` buffers.

---

### Reduction Operations

These functions summarize all elements of a matrix into a single value.

#### `mj.mean(a: Matrix): number`

Average value of all elements.

```ts
const avg = mj.mean(mj.matrix([[2, 4], [6, 8]]));
// (2+4+6+8) / 4 = 5
```

#### `mj.dotSum(a: Matrix): number`

Sum of all elements.

```ts
const total = mj.dotSum(mj.matrix([[1, 2], [3, 4]]));
// 1+2+3+4 = 10
```

#### `mj.dotMul(a: Matrix): number`

Product of all elements.

```ts
const prod = mj.dotMul(mj.matrix([[1, 2], [3, 4]]));
// 1*2*3*4 = 24
```

#### `mj.dotSub(a: Matrix): number`

Sequential subtraction of all elements from 0.

```ts
const res = mj.dotSub(mj.matrix([[1, 2], [3, 4]]));
// 0 - 1 - 2 - 3 - 4 = -10
```

#### `mj.dotDiv(a: Matrix): number`

Sequential division of all elements starting from 1.

```ts
const res = mj.dotDiv(mj.matrix([[2, 5]]));
// 1 / 2 / 5 = 0.1
```

---

### Axis Reduction

#### `mj.sumAxis(a: Matrix, axis: 0 | 1, out?): Matrix`

Sums values along a specific axis.

- **`axis 1`** — sums across columns; result shape `[rows, 1]`.
- **`axis 0`** — sums across rows; result shape `[1, cols]`.

```ts
const a = mj.matrix([[1, 2], [3, 4]]);
const sumRows = mj.sumAxis(a, 1);
// [[3], [7]]
```

---

### Matrix Initializers

All initializer functions accept a `shape: [rows, cols]` parameter.

#### `mj.zeros(shape): Matrix`

All elements set to `0`.

```ts
const z = mj.zeros([2, 3]);
// [[0, 0, 0], [0, 0, 0]]
```

#### `mj.ones(shape): Matrix`

All elements set to `1`.

```ts
const o = mj.ones([2, 2]);
// [[1, 1], [1, 1]]
```

#### `mj.random(shape): Matrix`

Uniform random values in `[0, 1)`.

```ts
const r = mj.random([3, 1]);
```

#### `mj.xavier(shape): Matrix`

Xavier (Glorot) initialization — keeps activation variance stable. Recommended for **Sigmoid** and **Tanh** activations.

```ts
const w = mj.xavier([128, 64]);
```

#### `mj.he(shape): Matrix`

He (Kaiming) initialization — optimised for **ReLU** activations to prevent vanishing gradients.

```ts
const w = mj.he([64, 10]);
```

---

### Specialized Deep-Learning Operations

#### `mj.convolution(a: Matrix, kernel: Matrix): Matrix`

Basic 2D convolution for spatial feature extraction.

```ts
const input = mj.matrix([
  [1, 1, 1, 0, 0],
  [0, 1, 1, 1, 0],
  [0, 0, 1, 1, 1],
  [0, 0, 1, 1, 0],
  [0, 1, 1, 0, 0]
]);

const kernel = mj.matrix([
  [1, 0, 1],
  [0, 1, 0],
  [1, 0, 1]
]);

const features = mj.convolution(input, kernel);
// [[4, 3, 4],
//  [2, 4, 3],
//  [2, 3, 4]]
```

#### `mj.clipGradients(a: Matrix, limit: number): void`

Clips all values in the matrix **in-place** to the range `[-limit, limit]`. Prevents exploding gradients.

```ts
const grads = mj.matrix([[0.5, 5.0], [-10.2, 0.1]]);
mj.clipGradients(grads, 1.0);
// [[0.5, 1.0], [-1.0, 0.1]]
```

#### `mj.reshape(a: Matrix, shape: [number, number]): Matrix`

Returns a new view of the matrix with a different shape (no data copy).

#### `mj.flatten(a: Matrix): Matrix`

Returns a flat row-vector view of the matrix.

```ts
const a = mj.matrix([[1, 2], [3, 4]]); // [2, 2]

const reshaped = mj.reshape(a, [1, 4]);
// [[1, 2, 3, 4]]

const flat = mj.flatten(a);
// [[1, 2, 3, 4]]
```

---

## Notes

- Most operations that have an `out` parameter will write the result directly to the provided buffer, avoiding heap allocation. Always pass a buffer whose `_data` does not alias the input buffers.
- `mj.map(a, f)` always returns a new `Matrix`, whereas `matrix.map(f)` (the `Matrix` instance method) is in-place.
- Native dispatch thresholds are adaptive; small matrices use the JS path even when the native backend is available.
