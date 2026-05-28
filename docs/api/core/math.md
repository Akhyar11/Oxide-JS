# 🧮 Math Primitives (mj Namespace) API Reference

The **[mj](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/core/src/math/index.ts)** namespace serves as the primary mathematics dispatcher for Oxide-JS. It contains highly optimized, vector-accelerated operations that dynamically select the **Rust Native Backend** (utilizing parallel threads via `Rayon` and dense multipliers via `matrixmultiply`) if installed, gracefully falling back to optimized JavaScript loop implementations when unavailable.

---

## 🏗️ Core Math Operations

### 1. Matrix Multiplication
```ts
mj.dotProduct(a: Matrix, b: Matrix): Matrix
```
Calculates the dot product of two matrices.
- **Math**: $C = A \cdot B$
- **Requirements**: Columns in matrix `a` must equal rows in matrix `b`.
- **Example**:
  ```ts
  import { mj } from "@oxide-js/core";

  const a = mj.matrix([[1, 2], [3, 4]]);
  const b = mj.matrix([[5, 6], [7, 8]]);
  const c = mj.dotProduct(a, b);
  c.print(); // Output matches: [[19, 22], [43, 50]]
  ```

### 2. Element-Wise Math Operations

#### `add(a: Matrix, b: Matrix | number): Matrix`
Element-wise addition. Returns a new matrix.
- **Example**:
  ```ts
  const a = mj.matrix([[1, 2]]);
  mj.add(a, 10).print(); // [[11, 12]]
  ```

#### `sub(a: Matrix, b: Matrix | number): Matrix`
Element-wise subtraction. Returns a new matrix.
- **Example**:
  ```ts
  const a = mj.matrix([[5, 10]]);
  mj.sub(a, 2).print(); // [[3, 8]]
  ```

#### `mul(a: Matrix, b: Matrix | number): Matrix`
Element-wise Hadamard multiplication. Returns a new matrix.
- **Example**:
  ```ts
  const a = mj.matrix([[2, 4]]);
  mj.mul(a, 3).print(); // [[6, 12]]
  ```

#### `div(a: Matrix, b: Matrix | number): Matrix`
Element-wise division. Throws on division by zero.
- **Example**:
  ```ts
  const a = mj.matrix([[10, 20]]);
  mj.div(a, 5).print(); // [[2, 4]]
  ```

#### `pow(a: Matrix, exponent: number): Matrix`
Element-wise raise to power.
- **Example**:
  ```ts
  const a = mj.matrix([[2, 3]]);
  mj.pow(a, 3).print(); // [[8, 27]]
  ```

#### `expm(a: Matrix): Matrix`
Element-wise exponential ($e^x$).
- **Example**:
  ```ts
  const a = mj.matrix([[0, 1]]);
  mj.expm(a).print(); // [[1.0, 2.71828]]
  ```

#### `logm(a: Matrix): Matrix`
Element-wise natural logarithm ($\ln(x)$).
- **Example**:
  ```ts
  const a = mj.matrix([[1, 2.71828]]);
  mj.logm(a).print(); // [[0.0, 1.0]]
  ```

#### `absm(a: Matrix): Matrix`
Element-wise absolute values.
- **Example**:
  ```ts
  const a = mj.matrix([[-1, -2]]);
  mj.absm(a).print(); // [[1, 2]]
  ```

---

### 3. In-Place Target Writing
Optimized variants that write results directly into target tensors, avoiding intermediate matrix allocations.

#### `addInto(target: Matrix, source: Matrix): void`
Adds `source` element-wise directly into `target`.
- **Example**:
  ```ts
  const target = mj.matrix([[1, 2]]);
  const source = mj.matrix([[10, 20]]);
  mj.addInto(target, source);
  target.print(); // [[11, 22]]
  ```

#### `subInto(target: Matrix, source: Matrix): void`
Subtracts `source` element-wise directly from `target`.
- **Example**:
  ```ts
  const target = mj.matrix([[10, 20]]);
  const source = mj.matrix([[2, 3]]);
  mj.subInto(target, source);
  target.print(); // [[8, 17]]
  ```

---

## 📊 Dimension Reduction & Concatenation

### 1. `sumAxis(matrix: Matrix, axis: 0 | 1): Matrix`
Sums elements along a specified dimension axis.
- **`axis = 0`**: Columns summation. Reduces shape from `[R, C]` to `[1, C]`.
- **`axis = 1`**: Rows summation. Reduces shape from `[R, C]` to `[R, 1]`.
- **Example**:
  ```ts
  const a = mj.matrix([[1, 2], [3, 4]]);
  mj.sumAxis(a, 0).print(); // Columns sum: [[4, 6]]
  mj.sumAxis(a, 1).print(); // Rows sum: [[3], [7]]
  ```

### 2. `mean(matrix: Matrix): number`
Computes the global scalar mean value across all elements of the matrix.
- **Example**:
  ```ts
  const a = mj.matrix([[1, 2], [3, 4]]);
  console.log(mj.mean(a)); // 2.5
  ```

### 3. `concat(matrices: Matrix[], axis: 0 | 1): Matrix`
Concatenates an array of matrices along the specified axis (0 for vertical stack, 1 for horizontal stack).
- **Example**:
  ```ts
  const a = mj.matrix([[1, 2]]);
  const b = mj.matrix([[3, 4]]);
  mj.concat([a, b], 0).print(); // Vertical stack: [[1, 2], [3, 4]]
  mj.concat([a, b], 1).print(); // Horizontal stack: [[1, 2, 3, 4]]
  ```

### 4. `argmax(matrix: Matrix, axis?: 0 | 1): number | Matrix`
Finds the index of the maximum value.
- **`axis = undefined`**: Returns the global flat index of the maximum element.
- **`axis = 0`**: Returns the row index of the max value for each column (shape `[1, C]`).
- **`axis = 1`**: Returns the column index of the max value for each row (shape `[R, 1]`).
- **Example**:
  ```ts
  const a = mj.matrix([[1, 5, 2], [4, 3, 6]]);
  console.log(mj.argmax(a)); // 5 (flattened index of 6)
  mj.argmax(a, 1).print(); // [[1], [2]] (max indices per row)
  ```

---

## 🎨 Initialization Methods

### 1. `matrix(array: number[][]): Matrix`
Alias constructor. Creates a matrix from a 2D array.
- **Example**: `const m = mj.matrix([[1, 2], [3, 4]])`

### 2. `zeros(shape: [number, number]): Matrix`
Generates a new zero-initialized matrix of the given shape.
- **Example**: `mj.zeros([2, 3]).print(); // 2 rows, 3 columns of zeros`

### 3. `ones(shape: [number, number]): Matrix`
Generates a new matrix filled with ones.
- **Example**: `mj.ones([1, 4]).print(); // [[1, 1, 1, 1]]`

### 4. `random(shape: [number, number]): Matrix`
Generates a matrix filled with random numbers distributed between `[-0.5, 0.5]`.
- **Example**: `const r = mj.random([2, 2]);`

### 5. `xavier(shape: [number, number]): Matrix`
Initializes weights using the **Xavier Normal (Glorot)** distribution.
- **Math**: $\sigma = \sqrt{\frac{2}{\text{in} + \text{out}}}$
- **Best Use**: Sigmoid or Tanh active layers.
- **Example**: `const w = mj.xavier([64, 32]);`

### 6. `he(shape: [number, number]): Matrix`
Initializes weights using the **He Normal** distribution.
- **Math**: $\sigma = \sqrt{\frac{2}{\text{in}}}$
- **Best Use**: ReLU or Leaky ReLU active layers.
- **Example**: `const w = mj.he([64, 32]);`

---

## ⚙️ Advanced Signal & Neural Primitives

### 1. `addBias(matrix: Matrix, bias: Matrix): Matrix`
Adds a 1D bias vector to every row of a 2D matrix.
- **Shape Constraints**: Bias must be `[1, C]` where `C` matches columns in the matrix.
- **Example**:
  ```ts
  const a = mj.matrix([[1, 2], [3, 4]]);
  const bias = mj.matrix([[10, 20]]);
  mj.addBias(a, bias).print(); // [[11, 22], [13, 24]]
  ```

### 2. `convolution(input: Matrix, kernel: Matrix, stride?: number, padding?: number): Matrix`
Applies a discrete 2D spatial convolution operation on the input matrix using a specialized sliding kernel window.
- **Example**:
  ```ts
  const input = mj.matrix([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
  ]);
  const kernel = mj.matrix([
    [1, 0],
    [0, 1]
  ]);
  const output = mj.convolution(input, kernel, 1, 0);
  output.print();
  ```

### 3. `clipGradients(matrix: Matrix, clipValue: number): Matrix`
Clips values within the range `[-clipValue, clipValue]`.
- **Best Use**: Preventing exploding gradients in recurrent networks (LSTMs/GRUs).
- **Example**:
  ```ts
  const grads = mj.matrix([[-10.0, 5.0, 0.2]]);
  mj.clipGradients(grads, 2.0).print(); // [[-2.0, 2.0, 0.2]]
  ```

### 4. `norm(matrix: Matrix): Matrix` & `normScalar(matrix: Matrix): number`
Calculates L2 Euclidean normalization.
- **Example**:
  ```ts
  const a = mj.matrix([[3, 4]]);
  console.log(mj.normScalar(a)); // 5 (L2 norm)
  mj.norm(a).print();            // Normalize: [[0.6, 0.8]]
  ```

### 5. `threshold(matrix: Matrix, threshold: number, leak: number): Matrix`
Applies a thresholding function to a matrix, often used in Spiking Neural Networks (SNNs) as a Leaky Integrate-and-Fire output function. Emits `1.0` if $x \ge threshold$, otherwise emits $leak$ (usually 0.0).
- **Example**:
  ```ts
  const a = mj.matrix([[0.5, 1.2]]);
  mj.threshold(a, 1.0, 0.0).print(); // [[0.0, 1.0]]
  ```

---

## 🔗 Integrated Namespaces
For convenience, `mj` also exposes aliases to other core utilities:
- **Activations**: `mj.sigmoid`, `mj.tanh`, `mj.relu`, `mj.lRelu`, `mj.softmax`, `mj.threshold`, etc. (See [Activation Docs](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/docs/api/core/activation.md)).
- **Cost Loss Wrappers**: `mj.mse`, `mj.crossEntropy`, `mj.binaryCrossEntropy`, `mj.softmaxCrossEntropy` (Returns scalar values).
- **Optimizers**: `mj.SGD`, `mj.Adam`, `mj.NAG`, `mj.AdaGrad`, `mj.Momentum`.
