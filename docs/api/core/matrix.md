# 🧮 Matrix API Reference

The **[Matrix](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/core/src/matrix/index.ts)** class is the primary data structure in the **Oxide-JS** engine. It is backed by a flat, contiguous `Float32Array` memory buffer to ensure maximum mathematical throughput and cache locality, while avoiding garbage collection bottlenecks.

---

## 📌 Properties

| Property | Type | Access | Description |
| :--- | :--- | :--- | :--- |
| **`_data`** | `Float32Array` | Public | Flat, linear memory buffer holding elements. Access via `i * cols + j`. |
| **`_shape`** | `[number, number]` | Public | 2D dimension tuple represented as `[rows, cols]`. |
| **`_version`** | `number` | Public | Modification version count. Automatically increments on in-place updates. |
| **`grad`** | `Matrix \| null` | Public | Stores computed partial derivatives during backward automatic differentiation. |
| **`requiresGrad`** | `boolean` | Public | If `true`, the `Tape` auto-diff engine tracks mathematical operations on this matrix. |
| **`name`** | `string \| undefined` | Public | Optional identifier for debugging or graph visualization. |

### 📝 Property Access Example
```ts
import { Matrix } from "@oxide-js/core";

const mat = new Matrix({ array: [[5, 6], [7, 8]] });
console.log(mat._shape);      // [2, 2]
console.log(mat._data);       // Float32Array [ 5, 6, 7, 8 ]
console.log(mat._version);    // 0
console.log(mat.requiresGrad); // false
```

---

## 🛠️ Constructors & Static Initializers

### 1. `new Matrix({ array: matrix2d })`
Creates a matrix from a standard 2D JavaScript number array.
- **Arguments**:
  - `config: { array: number[][] }` - A standard 2D Javascript nested number array.
- **Example**:
  ```ts
  import { Matrix } from "@oxide-js/core";
  const m = new Matrix({ array: [[1, 2], [3, 4]] });
  ```

### 2. `Matrix.fromFlat(data: Float32Array | number[], shape: [number, number]): Matrix`
Constructs a matrix from a flat memory array, saving performance by avoiding nested array allocations.
- **Arguments**:
  - `data` - A flat typed array or array-like containing linear elements.
  - `shape` - A dimension tuple `[rows, cols]`.
- **Example**:
  ```ts
  import { Matrix } from "@oxide-js/core";
  const m = Matrix.fromFlat(new Float32Array([1, 2, 3, 4]), [2, 2]);
  ```

---

## 🔑 Accessors

### 1. `get(i: number, j: number): number`
Performs flat index retrieval mapping 2D indices `(i, j)` to flat indices `i * cols + j`.
- **Performance**: Extremely fast lookup, O(1).
- **Example**:
  ```ts
  const m = new Matrix({ array: [[10, 20], [30, 40]] });
  console.log(m.get(0, 1)); // 20
  console.log(m.get(1, 0)); // 30
  ```

### 2. `set(i: number, j: number, val: number): void`
Sets value at index `(i, j)`. Automatically increments `_version` to alert the autodiff engine of mutations.
- **Example**:
  ```ts
  const m = new Matrix({ array: [[1, 2], [3, 4]] });
  m.set(0, 1, 99);
  console.log(m.get(0, 1)); // 99
  console.log(m._version); // 1
  ```

### 3. `get _value: number[][]` (Getter)
Converts flat storage back to standard JS nested arrays for backward compatibility.
> [!WARNING]
> **Performance Hit:** Avoid calling this getter in computational loops, as it allocates fresh nested JavaScript arrays on every call. Use `_data` directly for performance.
- **Example**:
  ```ts
  const m = Matrix.fromFlat([1, 2, 3, 4], [2, 2]);
  console.log(m._value); // [[1, 2], [3, 4]]
  ```

### 4. `set _value(arr: number[][])` (Setter)
Copies nested 2D array contents into flat storage. Re-allocates internal `_data` and increments `_version`.
- **Example**:
  ```ts
  const m = Matrix.fromFlat([0, 0, 0, 0], [2, 2]);
  m._value = [[5, 6], [7, 8]];
  console.log(m.get(0, 0)); // 5
  ```

---

## 🌀 Instance Methods

### 🧪 Basic Math Ops (Out-of-place equivalents are defined on `mj`)
These perform operations and return a fresh `Matrix`.

#### 1. `add(other: Matrix | number): void`
Performs element-wise addition with a scalar or another matrix of the same shape.
- **Example**:
  ```ts
  const m = new Matrix({ array: [[1, 2]] });
  m.add(10); // Adds scalar 10 element-wise
  ```

#### 2. `sub(other: Matrix | number): void`
Performs element-wise subtraction.
- **Example**:
  ```ts
  const m = new Matrix({ array: [[5, 10]] });
  m.sub(2); // Subtracts scalar 2 element-wise
  ```

#### 3. `mul(other: Matrix | number): void`
Performs element-wise Hadamard multiplication.
- **Example**:
  ```ts
  const m = new Matrix({ array: [[2, 3]] });
  m.mul(3); // Hadamard element-wise multiply by 3
  ```

#### 4. `div(other: Matrix | number): void`
Performs element-wise division. Throws if dividing by zero.
- **Example**:
  ```ts
  const m = new Matrix({ array: [[10, 20]] });
  m.div(5); // Divides element-wise by 5
  ```

---

### ⚡ Accelerated In-Place Operations
These operations modify internal `_data` directly without allocating new objects. They automatically dispatch to optimized **Rust native kernels** when available.

#### 1. `addInPlace(other: Matrix | number): void`
In-place element-wise addition.
- **Example**:
  ```ts
  const m1 = new Matrix({ array: [[1, 2]] });
  const m2 = new Matrix({ array: [[10, 20]] });
  m1.addInPlace(m2);
  m1.print(); // [[11, 22]]
  ```

#### 2. `subInPlace(other: Matrix | number): void`
In-place element-wise subtraction.
- **Example**:
  ```ts
  const m = new Matrix({ array: [[10, 20]] });
  m.subInPlace(5);
  m.print(); // [[5, 15]]
  ```

#### 3. `mulInPlace(other: Matrix | number): void`
In-place element-wise multiplication.
- **Example**:
  ```ts
  const m = new Matrix({ array: [[2, 4]] });
  m.mulInPlace(3);
  m.print(); // [[6, 12]]
  ```

---

## 📐 Shape Restructuring

#### 1. `flatten(): void`
Collapses shape interpretation into a single row vector with shape `[1, rows * cols]` without modifying the underlying linear flat array buffer.
- **Example**:
  ```ts
  const m = new Matrix({ array: [[1, 2], [3, 4]] });
  m.flatten();
  console.log(m._shape); // [1, 4]
  ```

#### 2. `reshape(shape: [number, number]): void`
Changes the dimension mapping to a new target shape.
- **Constraint**: Total element count (`shape[0] * shape[1]`) must equal original element count.
- **Example**:
  ```ts
  const m = new Matrix({ array: [[1, 2, 3, 4]] });
  m.reshape([2, 2]);
  console.log(m._shape); // [2, 2]
  ```

---

## 👥 Cloning & Detachment

#### 1. `clone(): Matrix`
Performs a deep-copy of the internal `_data` and replicates the `_shape` and `requiresGrad` configuration into a fresh `Matrix` object.
- **Example**:
  ```ts
  const m1 = new Matrix({ array: [[1, 2]] });
  const m2 = m1.clone();
  m2.set(0, 0, 99);
  console.log(m1.get(0, 0)); // 1 (unchanged)
  ```

#### 2. `copyFrom(other: Matrix): void`
Copies data directly from another matrix's linear buffer without allocating a new matrix container. Throws if sizes mismatch.
- **Example**:
  ```ts
  const m1 = new Matrix({ array: [[0, 0]] });
  const m2 = new Matrix({ array: [[9, 9]] });
  m1.copyFrom(m2);
  m1.print(); // [[9, 9]]
  ```

#### 3. `detach(): Matrix`
Clones the matrix, zeroes its gradient tracker (`grad = null`), and disables gradient recording (`requiresGrad = false`). Extremely useful when freezing layers or during validation loops.
- **Example**:
  ```ts
  const x = new Matrix({ array: [[1, 2]] });
  x.requiresGrad = true;
  const detachedX = x.detach();
  console.log(detachedX.requiresGrad); // false
  console.log(detachedX.grad);         // null
  ```

#### 4. `clearGrad(): void`
Resets the partial derivatives record (`this.grad = null`). Must be called before starting backward propagation.
- **Example**:
  ```ts
  const x = new Matrix({ array: [[1, 2]] });
  x.grad = new Matrix({ array: [[0.1, 0.2]] });
  x.clearGrad();
  console.log(x.grad); // null
  ```

---

## 🖨️ Utilities

### `print(): void`
Formats and prints internal values to the console using a clean, visually structured table layout.
- **Example**:
  ```ts
  const m = new Matrix({ array: [[1, 2], [3, 4]] });
  m.print();
  // Prints beautifully to the terminal:
  // ┌─────────┬───┬───┐
  // │ (index) │ 0 │ 1 │
  // ├─────────┼───┼───┤
  // │    0    │ 1 │ 2 │
  // │    1    │ 3 │ 4 │
  // └─────────┴───┴───┘
  ```
