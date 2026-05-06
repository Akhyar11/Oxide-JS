# Matrix

The `Matrix` class is the core data structure of **Oxide-JS**. It uses **`Float32Array`** for flat storage, providing memory efficiency and maximum access speed.

## Import

```ts
import { Matrix } from "@oxide-js/core"
```

## Overview

`Matrix` backs all numerical operations in the library. Every layer, model, and math primitive works with `Matrix` objects end-to-end. The flat `Float32Array` layout allows zero-copy access in hot paths via `_data`.

---

## API Reference

### Properties

#### `_data: Float32Array`

Flat data buffer. Element at row `i`, column `j` is stored at index `i * cols + j`.

#### `_shape: [rows, cols]`

Matrix dimensions, e.g. `[2, 3]` for 2 rows and 3 columns.

---

### Initialization & Creation

#### `constructor({ array: number[][] })`

Creates a matrix from a standard 2D array.

```ts
const m = new Matrix({
  array: [
    [1, 2],
    [3, 4]
  ]
});
// _data = Float32Array([1, 2, 3, 4]), _shape = [2, 2]
```

#### `static fromFlat(data: Float32Array, shape: [number, number]): Matrix`

Creates a matrix directly from flat data. Faster than the constructor because it skips nested-array conversion.

```ts
const rawData = new Float32Array([10, 20, 30, 40]);
const m = Matrix.fromFlat(rawData, [2, 2]);
// [[10, 20],
//  [30, 40]]
```

---

### Element Access & Modification

#### `get(i, j): number`

Returns the value at row `i`, column `j`.

#### `set(i, j, val): void`

Sets the value at row `i`, column `j`.

```ts
const m = new Matrix({ array: [[1, 2], [3, 4]] });

console.log(m.get(0, 1)); // 2

m.set(1, 0, 99);
// [[1,  2],
//  [99, 4]]
```

#### `getCol(index): Float32Array`

Returns the column at the given index as a typed array.

#### `setCol(index, data: Float32Array): void`

Replaces the column at the given index with the provided data.

```ts
const m = new Matrix({ array: [[1, 2], [3, 4]] });

const col1 = m.getCol(1);
// Float32Array([2, 4])

m.setCol(0, new Float32Array([10, 30]));
// [[10, 2],
//  [30, 4]]
```

---

### Element-wise Operations (In-place on Instance)

#### `add(a)`, `sub(a)`, `mul(a)`, `div(a)`

Basic arithmetic that **directly modifies the current matrix instance**.

> [!IMPORTANT]
> Unlike `mj.add` / `mj.sub` / `mj.mul` / `mj.div`, these instance methods do **not** return a new matrix.

```ts
const a = new Matrix({ array: [[1, 2], [3, 4]] });

a.add(10);
// [[11, 12], [13, 14]]

const b = new Matrix({ array: [[1, 2], [3, 4]] });
b.mul(b);
// [[1, 4], [9, 16]]  (Hadamard product in-place)
```

---

### In-Place Buffer Operations

Modify data directly on the original `_data` buffer without allocating a new matrix. Automatically accelerated when the Rust backend is available.

#### `addInPlace(other)`, `subInPlace(other)`, `mulInPlace(other)`

```ts
const m = new Matrix({ array: [[1, 2], [3, 4]] });
m.addInPlace(5);
// [[6, 7],
//  [8, 9]]
```

> **Note:** For `addInPlace` / `subInPlace`, the output buffer must not alias the input buffer. See [Native Backend](./native-backend.md) for details.

---

### Transformation & Utilities

#### `reshape(shape: [number, number]): void`

Changes the dimension interpretation without reordering data in memory.

#### `flatten(): void`

Flattens the matrix to a single row.

```ts
const m = new Matrix({ array: [[1, 2], [3, 4]] }); // [2, 2]

m.reshape([1, 4]);
// [[1, 2, 3, 4]]  — 1 row, 4 columns

m.flatten();
// [[1, 2, 3, 4]]  — flat vector
```

#### `clone(): Matrix`

Returns a new `Matrix` with a deep copy of the current data.

#### `map(func: (v: number) => number): void`

Applies a custom function to every element **in-place**.

```ts
const original = new Matrix({ array: [[1, 2], [3, 4]] });

const copy = original.clone();

original.map(v => v * 2);
// original: [[2, 4], [6, 8]]
```

> **Note:** `clone()` returns a new `Matrix`. `map(func)` modifies the current matrix and returns nothing.

#### `print(): void`

Displays the matrix in table format in the console for debugging.

```ts
m.print();
// ┌─────────┬────┬────┐
// │ (index) │ 0  │ 1  │
// ├─────────┼────┼────┤
// │    0    │ 2  │ 4  │
// │    1    │ 6  │ 8  │
// └─────────┴────┴────┘
```

---

## Notes

- Prefer `_data` (flat typed array) for performance-critical hot paths. `_value` exists for backward compatibility but allocates.
- The data layout is row-major: element at `(i, j)` is at `_data[i * cols + j]`.
- Most layer and model APIs accept and return `Matrix` objects directly.
