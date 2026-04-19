# Math and Matrix

## Matrix

`Matrix` menyimpan:
- `_data: Float32Array`
- `_shape: [rows, cols]`

Utility utama:
- `get(i,j)`, `set(i,j,val)`
- `getCol(colIndex)`, `setCol(colIndex, data)`
- `reshape(shape)`, `flatten()`
- `clone()`, `copyFrom(other)`
- `addInPlace/subInPlace/mulInPlace`

## Shape/layout data
- Flat row-major: `idx = row * cols + col`
- `_value` getter/setter ada untuk kompatibilitas tetapi mengalokasikan array baru.

## Operasi math penting
- Linear algebra: `dotProduct`, `transpose`
- Element-wise: `add`, `sub`, `mul`, `div`
- Utility: `zeros`, `ones`, `random`, `xavier`, `he`
- Stabilitas training: `addBias`, `sumAxis`, `clipGradients`

## Contoh penggunaan
```ts
import mj from "../src/math";

const x = mj.matrix([[1, 2], [3, 4]]);
const w = mj.xavier([3, 2]);
const b = mj.zeros([3, 1]);

const z = mj.dotProduct(w, x); // [3,2]
mj.addBias(z, b);
const rowSum = mj.sumAxis(z, 1);

mj.clipGradients(z, 1.0);
console.log(rowSum._shape);
```
