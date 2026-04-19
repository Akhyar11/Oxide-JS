import { MatrixCollection } from "../@types/type";
import Matrix from "../matrix";
import { isNativeAvailable, addNative } from "./rust_backend";

const ensureOutputShape = (out: Matrix, rows: number, cols: number): void => {
  if (out._shape[0] !== rows || out._shape[1] !== cols) {
    throw new Error(`Output matrix shape mismatch: expected [${rows}x${cols}], got [${out._shape[0]}x${out._shape[1]}]`);
  }
};

const ensureNoAliasing = (a: Matrix, b: Matrix, out: Matrix): void => {
  if (out._data === a._data || out._data === b._data) {
    throw new Error(
      "add(a, b, out) tidak mengizinkan aliasing: `out` harus memiliki buffer terpisah dari `a` dan `b`."
    );
  }
};

/**
 * Penjumlahan a dan b — DIOPTIMASI
 */
export default function add(a: MatrixCollection, b: MatrixCollection, out?: Matrix): Matrix {
  if (typeof a === "number") {
    const bm = b as Matrix;
    const result = out ? out._data : new Float32Array(bm._data.length);
    if (out) ensureOutputShape(out, bm._shape[0], bm._shape[1]);
    for (let i = 0; i < bm._data.length; i++) result[i] = a + bm._data[i];
    return out || Matrix.fromFlat(result, [bm._shape[0], bm._shape[1]]);
  }
  if (typeof b === "number") {
    const am = a as Matrix;
    const result = out ? out._data : new Float32Array(am._data.length);
    if (out) ensureOutputShape(out, am._shape[0], am._shape[1]);
    for (let i = 0; i < am._data.length; i++) result[i] = am._data[i] + b;
    return out || Matrix.fromFlat(result, [am._shape[0], am._shape[1]]);
  }
  const am = a as Matrix;
  const bm = b as Matrix;
  if (am._shape[0] !== bm._shape[0] || am._shape[1] !== bm._shape[1]) {
    throw new Error(`bentuk dari a harus sama dengan matrix ${a._shape} != ${b._shape}`);
  }
  if (out) {
    ensureOutputShape(out, am._shape[0], am._shape[1]);
    ensureNoAliasing(am, bm, out);
  }

  const resultData = out ? out._data : new Float32Array(am._data.length);

  // USE NATIVE IF AVAILABLE
  if (isNativeAvailable()) {
    addNative(am._data, bm._data, resultData);
    return out || Matrix.fromFlat(resultData, [am._shape[0], am._shape[1]]);
  }

  for (let i = 0; i < am._data.length; i++) resultData[i] = am._data[i] + bm._data[i];
  return out || Matrix.fromFlat(resultData, [am._shape[0], am._shape[1]]);
}

/**
 * Variasi eksplisit untuk penjumlahan matrix ke buffer output.
 * Kontrak: `out` harus berbentuk sama dengan input DAN tidak boleh alias dengan buffer `a`/`b`.
 */
export function addInto(a: Matrix, b: Matrix, out: Matrix): Matrix {
  return add(a, b, out);
}
