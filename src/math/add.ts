import { MatrixCollection } from "../@types/type";
import Matrix from "../matrix";
import { isNativeAvailable, addNative } from "./rust_backend";

/**
 * Penjumlahan a dan b — DIOPTIMASI
 */
export default function add(a: MatrixCollection, b: MatrixCollection): Matrix {
  if (typeof a === "number") {
    const bm = b as Matrix;
    const result = new Float32Array(bm._data.length);
    for (let i = 0; i < bm._data.length; i++) result[i] = a + bm._data[i];
    return Matrix.fromFlat(result, [bm._shape[0], bm._shape[1]]);
  }
  if (typeof b === "number") {
    const result = new Float32Array(a._data.length);
    for (let i = 0; i < a._data.length; i++) result[i] = a._data[i] + b;
    return Matrix.fromFlat(result, [a._shape[0], a._shape[1]]);
  }
  if (a._shape[0] !== b._shape[0] || a._shape[1] !== b._shape[1]) {
    throw new Error(`bentuk dari a harus sama dengan matrix ${a._shape} != ${b._shape}`);
  }

  // USE NATIVE IF AVAILABLE
  if (isNativeAvailable()) {
    const resultData = new Float32Array(a._data.length);
    addNative(a._data, b._data, resultData);
    return Matrix.fromFlat(resultData, [a._shape[0], a._shape[1]]);
  }

  const result = new Float32Array(a._data.length);
  for (let i = 0; i < a._data.length; i++) result[i] = a._data[i] + b._data[i];
  return Matrix.fromFlat(result, [a._shape[0], a._shape[1]]);
}
