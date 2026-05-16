import Matrix from "../matrix/index.js";

/**
 * Memetakan matrix kedalam function — DIOPTIMASI
 */
export default function map(
  a: Matrix,
  func: (value: number, index: number) => number
): Matrix {
  const result = new Float32Array(a._data.length);
  for (let i = 0; i < a._data.length; i++) {
    result[i] = func(a._data[i], i);
  }
  return Matrix.fromFlat(result, [a._shape[0], a._shape[1]]);
}
