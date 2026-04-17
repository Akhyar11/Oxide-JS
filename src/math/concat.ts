import Matrix from "../matrix";

/**
 * Menggabungkan dua buah matrix, pastikan matrix sudah di flatten atau berbentuk [1, n] — DIOPTIMASI
 * @param a Matrix
 * @param b Matrix
 * @returns Matrix
 */
export default function concat(a: Matrix, b: Matrix): Matrix {
  if (a._shape[0] !== 1 || b._shape[0] !== 1) {
    throw new Error(`pastikan matrix sudah di flatten atau berbentuk [1, n]`);
  }
  const result = new Float32Array(a._data.length + b._data.length);
  // Float32Array .set runs at native speed in V8
  result.set(a._data);
  result.set(b._data, a._data.length);
  return Matrix.fromFlat(result, [1, result.length]);
}
