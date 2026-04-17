import Matrix from "../matrix";

/**
 * Meratakan matrix menjadi ukuran [n, 1] — DIOPTIMASI
 * Dengan Float32Array flat, flatten hanya ubah shape
 */
export default function flatten(a: Matrix): Matrix {
  const n = a._data.length;
  const result = new Float32Array(a._data);
  return Matrix.fromFlat(result, [n, 1]);
}
