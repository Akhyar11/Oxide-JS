import Matrix from "../matrix";

/**
 * Log natural (ln) => Matrix log(a) — DIOPTIMASI
 * @param a Matrix
 * @returns Matrix
 */
export default function logm(a: Matrix): Matrix {
  const result = new Float32Array(a._data.length);
  const data = a._data;
  for (let i = 0; i < data.length; i++) {
    const val = data[i];
    result[i] = Math.log(val <= 0 ? 1e-15 : val);
  }
  return Matrix.fromFlat(result, [a._shape[0], a._shape[1]]);
}
