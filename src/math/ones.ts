import { MatrixShape } from "../@types/type";
import Matrix from "../matrix";

/**
 * Memberikan nilai matrix 1 dengan ukuran [n, n] — DIOPTIMASI
 * @param shape [number, number]
 * @returns Matrix
 */
export default function ones(shape: MatrixShape): Matrix {
  const n = shape[0] * shape[1];
  const data = new Float32Array(n);
  // Array.fill sangat dioptimasi di V8
  data.fill(1);
  return Matrix.fromFlat(data, shape);
}
