import { MatrixShape } from "../@types/type";
import Matrix from "../matrix";

/**
 * Xavier Uniform Initialization (Glorot Initialization)
 * Formula: U(-sqrt(6 / (n_in + n_out)), sqrt(6 / (n_in + n_out)))
 */
export default function xavier(shape: MatrixShape): Matrix {
  const n_in = shape[1];
  const n_out = shape[0];
  const limit = Math.sqrt(6 / (n_in + n_out));
  const n = n_in * n_out;
  const data = new Float64Array(n);
  
  for (let i = 0; i < n; i++) {
    data[i] = (Math.random() * 2 - 1) * limit;
  }
  
  return Matrix.fromFlat(data, shape);
}
