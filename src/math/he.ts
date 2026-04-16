import { MatrixShape } from "../@types/type";
import Matrix from "../matrix";

/**
 * He Uniform Initialization
 * Formula: U(-sqrt(6 / n_in), sqrt(6 / n_in))
 */
export default function he(shape: MatrixShape): Matrix {
  const n_in = shape[1];
  const n = shape[0] * n_in;
  const limit = Math.sqrt(6 / n_in);
  const data = new Float64Array(n);
  
  for (let i = 0; i < n; i++) {
    data[i] = (Math.random() * 2 - 1) * limit;
  }
  
  return Matrix.fromFlat(data, shape);
}
