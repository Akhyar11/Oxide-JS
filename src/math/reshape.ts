import { MatrixShape } from "../@types/type";
import Matrix from "../matrix";

/**
 * Reshape matrix — DIOPTIMASI
 * Karena _data sudah flat, reshape hanya mengubah interpretasi shape
 */
export default function reshape(a: Matrix, shape: MatrixShape): Matrix {
  if (a._shape[0] * a._shape[1] !== shape[0] * shape[1]) {
    throw new Error(
      `panjang dari a tidak sama dengan bentuk yang diinginkan ${a._shape[0] * a._shape[1]}!=${shape[0] * shape[1]}`
    );
  }
  // Data sudah flat dan urut — hanya copy dan ubah shape
  const result = new Float32Array(a._data);
  return Matrix.fromFlat(result, shape);
}
