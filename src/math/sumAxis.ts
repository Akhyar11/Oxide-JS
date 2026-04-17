import Matrix from "../matrix";
import { isNativeAvailable, sumAxisNative } from "./rust_backend";

/**
 * Menjumlahkan matrix sepanjang axis tertentu
 * @param a Matrix
 * @param axis 1 untuk baris (hasil [rows x 1]), 0 untuk kolom (hasil [1 x cols])
 * @param out Optional Matrix output
 */
export default function sumAxis(a: Matrix, axis: number, out?: Matrix): Matrix {
  const [rows, cols] = a._shape;
  const outShape: [number, number] = axis === 1 ? [rows, 1] : [1, cols];
  const result = out || Matrix.fromFlat(new Float32Array(outShape[0] * outShape[1]), outShape);

  if (isNativeAvailable()) {
    sumAxisNative(a._data, rows, cols, axis, result._data);
  } else {
    const data = a._data;
    const res = result._data;
    res.fill(0);
    if (axis === 1) {
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) res[i] += data[i * cols + j];
      }
    } else {
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) res[j] += data[i * cols + j];
      }
    }
  }
  return result;
}
