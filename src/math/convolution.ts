import Matrix from "../matrix";
import zeros from "./zeros";
import { isNativeAvailable, convolutionNative } from "./rust_backend";

/**
 * Menghitung convolution dari matrix a dengan kernel — DIOPTIMASI
 * @param a Matrix
 * @param kernel Matrix
 * @returns Matrix
 */
export default function convolution(a: Matrix, kernel: Matrix): Matrix {
  const aRows = a._shape[0], aCols = a._shape[1];
  const kRows = kernel._shape[0], kCols = kernel._shape[1];
  const outRows = aRows - kRows + 1;
  const outCols = aCols - kCols + 1;

  if (isNativeAvailable()) {
    const res = convolutionNative(a._data, aRows, aCols, kernel._data, kRows, kCols);
    return Matrix.fromFlat(res, [outRows, outCols]);
  }

  const matrix = zeros([outRows, outCols]);
  const aData = a._data;
  const kData = kernel._data;
  const outData = matrix._data;

  for (let i = 0; i < outRows; i++) {
    for (let j = 0; j < outCols; j++) {
      let sum = 0;
      for (let k = 0; k < kRows; k++) {
        const aOffset = (i + k) * aCols + j;
        const kOffset = k * kCols;
        for (let l = 0; l < kCols; l++) {
          sum += aData[aOffset + l] * kData[kOffset + l];
        }
      }
      outData[i * outCols + j] = sum;
    }
  }

  return matrix;
}
