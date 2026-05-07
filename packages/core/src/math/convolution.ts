import Matrix from "../matrix/index.js";
import zeros from "./zeros.js";
import { isNativeAvailable, convolutionNative } from "./rust_backend.js";
import { engine } from "../autodiff/engine.js";
import mj from "./index.js";

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

  let matrix: Matrix;
  if (isNativeAvailable()) {
    const res = convolutionNative(a._data, aRows, aCols, kernel._data, kRows, kCols);
    matrix = Matrix.fromFlat(res, [outRows, outCols]);
  } else {
    matrix = zeros([outRows, outCols]);
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
  }

  // RECORD FOR AUTO-DIFF
  const tape = engine.tape;
  if (tape) {
    tape.record([a, kernel], [matrix], (grad: Matrix) => {
      // dL/dKernel = convolution(input, grad)
      const dKernel = convolution(a, grad);
      if (kernel.grad) kernel.grad.addInPlace(dKernel);
      else kernel.grad = dKernel;

      // dL/dInput = "full" convolution(grad, flipped_kernel)
      const flippedK = Matrix.fromFlat(new Float32Array(kRows * kCols), [kRows, kCols]);
      for (let i = 0; i < kRows; i++) {
        for (let j = 0; j < kCols; j++) {
          flippedK._data[(kRows - 1 - i) * kCols + (kCols - 1 - j)] = kernel._data[i * kCols + j];
        }
      }

      // dInput = full convolution of grad with flipped kernel
      // "Full" convolution = padding grad with (kRows-1, kCols-1) on each side
      const pGrad = mj.zeros([outRows + 2 * (kRows - 1), outCols + 2 * (kCols - 1)]);
      const pgData = pGrad._data;
      const pgCols = pGrad._shape[1];
      const gData = grad._data;
      const gCols = grad._shape[1];
      for (let i = 0; i < outRows; i++) {
        for (let j = 0; j < outCols; j++) {
          pgData[(i + kRows - 1) * pgCols + (j + kCols - 1)] = gData[i * gCols + j];
        }
      }

      const gradA = convolution(pGrad, flippedK);
      if (a.grad) a.grad.addInPlace(gradA);
      else a.grad = gradA;
    });
  }

  return matrix;
}
