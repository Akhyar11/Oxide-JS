import Matrix from "../matrix/index.js";
import { isNativeAvailable, dotProductNative } from "./rust_backend.js";
import { engine } from "../autodiff/engine.js";

/**
 * Perkalian product matrix a dan b — DIOPTIMASI
 * Mendukung transposisi "on-the-fly" tanpa alokasi tambahan.
 * 
 * @param a Matrix
 * @param b Matrix
 * @param out Optional Matrix to store result
 * @param transA Jika true, anggap a adalah a^T
 * @param transB Jika true, anggap b adalah b^T
 * @returns Matrix
 */
export default function dotProduct(
  a: Matrix,
  b: Matrix,
  out?: Matrix,
  transA: boolean = false,
  transB: boolean = false
): Matrix {
  const aRowsOrig = a._shape[0], aColsOrig = a._shape[1];
  const bRowsOrig = b._shape[0], bColsOrig = b._shape[1];

  const aRows = transA ? aColsOrig : aRowsOrig;
  const aCols = transA ? aRowsOrig : aColsOrig;
  const bRows = transB ? bColsOrig : bRowsOrig;
  const bCols = transB ? bRowsOrig : bColsOrig;

  if (aCols !== bRows) {
    throw new Error(`Dimensi matrix tidak cocok untuk dot product: [${aRows}x${aCols}] * [${bRows}x${bCols}]`);
  }

  if (out) {
    if (out._shape[0] !== aRows || out._shape[1] !== bCols) {
      throw new Error(`Output matrix shape mismatch: expected [${aRows}x${bCols}], got [${out._shape[0]}x${out._shape[1]}]`);
    }
  }

  let res: Matrix;
  if (isNativeAvailable()) {
    res = out || Matrix.fromFlat(new Float32Array(aRows * bCols), [aRows, bCols]);
    dotProductNative(
      a._data,
      aRowsOrig,
      aColsOrig,
      b._data,
      bRowsOrig,
      bColsOrig,
      transA,
      transB,
      res._data
    );
  } else {
    const resultData = out ? out._data : new Float32Array(aRows * bCols);
    const aData = a._data;
    const bData = b._data;

    // Kasus 1: Standar A * B (atau A^T * B) - Gunakan i-k-j untuk cache friendliness pada B
    if (!transB) {
      if (out) resultData.fill(0);
      for (let i = 0; i < aRows; i++) {
        const rOffset = i * bCols;
        for (let k = 0; k < aCols; k++) {
          const aik = transA ? aData[k * aRows + i] : aData[i * aCols + k];
          if (aik === 0) continue;

          const kOffset = k * bCols;
          let j = 0;
          const jBound = bCols - 8;
          for (; j <= jBound; j += 8) {
            resultData[rOffset + j] += aik * bData[kOffset + j];
            resultData[rOffset + j + 1] += aik * bData[kOffset + j + 1];
            resultData[rOffset + j + 2] += aik * bData[kOffset + j + 2];
            resultData[rOffset + j + 3] += aik * bData[kOffset + j + 3];
            resultData[rOffset + j + 4] += aik * bData[kOffset + j + 4];
            resultData[rOffset + j + 5] += aik * bData[kOffset + j + 5];
            resultData[rOffset + j + 6] += aik * bData[kOffset + j + 6];
            resultData[rOffset + j + 7] += aik * bData[kOffset + j + 7];
          }
          for (; j < bCols; j++) {
            resultData[rOffset + j] += aik * bData[kOffset + j];
          }
        }
      }
    }
    // Kasus 2: A * B^T (atau A^T * B^T) - Gunakan i-j-k untuk cache friendliness pada A dan B
    else {
      for (let i = 0; i < aRows; i++) {
        const rOffset = i * bCols;
        for (let j = 0; j < bCols; j++) {
          let sum = 0;
          // Loop k
          let k = 0;
          const kBound = aCols - 8;
          for (; k <= kBound; k += 8) {
            const aik0 = transA ? aData[k * aRows + i] : aData[i * aCols + k];
            const bjk0 = bData[j * aCols + k];
            sum += aik0 * bjk0;

            const aik1 = transA ? aData[(k + 1) * aRows + i] : aData[i * aCols + (k + 1)];
            const bjk1 = bData[j * aCols + (k + 1)];
            sum += aik1 * bjk1;

            const aik2 = transA ? aData[(k + 2) * aRows + i] : aData[i * aCols + (k + 2)];
            const bjk2 = bData[j * aCols + (k + 2)];
            sum += aik2 * bjk2;

            const aik3 = transA ? aData[(k + 3) * aRows + i] : aData[i * aCols + (k + 3)];
            const bjk3 = bData[j * aCols + (k + 3)];
            sum += aik3 * bjk3;

            const aik4 = transA ? aData[(k + 4) * aRows + i] : aData[i * aCols + (k + 4)];
            const bjk4 = bData[j * aCols + (k + 4)];
            sum += aik4 * bjk4;

            const aik5 = transA ? aData[(k + 5) * aRows + i] : aData[i * aCols + (k + 5)];
            const bjk5 = bData[j * aCols + (k + 5)];
            sum += aik5 * bjk5;

            const aik6 = transA ? aData[(k + 6) * aRows + i] : aData[i * aCols + (k + 6)];
            const bjk6 = bData[j * aCols + (k + 6)];
            sum += aik6 * bjk6;

            const aik7 = transA ? aData[(k + 7) * aRows + i] : aData[i * aCols + (k + 7)];
            const bjk7 = bData[j * aCols + (k + 7)];
            sum += aik7 * bjk7;
          }
          for (; k < aCols; k++) {
            const aik = transA ? aData[k * aRows + i] : aData[i * aCols + k];
            sum += aik * bData[j * aCols + k];
          }
          resultData[rOffset + j] = sum;
        }
      }
    }
    res = out ? out : Matrix.fromFlat(resultData, [aRows, bCols]);
  }

  // RECORD FOR AUTO-DIFF
  if (engine.tape) {
    engine.record([a, b], [res], (grad: Matrix) => {
      const gA = !transA
        ? dotProduct(grad, b, undefined, false, !transB)
        : dotProduct(b, grad, undefined, transB, true);
      const gB = !transB
        ? dotProduct(a, grad, undefined, !transA, false)
        : dotProduct(grad, a, undefined, true, transA);
      return [gA, gB];
    }, { saveInput: false, saveOutput: false, requireInputStability: true });
  }

  return res;
}
