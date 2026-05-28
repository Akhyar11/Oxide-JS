import { Matrix } from "@oxide-js/core";
import { engine } from "@oxide-js/core";
import { isNativeAvailable, dotProductAddOnlyNativeWrapper } from "../native_backend.js";

/**
 * Perkalian product matrix a dan b KHUSUS UNTUK SNN (Add-Only)
 * Salah satu matriks HARUS berupa matriks biner (hanya berisi 0 dan 1).
 * Jika 0 maka di-skip, jika 1 maka cukup tambahkan nilainya tanpa dikalikan.
 * 
 * @param a Matrix
 * @param b Matrix
 * @param out Optional Matrix to store result
 * @param transA Jika true, anggap a adalah a^T
 * @param transB Jika true, anggap b adalah b^T
 * @returns Matrix
 */
export default function dotProductAddOnly(
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

  // Verifikasi kondisi biner: salah satu matrix harus berupa 0 dan 1
  let aIsBinary = true;
  for (let i = 0; i < a._data.length; i++) {
    const val = a._data[i];
    if (val !== 0 && val !== 1) {
      aIsBinary = false;
      break;
    }
  }

  let bIsBinary = true;
  if (!aIsBinary) {
    for (let i = 0; i < b._data.length; i++) {
      const val = b._data[i];
      if (val !== 0 && val !== 1) {
        bIsBinary = false;
        break;
      }
    }
  }

  if (!aIsBinary && !bIsBinary) {
    throw new Error("SNN Error: Kedua matriks adalah floating-point. Setidaknya salah satu matriks harus hanya berisi 0 dan 1.");
  }

  const resultData = out ? out._data : new Float32Array(aRows * bCols);
  const aData = a._data;
  const bData = b._data;

  if (isNativeAvailable()) {
    dotProductAddOnlyNativeWrapper(
      aData,
      aRowsOrig,
      aColsOrig,
      bData,
      bRowsOrig,
      bColsOrig,
      transA,
      transB,
      resultData
    );
  } else {
    // Standar A * B (atau A^T * B)
    if (!transB) {
    if (out) resultData.fill(0);
    for (let i = 0; i < aRows; i++) {
      const rOffset = i * bCols;
      for (let k = 0; k < aCols; k++) {
        const aik = transA ? aData[k * aRows + i] : aData[i * aCols + k];
        
        // Skip awal jika kita tahu aik = 0 (berlaku untuk kedua kasus binary)
        if (aik === 0) continue;

        const kOffset = k * bCols;
        let j = 0;
        const jBound = bCols - 8;

        if (aIsBinary) {
          // aik pasti 1 di sini
          for (; j <= jBound; j += 8) {
            resultData[rOffset + j] += bData[kOffset + j];
            resultData[rOffset + j + 1] += bData[kOffset + j + 1];
            resultData[rOffset + j + 2] += bData[kOffset + j + 2];
            resultData[rOffset + j + 3] += bData[kOffset + j + 3];
            resultData[rOffset + j + 4] += bData[kOffset + j + 4];
            resultData[rOffset + j + 5] += bData[kOffset + j + 5];
            resultData[rOffset + j + 6] += bData[kOffset + j + 6];
            resultData[rOffset + j + 7] += bData[kOffset + j + 7];
          }
          for (; j < bCols; j++) {
            resultData[rOffset + j] += bData[kOffset + j];
          }
        } else {
          // bIsBinary = true, aik adalah float biasa
          for (; j <= jBound; j += 8) {
            if (bData[kOffset + j] === 1) resultData[rOffset + j] += aik;
            if (bData[kOffset + j + 1] === 1) resultData[rOffset + j + 1] += aik;
            if (bData[kOffset + j + 2] === 1) resultData[rOffset + j + 2] += aik;
            if (bData[kOffset + j + 3] === 1) resultData[rOffset + j + 3] += aik;
            if (bData[kOffset + j + 4] === 1) resultData[rOffset + j + 4] += aik;
            if (bData[kOffset + j + 5] === 1) resultData[rOffset + j + 5] += aik;
            if (bData[kOffset + j + 6] === 1) resultData[rOffset + j + 6] += aik;
            if (bData[kOffset + j + 7] === 1) resultData[rOffset + j + 7] += aik;
          }
          for (; j < bCols; j++) {
            if (bData[kOffset + j] === 1) resultData[rOffset + j] += aik;
          }
        }
      }
    }
  } 
  // A * B^T (atau A^T * B^T)
  else {
    for (let i = 0; i < aRows; i++) {
      const rOffset = i * bCols;
      for (let j = 0; j < bCols; j++) {
        let sum = 0;
        let k = 0;
        const kBound = aCols - 8;

        if (aIsBinary) {
          for (; k <= kBound; k += 8) {
            const aik0 = transA ? aData[k * aRows + i] : aData[i * aCols + k];
            if (aik0 === 1) sum += bData[j * aCols + k];
            
            const aik1 = transA ? aData[(k + 1) * aRows + i] : aData[i * aCols + (k + 1)];
            if (aik1 === 1) sum += bData[j * aCols + (k + 1)];
            
            const aik2 = transA ? aData[(k + 2) * aRows + i] : aData[i * aCols + (k + 2)];
            if (aik2 === 1) sum += bData[j * aCols + (k + 2)];
            
            const aik3 = transA ? aData[(k + 3) * aRows + i] : aData[i * aCols + (k + 3)];
            if (aik3 === 1) sum += bData[j * aCols + (k + 3)];
            
            const aik4 = transA ? aData[(k + 4) * aRows + i] : aData[i * aCols + (k + 4)];
            if (aik4 === 1) sum += bData[j * aCols + (k + 4)];
            
            const aik5 = transA ? aData[(k + 5) * aRows + i] : aData[i * aCols + (k + 5)];
            if (aik5 === 1) sum += bData[j * aCols + (k + 5)];
            
            const aik6 = transA ? aData[(k + 6) * aRows + i] : aData[i * aCols + (k + 6)];
            if (aik6 === 1) sum += bData[j * aCols + (k + 6)];
            
            const aik7 = transA ? aData[(k + 7) * aRows + i] : aData[i * aCols + (k + 7)];
            if (aik7 === 1) sum += bData[j * aCols + (k + 7)];
          }
          for (; k < aCols; k++) {
            const aik = transA ? aData[k * aRows + i] : aData[i * aCols + k];
            if (aik === 1) sum += bData[j * aCols + k];
          }
        } else {
          // bIsBinary = true
          for (; k <= kBound; k += 8) {
            if (bData[j * aCols + k] === 1) {
              sum += transA ? aData[k * aRows + i] : aData[i * aCols + k];
            }
            if (bData[j * aCols + (k + 1)] === 1) {
              sum += transA ? aData[(k + 1) * aRows + i] : aData[i * aCols + (k + 1)];
            }
            if (bData[j * aCols + (k + 2)] === 1) {
              sum += transA ? aData[(k + 2) * aRows + i] : aData[i * aCols + (k + 2)];
            }
            if (bData[j * aCols + (k + 3)] === 1) {
              sum += transA ? aData[(k + 3) * aRows + i] : aData[i * aCols + (k + 3)];
            }
            if (bData[j * aCols + (k + 4)] === 1) {
              sum += transA ? aData[(k + 4) * aRows + i] : aData[i * aCols + (k + 4)];
            }
            if (bData[j * aCols + (k + 5)] === 1) {
              sum += transA ? aData[(k + 5) * aRows + i] : aData[i * aCols + (k + 5)];
            }
            if (bData[j * aCols + (k + 6)] === 1) {
              sum += transA ? aData[(k + 6) * aRows + i] : aData[i * aCols + (k + 6)];
            }
            if (bData[j * aCols + (k + 7)] === 1) {
              sum += transA ? aData[(k + 7) * aRows + i] : aData[i * aCols + (k + 7)];
            }
          }
          for (; k < aCols; k++) {
            if (bData[j * aCols + k] === 1) {
              sum += transA ? aData[k * aRows + i] : aData[i * aCols + k];
            }
          }
        }
        resultData[rOffset + j] = sum;
      }
    }
    }
  }

  const res = out ? out : Matrix.fromFlat(resultData, [aRows, bCols]);

  // RECORD FOR AUTO-DIFF
  // Asumsikan engine tersedia dari core
  if (engine && engine.tape) {
    engine.record([a, b], [res], (grad: Matrix) => {
      const gA = !transA
        ? dotProductAddOnly(grad, b, undefined, false, !transB)
        : dotProductAddOnly(b, grad, undefined, transB, true);
      const gB = !transB
        ? dotProductAddOnly(a, grad, undefined, !transA, false)
        : dotProductAddOnly(grad, a, undefined, true, transA);
      return [gA, gB];
    }, { saveInput: false, saveOutput: false, requireInputStability: true });
  }

  return res;
}
