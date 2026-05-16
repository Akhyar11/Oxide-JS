import Matrix from "../matrix/index.js";
import { engine } from "../autodiff/engine.js";
import mj from "./index.js";

/**
 * Transposisi matrix [i, j] => [j, i] — DIOPTIMASI
 * @param a Matrix asalnya
 * @param out Matrix penampung hasil (opsional)
 */
export default function transpose(a: Matrix, out?: Matrix): Matrix {
  const [rows, cols] = a._shape;
  
  if (out) {
    if (out._shape[0] !== cols || out._shape[1] !== rows) {
      throw new Error("Output matrix shape mismatch for transpose");
    }
  }

  const resultData = out ? out._data : new Float32Array(rows * cols);
  const aData = a._data;

  // Optimasi: Membaca secara sequential dari 'a' untuk cache friendliness
  for (let i = 0; i < rows; i++) {
    const iOffset = i * cols;
    for (let j = 0; j < cols; j++) {
      resultData[j * rows + i] = aData[iOffset + j];
    }
  }

  const res = out || Matrix.fromFlat(resultData, [cols, rows]);

  // RECORD FOR AUTO-DIFF
  engine.record([a], [res], (grad: Matrix) => [mj.transpose(grad)], { saveInput: false, saveOutput: false });

  return res;
}
