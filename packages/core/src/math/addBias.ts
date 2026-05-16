import Matrix from "../matrix/index.js";
import { isNativeAvailable, addBiasNative } from "./rust_backend.js";
import { engine } from "../autodiff/engine.js";
import mj from "./index.js";

/**
 * Menambahkan bias ke matrix secara in-place (broadcasting)
 * @param a Matrix [rows x cols]
 * @param bias Matrix [rows x 1]
 */
export default function addBias(a: Matrix, bias: Matrix): void {
  const [rows, cols] = a._shape;
  const [bRows, bCols] = bias._shape;
  
  if (rows !== bRows || bCols !== 1) {
      throw new Error(`Bias shape mismatch: expected [${rows},1], got [${bRows},${bCols}]`);
  }

  if (isNativeAvailable()) {
    addBiasNative(a._data, bias._data, rows, cols);
  } else {
    const data = a._data;
    const bData = bias._data;
    for (let j = 0; j < cols; j++) {
      for (let i = 0; i < rows; i++) {
        data[i * cols + j] += bData[i];
      }
    }
  }

  engine.record([a, bias], [a], (grad: Matrix) => {
    const gradBias = mj.sumAxis(grad, 1);
    // `a` adalah input sekaligus output (in-place), grad untuk `a`
    // sudah tersimpan sebagai output grad di tensor yang sama.
    return [null, gradBias];
  }, { saveInput: false, saveOutput: false });
}
