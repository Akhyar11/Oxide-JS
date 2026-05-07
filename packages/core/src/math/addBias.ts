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
  
  if (rows !== bRows || (bCols !== 1 && bCols !== cols)) {
      throw new Error(`Bias shape mismatch: ${a._shape} vs ${bias._shape}`);
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

  // RECORD FOR AUTO-DIFF
  const tape = engine.tape;
  if (tape) {
    tape.record([a, bias], [a], (grad: Matrix) => {
      // dL/da = grad * 1 (sudah di a.grad)
      // dL/dbias = sum(grad over columns)
      const gradBias = mj.sumAxis(grad, 1);
      if (bias.grad) bias.grad.addInPlace(gradBias);
      else bias.grad = gradBias;
    });
  }
}
