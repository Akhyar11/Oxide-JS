import Matrix from "../matrix/index.js";
import { isNativeAvailable, addBiasRowNative } from "./rust_backend.js";
import { engine } from "../autodiff/engine.js";
import mj from "./index.js";

/**
 * Menambahkan bias ke matrix secara in-place per-kolom (broadcasting axis=1).
 * 
 * Untuk Dense layer: dot shape [batch, units], bias shape [units, 1]
 * Operasi: dot[r][c] += bias[c] untuk semua r (batch)
 * 
 * Berbeda dengan addBias yang broadcast di axis=0 (bias shape [rows, 1]):
 * - addBias:    a[r][c] += bias[r]  → bias per-row
 * - addBiasRow: a[r][c] += bias[c]  → bias per-column (untuk Dense layer)
 * 
 * @param a Matrix [rows x cols]
 * @param bias Matrix [cols x 1]
 */
export default function addBiasRow(a: Matrix, bias: Matrix): void {
  const [rows, cols] = a._shape;
  const [bRows, bCols] = bias._shape;
  
  if (cols !== bRows || bCols !== 1) {
      throw new Error(`addBiasRow: Bias shape mismatch: expected [${cols},1], got [${bRows},${bCols}]`);
  }

  if (isNativeAvailable()) {
    addBiasRowNative(a._data, bias._data, rows, cols);
  } else {
    const data = a._data;
    const bData = bias._data;
    for (let r = 0; r < rows; r++) {
      const offset = r * cols;
      for (let c = 0; c < cols; c++) {
        data[offset + c] += bData[c];
      }
    }
  }

  if (engine.tape) {
    engine.record([a, bias], [a], (grad: Matrix) => {
      // Gradient untuk bias = sum gradient di axis 0 (batch dimension)
      // sumAxis(grad, 0) menghasilkan [1, cols], kita perlu reshape ke [cols, 1]
      const gradSum = mj.sumAxis(grad, 0);
      const gradBias = mj.reshape(gradSum, [cols, 1]);
      // `a` adalah input sekaligus output (in-place), grad sudah di tensor yang sama
      return [null, gradBias];
    }, { saveInput: false, saveOutput: false });
  }
}
