import Matrix from "../matrix";
import { isNativeAvailable, addBiasNative } from "./rust_backend";

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
}
