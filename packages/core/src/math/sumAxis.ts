import Matrix from "../matrix/index.js";
import { isNativeAvailable, shouldUseNativeElementwise, sumAxisNative } from "./rust_backend.js";
import { engine } from "../autodiff/engine.js";
import mj from "./index.js";

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

  if (out) {
    if (out._shape[0] !== outShape[0] || out._shape[1] !== outShape[1]) {
      throw new Error(`sumAxis output shape mismatch: expected [${outShape[0]}x${outShape[1]}], got [${out._shape[0]}x${out._shape[1]}]`);
    }
  }

  if (isNativeAvailable() && shouldUseNativeElementwise(rows * cols)) {
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

  // RECORD FOR AUTO-DIFF
  const tape = engine.tape;
  if (tape) {
    tape.record([a], [result], (grad: Matrix) => {
      // dL/da = broadcast(grad) ke shape original a
      const gradA = Matrix.fromFlat(new Float32Array(rows * cols), [rows, cols]);
      const gaData = gradA._data;
      const gData = grad._data;
      
      if (axis === 1) {
        for (let i = 0; i < rows; i++) {
          const gVal = gData[i];
          for (let j = 0; j < cols; j++) gaData[i * cols + j] = gVal;
        }
      } else {
        for (let i = 0; i < rows; i++) {
          for (let j = 0; j < cols; j++) gaData[i * cols + j] = gData[j];
        }
      }
      
      if (a.grad) a.grad.addInPlace(gradA);
      else a.grad = gradA;
    });
  }

  return result;
}
