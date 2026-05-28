import Matrix from "../matrix/index.js";
import { engine } from "../autodiff/engine.js";
import mj from "./index.js";
import { isNativeAvailable, logmNative } from "./rust_backend.js";

/**
 * Log natural (ln) => Matrix log(a) — DIOPTIMASI
 * @param a Matrix
 * @returns Matrix
 */
export default function logm(a: Matrix): Matrix {
  const resultData = new Float32Array(a._data.length);
  const data = a._data;

  if (isNativeAvailable()) {
    logmNative(data, resultData);
  } else {
    for (let i = 0; i < data.length; i++) {
      const val = data[i];
      resultData[i] = Math.log(val <= 0 ? 1e-15 : val);
    }
  }

  const res = Matrix.fromFlat(resultData, [a._shape[0], a._shape[1]]);

  // RECORD FOR AUTO-DIFF
  if (engine.tape) {
    engine.record([a], [res], (grad: Matrix) => [mj.div(grad, a)], { saveInput: true, saveOutput: false });
  }

  return res;
}
