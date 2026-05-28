import Matrix from "../matrix/index.js";
import { engine } from "../autodiff/engine.js";
import mj from "./index.js";
import { isNativeAvailable, expmNative } from "./rust_backend.js";

/**
 * Matrix a => Matrix exp(a) — DIOPTIMASI
 * @param a Matrix
 * @returns Matrix
 */
export default function expm(a: Matrix): Matrix {
  const resultData = new Float32Array(a._data.length);
  const data = a._data;

  if (isNativeAvailable()) {
    expmNative(data, resultData);
  } else {
    for (let i = 0; i < data.length; i++) {
      resultData[i] = Math.exp(data[i]);
    }
  }

  const res = Matrix.fromFlat(resultData, [a._shape[0], a._shape[1]]);

  // RECORD FOR AUTO-DIFF
  if (engine.tape) {
    engine.record([a], [res], (grad: Matrix) => [mj.mul(grad, res)], { saveInput: false, saveOutput: true });
  }

  return res;
}
