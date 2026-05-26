import Matrix from "../matrix/index.js";
import { engine } from "../autodiff/engine.js";
import mj from "./index.js";
import { isNativeAvailable, powNative } from "./rust_backend.js";

/**
 * Pangkat elemen-wise: result = a ^ n
 */
export default function pow(a: Matrix, n: number, out?: Matrix): Matrix {
  const resultData = out ? out._data : new Float32Array(a._data.length);
  const aData = a._data;

  if (isNativeAvailable()) {
    powNative(aData, n, resultData);
  } else {
    for (let i = 0; i < aData.length; i++) {
      resultData[i] = Math.pow(aData[i], n);
    }
  }

  const res = out || Matrix.fromFlat(resultData, [...a._shape]);

  // RECORD FOR AUTO-DIFF
  if (engine.tape) {
    engine.record([a], [res], (grad: Matrix) => {
      const gradABase = mj.map(a, (val) => n * Math.pow(val, n - 1));
      return [mj.mul(grad, gradABase)];
    }, { saveInput: true, saveOutput: false });
  }

  return res;
}
