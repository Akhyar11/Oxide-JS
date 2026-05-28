import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { engine } from "../autodiff/engine.js";
import { isNativeAvailable, lReluNative } from "../math/rust_backend.js";

export default function lRelu(a: Matrix): Matrix {
  const result = mj.zeros([...a._shape]);
  const dResult = mj.zeros([...a._shape]);

  if (isNativeAvailable()) {
    lReluNative(a._data, result._data, dResult._data);
  } else {
    for (let i = 0; i < a._data.length; i++) {
      const val = a._data[i];
      if (val < 0) {
        result._data[i] = val * 1e-5;
        dResult._data[i] = 1e-5;
      } else {
        result._data[i] = val;
        dResult._data[i] = 1.0;
      }
    }
  }

  if (engine.tape) {
    engine.record(
      [a],
      [result],
      (grad: Matrix) => [mj.mul(grad, dResult)],
      { saveInput: false, saveOutput: false }
    );
  }

  return result;
}