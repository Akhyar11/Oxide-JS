import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { engine } from "../autodiff/engine.js";
import { isNativeAvailable, thresholdNative } from "../math/rust_backend.js";

export default function threshold(a: Matrix, thresholdValue: number = 0.0, leak: number = 0.0): Matrix {
  const result = mj.zeros([...a._shape]);
  const dResult = mj.zeros([...a._shape]);

  if (isNativeAvailable()) {
    thresholdNative(a._data, thresholdValue, leak, result._data, dResult._data);
  } else {
    for (let i = 0; i < a._data.length; i++) {
      result._data[i] = a._data[i] > thresholdValue ? 1.0 : 0.0;
      dResult._data[i] = a._data[i] > thresholdValue ? leak : 1.0;
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
