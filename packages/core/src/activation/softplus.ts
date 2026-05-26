import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { engine } from "../autodiff/engine.js";
import { isNativeAvailable, softplusNative } from "../math/rust_backend.js";

export default function softplus(a: Matrix): Matrix {
  const result = mj.zeros([...a._shape]);
  const dResult = mj.zeros([...a._shape]);

  if (isNativeAvailable()) {
    softplusNative(a._data, result._data, dResult._data);
  } else {
    throw new Error("softplus JS fallback not implemented yet. Please run with native backend.");
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
