import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { tanhNative, isNativeAvailable } from "../math/rust_backend.js";
import { engine } from "../autodiff/engine.js";

export default function tanh(a: Matrix): Matrix {
  let result: Matrix;

  if (isNativeAvailable()) {
    const res = new Float32Array(a._data.length);
    const grad = new Float32Array(a._data.length);
    tanhNative(a._data, res, grad);
    result = Matrix.fromFlat(res, a._shape);
  } else {
    result = mj.map(a, (val) => Math.tanh(val));
  }
  const dResult = mj.sub(1, mj.mul(result, result));

  const tape = engine.tape;
  if (tape) {
    tape.record([a], [result], (grad: Matrix) => {
      const gradA = mj.mul(grad, dResult);
      if (a.grad) a.grad.addInPlace(gradA);
      else a.grad = gradA;
    }, { saveInput: false, saveOutput: true });
  }

  return result;
}
