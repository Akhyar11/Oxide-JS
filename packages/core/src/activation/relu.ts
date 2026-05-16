import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { reluNative, isNativeAvailable } from "../math/rust_backend.js";
import { engine } from "../autodiff/engine.js";

export default function relu(a: Matrix): Matrix {
  let result: Matrix;

  if (isNativeAvailable()) {
    const res = new Float32Array(a._data.length);
    const grad = new Float32Array(a._data.length);
    reluNative(a._data, res, grad);
    result = Matrix.fromFlat(res, a._shape);
  } else {
    result = mj.map(a, (val) => (val < 0 ? 0 : val));
  }
  const dResult = mj.map(a, (val) => (val < 0 ? 0 : 1));

  const tape = engine.tape;
  if (tape) {
    tape.record([a], [result], (grad: Matrix) => {
      const gradA = mj.mul(grad, dResult);
      if (a.grad) a.grad.addInPlace(gradA);
      else a.grad = gradA;
    }, { saveInput: true, saveOutput: false });
  }

  return result;
}
