import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { reluNative, isNativeAvailable } from "../math/rust_backend.js";
import { engine } from "../autodiff/engine.js";

export default function relu(a: Matrix): Matrix {
  let result: Matrix;
  let dResult: Matrix;

  if (isNativeAvailable()) {
    const res = new Float32Array(a._data.length);
    const grad = new Float32Array(a._data.length);

    reluNative(a._data, res, grad);

    result = Matrix.fromFlat(res, a._shape);
    dResult = Matrix.fromFlat(grad, a._shape);
  } else {
    result = mj.map(a, (val) => (val < 0 ? 0 : val));
    dResult = mj.map(a, (val) => (val < 0 ? 0 : 1));
  }

  engine.record(
    [a],
    [result],
    (grad: Matrix) => [mj.mul(grad, dResult)],
    { saveInput: false, saveOutput: false }
  );

  return result;
}