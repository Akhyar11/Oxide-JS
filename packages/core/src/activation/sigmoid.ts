import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { sigmoidNative, isNativeAvailable } from "../math/rust_backend.js";
import { engine } from "../autodiff/engine.js";

export default function sigmoid(a: Matrix): Matrix {
  let result: Matrix;

  if (isNativeAvailable()) {
    const res = new Float32Array(a._data.length);
    const grad = new Float32Array(a._data.length);
    sigmoidNative(a._data, res, grad);
    result = Matrix.fromFlat(res, a._shape);
  } else {
    result = mj.map(a, (val) => 1 / (1 + Math.exp(-val)));
  }
  const dResult = mj.mul(result, mj.sub(1, result));

  if (engine.tape) {
    engine.record([a], [result], (grad: Matrix) => [mj.mul(grad, dResult)], { saveInput: false, saveOutput: true });
  }

  return result;
}
