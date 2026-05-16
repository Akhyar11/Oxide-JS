import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { engine } from "../autodiff/engine.js";

export default function lRelu(a: Matrix): Matrix {
  const result = mj.map(a, (val) => (val < 0 ? val * 1e-5 : val));
  const dResult = mj.map(a, (val) => (val < 0 ? 1e-5 : 1));

  engine.record(
    [a],
    [result],
    (grad: Matrix) => [mj.mul(grad, dResult)],
    { saveInput: false, saveOutput: false }
  );

  return result;
}