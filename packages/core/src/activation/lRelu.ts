import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { engine } from "../autodiff/engine.js";

export default function lRelu(a: Matrix): Matrix {
  const result = mj.map(a, (val) => (val < 0 ? val * 1e-5 : val));

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
