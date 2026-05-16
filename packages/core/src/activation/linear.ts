import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { engine } from "../autodiff/engine.js";

export default function linear(a: Matrix): Matrix {
  const result = mj.map(a, (val) => val);

  const tape = engine.tape;
  if (tape) {
    tape.record([a], [result], (grad: Matrix) => {
      if (a.grad) a.grad.addInPlace(grad);
      else a.grad = grad;
    }, { saveInput: false, saveOutput: false });
  }

  return result;
}
