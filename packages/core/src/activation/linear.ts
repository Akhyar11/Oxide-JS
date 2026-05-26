import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { engine } from "../autodiff/engine.js";

export default function linear(a: Matrix): Matrix {
  const result = mj.map(a, (val) => val);

  if (engine.tape) {
    engine.record([a], [result], (grad: Matrix) => [grad], { saveInput: false, saveOutput: false });
  }

  return result;
}
