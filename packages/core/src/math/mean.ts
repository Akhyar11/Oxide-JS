import Matrix from "../matrix/index.js";
import { engine } from "../autodiff/engine.js";
import mj from "./index.js";

/**
 * Merata rata nilai matrix — DIOPTIMASI
 * @param a Matrix
 * @returns Matrix [1, 1]
 */
export default function mean(a: Matrix): Matrix {
  let value: number = 0;
  const data = a._data;
  const n = data.length;
  
  for (let i = 0; i < n; i++) {
    value += data[i];
  }

  const res = mj.matrix([[value / n]]);

  // RECORD FOR AUTO-DIFF
  const tape = engine.tape;
  if (tape) {
    tape.record([a], [res], (grad: Matrix) => {
      // dL/da = grad * (1/n) untuk setiap elemen
      const gradVal = grad._data[0] / n;
      const gradA = mj.ones(a._shape);
      for (let i = 0; i < gradA._data.length; i++) {
        gradA._data[i] *= gradVal;
      }
      
      if (a.grad) a.grad.addInPlace(gradA);
      else a.grad = gradA;
    });
  }

  return res;
}
