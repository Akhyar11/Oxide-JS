import Matrix from "../matrix/index.js";
import { engine } from "../autodiff/engine.js";
import mj from "./index.js";

/**
 * Pangkat elemen-wise: result = a ^ n
 */
export default function pow(a: Matrix, n: number, out?: Matrix): Matrix {
  const resultData = out ? out._data : new Float32Array(a._data.length);
  const aData = a._data;

  for (let i = 0; i < aData.length; i++) {
    resultData[i] = Math.pow(aData[i], n);
  }

  const res = out || Matrix.fromFlat(resultData, [...a._shape]);

  // RECORD FOR AUTO-DIFF
  const tape = engine.tape;
  if (tape) {
    tape.record([a], [res], (grad: Matrix) => {
      // dL/da = grad * (n * a^(n-1))
      // Kita hitung n * a^(n-1)
      const gradA_base = mj.map(a, (val) => n * Math.pow(val, n - 1));
      const gradA = mj.mul(grad, gradA_base);
      
      if (a.grad) a.grad.addInPlace(gradA);
      else a.grad = gradA;
    });
  }

  return res;
}
