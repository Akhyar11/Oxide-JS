import Matrix from "../matrix/index.js";
import { engine } from "../autodiff/engine.js";

/**
 * Menggabungkan dua buah matrix, pastikan matrix sudah di flatten atau berbentuk [1, n] — DIOPTIMASI
 * @param a Matrix
 * @param b Matrix
 * @returns Matrix
 */
export default function concat(a: Matrix, b: Matrix): Matrix {
  if (a._shape[0] !== 1 || b._shape[0] !== 1) {
    throw new Error(`pastikan matrix sudah di flatten atau berbentuk [1, n]`);
  }
  const result = new Float32Array(a._data.length + b._data.length);
  // Float32Array .set runs at native speed in V8
  result.set(a._data);
  result.set(b._data, a._data.length);
  const res = Matrix.fromFlat(result, [1, result.length]);

  // RECORD FOR AUTO-DIFF
  const tape = engine.tape;
  if (tape) {
    tape.record([a, b], [res], (grad: Matrix) => {
      // dL/da = grad[0 : a.len]
      const gradAData = grad._data.subarray(0, a._data.length);
      const gradA = Matrix.fromFlat(new Float32Array(gradAData), [...a._shape]);
      if (a.grad) a.grad.addInPlace(gradA);
      else a.grad = gradA;

      // dL/db = grad[a.len : ]
      const gradBData = grad._data.subarray(a._data.length);
      const gradB = Matrix.fromFlat(new Float32Array(gradBData), [...b._shape]);
      if (b.grad) b.grad.addInPlace(gradB);
      else b.grad = gradB;
    });
  }

  return res;
}
