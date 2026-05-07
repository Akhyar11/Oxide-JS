import { MatrixShape } from "../@types/type.js";
import Matrix from "../matrix/index.js";
import { engine } from "../autodiff/engine.js";
import mj from "./index.js";

/**
 * Reshape matrix — DIOPTIMASI
 * Karena _data sudah flat, reshape hanya mengubah interpretasi shape
 */
export default function reshape(a: Matrix, shape: MatrixShape): Matrix {
  if (a._shape[0] * a._shape[1] !== shape[0] * shape[1]) {
    throw new Error(
      `panjang dari a tidak sama dengan bentuk yang diinginkan ${a._shape[0] * a._shape[1]}!=${shape[0] * shape[1]}`
    );
  }
  // Data sudah flat dan urut — hanya copy dan ubah shape
  const originalShape = [...a._shape] as MatrixShape;
  const resultData = new Float32Array(a._data);
  const res = Matrix.fromFlat(resultData, shape);

  // RECORD FOR AUTO-DIFF
  const tape = engine.tape;
  if (tape) {
    tape.record([a], [res], (grad: Matrix) => {
      // dL/da = reshape(grad) back to originalShape
      const gradA = mj.reshape(grad, originalShape);
      if (a.grad) a.grad.addInPlace(gradA);
      else a.grad = gradA;
    });
  }

  return res;
}
