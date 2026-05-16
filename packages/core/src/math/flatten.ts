import Matrix from "../matrix/index.js";
import { engine } from "../autodiff/engine.js";
import mj from "./index.js";
import { MatrixShape } from "../@types/type.js";

/**
 * Meratakan matrix menjadi ukuran [n, 1] — DIOPTIMASI
 * Dengan Float32Array flat, flatten hanya ubah shape
 */
export default function flatten(a: Matrix): Matrix {
  const n = a._data.length;
  const originalShape = [...a._shape] as MatrixShape;
  const resultData = new Float32Array(a._data);
  const res = Matrix.fromFlat(resultData, [n, 1]);

  // RECORD FOR AUTO-DIFF
  engine.record([a], [res], (grad: Matrix) => [mj.reshape(grad, originalShape)], { saveInput: false, saveOutput: false });

  return res;
}
