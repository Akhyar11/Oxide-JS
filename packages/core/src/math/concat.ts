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
  const aLen = a._data.length;
  const aShape: [number, number] = [a._shape[0], a._shape[1]];
  const bShape: [number, number] = [b._shape[0], b._shape[1]];
  const result = new Float32Array(a._data.length + b._data.length);
  // Float32Array .set runs at native speed in V8
  result.set(a._data);
  result.set(b._data, a._data.length);
  const res = Matrix.fromFlat(result, [1, result.length]);

  if (engine.tape) {
    engine.record([a, b], [res], (grad: Matrix) => {
      const gradAData = grad._data.subarray(0, aLen);
      const gradA = Matrix.fromFlat(new Float32Array(gradAData), [...aShape]);
      const gradBData = grad._data.subarray(aLen);
      const gradB = Matrix.fromFlat(new Float32Array(gradBData), [...bShape]);
      return [gradA, gradB];
    }, { saveInput: false, saveOutput: false });
  }

  return res;
}
