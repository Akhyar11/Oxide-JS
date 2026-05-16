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
  const aShape: [number, number] = [a._shape[0], a._shape[1]];
  
  for (let i = 0; i < n; i++) {
    value += data[i];
  }

  const res = mj.matrix([[value / n]]);

  // RECORD FOR AUTO-DIFF
  engine.record([a], [res], (grad: Matrix) => {
    const gradVal = grad._data[0] / n;
    const gradA = mj.ones(aShape);
    for (let i = 0; i < gradA._data.length; i++) {
      gradA._data[i] *= gradVal;
    }
    return [gradA];
  }, { saveInput: false, saveOutput: false });

  return res;
}
