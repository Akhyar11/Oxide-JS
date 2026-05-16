import Matrix from "../matrix/index.js";
import { engine } from "../autodiff/engine.js";

function computeDotSum(a: Matrix): number {
  let value: number = 0;
  const data = a._data;
  const n = data.length;
  for (let i = 0; i < n; i++) {
    value += data[i];
  }
  return value;
}

/**
 * Penjumlahan matrix a dengan dirinya sendiri mengebalikan number — DIOPTIMASI
 * @param a Matrix
 * @returns Number
 */
export default function dotSum(a: Matrix): number {
  return computeDotSum(a);
}

export function dotSumScalar(a: Matrix): Matrix {
  const out = Matrix.fromFlat(new Float32Array([computeDotSum(a)]), [1, 1]);
  const aShape: [number, number] = [a._shape[0], a._shape[1]];

  engine.record([a], [out], (grad: Matrix) => {
    const g = grad._data[0];
    const gradA = Matrix.fromFlat(new Float32Array(a._data.length), aShape);
    gradA._data.fill(g);
    return [gradA];
  }, { saveInput: false, saveOutput: false });

  return out;
}
