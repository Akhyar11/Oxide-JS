import Matrix from "../matrix/index.js";
import { engine } from "../autodiff/engine.js";

function computeDotDiv(a: Matrix): number {
  let value: number = 1;
  const data = a._data;
  const n = data.length;
  for (let i = 0; i < n; i++) {
    value /= data[i];
  }
  return value;
}

/**
 * Pembagian matrix a dengan dirinya sendiri mengebalikan number — DIOPTIMASI
 * @param a Matrix
 * @returns Number
 */
export default function dotDiv(a: Matrix): number {
  return computeDotDiv(a);
}

export function dotDivScalar(a: Matrix): Matrix {
  const data = a._data;
  const n = data.length;
  const value = computeDotDiv(a);
  const out = Matrix.fromFlat(new Float32Array([value]), [1, 1]);
  const aShape: [number, number] = [a._shape[0], a._shape[1]];

  engine.record([a], [out], (grad: Matrix) => {
    const upstream = grad._data[0];
    const gradA = Matrix.fromFlat(new Float32Array(n), aShape);
    for (let i = 0; i < n; i++) {
      gradA._data[i] = upstream * (-value / data[i]);
    }
    return [gradA];
  }, { saveInput: true, saveOutput: false });

  return out;
}
