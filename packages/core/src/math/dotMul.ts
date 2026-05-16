import Matrix from "../matrix/index.js";
import { engine } from "../autodiff/engine.js";

function computeDotMul(a: Matrix): number {
  let value: number = 1;
  const data = a._data;
  const n = data.length;
  for (let i = 0; i < n; i++) {
    value *= data[i];
  }
  return value;
}

/**
 * Perkalian matrix a dengan dirinya sendiri mengebalikan number — DIOPTIMASI
 * @param a Matrix
 * @returns Number
 */
export default function dotMul(a: Matrix): number {
  return computeDotMul(a);
}

export function dotMulScalar(a: Matrix): Matrix {
  const data = a._data;
  const n = data.length;
  const value = computeDotMul(a);
  const out = Matrix.fromFlat(new Float32Array([value]), [1, 1]);
  const aShape: [number, number] = [a._shape[0], a._shape[1]];

  engine.record([a], [out], (grad: Matrix) => {
    const upstream = grad._data[0];
    const gradA = Matrix.fromFlat(new Float32Array(n), aShape);
    const prefix = new Float32Array(n + 1);
    const suffix = new Float32Array(n + 1);

    prefix[0] = 1;
    for (let i = 0; i < n; i++) {
      prefix[i + 1] = prefix[i] * data[i];
    }
    suffix[n] = 1;
    for (let i = n - 1; i >= 0; i--) {
      suffix[i] = suffix[i + 1] * data[i];
    }

    for (let i = 0; i < n; i++) {
      gradA._data[i] = upstream * prefix[i] * suffix[i + 1];
    }
    return [gradA];
  }, { saveInput: true, saveOutput: false });

  return out;
}
