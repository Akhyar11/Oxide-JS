import Matrix from "../matrix/index.js";
import { engine } from "../autodiff/engine.js";

function computeNorm(a: Matrix): number {
  let sum = 0;
  const data = a._data;
  for (let i = 0; i < data.length; i++) {
    sum += data[i] ** 2;
  }
  return Math.sqrt(sum);
}

export default function norm(a: Matrix): number {
  return computeNorm(a);
}

export function normScalar(a: Matrix): Matrix {
  const value = computeNorm(a);
  const out = Matrix.fromFlat(new Float32Array([value]), [1, 1]);
  const aShape: [number, number] = [a._shape[0], a._shape[1]];

  engine.record([a], [out], (grad: Matrix) => {
    const upstream = grad._data[0];
    const gradA = Matrix.fromFlat(new Float32Array(a._data.length), aShape);
    if (value === 0) {
      return [gradA];
    }
    const scale = upstream / value;
    for (let i = 0; i < a._data.length; i++) {
      gradA._data[i] = scale * a._data[i];
    }
    return [gradA];
  }, { saveInput: true, saveOutput: false });

  return out;
}
