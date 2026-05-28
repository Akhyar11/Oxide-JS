import { MatrixCollection } from "../@types/type.js";
import Matrix from "../matrix/index.js";
import { divNative, isNativeAvailable } from "./rust_backend.js";
import { engine } from "../autodiff/engine.js";
import mj from "./index.js";

/**
 * Membagi matrix dengan a — DIOPTIMASI
 */
export default function div(a: MatrixCollection, b: MatrixCollection): Matrix {
  if (typeof a === "number") {
    const bm = b as Matrix;
    const result = new Float32Array(bm._data.length);
    for (let i = 0; i < bm._data.length; i++) {
        if (bm._data[i] === 0) throw new Error(`Pembagian dengan nol pada indeks [${i}]`);
        result[i] = a / bm._data[i];
    }
    const res = Matrix.fromFlat(result, [bm._shape[0], bm._shape[1]]);
    if (engine.tape) {
      engine.record([bm], [res], (grad: Matrix) => {
        const denomSquared = mj.mul(bm, bm);
        const scale = mj.div(a * -1, denomSquared);
        return [mj.mul(grad, scale)];
      }, { saveInput: true, saveOutput: false });
    }
    return res;
  }
  if (typeof b === "number") {
    if (b === 0) throw new Error("Pembagian dengan nol (scalar = 0) tidak diizinkan");
    const result = new Float32Array(a._data.length);
    for (let i = 0; i < a._data.length; i++) result[i] = a._data[i] / b;
    const res = Matrix.fromFlat(result, [a._shape[0], a._shape[1]]);
    if (engine.tape) {
      engine.record([a], [res], (grad: Matrix) => [mj.div(grad, b)], { saveInput: false, saveOutput: false });
    }
    return res;
  }
  const am = a as Matrix;
  const bm = b as Matrix;
  if (am._shape[0] !== bm._shape[0] || am._shape[1] !== bm._shape[1]) {
    throw new Error(`bentuk dari a harus sama dengan matrix ${am._shape} != ${bm._shape}`);
  }

  const resultData = new Float32Array(am._data.length);
  for (let i = 0; i < bm._data.length; i++) {
    if (bm._data[i] === 0) throw new Error(`Pembagian dengan nol pada indeks [${i}]`);
  }

  // USE NATIVE IF AVAILABLE
  if (isNativeAvailable()) {
    divNative(am._data, bm._data, resultData);
  } else {
    for (let i = 0; i < am._data.length; i++) { if (bm._data[i] === 0) throw new Error("Division by zero"); resultData[i] = am._data[i] === 0 ? 0 : am._data[i] / bm._data[i]; }
  }
  const res = Matrix.fromFlat(resultData, [am._shape[0], am._shape[1]]);

  // RECORD FOR AUTO-DIFF
  if (engine.tape) {
    engine.record([am, bm], [res], (grad: Matrix) => {
      const gradA = mj.div(grad, bm);
      const bSquared = mj.mul(bm, bm);
      const negA = mj.mul(am, -1);
      const gradBBase = mj.div(negA, bSquared);
      const gradB = mj.mul(grad, gradBBase);
      return [gradA, gradB];
    }, { saveInput: true, saveOutput: false });
  }

  return res;
}
