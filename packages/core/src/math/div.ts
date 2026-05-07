import { MatrixCollection } from "../@types/type.js";
import Matrix from "../matrix/index.js";
import { divNative, isNativeAvailable, shouldUseNativeElementwise } from "./rust_backend.js";
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
    return Matrix.fromFlat(result, [bm._shape[0], bm._shape[1]]);
  }
  if (typeof b === "number") {
    if (b === 0) throw new Error("Pembagian dengan nol (scalar = 0) tidak diizinkan");
    const result = new Float32Array(a._data.length);
    for (let i = 0; i < a._data.length; i++) result[i] = a._data[i] / b;
    return Matrix.fromFlat(result, [a._shape[0], a._shape[1]]);
  }
  const am = a as Matrix;
  const bm = b as Matrix;
  if (am._shape[0] !== bm._shape[0] || am._shape[1] !== bm._shape[1]) {
    throw new Error(`bentuk dari a harus sama dengan matrix ${am._shape} != ${bm._shape}`);
  }

  const resultData = new Float32Array(am._data.length);

  // USE NATIVE IF AVAILABLE
  if (isNativeAvailable() && shouldUseNativeElementwise(am._data.length)) {
    divNative(am._data, bm._data, resultData);
  } else {
    for (let i = 0; i < am._data.length; i++) {
      if (bm._data[i] === 0) throw new Error(`Pembagian dengan nol pada indeks [${i}]`);
      resultData[i] = am._data[i] / bm._data[i];
    }
  }
  const res = Matrix.fromFlat(resultData, [am._shape[0], am._shape[1]]);

  // RECORD FOR AUTO-DIFF
  const tape = engine.tape;
  if (tape) {
    tape.record([am, bm], [res], (grad: Matrix) => {
      // dL/da = grad / b
      const gradA = mj.div(grad, bm);
      if (am.grad) am.grad.addInPlace(gradA);
      else am.grad = gradA;

      // dL/db = grad * (-a / b^2)
      const bSquared = mj.mul(bm, bm);
      const negA = mj.mul(am, -1);
      const gradB_base = mj.div(negA, bSquared);
      const gradB = mj.mul(grad, gradB_base);
      if (bm.grad) bm.grad.addInPlace(gradB);
      else bm.grad = gradB;
    });
  }

  return res;
}
