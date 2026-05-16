import { MatrixCollection } from "../@types/type.js";
import Matrix from "../matrix/index.js";
import { isNativeAvailable, mulNative } from "./rust_backend.js";
import { engine } from "../autodiff/engine.js";
import mj from "./index.js";

/**
 * Perkalian element-wise a dan b — DIOPTIMASI
 */
export default function mul(a: MatrixCollection, b: MatrixCollection, out?: Matrix): Matrix {
  if (typeof a === "number") {
    const bm = b as Matrix;
    const resultData = out ? out._data : new Float32Array(bm._data.length);
    for (let i = 0; i < bm._data.length; i++) resultData[i] = a * bm._data[i];
    const res = out || Matrix.fromFlat(resultData, [bm._shape[0], bm._shape[1]]);
    engine.record([bm], [res], (grad: Matrix) => [mj.mul(grad, a)]);
    return res;
  }
  if (typeof b === "number") {
    const am = a as Matrix;
    const resultData = out ? out._data : new Float32Array(am._data.length);
    for (let i = 0; i < am._data.length; i++) resultData[i] = am._data[i] * b;
    const res = out || Matrix.fromFlat(resultData, [am._shape[0], am._shape[1]]);
    engine.record([am], [res], (grad: Matrix) => [mj.mul(grad, b)]);
    return res;
  }
  const am = a as Matrix;
  const bm = b as Matrix;
  if (am._shape[0] !== bm._shape[0] || am._shape[1] !== bm._shape[1]) {
    console.error(`mj.mul shape mismatch: am:[${am._shape}], bm:[${bm._shape}]`);
    throw new Error(`bentuk dari a harus sama dengan matrix ${am._shape} != ${bm._shape}`);
  }

  if (out) {
    if (out._shape[0] !== am._shape[0] || out._shape[1] !== am._shape[1]) {
      throw new Error(`Output matrix shape mismatch: expected [${am._shape[0]}x${am._shape[1]}], got [${out._shape[0]}x${out._shape[1]}]`);
    }
  }

  const resultData = out ? out._data : new Float32Array(am._data.length);

  // USE NATIVE IF AVAILABLE
  if (isNativeAvailable()) {
    mulNative(am._data, bm._data, resultData);
  } else {
    for (let i = 0; i < am._data.length; i++) resultData[i] = am._data[i] * bm._data[i];
  }

  const res = out || Matrix.fromFlat(resultData, [am._shape[0], am._shape[1]]);

  // RECORD FOR AUTO-DIFF
  engine.record(
    [am, bm],
    [res],
    (grad: Matrix) => [mj.mul(grad, bm), mj.mul(grad, am)],
    { saveInput: true, saveOutput: false }
  );

  return res;
}
