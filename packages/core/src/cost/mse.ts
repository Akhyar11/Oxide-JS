import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { isNativeAvailable, mseNative } from "../math/rust_backend.js";

export default function MeanSquaredError(
  yTrue: Matrix,
  yPred: Matrix
): [number, Matrix] {
  let result: number;
  if (isNativeAvailable()) {
    result = mseNative(yTrue._data, yPred._data)[0];
  } else {
    result = mj.mean(mj.map(mj.sub(yTrue, yPred), (v) => v ** 2))._data[0];
  }
  
  const n = yTrue._shape[0] * yTrue._shape[1];
  const dResult = mj.mul(2 / n, mj.sub(yPred, yTrue));
  return [result, dResult];
}
