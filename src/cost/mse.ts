import mj from "../math";
import Matrix from "../matrix";
import { isNativeAvailable, mseNative } from "../math/rust_backend";

export default function MeanSquerError(
  yTrue: Matrix,
  yPred: Matrix
): [number, Matrix] {
  let result: number;
  if (isNativeAvailable()) {
    result = mseNative(yTrue._data, yPred._data)[0];
  } else {
    result = mj.mean(mj.map(mj.sub(yTrue, yPred), (v) => v ** 2));
  }
  
  const n = yTrue._shape[0] * yTrue._shape[1];
  const dResult = mj.mul(2 / n, mj.sub(yPred, yTrue));
  return [result, dResult];
}
