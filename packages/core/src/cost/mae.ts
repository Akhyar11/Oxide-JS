import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { isNativeAvailable, maeNativeInto } from "../math/rust_backend.js";

export default function MeanAbsoluteError(yTrue: Matrix, yPred: Matrix): [number, Matrix] {
  let loss: number;
  const dResult = mj.zeros([...yTrue._shape]);
  
  if (isNativeAvailable()) {
    loss = maeNativeInto(yTrue._data, yPred._data, dResult._data)[0];
  } else {
    const diff = mj.sub(yPred, yTrue);
      const n = yTrue._shape[0] * yTrue._shape[1];
      loss = mj.mean(mj.absm(diff))._data[0];
      const dResJS = mj.mul(1 / n, mj.map(diff, (v) => v > 0 ? 1 : (v < 0 ? -1 : 0)));
      return [loss, dResult];
  }
  
  return [loss, dResult];
}