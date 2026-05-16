import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { isNativeAvailable, hingeNativeInto } from "../math/rust_backend.js";

export default function HingeLoss(yTrue: Matrix, yPred: Matrix): [number, Matrix] {
  let loss: number;
  const dResult = mj.zeros([...yTrue._shape]);
  
  if (isNativeAvailable()) {
    loss = hingeNativeInto(yTrue._data, yPred._data, dResult._data)[0];
  } else {
    const n = yTrue._shape[0] * yTrue._shape[1];
      // yTrue is expected to be -1 or 1
      const lossMatrix = mj.map(mj.mul(yTrue, yPred), (v) => Math.max(0, 1 - v));
      loss = mj.mean(lossMatrix)._data[0];
      
      const dResJS = mj.map(mj.mul(yTrue, yPred), (v, i) => v < 1 ? -yTrue._data[i] / n : 0);
      
      return [loss, dResult];
  }
  
  return [loss, dResult];
}