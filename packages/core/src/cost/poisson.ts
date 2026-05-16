import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { isNativeAvailable, poissonNativeInto } from "../math/rust_backend.js";

export default function PoissonLoss(yTrue: Matrix, yPred: Matrix): [number, Matrix] {
  let loss: number;
  const dResult = mj.zeros([...yTrue._shape]);
  
  if (isNativeAvailable()) {
    loss = poissonNativeInto(yTrue._data, yPred._data, dResult._data)[0];
  } else {
    const n = yTrue._shape[0] * yTrue._shape[1];
    const lossMatrix = mj.map(yPred, (pred, i) => {
      const trueV = yTrue._data[i];
      return pred - trueV * Math.log(Math.max(pred, 1e-7));
    });
    loss = mj.mean(lossMatrix)._data[0];
  
    const dResultJS = mj.map(yPred, (pred, i) => {
      const trueV = yTrue._data[i];
      return (1 - trueV / Math.max(pred, 1e-7)) / n;
    });
    dResult._data.set(dResultJS._data);
  }
  
  return [loss, dResult];
}