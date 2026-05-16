import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { isNativeAvailable, logcoshNativeInto } from "../math/rust_backend.js";

export default function LogCoshLoss(yTrue: Matrix, yPred: Matrix): [number, Matrix] {
  let loss: number;
  const dResult = mj.zeros([...yTrue._shape]);
  
  if (isNativeAvailable()) {
    loss = logcoshNativeInto(yTrue._data, yPred._data, dResult._data)[0];
  } else {
    const diff = mj.sub(yPred, yTrue);
      const n = yTrue._shape[0] * yTrue._shape[1];
      
      const lossMatrix = mj.map(diff, (e) => Math.log(Math.cosh(e) + 1e-12));
      loss = mj.mean(lossMatrix)._data[0];
      
      const dResJS = mj.map(diff, (e) => Math.tanh(e) / n);
      
      return [loss, dResult];
  }
  
  return [loss, dResult];
}