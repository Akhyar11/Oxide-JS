import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { isNativeAvailable, huberNativeInto } from "../math/rust_backend.js";

export default function HuberLoss(yTrue: Matrix, yPred: Matrix, delta = 1.0): [number, Matrix] {
  const diff = mj.sub(yPred, yTrue);
  const n = yTrue._shape[0] * yTrue._shape[1];
  
  let loss: number;
  const dResult = mj.zeros([...yTrue._shape]);
  
  if (isNativeAvailable()) {
    loss = huberNativeInto(yTrue._data, yPred._data, dResult._data, delta)[0];
  } else {
    const lossMatrix = mj.map(diff, (e) => {
      const absE = Math.abs(e);
      return absE <= delta ? 0.5 * e * e : delta * (absE - 0.5 * delta);
    });
    loss = mj.mean(lossMatrix)._data[0];
    
    const dResJS = mj.map(diff, (e) => {
      const absE = Math.abs(e);
      return (absE <= delta ? e : delta * Math.sign(e)) / n;
    });
    dResult._data.set(dResJS._data);
  }
  
  return [loss, dResult];
}