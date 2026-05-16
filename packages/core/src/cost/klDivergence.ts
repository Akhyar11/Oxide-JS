import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { isNativeAvailable, kldivergenceNativeInto } from "../math/rust_backend.js";

export default function KLDivergence(yTrue: Matrix, yPred: Matrix): [number, Matrix] {
  let loss: number;
  const dResult = mj.zeros([...yTrue._shape]);
  
  if (isNativeAvailable()) {
    loss = kldivergenceNativeInto(yTrue._data, yPred._data, dResult._data)[0];
  } else {
    const n = yTrue._shape[0] * yTrue._shape[1];
    const lossMatrix = mj.map(yTrue, (v, i) => {
      const pred = Math.max(yPred._data[i], 1e-7);
      const trueV = Math.max(v, 1e-7);
      return trueV * Math.log(trueV / pred);
    });
    loss = mj.mean(lossMatrix)._data[0];
  
    const dResultJS = mj.map(yTrue, (v, i) => {
      const pred = Math.max(yPred._data[i], 1e-7);
      return -v / pred / n;
    });
    dResult._data.set(dResultJS._data);
  }
  
  return [loss, dResult];
}