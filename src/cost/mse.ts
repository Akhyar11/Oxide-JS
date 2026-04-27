import mj from "../math";
import Matrix from "../matrix";
import { isNativeAvailable, mseNative } from "../math/rust_backend";

export default function MeanSquerError(
  yTrue: Matrix,
  yPred: Matrix,
  dResult?: Matrix
): [number, Matrix] {
  const yTrueData = yTrue._data;
  const yPredData = yPred._data;
  const n = yTrueData.length;

  let loss = 0;
  if (isNativeAvailable()) {
    loss = mseNative(yTrueData, yPredData)[0];
  } else {
    for (let i = 0; i < n; i++) {
        const diff = yTrueData[i] - yPredData[i];
        loss += diff * diff;
    }
    loss /= n;
  }
  
  const grad = dResult || mj.zeros(yTrue._shape);
  const gradData = grad._data;
  const factor = 2 / n;
  for (let i = 0; i < n; i++) {
      gradData[i] = factor * (yPredData[i] - yTrueData[i]);
  }
  
  return [loss, grad];
}
