import mj from "../math";
import Matrix from "../matrix";

/**
 * Binary Cross-Entropy Loss
 * L = -1/N * Σ [y*log(ŷ) + (1-y)*log(1-ŷ)]
 * Gradient = (ŷ - y) / (N * ŷ * (1-ŷ))
 */
export function BinaryCrossEntropy(
  yTrue: Matrix,
  yPred: Matrix,
  dResult?: Matrix
): [number, Matrix] {
  const n = yTrue._shape[0] * yTrue._shape[1];
  const epsilon = 1e-15; // hindari log(0)
  const yData = yTrue._data;
  const pData = yPred._data;
  const grad = dResult || mj.zeros(yTrue._shape);
  const gradData = grad._data;

  let loss = 0;
  for (let i = 0; i < yData.length; i++) {
    const y = yData[i];
    const p = Math.max(epsilon, Math.min(1 - epsilon, pData[i]));
    loss += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
    gradData[i] = (p - y) / (n * p * (1 - p));
  }
  loss /= n;

  return [loss, grad];
}

/**
 * Categorical Cross-Entropy Loss (multi-class)
 * L = -1/N * Σ y*log(ŷ)
 * Gradient = -(y/ŷ) / N
 * Biasanya dipakai dengan Softmax di output layer
 */
export default function CategoricalCrossEntropy(
  yTrue: Matrix,
  yPred: Matrix,
  dResult?: Matrix
): [number, Matrix] {
  const n = yTrue._shape[0] * yTrue._shape[1];
  const epsilon = 1e-15;
  const yData = yTrue._data;
  const pData = yPred._data;
  const grad = dResult || mj.zeros(yTrue._shape);
  const gradData = grad._data;

  let loss = 0;
  for (let i = 0; i < yData.length; i++) {
    const y = yData[i];
    const p = Math.max(epsilon, pData[i]);
    loss += -(y * Math.log(p));
    gradData[i] = -y / (p * n);
  }
  loss /= n;

  return [loss, grad];
}
