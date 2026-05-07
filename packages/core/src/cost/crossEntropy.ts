import Matrix from "../matrix/index.js";

/**
 * Binary Cross-Entropy Loss
 * L = -1/N * Σ [y*log(ŷ) + (1-y)*log(1-ŷ)]
 * Gradient = (ŷ - y) / (N * ŷ * (1-ŷ))
 */
export function BinaryCrossEntropy(
  yTrue: Matrix,
  yPred: Matrix
): [number, Matrix] {
  const n = yTrue._shape[0] * yTrue._shape[1];
  const epsilon = 1e-15; // hindari log(0)
  const yData = yTrue._data;
  const pData = yPred._data;
  const gradData = new Float32Array(yData.length);

  let loss = 0;
  for (let i = 0; i < yData.length; i++) {
    const y = yData[i];
    const p = Math.max(epsilon, Math.min(1 - epsilon, pData[i]));
    loss += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
    gradData[i] = (p - y) / (n * p * (1 - p));
  }
  loss /= n;

  return [loss, Matrix.fromFlat(gradData, [yTrue._shape[0], yTrue._shape[1]])];
}

/**
 * Categorical Cross-Entropy Loss (multi-class)
 * L = -1/N * Σ y*log(ŷ)
 * Gradient = -(y/ŷ) / N
 * Biasanya dipakai dengan Softmax di output layer
 */
export default function CategoricalCrossEntropy(
  yTrue: Matrix,
  yPred: Matrix
): [number, Matrix] {
  if (yTrue._shape[0] !== yPred._shape[0] || yTrue._shape[1] !== yPred._shape[1]) {
    throw new Error(
      `CategoricalCrossEntropy: shape mismatch yTrue=[${yTrue._shape[0]}, ${yTrue._shape[1]}] yPred=[${yPred._shape[0]}, ${yPred._shape[1]}]`
    );
  }

  const batchSize = yTrue._shape[1];
  if (!Number.isInteger(batchSize) || batchSize < 1) {
    throw new Error(`CategoricalCrossEntropy: batchSize harus >= 1, got ${batchSize}`);
  }

  const epsilon = 1e-15;
  const yData = yTrue._data;
  const pData = yPred._data;
  const gradData = new Float32Array(yData.length);

  let loss = 0;
  for (let i = 0; i < yData.length; i++) {
    const y = yData[i];
    const p = Math.max(epsilon, pData[i]);
    loss += -(y * Math.log(p));
    gradData[i] = -y / (p * batchSize);
  }
  loss /= batchSize;

  return [loss, Matrix.fromFlat(gradData, [yTrue._shape[0], yTrue._shape[1]])];
}
