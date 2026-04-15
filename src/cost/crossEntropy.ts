import mj from "../math";
import Matrix from "../matrix";

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

  let loss = 0;
  for (let i = 0; i < yTrue._shape[0]; i++) {
    for (let j = 0; j < yTrue._shape[1]; j++) {
      const y = yTrue._value[i][j];
      const p = Math.max(epsilon, Math.min(1 - epsilon, yPred._value[i][j]));
      loss += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
    }
  }
  loss /= n;

  // Gradient: (ŷ - y) / (N * ŷ * (1-ŷ))
  const dResult = mj.map(yPred, (p) => Math.max(epsilon, Math.min(1 - epsilon, p)));
  const gradient = mj.map(dResult, (p) => {
    return 0; // placeholder — dihitung manual di bawah
  });

  const gradArray: number[][] = [];
  for (let i = 0; i < yTrue._shape[0]; i++) {
    gradArray[i] = [];
    for (let j = 0; j < yTrue._shape[1]; j++) {
      const y = yTrue._value[i][j];
      const p = Math.max(epsilon, Math.min(1 - epsilon, yPred._value[i][j]));
      gradArray[i][j] = (p - y) / (n * p * (1 - p));
    }
  }

  return [loss, mj.matrix(gradArray)];
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
  const n = yTrue._shape[0] * yTrue._shape[1];
  const epsilon = 1e-15;

  let loss = 0;
  const gradArray: number[][] = [];

  for (let i = 0; i < yTrue._shape[0]; i++) {
    gradArray[i] = [];
    for (let j = 0; j < yTrue._shape[1]; j++) {
      const y = yTrue._value[i][j];
      const p = Math.max(epsilon, yPred._value[i][j]);
      loss += -(y * Math.log(p));
      gradArray[i][j] = -y / (p * n);
    }
  }
  loss /= n;

  return [loss, mj.matrix(gradArray)];
}
