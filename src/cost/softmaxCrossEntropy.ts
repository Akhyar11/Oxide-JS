import mj from "../math";
import Matrix from "../matrix";
import { softmaxOnly } from "../activation";

/**
 * Softmax Cross-Entropy Loss (Combined)
 * 
 * Menggabungkan Softmax activation + Categorical Cross-Entropy loss
 * menjadi satu fungsi. Ini PENTING karena gradient gabungannya:
 *   dL/dz = ŷ - y   (softmax output - target)
 * 
 * Jauh lebih stabil dan sederhana daripada menghitung terpisah:
 *   dCE/dŷ × dSoftmax/dz  ← numerically unstable!
 * 
 * PEMAKAIAN: Gunakan activation='linear' di Dense layer output,
 *            lalu set loss='softmaxCrossEntropy'
 *            Softmax akan diterapkan di sini (bukan di activation).
 * 
 * @param yTrue - One-hot target [numClasses, 1]
 * @param logits - Raw output dari Dense (SEBELUM softmax) [numClasses, 1]
 * @returns [loss, gradient]
 */
export default function SoftmaxCrossEntropy(
  yTrue: Matrix,
  logits: Matrix,
  dResult?: Matrix,
  tempProbs?: Matrix
): [number, Matrix] {
  // logits shape: [numClasses, batchSize]
  // yTrue shape: [1, batchSize] (sparse) or [numClasses, batchSize] (one-hot)
  
  const [numClasses, batchSize] = logits._shape;
  const grad = dResult || mj.zeros(logits._shape);
  
  // Hitung softmax langsung ke dResult buffer karena grad = probs - y
  const probs = softmaxOnly(logits, false, grad);
  const pData = probs._data;
  
  const epsilon = 1e-15;
  const isSparseTarget = yTrue._shape[0] === 1;

  let totalLoss = 0;

  if (isSparseTarget) {
    const yData = yTrue._data;
    for (let b = 0; b < batchSize; b++) {
      const classIndex = Math.floor(yData[b]);
      if (classIndex < 0 || classIndex >= numClasses) {
        throw new Error(`Class index '${classIndex}' at batch ${b} di luar range logits (0 - ${numClasses - 1})`);
      }

      const p = Math.max(epsilon, pData[classIndex * batchSize + b]);
      totalLoss -= Math.log(p);
      
      // Gradient: probs - y (y is 1 for the target class)
      pData[classIndex * batchSize + b] -= 1;
    }
  } else {
    const yData = yTrue._data;
    for (let i = 0; i < yData.length; i++) {
      const y = yData[i];
      if (y === 0) continue;
      
      const p = Math.max(epsilon, pData[i]);
      totalLoss -= y * Math.log(p);
      pData[i] -= y;
    }
  }

  return [totalLoss / batchSize, grad];
}
