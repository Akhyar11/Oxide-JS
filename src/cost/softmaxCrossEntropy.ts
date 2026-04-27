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
  logits: Matrix
): [number, Matrix] {
  // logits shape: [numClasses, batchSize]
  // yTrue shape: [1, batchSize] (sparse) or [numClasses, batchSize] (one-hot)
  
  const [numClasses, batchSize] = logits._shape;
  const probs = softmaxOnly(logits, false);
  const epsilon = 1e-15;
  const pData = probs._data;
  const gradData = new Float32Array(pData);
  const isSparseTarget = yTrue._shape[0] === 1;

  let totalLoss = 0;

  if (isSparseTarget) {
    // Sparse case: yTrue is [1, batchSize]
    for (let b = 0; b < batchSize; b++) {
      const classIndex = Math.floor(yTrue._data[b]);
      if (classIndex < 0 || classIndex >= numClasses) {
        throw new Error(`Class index '${classIndex}' at batch ${b} di luar range logits (0 - ${numClasses - 1})`);
      }

      const p = Math.max(epsilon, pData[classIndex * batchSize + b]);
      totalLoss -= Math.log(p);
      
      // Gradient: probs - y (y is 1 for the target class)
      gradData[classIndex * batchSize + b] -= 1;
    }
  } else {
    // One-hot case: yTrue is [numClasses, batchSize]
    const yData = yTrue._data;
    for (let i = 0; i < yData.length; i++) {
      const y = yData[i];
      if (y === 0) continue;
      
      const p = Math.max(epsilon, pData[i]);
      totalLoss -= y * Math.log(p);
      gradData[i] -= y;
    }
  }

  // Rata-ratakan loss dan gradient berdasarkan batch size
  const finalGrad = Matrix.fromFlat(gradData, [numClasses, batchSize]);
  for (let i = 0; i < gradData.length; i++) {
    gradData[i] /= batchSize;
  }
  return [totalLoss / batchSize, finalGrad];
}
