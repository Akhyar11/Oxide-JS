import mj from "../math";
import Matrix from "../matrix";
import { softmax } from "../activation";

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
  // 1. Terapkan softmax ke logits
  const [probs] = softmax(logits, false);

  // 2. Hitung loss.
  // Mendukung dua format target:
  // - one-hot [numClasses, 1]
  // - sparse class index [[classId]]
  const epsilon = 1e-15;
  const pData = probs._data;
  const gradData = new Float64Array(pData);
  const isSparseTarget = yTrue._shape[0] === 1 && yTrue._shape[1] === 1;

  if (isSparseTarget) {
    const classIndex = Math.floor(yTrue._data[0]);
    if (classIndex < 0 || classIndex >= logits._shape[0]) {
      throw new Error(`Class index '${classIndex}' di luar range logits (0 - ${logits._shape[0] - 1})`);
    }

    const n = logits._shape[0];
    const p = Math.max(epsilon, pData[classIndex]);
    gradData[classIndex] -= 1;
    for (let i = 0; i < gradData.length; i++) gradData[i] /= n;
    return [-(Math.log(p)) / n, Matrix.fromFlat(gradData, [logits._shape[0], logits._shape[1]])];
  }

  const n = yTrue._shape[0];
  let loss = 0;
  const yData = yTrue._data;
  for (let i = 0; i < yData.length; i++) {
    const y = yData[i];
    const p = Math.max(epsilon, pData[i]);
    loss += -(y * Math.log(p));
    gradData[i] = (pData[i] - y) / n;
  }
  loss /= n;
  return [loss, Matrix.fromFlat(gradData, [logits._shape[0], logits._shape[1]])];
}
