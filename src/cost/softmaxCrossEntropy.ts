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

  // 2. Hitung loss: -sum(y * log(ŷ)) / N
  const epsilon = 1e-15;
  const n = yTrue._shape[0];
  let loss = 0;

  for (let i = 0; i < yTrue._shape[0]; i++) {
    for (let j = 0; j < yTrue._shape[1]; j++) {
      const y = yTrue._value[i][j];
      const p = Math.max(epsilon, probs._value[i][j]);
      loss += -(y * Math.log(p));
    }
  }
  loss /= n;

  // 3. Gradient gabungan: (ŷ - y) / N
  //    Ini adalah keajaiban softmax+CE — gradient-nya sangat sederhana!
  const gradArray: number[][] = [];
  for (let i = 0; i < yTrue._shape[0]; i++) {
    gradArray[i] = [];
    for (let j = 0; j < yTrue._shape[1]; j++) {
      gradArray[i][j] = (probs._value[i][j] - yTrue._value[i][j]) / n;
    }
  }

  return [loss, mj.matrix(gradArray)];
}
