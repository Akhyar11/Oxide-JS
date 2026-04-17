import { MatrixShape } from "../@types/type";
import Matrix from "../matrix";

/**
 * Memberikan nilai matrix random -1 sampai 1 — DIOPTIMASI
 */
export default function random(shape: MatrixShape): Matrix {
  const n = shape[0] * shape[1];
  const data = new Float32Array(n);
  // Gunakan scale 0.1 agar tidak meledak di model dalam (Xavier-like approximation)
  const scale = 0.1; 
  for (let i = 0; i < n; i++) {
    data[i] = (Math.random() * 2 - 1) * scale;
  }
  return Matrix.fromFlat(data, shape);
}
