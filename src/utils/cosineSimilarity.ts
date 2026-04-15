import mj from "../math";
import Matrix from "../matrix";

export default function cosineSimilarity(a: Matrix, b: Matrix): number {
  const flatA = mj.flatten(a);
  const flatB = mj.flatten(b);
  const magnitudeA = mj.norm(flatA);
  const magnitudeB = mj.norm(flatB);
  if (magnitudeA === 0 || magnitudeB === 0) {
    throw new Error("cosineSimilarity: salah satu vector adalah zero vector (magnitude=0)");
  }
  const dotProduct = mj.dotProduct(flatA, mj.transpose(flatB));
  return dotProduct._value[0][0] / (magnitudeA * magnitudeB);
}
