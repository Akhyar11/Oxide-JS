import mj from "../math";
import Matrix from "../matrix";
import { isNativeAvailable, sgdUpdateNative, sgdSparseUpdateNative, shouldUseNativeOptimizer } from "../math/rust_backend";

export default class SGD {
  private updateBuffer: Matrix | null = null;

  calculate(a: Matrix, alpha: number): Matrix {
    if (isNativeAvailable() && shouldUseNativeOptimizer(a._data.length)) {
      if (!this.updateBuffer || this.updateBuffer._data.length !== a._data.length) {
        this.updateBuffer = mj.zeros(a._shape);
      }
      sgdUpdateNative(a._data, this.updateBuffer._data, alpha);
      return this.updateBuffer;
    }
    return mj.mul(a, alpha);
  }

  updateSparse(target: Matrix, grad: Matrix, alpha: number, indices: Int32Array): void {
    if (isNativeAvailable() && shouldUseNativeOptimizer(grad._data.length)) {
      sgdSparseUpdateNative(
        indices,
        grad._data,
        target._data,
        alpha,
        target._shape[1],
        target._shape[0]
      );
      return;
    }

    const targetData = target._data;
    const gradData = grad._data;
    const vocabSize = target._shape[1];
    const embeddingDim = target._shape[0];
    const numUnique = indices.length;

    for (let j = 0; j < numUnique; j++) {
      const tokenIndex = indices[j];
      for (let i = 0; i < embeddingDim; i++) {
        const fullIdx = i * vocabSize + tokenIndex;
        const gradIdx = i * numUnique + j;
        targetData[fullIdx] -= alpha * gradData[gradIdx];
      }
    }
  }
}
