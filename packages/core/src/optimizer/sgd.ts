import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { isNativeAvailable, sgdUpdateNative, sgdSparseUpdateNative, shouldUseNativeOptimizer, embeddingSgdBackwardUpdateNative } from "../math/rust_backend.js";

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

  /**
   * Fused embedding backward + SGD update via single NAPI call.
   * @returns true when the fused native path ran, false to signal fallback.
   */
  updateEmbeddingSparseNative(
    target: Matrix,
    indices: Int32Array,
    errData: Float32Array,
    alpha: number,
    embeddingDim: number,
    vocabSize: number,
    padTokenId: number | null
  ): boolean {
    if (!isNativeAvailable()) return false;
    return embeddingSgdBackwardUpdateNative(
      indices,
      errData,
      target._data,
      alpha,
      vocabSize,
      embeddingDim,
      padTokenId
    );
  }

  /**
   * Menerapkan gradien ke matrix target secara in-place.
   */
  apply(target: Matrix, alpha: number): void {
    if (!target.grad) return;
    const update = this.calculate(target.grad, alpha);
    target.subInPlace(update);
    target.grad = null;
  }
}

