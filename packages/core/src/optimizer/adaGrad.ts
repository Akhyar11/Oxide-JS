import { MatrixShape } from "../@types/type.js";
import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { isNativeAvailable, adagradUpdateNative, adagradSparseUpdateNative, shouldUseNativeOptimizer, embeddingAdagradBackwardUpdateNative } from "../math/rust_backend.js";

export default class AdaGrad {
  shape: MatrixShape;
  sumGradien: Matrix;
  epsilon: number = 0.1;
  private updateBuffer: Matrix;
  constructor(shape: MatrixShape, epsilon: number) {
    this.shape = shape;
    this.sumGradien = mj.zeros(this.shape);
    this.updateBuffer = mj.zeros(this.shape);
    this.epsilon = epsilon;
  }

  calculate(a: Matrix, alpha: number) {
    const gradData = a._data;
    const sumData = this.sumGradien._data;
    const updateData = this.updateBuffer._data;

    if (isNativeAvailable() && shouldUseNativeOptimizer(gradData.length)) {
      adagradUpdateNative(gradData, sumData, updateData, alpha, this.epsilon);
      return this.updateBuffer;
    }

    for (let i = 0; i < gradData.length; i++) {
      const grad = gradData[i];
      const accumulated = sumData[i] + grad * grad;
      sumData[i] = accumulated;
      updateData[i] = alpha * grad / Math.sqrt(accumulated + this.epsilon);
    }

    return this.updateBuffer;
  }

  updateSparse(target: Matrix, grad: Matrix, alpha: number, indices: Int32Array): void {
    if (isNativeAvailable() && shouldUseNativeOptimizer(grad._data.length)) {
      adagradSparseUpdateNative(
        indices,
        grad._data,
        target._data,
        this.sumGradien._data,
        alpha,
        this.epsilon,
        target._shape[1],
        target._shape[0]
      );
      return;
    }

    const targetData = target._data;
    const gradData = grad._data;
    const sumData = this.sumGradien._data;

    const vocabSize = target._shape[1];
    const embeddingDim = target._shape[0];
    const numUnique = indices.length;

    for (let j = 0; j < numUnique; j++) {
      const tokenIndex = indices[j];
      for (let i = 0; i < embeddingDim; i++) {
        const fullIdx = i * vocabSize + tokenIndex;
        const gradIdx = i * numUnique + j;

        const g = gradData[gradIdx];
        const accumulated = sumData[fullIdx] + g * g;
        sumData[fullIdx] = accumulated;
        targetData[fullIdx] -= alpha * g / Math.sqrt(accumulated + this.epsilon);
      }
    }
  }

  /**
   * Fused embedding backward + AdaGrad update via single NAPI call.
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
    return embeddingAdagradBackwardUpdateNative(
      indices,
      errData,
      target._data,
      this.sumGradien._data,
      alpha,
      this.epsilon,
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

