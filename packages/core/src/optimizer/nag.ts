import { MatrixShape } from "../@types/type.js";
import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { isNativeAvailable, nagUpdateNative, nagSparseUpdateNative, shouldUseNativeOptimizer, embeddingNagBackwardUpdateNative } from "../math/rust_backend.js";

export default class NAG {
  prevGradien: Matrix;
  beta = 0.9;
  private updateBuffer: Matrix | null = null;

  constructor(shape: MatrixShape) {
    this.prevGradien = mj.zeros(shape);
  }

  calculate(a: Matrix, alpha: number) {
    if (isNativeAvailable() && shouldUseNativeOptimizer(a._data.length)) {
      if (!this.updateBuffer || this.updateBuffer._data.length !== a._data.length) {
        this.updateBuffer = mj.zeros(a._shape);
      }
      nagUpdateNative(a._data, this.prevGradien._data, this.updateBuffer._data, alpha, this.beta);
      return this.updateBuffer;
    }

    // NAG: v_t = β * v_{t-1} + α * (g - β * v_{t-1})  (element-wise)
    const betaVelocity = mj.mul(this.beta, this.prevGradien);
    const wUpdate = mj.sub(a, betaVelocity);
    const newGradien = mj.add(betaVelocity, mj.mul(alpha, wUpdate));
    this.prevGradien = newGradien;
    return newGradien;
  }

  updateSparse(target: Matrix, grad: Matrix, alpha: number, indices: Int32Array): void {
    if (isNativeAvailable() && shouldUseNativeOptimizer(grad._data.length)) {
      nagSparseUpdateNative(
        indices,
        grad._data,
        target._data,
        this.prevGradien._data,
        alpha,
        this.beta,
        target._shape[1],
        target._shape[0]
      );
      return;
    }

    const targetData = target._data;
    const gradData = grad._data;
    const prevData = this.prevGradien._data;

    const vocabSize = target._shape[1];
    const embeddingDim = target._shape[0];
    const numUnique = indices.length;

    for (let j = 0; j < numUnique; j++) {
      const tokenIndex = indices[j];
      for (let i = 0; i < embeddingDim; i++) {
        const fullIdx = i * vocabSize + tokenIndex;
        const gradIdx = i * numUnique + j;

        const g = gradData[gradIdx];
        const vOld = prevData[fullIdx];
        const vNew = this.beta * vOld + alpha * (g - this.beta * vOld);
        prevData[fullIdx] = vNew;
        targetData[fullIdx] -= vNew;
      }
    }
  }

  /**
   * Fused embedding backward + NAG update via single NAPI call.
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
    return embeddingNagBackwardUpdateNative(
      indices,
      errData,
      target._data,
      this.prevGradien._data,
      alpha,
      this.beta,
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

