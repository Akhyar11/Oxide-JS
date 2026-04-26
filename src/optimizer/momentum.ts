import { MatrixShape } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";
import { isNativeAvailable, momentumUpdateNative, momentumSparseUpdateNative, shouldUseNativeOptimizer } from "../math/rust_backend";

export default class Momentum {
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
      momentumUpdateNative(a._data, this.prevGradien._data, this.updateBuffer._data, alpha, this.beta);
      return this.updateBuffer;
    }

    // v_t = β * v_{t-1} + α * gradient  (element-wise)
    const betaVelocity = mj.mul(this.beta, this.prevGradien);
    const alphaGrad = mj.mul(alpha, a);
    const newGradien = mj.add(betaVelocity, alphaGrad);
    this.prevGradien = newGradien;
    return newGradien;
  }

  updateSparse(target: Matrix, grad: Matrix, alpha: number, indices: Int32Array): void {
    if (isNativeAvailable() && shouldUseNativeOptimizer(grad._data.length)) {
      momentumSparseUpdateNative(
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
        const v = this.beta * prevData[fullIdx] + alpha * g;
        prevData[fullIdx] = v;
        targetData[fullIdx] -= v;
      }
    }
  }
}
