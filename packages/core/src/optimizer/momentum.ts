import { MatrixShape } from "../@types/type.js";
import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { isNativeAvailable, momentumUpdateNative } from "../math/rust_backend.js";

export default class Momentum {
  prevGradien: Matrix;
  beta = 0.9;
  private updateBuffer: Matrix | null = null;

  constructor(shape: MatrixShape) {
    this.prevGradien = mj.zeros(shape);
  }

  calculate(a: Matrix, alpha: number) {
    if (isNativeAvailable()) {
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
    const targetData = target._data;
    const gradData = grad._data;
    const prevData = this.prevGradien._data;
    const cols = target._shape[1];
    const rows = target._shape[0];
    const numUnique = indices.length;

    for (let j = 0; j < numUnique; j++) {
      const tokenIndex = indices[j];
      for (let i = 0; i < rows; i++) {
        const fullIdx = i * cols + tokenIndex;
        const gradIdx = i * numUnique + j;

        const g = gradData[gradIdx];
        const v = this.beta * prevData[fullIdx] + alpha * g;
        prevData[fullIdx] = v;
        targetData[fullIdx] -= v;
      }
    }
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
