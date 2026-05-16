import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { isNativeAvailable, sgdUpdateNative } from "../math/rust_backend.js";

export default class SGD {
  private updateBuffer: Matrix | null = null;

  calculate(a: Matrix, alpha: number): Matrix {
    if (isNativeAvailable()) {
      if (!this.updateBuffer || this.updateBuffer._data.length !== a._data.length) {
        this.updateBuffer = mj.zeros(a._shape);
      }
      sgdUpdateNative(a._data, this.updateBuffer._data, alpha);
      return this.updateBuffer;
    }
    return mj.mul(a, alpha);
  }

  updateSparse(target: Matrix, grad: Matrix, alpha: number, indices: Int32Array): void {
    const targetData = target._data;
    const gradData = grad._data;
    const cols = target._shape[1];
    const rows = target._shape[0];
    const numUnique = indices.length;

    for (let j = 0; j < numUnique; j++) {
      const tokenIndex = indices[j];
      for (let i = 0; i < rows; i++) {
        const fullIdx = i * cols + tokenIndex;
        const gradIdx = i * numUnique + j;
        targetData[fullIdx] -= alpha * gradData[gradIdx];
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
