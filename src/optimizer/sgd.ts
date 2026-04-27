import mj from "../math";
import Matrix from "../matrix";
import { isNativeAvailable, sgdUpdateNative, sgdSparseUpdateNative, shouldUseNativeOptimizer } from "../math/rust_backend";
import { MemoryManager } from "../utils/memory";
import { MatrixShape } from "../@types/type";

export default class SGD {
  private updateBufferData: any = new Float32Array(0);
  private updateBuffer: Matrix;

  constructor() {
    this.updateBuffer = mj.matrix([]);
  }

  private ensureBuffers(size: number, shape: MatrixShape) {
    this.updateBufferData = MemoryManager.ensureCapacity(this.updateBufferData, size) as any;
    this.updateBuffer = Matrix.fromFlat(this.updateBufferData.subarray(0, size) as any, shape);
  }

  calculate(a: Matrix, alpha: number): Matrix {
    const size = a._data.length;
    this.ensureBuffers(size, a._shape);

    const gradData = a._data;
    const updateData = this.updateBuffer._data;

    if (isNativeAvailable() && shouldUseNativeOptimizer(size)) {
      sgdUpdateNative(gradData, updateData, alpha);
      return this.updateBuffer;
    }

    // fallback JS: update = gradient * alpha
    for (let i = 0; i < size; i++) {
        updateData[i] = gradData[i] * alpha;
    }
    return this.updateBuffer;
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

  dispose(): void {
    this.updateBufferData = new Float32Array(0);
    this.updateBuffer = mj.matrix([]);
  }
}
