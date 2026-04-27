import { MatrixShape } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";
import { isNativeAvailable, nagUpdateNative, nagSparseUpdateNative, shouldUseNativeOptimizer } from "../math/rust_backend";
import { MemoryManager } from "../utils/memory";

export default class NAG {
  private prevGradienData: any = new Float32Array(0);
  private updateBufferData: any = new Float32Array(0);
  private prevGradien: Matrix;
  private updateBuffer: Matrix;
  beta = 0.9;

  constructor(shape: MatrixShape) {
    this.prevGradien = mj.matrix([]);
    this.updateBuffer = mj.matrix([]);
  }

  private ensureBuffers(size: number, shape: MatrixShape) {
    this.prevGradienData = MemoryManager.ensureCapacity(this.prevGradienData, size) as any;
    this.updateBufferData = MemoryManager.ensureCapacity(this.updateBufferData, size) as any;
    
    this.prevGradien = Matrix.fromFlat(this.prevGradienData.subarray(0, size) as any, shape);
    this.updateBuffer = Matrix.fromFlat(this.updateBufferData.subarray(0, size) as any, shape);
  }

  calculate(a: Matrix, alpha: number) {
    const size = a._data.length;
    this.ensureBuffers(size, a._shape);

    const gradData = a._data;
    const prevData = this.prevGradien._data;
    const updateData = this.updateBuffer._data;

    if (isNativeAvailable() && shouldUseNativeOptimizer(size)) {
      nagUpdateNative(gradData, prevData, updateData, alpha, this.beta);
      return this.updateBuffer;
    }

    // fallback JS: v_t = β * v_{t-1} + alpha * (g - β * v_{t-1})
    for (let i = 0; i < size; i++) {
        const vOld = prevData[i];
        const vNew = this.beta * vOld + alpha * (gradData[i] - this.beta * vOld);
        prevData[i] = vNew;
        updateData[i] = vNew;
    }

    return this.updateBuffer;
  }

  updateSparse(target: Matrix, grad: Matrix, alpha: number, indices: Int32Array): void {
    const vocabSize = target._shape[1];
    const embeddingDim = target._shape[0];
    const size = vocabSize * embeddingDim;

    if (this.prevGradienData.length < size) {
        this.prevGradienData = MemoryManager.ensureCapacity(this.prevGradienData, size) as any;
        this.prevGradien = Matrix.fromFlat(this.prevGradienData.subarray(0, size) as any, target._shape);
    }

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

  dispose(): void {
    this.prevGradienData = new Float32Array(0);
    this.updateBufferData = new Float32Array(0);
    this.prevGradien = mj.matrix([]);
    this.updateBuffer = mj.matrix([]);
  }
}
