import { MatrixShape } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";
import { isNativeAvailable, adagradUpdateNative, adagradSparseUpdateNative, shouldUseNativeOptimizer } from "../math/rust_backend";
import { MemoryManager } from "../utils/memory";

export default class AdaGrad {
  shape: MatrixShape;
  epsilon: number = 0.1;
  private sumGradienData: any = new Float32Array(0);
  private updateBufferData: any = new Float32Array(0);
  private sumGradien: Matrix;
  private updateBuffer: Matrix;

  constructor(shape: MatrixShape, epsilon: number) {
    this.shape = shape;
    this.epsilon = epsilon;
    this.sumGradien = mj.matrix([]);
    this.updateBuffer = mj.matrix([]);
  }

  private ensureBuffers(size: number, shape: MatrixShape) {
    this.sumGradienData = MemoryManager.ensureCapacity(this.sumGradienData, size) as any;
    this.updateBufferData = MemoryManager.ensureCapacity(this.updateBufferData, size) as any;
    
    this.sumGradien = Matrix.fromFlat(this.sumGradienData.subarray(0, size) as any, shape);
    this.updateBuffer = Matrix.fromFlat(this.updateBufferData.subarray(0, size) as any, shape);
  }

  calculate(a: Matrix, alpha: number) {
    const size = a._data.length;
    this.ensureBuffers(size, a._shape);

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
      updateData[i] = (alpha * grad) / Math.sqrt(accumulated + this.epsilon);
    }

    return this.updateBuffer;
  }

  updateSparse(target: Matrix, grad: Matrix, alpha: number, indices: Int32Array): void {
    const vocabSize = target._shape[1];
    const embeddingDim = target._shape[0];
    const size = vocabSize * embeddingDim;
    
    // Untuk sparse update, kita perlu memastikan buffer sumGradien cukup besar untuk seluruh target
    if (this.sumGradienData.length < size) {
        this.sumGradienData = MemoryManager.ensureCapacity(this.sumGradienData, size) as any;
        this.sumGradien = Matrix.fromFlat(this.sumGradienData.subarray(0, size) as any, target._shape);
    }

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

  dispose(): void {
    this.sumGradienData = new Float32Array(0);
    this.updateBufferData = new Float32Array(0);
    this.sumGradien = mj.matrix([]);
    this.updateBuffer = mj.matrix([]);
  }
}
