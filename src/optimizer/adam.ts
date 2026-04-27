import { MatrixShape } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";
import { isNativeAvailable, adamUpdateNative, adamSparseUpdateNative, shouldUseNativeAdam } from "../math/rust_backend";
import { MemoryManager } from "../utils/memory";

/**
 * Adam Optimizer (Adaptive Moment Estimation)
 */
export default class Adam {
  private mData: any = new Float32Array(0);
  private vData: any = new Float32Array(0);
  private updateBufferData: any = new Float32Array(0);
  
  private m: Matrix;       // first moment (mean)
  private v: Matrix;       // second moment (variance)
  private t: number = 0;  // timestep
  private beta1: number;
  private beta2: number;
  private epsilon: number;
  private updateBuffer: Matrix; 

  constructor(
    shape: MatrixShape,
    beta1: number = 0.9,
    beta2: number = 0.999,
    epsilon: number = 1e-8
  ) {
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
    this.m = mj.matrix([]);
    this.v = mj.matrix([]);
    this.updateBuffer = mj.matrix([]);
  }

  private ensureBuffers(size: number, shape: MatrixShape) {
    this.mData = MemoryManager.ensureCapacity(this.mData, size) as any;
    this.vData = MemoryManager.ensureCapacity(this.vData, size) as any;
    this.updateBufferData = MemoryManager.ensureCapacity(this.updateBufferData, size) as any;

    this.m = Matrix.fromFlat(this.mData.subarray(0, size) as any, shape);
    this.v = Matrix.fromFlat(this.vData.subarray(0, size) as any, shape);
    this.updateBuffer = Matrix.fromFlat(this.updateBufferData.subarray(0, size) as any, shape);
  }

  calculate(a: Matrix, alpha: number): Matrix {
    this.t++;
    const size = a._data.length;
    this.ensureBuffers(size, a._shape);

    const gradData = a._data;
    const mData = this.m._data;
    const vData = this.v._data;
    const bufferData = this.updateBuffer._data;
    
    if (isNativeAvailable() && shouldUseNativeAdam(gradData.length)) {
      adamUpdateNative(
        gradData,
        mData,
        vData,
        bufferData,
        this.t,
        alpha,
        this.beta1,
        this.beta2,
        this.epsilon
      );
      return this.updateBuffer;
    }

    const oneMinusBeta1 = 1 - this.beta1;
    const oneMinusBeta2 = 1 - this.beta2;
    const biasCorrection1 = 1 / (1 - Math.pow(this.beta1, this.t));
    const biasCorrection2 = 1 / (1 - Math.pow(this.beta2, this.t));

    for (let i = 0; i < gradData.length; i++) {
      const g = gradData[i];
      const m = this.beta1 * mData[i] + oneMinusBeta1 * g;
      const v = this.beta2 * vData[i] + oneMinusBeta2 * g * g;
      mData[i] = m;
      vData[i] = v;

      const mHat = m * biasCorrection1;
      const vHat = v * biasCorrection2;
      bufferData[i] = alpha * mHat / (Math.sqrt(vHat) + this.epsilon);
    }

    return this.updateBuffer;
  }

  updateSparse(target: Matrix, grad: Matrix, alpha: number, indices: Int32Array): void {
    const vocabSize = target._shape[1];
    const embeddingDim = target._shape[0];
    const size = vocabSize * embeddingDim;

    if (this.mData.length < size) {
        this.mData = MemoryManager.ensureCapacity(this.mData, size) as any;
        this.vData = MemoryManager.ensureCapacity(this.vData, size) as any;
        this.m = Matrix.fromFlat(this.mData.subarray(0, size) as any, target._shape);
        this.v = Matrix.fromFlat(this.vData.subarray(0, size) as any, target._shape);
    }

    const targetData = target._data;
    const gradData = grad._data;
    const mData = this.m._data;
    const vData = this.v._data;

    if (isNativeAvailable() && shouldUseNativeAdam(gradData.length)) {
      adamSparseUpdateNative(
        indices,
        gradData,
        targetData,
        mData,
        vData,
        this.t + 1,
        alpha,
        this.beta1,
        this.beta2,
        this.epsilon,
        vocabSize,
        embeddingDim
      );
      this.t++;
      return;
    }

    this.t++;
    const oneMinusBeta1 = 1 - this.beta1;
    const oneMinusBeta2 = 1 - this.beta2;
    const biasCorrection1 = 1 / (1 - Math.pow(this.beta1, this.t));
    const biasCorrection2 = 1 / (1 - Math.pow(this.beta2, this.t));
    const numUnique = indices.length;

    for (let j = 0; j < numUnique; j++) {
      const tokenIndex = indices[j];
      for (let i = 0; i < embeddingDim; i++) {
        const fullIdx = i * vocabSize + tokenIndex;
        const gradIdx = i * numUnique + j;

        const g = gradData[gradIdx];
        const m = this.beta1 * mData[fullIdx] + oneMinusBeta1 * g;
        const v = this.beta2 * vData[fullIdx] + oneMinusBeta2 * g * g;
        mData[fullIdx] = m;
        vData[fullIdx] = v;

        const mHat = m * biasCorrection1;
        const vHat = v * biasCorrection2;
        targetData[fullIdx] -= alpha * mHat / (Math.sqrt(vHat) + this.epsilon);
      }
    }
  }

  dispose(): void {
    this.mData = new Float32Array(0);
    this.vData = new Float32Array(0);
    this.updateBufferData = new Float32Array(0);
    this.m = mj.matrix([]);
    this.v = mj.matrix([]);
    this.updateBuffer = mj.matrix([]);
  }
}
