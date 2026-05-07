import { MatrixShape } from "../@types/type.js";
import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { isNativeAvailable, adamUpdateNative, adamSparseUpdateNative, shouldUseNativeAdam, embeddingAdamBackwardUpdateNative } from "../math/rust_backend.js";

/**
 * Adam Optimizer (Adaptive Moment Estimation)
 * Formula:
 *   m_t = β1 * m_{t-1} + (1-β1) * g_t         ← first moment (mean)
 *   v_t = β2 * v_{t-1} + (1-β2) * g_t²         ← second moment (variance)
 *   m̂_t = m_t / (1 - β1^t)                     ← bias-corrected mean
 *   v̂_t = v_t / (1 - β2^t)                     ← bias-corrected variance
 *   θ_t = θ_{t-1} - α * m̂_t / (sqrt(v̂_t) + ε)
 */
export default class Adam {
  private m: Matrix;       // first moment (mean)
  private v: Matrix;       // second moment (variance)
  private t: number = 0;  // timestep
  private beta1: number;
  private beta2: number;
  private epsilon: number;
  private updateBuffer: Matrix; // Buffer untuk menampung hasil update (REUSE)

  constructor(
    shape: MatrixShape,
    beta1: number = 0.9,
    beta2: number = 0.999,
    epsilon: number = 1e-8
  ) {
    this.m = mj.zeros(shape);
    this.v = mj.zeros(shape);
    this.updateBuffer = mj.zeros(shape);
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
  }

  calculate(a: Matrix, alpha: number): Matrix {
    this.t++;
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
    const targetData = target._data;
    const gradData = grad._data;
    const mData = this.m._data;
    const vData = this.v._data;

    const vocabSize = target._shape[1];
    const embeddingDim = target._shape[0];

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

  /**
   * Fused embedding backward + Adam parameter update via single NAPI call.
   *
   * Performs gradient aggregation AND Adam update entirely in Rust —
   * no Int32Array / Float32Array / Matrix is returned across the JS boundary.
   *
   * @returns true  when the fused native path ran successfully
   * @returns false when native is unavailable or the binary predates this symbol
   *               (caller should fall back to the existing split path)
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

    const succeeded = embeddingAdamBackwardUpdateNative(
      indices,
      errData,
      target._data,
      this.m._data,
      this.v._data,
      this.t + 1,   // Rust receives the incremented t (bias-correction uses t≥1)
      alpha,
      this.beta1,
      this.beta2,
      this.epsilon,
      vocabSize,
      embeddingDim,
      padTokenId
    );

    if (succeeded) {
      this.t++;
    }
    return succeeded;
  }

  /**
   * Menerapkan gradien ke matrix target secara in-place.
   * Mengambil gradien dari target.grad.
   */
  apply(target: Matrix, alpha: number): void {
    if (!target.grad) return;
    const update = this.calculate(target.grad, alpha);
    target.subInPlace(update);
    // Reset gradien setelah update
    target.grad = null;
  }
}
