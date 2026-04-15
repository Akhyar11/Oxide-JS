import { MatrixShape } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";

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

  constructor(
    shape: MatrixShape,
    beta1: number = 0.9,
    beta2: number = 0.999,
    epsilon: number = 1e-8
  ) {
    this.m = mj.zeros(shape);
    this.v = mj.zeros(shape);
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
  }

  calculate(a: Matrix, alpha: number): Matrix {
    this.t++;
    const gradData = a._data;
    const mData = this.m._data;
    const vData = this.v._data;
    const updateData = new Float64Array(gradData.length);
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
      updateData[i] = alpha * mHat / (Math.sqrt(vHat) + this.epsilon);
    }

    return Matrix.fromFlat(updateData, [a._shape[0], a._shape[1]]);
  }
}
