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

    // m_t = β1 * m_{t-1} + (1-β1) * g
    const beta1G = mj.mul(this.beta1, this.m);
    const oneMinusBeta1G = mj.mul(1 - this.beta1, a);
    this.m = mj.add(beta1G, oneMinusBeta1G);

    // v_t = β2 * v_{t-1} + (1-β2) * g²
    const gSquared = mj.map(a, (val) => val ** 2);
    const beta2V = mj.mul(this.beta2, this.v);
    const oneMinusBeta2V = mj.mul(1 - this.beta2, gSquared);
    this.v = mj.add(beta2V, oneMinusBeta2V);

    // Bias correction
    const mHat = mj.mul(1 / (1 - Math.pow(this.beta1, this.t)), this.m);
    const vHat = mj.mul(1 / (1 - Math.pow(this.beta2, this.t)), this.v);

    // update = α * m̂ / (sqrt(v̂) + ε)
    const sqrtVHat = mj.map(vHat, (val) => Math.sqrt(val) + this.epsilon);
    const update = mj.mul(alpha, mj.div(mHat, sqrtVHat));

    return update;
  }
}
