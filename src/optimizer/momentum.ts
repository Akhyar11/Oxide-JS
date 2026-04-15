import { MatrixShape } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";

export default class Momentum {
  prevGradien: Matrix;
  beta = 0.9;
  constructor(shape: MatrixShape) {
    this.prevGradien = mj.zeros(shape);
  }
  calculate(a: Matrix, alpha: number) {
    // v_t = β * v_{t-1} + α * gradient  (element-wise)
    const betaVelocity = mj.mul(this.beta, this.prevGradien);
    const alphaGrad = mj.mul(alpha, a);
    const newGradien = mj.add(betaVelocity, alphaGrad);
    this.prevGradien = newGradien;
    return newGradien;
  }
}
