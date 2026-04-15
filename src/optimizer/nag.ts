import { MatrixShape } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";

export default class NAG {
  prevGradien: Matrix;
  beta = 0.9;
  constructor(shape: MatrixShape) {
    this.prevGradien = mj.zeros(shape);
  }

  calculate(a: Matrix, alpha: number) {
    // NAG: v_t = β * v_{t-1} + α * (g - β * v_{t-1})  (element-wise)
    const betaVelocity = mj.mul(this.beta, this.prevGradien);
    const wUpdate = mj.sub(a, betaVelocity);
    const newGradien = mj.add(betaVelocity, mj.mul(alpha, wUpdate));
    this.prevGradien = newGradien;
    return newGradien;
  }
}
