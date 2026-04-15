import { MatrixShape } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";

export default class AdaGrad {
  shape: MatrixShape;
  sumGradien: Matrix;
  epsilon: number = 0.1;
  constructor(shape: MatrixShape, epsilon: number) {
    this.shape = shape;
    this.sumGradien = mj.zeros(this.shape);
    this.epsilon = epsilon;
  }

  calculate(a: Matrix, alpha: number) {
    // AdaGrad: G_t = G_{t-1} + g_t²  (akumulasi kuadrat gradient)
    const squaredGradient = mj.map(a, (val) => val ** 2);
    const sumGradien = mj.add(this.sumGradien, squaredGradient);
    const addEpsilon = mj.add(sumGradien, this.epsilon);
    const sqrtGradien = mj.map(addEpsilon, (val) => Math.sqrt(val));
    const newAlpha = mj.div(alpha, sqrtGradien);
    this.sumGradien = sumGradien;
    return mj.mul(newAlpha, a);
  }
}
