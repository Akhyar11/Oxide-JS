import mj from "../math";
import Matrix from "../matrix";

export function sigmoid(a: Matrix): [Matrix, Matrix] {
  const result = mj.map(a, (val) => 1 / (1 + Math.exp(-val)));
  const dResult = mj.map(result, (val) => val * (1 - val));
  return [result, dResult];
}

export function tanh(a: Matrix): [Matrix, Matrix] {
  const result = mj.map(a, (val) => Math.tanh(val));
  const dResult = mj.map(result, (val) => 1 - val ** 2);
  return [result, dResult];
}

export function relu(a: Matrix): [Matrix, Matrix] {
  const result = mj.map(a, (val) => (val < 0 ? 0 : val));
  // gradient dihitung dari input 'a' asli: 0 jika a<=0, 1 jika a>0
  const dResult = mj.map(a, (val) => (val > 0 ? 1 : 0));
  return [result, dResult];
}

export function lRelu(a: Matrix): [Matrix, Matrix] {
  const result = mj.map(a, (val) => (val < 0 ? val * 1e-5 : val));
  const dResult = mj.map(a, (val) => (val < 0 ? 1e-5 : 1));
  return [result, dResult];
}

export default function linear(a: Matrix): [Matrix, Matrix] {
  // Buat salinan baru agar tidak terjadi aliasing yang merusak backward pass
  const result = mj.map(a, (val) => val);
  const dResult = mj.ones(a._shape);
  return [result, dResult];
}

/**
 * Fungsi non linear softmax dengan kembalian array [softmax, dSoftmax]
 * @param a Matrix
 * @param row Boolean default False
 * @returns [Matrix, Matrix]
 */
export function softmax(a: Matrix, row = false): [Matrix, Matrix] {
  if (row) {
    const result = a._value.map((row) => {
      const maxVal = Math.max(...row);
      const expValues = row.map((value) => Math.exp(value - maxVal));
      const sumExpValues = expValues.reduce((a, b) => a + b, 0);
      return expValues.map((value) => value / sumExpValues);
    });

    const softmaxMatrix = mj.matrix(result);
    return [softmaxMatrix, softmaxGradien(softmaxMatrix)];
  } else {
    const reshapeMatrix = mj.reshape(a, [a._shape[1], a._shape[0]]);
    const result = reshapeMatrix._value.map((row) => {
      const maxVal = Math.max(...row);
      const expValues = row.map((value) => Math.exp(value - maxVal));
      const sumExpValues = expValues.reduce((a, b) => a + b, 0);
      return expValues.map((value) => value / sumExpValues);
    });

    const softmaxMatrix = mj.reshape(mj.matrix(result), a._shape);
    return [softmaxMatrix, softmaxGradien(softmaxMatrix)];
  }
}

export function softmaxGradien(a: Matrix) {
  const gradSoftmax = mj.zeros(a._shape);
  for (let i = 0; i < a._shape[0]; i++) {
    for (let j = 0; j < a._shape[1]; j++) {
      // Turunan diagonal Jacobian softmax: s_i * (1 - s_i)
      // Untuk backprop praktis, elemen off-diagonal jarang dibutuhkan secara eksplisit
      // karena digabung dengan chain rule dari loss
      gradSoftmax._value[i][j] = a._value[i][j] * (1 - a._value[i][j]);
    }
  }

  return gradSoftmax;
}
