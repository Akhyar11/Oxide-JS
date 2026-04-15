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
  const [rows, cols] = a._shape;
  const input = a._data;

  if (row) {
    const result = new Float64Array(input.length);
    for (let i = 0; i < rows; i++) {
      const offset = i * cols;
      let maxVal = -Infinity;
      for (let j = 0; j < cols; j++) {
        const value = input[offset + j];
        if (value > maxVal) maxVal = value;
      }

      let sumExp = 0;
      for (let j = 0; j < cols; j++) {
        const expValue = Math.exp(input[offset + j] - maxVal);
        result[offset + j] = expValue;
        sumExp += expValue;
      }

      for (let j = 0; j < cols; j++) {
        result[offset + j] /= sumExp;
      }
    }

    const softmaxMatrix = Matrix.fromFlat(result, [rows, cols]);
    return [softmaxMatrix, softmaxGradien(softmaxMatrix)];
  } else {
    const result = new Float64Array(input.length);
    for (let j = 0; j < cols; j++) {
      let maxVal = -Infinity;
      for (let i = 0; i < rows; i++) {
        const value = input[i * cols + j];
        if (value > maxVal) maxVal = value;
      }

      let sumExp = 0;
      for (let i = 0; i < rows; i++) {
        const idx = i * cols + j;
        const expValue = Math.exp(input[idx] - maxVal);
        result[idx] = expValue;
        sumExp += expValue;
      }

      for (let i = 0; i < rows; i++) {
        result[i * cols + j] /= sumExp;
      }
    }

    const softmaxMatrix = Matrix.fromFlat(result, [rows, cols]);
    return [softmaxMatrix, softmaxGradien(softmaxMatrix)];
  }
}

export function softmaxGradien(a: Matrix) {
  const gradData = new Float64Array(a._data.length);
  for (let i = 0; i < a._data.length; i++) {
    const value = a._data[i];
    gradData[i] = value * (1 - value);
  }

  return Matrix.fromFlat(gradData, [a._shape[0], a._shape[1]]);
}
