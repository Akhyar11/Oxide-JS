import mj from "../math";
import Matrix from "../matrix";
import { isNativeAvailable, softmaxNative, softmaxBackwardNative, sigmoidNative, reluNative, tanhNative } from "../math/rust_backend";

export function sigmoid(a: Matrix): [Matrix, Matrix] {
  if (isNativeAvailable()) {
    const res = new Float32Array(a._data.length);
    const grad = new Float32Array(a._data.length);
    sigmoidNative(a._data, res, grad);
    return [Matrix.fromFlat(res, a._shape), Matrix.fromFlat(grad, a._shape)];
  }
  const result = mj.map(a, (val) => 1 / (1 + Math.exp(-val)));
  const dResult = mj.map(result, (val) => val * (1 - val));
  return [result, dResult];
}

export function tanh(a: Matrix): [Matrix, Matrix] {
  if (isNativeAvailable()) {
    const res = new Float32Array(a._data.length);
    const grad = new Float32Array(a._data.length);
    tanhNative(a._data, res, grad);
    return [Matrix.fromFlat(res, a._shape), Matrix.fromFlat(grad, a._shape)];
  }
  const result = mj.map(a, (val) => Math.tanh(val));
  const dResult = mj.map(result, (val) => 1 - val ** 2);
  return [result, dResult];
}

export function relu(a: Matrix): [Matrix, Matrix] {
  if (isNativeAvailable()) {
    const res = new Float32Array(a._data.length);
    const grad = new Float32Array(a._data.length);
    reluNative(a._data, res, grad);
    return [Matrix.fromFlat(res, a._shape), Matrix.fromFlat(grad, a._shape)];
  }
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

function ensureSoftmaxShape(out: Matrix, rows: number, cols: number) {
  if (out._shape[0] !== rows || out._shape[1] !== cols) {
    throw new Error(`Softmax output shape mismatch: expected [${rows}x${cols}], got [${out._shape[0]}x${out._shape[1]}]`);
  }
}

export function softmaxInto(a: Matrix, out: Matrix, row = false): Matrix {
  const [rows, cols] = a._shape;
  ensureSoftmaxShape(out, rows, cols);

  if (isNativeAvailable()) {
    softmaxNative(a._data, rows, cols, row, out._data);
    return out;
  }

  const input = a._data;
  const result = out._data;

  if (row) {
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

      if (!Number.isFinite(sumExp) || sumExp <= 0) {
        const uniform = 1 / cols;
        for (let j = 0; j < cols; j++) result[offset + j] = uniform;
        continue;
      }

      for (let j = 0; j < cols; j++) {
        result[offset + j] /= sumExp;
      }
    }
  } else {
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

      if (!Number.isFinite(sumExp) || sumExp <= 0) {
        const uniform = 1 / rows;
        for (let i = 0; i < rows; i++) result[i * cols + j] = uniform;
        continue;
      }

      for (let i = 0; i < rows; i++) {
        result[i * cols + j] /= sumExp;
      }
    }
  }

  return out;
}

export function softmaxOnly(a: Matrix, row = false): Matrix {
  const [rows, cols] = a._shape;
  return softmaxInto(a, Matrix.fromFlat(new Float32Array(rows * cols), [rows, cols]), row);
}

/**
 * Fungsi non linear softmax dengan kembalian array [softmax, dSoftmax]
 * @param a Matrix
 * @param row Boolean default False
 * @returns [Matrix, Matrix]
 */
export function softmax(a: Matrix, row = false): [Matrix, Matrix] {
  const softmaxMatrix = softmaxOnly(a, row);
  return [softmaxMatrix, softmaxGradient(softmaxMatrix)];
}

/**
 * Menghitung Jacobian-vector product untuk backpropagation Softmax.
 * Rumus: dL/dz_i = S_i * (dL/dS_i - Σ(S_j * dL/dS_j))
 * 
 * @param s Matrix - Output dari softmax (probs)
 * @param g Matrix - Gradient dari layer setelahnya (incoming error)
 * @param row Boolean - Apakah softmax dihitung per baris (default false)
 */
export function softmaxBackwardInto(s: Matrix, g: Matrix, out: Matrix, row = false): Matrix {
  const [rows, cols] = s._shape;
  ensureSoftmaxShape(out, rows, cols);

  if (isNativeAvailable()) {
    softmaxBackwardNative(s._data, g._data, rows, cols, row, out._data);
    return out;
  }

  const resultData = out._data;
  const sData = s._data;
  const gData = g._data;

  if (row) {
    for (let i = 0; i < rows; i++) {
      const offset = i * cols;
      let sumGradS = 0;
      for (let j = 0; j < cols; j++) {
        const idx = offset + j;
        sumGradS += sData[idx] * gData[idx];
      }
      for (let j = 0; j < cols; j++) {
        const idx = offset + j;
        resultData[idx] = sData[idx] * (gData[idx] - sumGradS);
      }
    }
  } else {
    for (let j = 0; j < cols; j++) {
      let sumGradS = 0;
      for (let i = 0; i < rows; i++) {
        const idx = i * cols + j;
        sumGradS += sData[idx] * gData[idx];
      }
      for (let i = 0; i < rows; i++) {
        const idx = i * cols + j;
        resultData[idx] = sData[idx] * (gData[idx] - sumGradS);
      }
    }
  }

  return out;
}

export function softmaxBackward(s: Matrix, g: Matrix, row = false): Matrix {
  const [rows, cols] = s._shape;
  return softmaxBackwardInto(s, g, Matrix.fromFlat(new Float32Array(rows * cols), [rows, cols]), row);
}

/**
 * Kembalikan diagonal dari Jacobian Softmax (Aproksimasi elemen-wise).
 * CATATAN: Ini tidak akurat untuk backprop penuh, disarankan gunakan softmaxBackward.
 */
export function softmaxGradient(a: Matrix) {
  const gradData = new Float32Array(a._data.length);
  for (let i = 0; i < a._data.length; i++) {
    const value = a._data[i];
    gradData[i] = value * (1 - value);
  }

  return Matrix.fromFlat(gradData, [a._shape[0], a._shape[1]]);
}
