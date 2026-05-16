import Matrix from "../matrix/index.js";
import { isNativeAvailable, softmaxNative, softmaxBackwardNative } from "../math/rust_backend.js";
import { engine } from "../autodiff/engine.js";

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

export function softmaxBackwardInto(s: Matrix, g: Matrix, out: Matrix, row = false): Matrix {
  const [rows, cols] = s._shape;
  if (g._shape[0] !== rows || g._shape[1] !== cols) {
    throw new Error(`softmaxBackwardInto: shape mismatch between s [${rows}x${cols}] and g [${g._shape[0]}x${g._shape[1]}]`);
  }
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

export default function softmax(a: Matrix, row = false): Matrix {
  const [rows, cols] = a._shape;
  const out = Matrix.fromFlat(new Float32Array(rows * cols), [rows, cols]);

  softmaxInto(a, out, row);

  const tape = engine.tape;
  if (tape) {
    tape.record([a], [out], (grad: Matrix) => {
      const gradA = softmaxBackward(out, grad, row);
      if (a.grad) a.grad.addInPlace(gradA);
      else a.grad = gradA;
    }, { saveInput: false, saveOutput: true });
  }

  return out;
}
