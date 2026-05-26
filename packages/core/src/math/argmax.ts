import Matrix from "../matrix/index.js";

/**
 * Mencari indeks nilai maksimum sepanjang axis tertentu
 * @param a Matrix
 * @param axis 1 untuk baris (hasil [rows x 1]), 0 untuk kolom (hasil [1 x cols]), undefined untuk seluruh matrix (hasil number)
 */
export default function argmax(a: Matrix, axis?: number): Matrix | number {
  const [rows, cols] = a._shape;
  const data = a._data;

  if (axis === undefined) {
    if (data.length === 0) return -1;
    let maxIdx = 0;
    let maxVal = data[0];
    for (let i = 1; i < data.length; i++) {
      if (data[i] > maxVal) {
        maxVal = data[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }

  const outShape: [number, number] = axis === 1 ? [rows, 1] : [1, cols];
  const result = Matrix.fromFlat(new Float32Array(outShape[0] * outShape[1]), outShape);
  const res = result._data;

  if (axis === 1) {
    for (let i = 0; i < rows; i++) {
      let maxIdx = 0;
      let maxVal = data[i * cols];
      for (let j = 1; j < cols; j++) {
        const val = data[i * cols + j];
        if (val > maxVal) {
          maxVal = val;
          maxIdx = j;
        }
      }
      res[i] = maxIdx;
    }
  } else if (axis === 0) {
    for (let j = 0; j < cols; j++) {
      let maxIdx = 0;
      let maxVal = data[j];
      for (let i = 1; i < rows; i++) {
        const val = data[i * cols + j];
        if (val > maxVal) {
          maxVal = val;
          maxIdx = i;
        }
      }
      res[j] = maxIdx;
    }
  } else {
    throw new Error(`Invalid axis: ${axis}. axis must be 0 or 1.`);
  }

  return result;
}
