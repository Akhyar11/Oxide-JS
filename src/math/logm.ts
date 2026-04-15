import Matrix from "../matrix";

/**
 * Log natural (ln) => Matrix log(a)  — Math.log() = ln, BUKAN log10
 * @param a Matrix
 * @returns Matrix
 */
export default function logm(a: Matrix): Matrix {
  const array: number[][] = new Array(a._shape[0]);
  for (let i = 0; i < a._shape[0]; i++) {
    const row = new Array(a._shape[1]);
    for (let j = 0; j < a._shape[1]; j++) {
      const val = a._value[i][j];
      // Guard: log(x<=0) = -Infinity/NaN, tambah epsilon kecil jika nol
      row[j] = Math.log(val <= 0 ? 1e-15 : val);
    }
    array[i] = row;
  }

  return new Matrix({ array });
}
