import Matrix from "../matrix";

/**
 * Perkalian matrix a dengan dirinya sendiri mengebalikan number
 * @param a Matrix
 * @returns Number
 */
export default function dotMul(a: Matrix): number {
  let value: number = 1;

  if (a._shape[0] === 1) {
    for (let j = 0; j < a._shape[1]; j++) {
      value *= a._value[0][j];
    }
  } else if (a._shape[1] === 1) {
    for (let i = 0; i < a._shape[0]; i++) {
      value *= a._value[i][0];
    }
  } else {
    for (let i = 0; i < a._shape[0]; i++) {
      for (let j = 0; j < a._shape[1]; j++) {
        value *= a._value[i][j];
      }
    }
  }

  return value;
}
