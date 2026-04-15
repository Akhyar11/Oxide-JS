import Matrix from "../matrix";

export default function norm(a: Matrix): number {
  let sum = 0;
  for (let i = 0; i < a._shape[0]; i++) {
    for (let j = 0; j < a._shape[1]; j++) {
      sum += a._value[i][j] ** 2;
    }
  }
  return Math.sqrt(sum);
}
