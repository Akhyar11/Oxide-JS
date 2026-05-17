import { expect } from "vitest";
import { Matrix } from "@oxide-js/core";

export function expectMatrixShape(matrix: Matrix, shape: number[]) {
  expect(matrix._shape).toEqual(shape);
}

export function expectMatrixCloseTo(
  matrix: Matrix,
  expected: number[],
  precision = 5
) {
  expect(Array.from(matrix._data).length).toBe(expected.length);

  Array.from(matrix._data).forEach((value, index) => {
    expect(value).toBeCloseTo(expected[index], precision);
  });
}

export function mat(data: number[], shape: [number, number]) {
  return Matrix.fromFlat(new Float32Array(data), shape);
}
