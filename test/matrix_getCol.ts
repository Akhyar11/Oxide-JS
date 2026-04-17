import Matrix from "../src/matrix";

function assert(condition: boolean, message: string) {
  if (!condition) {
    throw new Error(message);
  }
}

const matrix = new Matrix({
  array: [
    [1, 2, 3],
    [4, 5, 6],
  ],
});

const column = matrix.getCol(1);

assert(column instanceof Float32Array, "getCol should return a Float32Array");
assert(column.length === 2, `expected column length 2, got ${column.length}`);
assert(column[0] === 2, `expected first value 2, got ${column[0]}`);
assert(column[1] === 5, `expected second value 5, got ${column[1]}`);

console.log("matrix_getCol passed");
