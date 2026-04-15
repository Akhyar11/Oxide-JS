import { MatrixCollection } from "../@types/type";
import Matrix from "../matrix";
import map from "./map";
import zeros from "./zeros";

/**
 * Pembagian a dan b
 * @param a Matrix | Number
 * @param b Matrix | Number
 * @returns Matrix
 */
export default function div(a: MatrixCollection, b: MatrixCollection): Matrix {
  try {
    let array: number[][] | Matrix = [];
    if (typeof a === "number") {
      array = map(b as Matrix, (val) => a / val);
    } else if (typeof b === "number") {
      array = map(a, (val) => val / b);
    } else {
      if (a._shape[0] !== b._shape[0] || a._shape[1] !== b._shape[1]) {
        throw new Error(
          `bentuk dari a harus sama dengan matrix ${a._shape} != ${b._shape}`
        );
      }

      array = zeros(a._shape);
      for (let i = 0; i < a._shape[0]; i++) {
        for (let j = 0; j < a._shape[1]; j++) {
          array._value[i][j] = a._value[i][j] / b._value[i][j];
        }
      }
    }

    return array instanceof Matrix ? array : new Matrix({ array });
  } catch (err) {
    throw err;
  }
}
