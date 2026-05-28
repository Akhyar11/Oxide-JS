import { execFileSync } from "node:child_process";
import { describe, expect, it, vi } from "vitest";

import Matrix from "../src/matrix/index.js";
import mj from "../src/math/index.js";
import matrix from "../src/math/matrix.js";
import absm from "../src/math/absm.js";
import add, { addInto } from "../src/math/add.js";
import addBias from "../src/math/addBias.js";
import clipGradients from "../src/math/clipGradients.js";
import concat from "../src/math/concat.js";
import convolution from "../src/math/convolution.js";
import div from "../src/math/div.js";
import dotDiv from "../src/math/dotDiv.js";
import dotMul from "../src/math/dotMul.js";
import dotProduct from "../src/math/dotProduct.js";
import dotSub from "../src/math/dotSub.js";
import dotSum from "../src/math/dotSum.js";
import expm from "../src/math/expm.js";
import flatten from "../src/math/flatten.js";
import he from "../src/math/he.js";
import logm from "../src/math/logm.js";
import map from "../src/math/map.js";
import mean from "../src/math/mean.js";
import mul from "../src/math/mul.js";
import norm from "../src/math/norm.js";
import ones from "../src/math/ones.js";
import pow from "../src/math/pow.js";
import random from "../src/math/random.js";
import reshape from "../src/math/reshape.js";
import sub, { subInto } from "../src/math/sub.js";
import sumAxis from "../src/math/sumAxis.js";
import transpose from "../src/math/transpose.js";
import xavier from "../src/math/xavier.js";
import zeros from "../src/math/zeros.js";
import { engine } from "../src/autodiff/engine.js";
import {
  addNative,
  divNative,
  dotProductNative,
  isNativeAvailable,
  mulNative,
  setForceDisableNative,
  subNative,
} from "../src/math/rust_backend.js";

function expectMatrixValue(actual: Matrix, expected: number[][], precision = 5) {
  expect(actual._shape).toEqual([expected.length, expected[0]?.length ?? 0]);
  for (let i = 0; i < expected.length; i++) {
    for (let j = 0; j < expected[i].length; j++) {
      expect(actual.get(i, j)).toBeCloseTo(expected[i][j], precision);
    }
  }
}

describe("math primitives", () => {
  it("creates matrices from matrix()", () => {
    const m = matrix([[1, 2], [3, 4]]);
    expect(m).toBeInstanceOf(Matrix);
    expectMatrixValue(m, [[1, 2], [3, 4]]);
  });

  it("exposes core helpers via mj barrel", () => {
    expect(typeof mj.add).toBe("function");
    expect(typeof mj.dotProduct).toBe("function");
    expect(typeof mj.zeros).toBe("function");
    expectMatrixValue(mj.sub(mj.matrix([[3]]), 1), [[2]]);
  });

  it("computes absolute values", () => {
    expectMatrixValue(absm(matrix([[-1, 2], [-3, 4]])), [[1, 2], [3, 4]]);
  });

  it("adds matrices, scalars, and addInto output", () => {
    const a = matrix([[1, 2], [3, 4]]);
    const b = matrix([[10, 20], [30, 40]]);
    const out = zeros([2, 2]);

    expectMatrixValue(add(a, b), [[11, 22], [33, 44]]);
    expectMatrixValue(add(2, a), [[3, 4], [5, 6]]);
    expectMatrixValue(add(a, 2), [[3, 4], [5, 6]]);
    expect(addInto(a, b, out)).toBe(out);
    expectMatrixValue(out, [[11, 22], [33, 44]]);
    expect(() => addInto(a, b, a)).toThrow(/aliasing/);
    expect(() => addInto(a, b, zeros([1, 1]))).toThrow(/Output matrix shape mismatch/);
  });

  it("subtracts matrices, scalars, and subInto output", () => {
    const a = matrix([[10, 20], [30, 40]]);
    const b = matrix([[1, 2], [3, 4]]);
    const out = zeros([2, 2]);

    expectMatrixValue(sub(a, b), [[9, 18], [27, 36]]);
    expectMatrixValue(sub(5, b), [[4, 3], [2, 1]]);
    expectMatrixValue(sub(a, 5), [[5, 15], [25, 35]]);
    expect(subInto(a, b, out)).toBe(out);
    expectMatrixValue(out, [[9, 18], [27, 36]]);
    expect(() => subInto(a, b, a)).toThrow(/aliasing/);
    expect(() => subInto(a, b, zeros([1, 1]))).toThrow(/Output matrix shape mismatch/);
  });

  it("multiplies element-wise with scalars and matrices", () => {
    const a = matrix([[1, 2], [3, 4]]);
    const b = matrix([[2, 3], [4, 5]]);

    expectMatrixValue(mul(a, b), [[2, 6], [12, 20]]);
    expectMatrixValue(mul(3, a), [[3, 6], [9, 12]]);
    expectMatrixValue(mul(a, 3), [[3, 6], [9, 12]]);
    expect(() => mul(a, b, zeros([1, 1]))).toThrow(/Output matrix shape mismatch/);
  });

  it("divides with scalars and matrices and rejects zero divisors", () => {
    const a = matrix([[8, 6], [4, 2]]);
    const b = matrix([[2, 3], [4, 1]]);

    expectMatrixValue(div(a, b), [[4, 2], [1, 2]]);
    expectMatrixValue(div(a, 2), [[4, 3], [2, 1]]);
    expectMatrixValue(div(12, b), [[6, 4], [3, 12]]);
    expect(() => div(a, matrix([[1, 0], [1, 1]]))).toThrow(/Pembagian dengan nol/);
  });

  it("adds row bias in place and rejects invalid bias shapes", () => {
    const a = matrix([[1, 2], [3, 4]]);
    const bias = matrix([[10], [20]]);

    addBias(a, bias);
    expectMatrixValue(a, [[11, 12], [23, 24]]);

    expect(() => addBias(matrix([[1, 2], [3, 4]]), matrix([[10, 20], [30, 40]]))).toThrow(/Bias shape mismatch/);
  });

  it("clips gradients in place", () => {
    const a = matrix([[-5, -1], [2, 10]]);
    clipGradients(a, 3);
    expectMatrixValue(a, [[-3, -1], [2, 3]]);
    expect(() => clipGradients(a, -1)).toThrow(/non-negative/);
  });

  it("concatenates row vectors", () => {
    expectMatrixValue(concat(matrix([[1, 2]]), matrix([[3, 4, 5]])), [[1, 2, 3, 4, 5]]);
    expect(() => concat(matrix([[1], [2]]), matrix([[3]]) )).toThrow(/flatten/);
  });

  it("computes valid convolution", () => {
    const input = matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    const kernel = matrix([[1, 0], [0, -1]]);
    expectMatrixValue(convolution(input, kernel), [[-4, -4], [-4, -4]]);
  });

  it("computes scalar reductions", () => {
    const a = matrix([[1, 2], [3, 4]]);
    expect(dotDiv(a)).toBeCloseTo(1 / 24, 8);
    expect(dotMul(a)).toBe(24);
    expect(dotSub(a)).toBe(-10);
    expect(dotSum(a)).toBe(10);
    expect(norm(a)).toBeCloseTo(Math.sqrt(30), 8);
  });

  it("computes dot products with and without transpose", () => {
    const a = matrix([[1, 2, 3], [4, 5, 6]]);
    const b = matrix([[7, 8], [9, 10], [11, 12]]);
    const out = zeros([2, 2]);

    expectMatrixValue(dotProduct(a, b), [[58, 64], [139, 154]]);
    expect(dotProduct(a, b, out)).toBe(out);
    expectMatrixValue(out, [[58, 64], [139, 154]]);

    const c = matrix([[1, 2, 3], [4, 5, 6]]);
    const d = matrix([[7, 8, 9], [10, 11, 12]]);
    expectMatrixValue(dotProduct(c, d, undefined, false, true), [[50, 68], [122, 167]]);
    expectMatrixValue(dotProduct(c, c, undefined, true, false), [[17, 22, 27], [22, 29, 36], [27, 36, 45]]);
    expect(() => dotProduct(a, b, zeros([1, 1]))).toThrow(/Output matrix shape mismatch/);
    expect(() => dotProduct(matrix([[1, 2]]), matrix([[1, 2]]))).toThrow(/Dimensi matrix tidak cocok/);
  });

  it("computes exp and log transforms", () => {
    expectMatrixValue(expm(matrix([[0, 1]])), [[1, Math.E]]);
    const logged = logm(matrix([[1, Math.E], [0, -1]]));
    expect(logged.get(0, 0)).toBeCloseTo(0, 6);
    expect(logged.get(0, 1)).toBeCloseTo(1, 6);
    expect(logged.get(1, 0)).toBeCloseTo(Math.log(1e-15), 6);
    expect(logged.get(1, 1)).toBeCloseTo(Math.log(1e-15), 6);
  });

  it("flattens, reshapes, transposes, and sums by axis", () => {
    const a = matrix([[1, 2], [3, 4]]);
    expectMatrixValue(flatten(a), [[1], [2], [3], [4]]);
    expectMatrixValue(reshape(a, [1, 4]), [[1, 2, 3, 4]]);
    expect(() => reshape(a, [3, 2])).toThrow(/tidak sama/);
    expectMatrixValue(transpose(a), [[1, 3], [2, 4]]);
    expectMatrixValue(sumAxis(a, 1), [[3], [7]]);
    expectMatrixValue(sumAxis(a, 0), [[4, 6]]);
    expect(() => transpose(a, zeros([1, 1]))).toThrow(/Output matrix shape mismatch/);
    expect(() => sumAxis(a, 1, zeros([1, 1]))).toThrow(/sumAxis output shape mismatch/);
  });

  it("maps, averages, and raises powers", () => {
    const a = matrix([[1, 2], [3, 4]]);
    expectMatrixValue(map(a, (v) => v * 10), [[10, 20], [30, 40]]);
    expectMatrixValue(mean(a), [[2.5]]);
    expectMatrixValue(pow(a, 2), [[1, 4], [9, 16]]);
  });

  it("computes argmax along axes or globally", () => {
    const a = matrix([[1, 5, 2], [4, 3, 6]]);
    
    // global argmax
    expect(mj.argmax(a)).toBe(5); // index of value 6 in flattened array
    
    // argmax along rows (axis 1)
    const rowArgmax = mj.argmax(a, 1) as Matrix;
    expectMatrixValue(rowArgmax, [[1], [2]]);
    
    // argmax along columns (axis 0)
    const colArgmax = mj.argmax(a, 0) as Matrix;
    expectMatrixValue(colArgmax, [[1, 0, 1]]);
  });


  it("creates ones and zeros", () => {
    expectMatrixValue(ones([2, 3]), [[1, 1, 1], [1, 1, 1]]);
    expectMatrixValue(zeros([2, 2]), [[0, 0], [0, 0]]);
  });

  it("creates random, he, and xavier matrices within expected bounds", () => {
    const randomSpy = vi.spyOn(Math, "random").mockReturnValue(0.75);

    const rand = random([2, 2]);
    expectMatrixValue(rand, [[0.05, 0.05], [0.05, 0.05]], 6);

    const heMatrix = he([2, 4]);
    const heLimit = Math.sqrt(6 / 4);
    for (const value of heMatrix._data) {
      expect(value).toBeLessThanOrEqual(heLimit);
      expect(value).toBeGreaterThanOrEqual(-heLimit);
    }

    const xavierMatrix = xavier([2, 4]);
    const xavierLimit = Math.sqrt(6 / (4 + 2));
    for (const value of xavierMatrix._data) {
      expect(value).toBeLessThanOrEqual(xavierLimit);
      expect(value).toBeGreaterThanOrEqual(-xavierLimit);
    }

    randomSpy.mockRestore();
  });

  it("covers activation and loss helpers exposed from mj", () => {
    expectMatrixValue(mj.sigmoid(matrix([[0]])), [[0.5]], 5);
    expectMatrixValue(mj.tanh(matrix([[0]])), [[0]], 5);
    expectMatrixValue(mj.relu(matrix([[-1, 2]])), [[0, 2]]);
    expectMatrixValue(mj.lRelu(matrix([[-1, 2]])), [[-1e-5, 2]], 7);
    expectMatrixValue(mj.linear(matrix([[3]])), [[3]]);
    expectMatrixValue(mj.softmax(matrix([[1], [2]])), [[0.26894143], [0.7310586]], 5);

    expect(mj.mse(matrix([[1]]), matrix([[3]]))).toBeCloseTo(4, 6);
    expect(mj.crossEntropy(matrix([[0.9], [0.1]]), matrix([[1], [0]]))).toBeGreaterThan(0);
    expect(mj.binaryCrossEntropy(matrix([[0.9]]), matrix([[1]]))).toBeGreaterThan(0);
    expect(mj.softmaxCrossEntropy(matrix([[1], [3]]), matrix([[0], [1]]))).toBeGreaterThan(0);
  });
});

describe("math autodiff hooks", () => {
  it("accumulates gradients for addBias into the bias column", () => {
    const a = matrix([[1, 2], [3, 4]]);
    const bias = matrix([[10], [20]]);
    const tape = engine.grad(() => {
      addBias(a, bias);
      return a;
    });

    tape.backward(tape.result);
    expectMatrixValue(bias.grad!, [[2], [2]]);
  });

  it("backpropagates through exp, mean, pow, transpose, reshape, flatten, and concat", () => {
    const a = matrix([[1, 2]]);
    const b = matrix([[3, 4]]);
    const tape = engine.grad(() => {
      const expA = expm(a);
      const powB = pow(b, 2);
      const joined = concat(expA, powB);
      const col = flatten(joined);
      const row = transpose(col);
      const reshaped = reshape(row, [1, 4]);
      return mean(reshaped);
    });

    tape.backward(tape.result);

    expect(a.grad!.get(0, 0)).toBeCloseTo(Math.E / 4, 5);
    expect(a.grad!.get(0, 1)).toBeCloseTo(Math.exp(2) / 4, 5);
    expect(b.grad!.get(0, 0)).toBeCloseTo(6 / 4, 5);
    expect(b.grad!.get(0, 1)).toBeCloseTo(8 / 4, 5);
  });

  it("backpropagates through log and sumAxis", () => {
    const a = matrix([[1, Math.E], [2, 4]]);
    const tape = engine.grad(() => mean(sumAxis(logm(a), 1)));

    tape.backward(tape.result);

    expect(a.grad!.get(0, 0)).toBeCloseTo(0.5, 5);
    expect(a.grad!.get(0, 1)).toBeCloseTo(0.5 / Math.E, 5);
    expect(a.grad!.get(1, 0)).toBeCloseTo(0.25, 5);
    expect(a.grad!.get(1, 1)).toBeCloseTo(0.125, 5);
  });
});

describe("rust backend wrapper in JS-only mode", () => {
  it("reports native as unavailable when disabled", () => {
    setForceDisableNative(true);
    expect(isNativeAvailable()).toBe(false);
    setForceDisableNative(false);
    expect(typeof isNativeAvailable()).toBe("boolean");
  });

  it("throws for native-only wrappers when native is unavailable", () => {
    const output = execFileSync(
      "npx",
      [
        "tsx",
        "--eval",
        "import { addNative, subNative, mulNative, divNative, dotProductNative } from './src/math/rust_backend.ts'; const data=new Float32Array([1,2,3,4]); const out=new Float32Array(4); try { addNative(data,data,out); subNative(data,data,out); mulNative(data,data,out); divNative(data,data,out); dotProductNative(data,2,2,data,2,2,false,false,out); console.log('NO_THROW'); } catch (e) { console.log(String(e)); }",
      ],
      {
        cwd: new URL("../", import.meta.url).pathname,
        env: { ...process.env, ML_DISABLE_NATIVE: "1" },
      }
    ).toString();

    expect(output).toMatch(/Native backend not available/);
  });
});
