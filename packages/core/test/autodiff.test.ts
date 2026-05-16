import { describe, expect, it } from "vitest";

import Matrix from "../src/matrix/index.js";
import Tape from "../src/autodiff/index.js";
import { engine } from "../src/autodiff/engine.js";
import mj from "../src/math/index.js";

describe("autodiff", () => {
  it("backpropagates through repeated use of the same tensor", () => {
    const x = mj.matrix([[2]]);
    const tape = engine.grad(() => {
      const y = mj.mul(x, x);
      return mj.add(y, y);
    });

    tape.backward(tape.result);

    expect(tape.result._value).toEqual([[8]]);
    expect(x.grad?._value).toEqual([[8]]);
  });

  it("backpropagates matrix multiplication gradients", () => {
    const a = mj.matrix([[1], [2]]);
    const b = mj.matrix([[3, 4]]);
    const tape = engine.grad(() => mj.dotProduct(a, b));

    tape.backward(tape.result);

    expect(a.grad?._value).toEqual([[7], [7]]);
    expect(b.grad?._value).toEqual([[3, 3]]);
  });

  it("supports custom upstream gradients in Tape.backward", () => {
    const x = mj.matrix([[2]]);
    const tape = engine.grad(() => mj.mul(x, x));

    const upstream = mj.matrix([[3]]);
    tape.backward(tape.result, upstream);

    expect(x.grad?._value).toEqual([[12]]);
  });

  it("backpropagates through scalar reducer wrappers", () => {
    const x = mj.matrix([[2], [3]]);

    const sumTape = engine.grad(() => mj.dotSumScalar(x));
    sumTape.backward(sumTape.result);
    expect(x.grad?._value).toEqual([[1], [1]]);

    x.clearGrad();
    const subTape = engine.grad(() => mj.dotSubScalar(x));
    subTape.backward(subTape.result);
    expect(x.grad?._value).toEqual([[-1], [-1]]);

    x.clearGrad();
    const mulTape = engine.grad(() => mj.dotMulScalar(x));
    mulTape.backward(mulTape.result);
    expect(x.grad?._value).toEqual([[3], [2]]);

    x.clearGrad();
    const divTape = engine.grad(() => mj.dotDivScalar(x));
    divTape.backward(divTape.result);
    expect(x.grad?._value?.[0][0]).toBeCloseTo(-1 / 12, 8);
    expect(x.grad?._value?.[1][0]).toBeCloseTo(-1 / 18, 8);

    x.clearGrad();
    const normTape = engine.grad(() => mj.normScalar(x));
    normTape.backward(normTape.result);
    const n = Math.sqrt(13);
    expect(x.grad?._value?.[0][0]).toBeCloseTo(2 / n, 6);
    expect(x.grad?._value?.[1][0]).toBeCloseTo(3 / n, 6);
  });

  it("supports nested tapes without clobbering the outer tape", () => {
    const x = mj.matrix([[2]]);
    const z = mj.matrix([[3]]);

    const outer = engine.grad(() => {
      const y = mj.mul(x, x);
      const inner = engine.grad(() => mj.mul(z, z));
      inner.backward(inner.result);

      expect(z.grad?._value).toEqual([[6]]);
      expect(x.grad).toBeNull();

      return y;
    });

    outer.backward(outer.result);

    expect(x.grad?._value).toEqual([[4]]);
    expect(z.grad?._value).toEqual([[6]]);
  });

  it("respects detach and does not propagate gradients through detached tensors", () => {
    const x = mj.matrix([[2]]);
    const tape = engine.grad(() => {
      const detached = x.detach();
      const frozen = mj.mul(detached, detached);
      const live = mj.mul(x, x);
      return mj.add(frozen, live);
    });

    tape.backward(tape.result);

    expect(x.grad?._value).toEqual([[4]]);
  });

  it("guards unstable live tensors when a tape node requires input stability", () => {
    const tape = new Tape();
    tape.watch();

    const input = Matrix.fromFlat(new Float32Array([2]), [1, 1]);
    const output = Matrix.fromFlat(new Float32Array([2]), [1, 1]);

    tape.record(
      [input],
      [output],
      (grad) => [grad],
      { saveInput: false, saveOutput: false, requireInputStability: true }
    );

    input.addInPlace(1);

    expect(() => tape.backward(output)).toThrow(/mutated after forward/i);
  });
});
