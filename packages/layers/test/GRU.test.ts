import { describe, expect, it } from "vitest";
import { GRU } from "../src/layers/GRU.js";
import { Matrix, engine } from "@oxide-js/core";
import { expectMatrixCloseTo, expectMatrixShape, mat } from "./helpers/matrix.js";

describe("GRU Layer Tests", () => {
  it("should create layer with default config", () => {
    const layer = new GRU({
      name: "gru_custom",
      units: 4,
      useBias: true,
      returnSequences: true,
      trainable: true
    });
    expect(layer.name).toBe("gru_custom");
    expect(layer.units).toBe(4);
    expect(layer.useBias).toBe(true);
    expect(layer.returnSequences).toBe(true);
    expect(layer.trainable).toBe(true);
  });

  it("should compute output shape when returnSequences = false", () => {
    const layer = new GRU({ units: 5, returnSequences: false, sequenceLength: 3, inputDim: 2 });
    expect(layer.computeOutputShape([2, 3, 2])).toEqual([2, 5]);
    expect(layer.computeOutputShape([6, 2])).toEqual([2, 5]);
  });

  it("should compute output shape when returnSequences = true", () => {
    const layer = new GRU({ units: 5, returnSequences: true, sequenceLength: 3, inputDim: 2 });
    expect(layer.computeOutputShape([2, 3, 2])).toEqual([6, 5]);
    expect(layer.computeOutputShape([6, 2])).toEqual([6, 5]);
  });

  it("should build parameters correctly", () => {
    const layer = new GRU({ units: 3, sequenceLength: 4, inputDim: 2 });
    layer.build([8, 2]);

    expect(layer.isBuilt).toBe(true);
    expect(layer.inputShape).toEqual([8, 2]);
    expect(layer.outputShape).toEqual([2, 3]); // returnSequences = false (default)

    expect(layer.kernel?._shape).toEqual([2, 9]); // 3 * units = 9
    expect(layer.recurrentKernel?._shape).toEqual([3, 9]);
    expect(layer.bias?._shape).toEqual([9, 1]);
  });

  it("should perform forward pass correctly and backward parity", () => {
    const layer = new GRU({
      units: 2,
      sequenceLength: 3,
      inputDim: 2,
      useBias: true
    });
    layer.build([3, 2]); // batch size = 1

    // Manually set weights for deterministic testing
    const kernel = layer.kernel!;
    kernel._data.fill(0.1);
    const recurrentKernel = layer.recurrentKernel!;
    recurrentKernel._data.fill(0.05);
    const bias = layer.bias!;
    bias._data.fill(0.01);

    const x = mat([1.0, 2.0, 0.5, -0.5, 0.0, 1.5], [3, 2]);
    x.requiresGrad = true;

    // Forward pass
    const y = layer.forward(x);
    expectMatrixShape(y, [1, 2]);

    // Backward pass
    const tape = engine.grad(() => {
      return layer.forward(x);
    });

    x.clearGrad();
    kernel.clearGrad();
    recurrentKernel.clearGrad();
    bias.clearGrad();

    tape.backward(tape.result);

    expect(x.grad).toBeDefined();
    expect(kernel.grad).toBeDefined();
    expect(recurrentKernel.grad).toBeDefined();
    expect(bias.grad).toBeDefined();

    expectMatrixShape(x.grad!, [3, 2]);
    expectMatrixShape(kernel.grad!, [2, 6]); // 3 * units = 6
    expectMatrixShape(recurrentKernel.grad!, [2, 6]);
    expectMatrixShape(bias.grad!, [6, 1]);
  });

  it("should perform forward and backward with returnSequences = true", () => {
    const layer = new GRU({
      units: 2,
      sequenceLength: 2,
      inputDim: 2,
      returnSequences: true,
      useBias: true
    });
    layer.build([4, 2]); // batch size = 2

    const x = mat([1, 2, 3, 4, 5, 6, 7, 8], [4, 2]);
    x.requiresGrad = true;

    const y = layer.forward(x);
    expectMatrixShape(y, [4, 2]);

    const tape = engine.grad(() => {
      return layer.forward(x);
    });

    x.clearGrad();
    tape.backward(tape.result);

    expect(x.grad).toBeDefined();
    expectMatrixShape(x.grad!, [4, 2]);
  });
});
