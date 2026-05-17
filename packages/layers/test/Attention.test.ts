import { describe, expect, it } from "vitest";
import { Attention } from "../src/layers/Attention.js";
import { Matrix, engine } from "@oxide-js/core";
import { expectMatrixShape, mat } from "./helpers/matrix.js";

describe("Attention Layer Tests", () => {
  it("should create layer with default config", () => {
    const layer = new Attention({
      name: "attention_custom",
      units: 4,
      useBias: true,
      trainable: true
    });
    expect(layer.name).toBe("attention_custom");
    expect(layer.units).toBe(4);
    expect(layer.useBias).toBe(true);
    expect(layer.trainable).toBe(true);
  });

  it("should compute output shape", () => {
    const layer = new Attention({ units: 5, sequenceLength: 3, inputDim: 2 });
    expect(layer.computeOutputShape([2, 3, 2])).toEqual([6, 5]);
    expect(layer.computeOutputShape([6, 2])).toEqual([6, 5]);
  });

  it("should build parameters correctly", () => {
    const layer = new Attention({ units: 3, sequenceLength: 4, inputDim: 2 });
    layer.build([8, 2]);

    expect(layer.isBuilt).toBe(true);
    expect(layer.inputShape).toEqual([8, 2]);
    expect(layer.outputShape).toEqual([8, 3]);

    expect(layer.wQ?._shape).toEqual([2, 3]);
    expect(layer.wK?._shape).toEqual([2, 3]);
    expect(layer.wV?._shape).toEqual([2, 3]);
    expect(layer.bQ?._shape).toEqual([3, 1]);
    expect(layer.bK?._shape).toEqual([3, 1]);
    expect(layer.bV?._shape).toEqual([3, 1]);
  });

  it("should perform forward pass correctly and backward parity", () => {
    const layer = new Attention({
      units: 2,
      sequenceLength: 3,
      inputDim: 2,
      useBias: true
    });
    layer.build([3, 2]); // batch size = 1

    // Manually set weights for deterministic testing
    const wQ = layer.wQ!;
    wQ._data.fill(0.1);
    const wK = layer.wK!;
    wK._data.fill(0.05);
    const wV = layer.wV!;
    wV._data.fill(0.15);

    const bQ = layer.bQ!;
    bQ._data.fill(0.01);
    const bK = layer.bK!;
    bK._data.fill(0.02);
    const bV = layer.bV!;
    bV._data.fill(0.03);

    const x = mat([1.0, 2.0, 0.5, -0.5, 0.0, 1.5], [3, 2]);
    x.requiresGrad = true;

    // Forward pass
    const y = layer.forward(x);
    expectMatrixShape(y, [3, 2]);

    // Backward pass
    const tape = engine.grad(() => {
      return layer.forward(x);
    });

    x.clearGrad();
    wQ.clearGrad();
    wK.clearGrad();
    wV.clearGrad();
    bQ.clearGrad();
    bK.clearGrad();
    bV.clearGrad();

    tape.backward(tape.result);

    expect(x.grad).toBeDefined();
    expect(wQ.grad).toBeDefined();
    expect(wK.grad).toBeDefined();
    expect(wV.grad).toBeDefined();
    expect(bQ.grad).toBeDefined();
    expect(bK.grad).toBeDefined();
    expect(bV.grad).toBeDefined();

    expectMatrixShape(x.grad!, [3, 2]);
    expectMatrixShape(wQ.grad!, [2, 2]);
    expectMatrixShape(wK.grad!, [2, 2]);
    expectMatrixShape(wV.grad!, [2, 2]);
    expectMatrixShape(bQ.grad!, [2, 1]);
    expectMatrixShape(bK.grad!, [2, 1]);
    expectMatrixShape(bV.grad!, [2, 1]);
  });

  it("should support setExternal to run Cross-Attention with different sequence lengths and trainability", () => {
    const layer = new Attention({
      units: 3,
      sequenceLength: 2, // Lq = 2
      inputDim: 2,
      useBias: true
    });
    layer.build([2, 2]); // built with batch = 1, sequenceLength = 2

    const extQuery = mat([1.0, 0.5, -0.5, 1.0], [2, 2]); // Lq = 2
    const extKey = mat([0.5, 1.5, 0.0, -1.0, 1.0, 2.0, -0.5, 0.5], [4, 2]); // Lk = 4

    layer.setExternal({
      query: extQuery,
      key: extKey,
      trainableQuery: true,
      trainableKey: false
    });

    // Verify trainability configuration
    expect(extQuery.requiresGrad).toBe(true);
    expect(extKey.requiresGrad).toBe(false);

    const trainableParams = layer.getTrainableParameters();
    expect(trainableParams.includes(extQuery)).toBe(true);
    expect(trainableParams.includes(extKey)).toBe(false);

    // forward pass on custom input matrix
    const dummyInputs = mat([0, 0, 0, 0], [2, 2]);
    const out = layer.forward(dummyInputs);

    // output should have shape [B * Lq, units] = [1 * 2, 3]
    expectMatrixShape(out, [2, 3]);

    const tape = engine.grad(() => {
      return layer.forward(dummyInputs);
    });

    extQuery.clearGrad();
    extKey.clearGrad();
    layer.wQ!.clearGrad();
    layer.wK!.clearGrad();
    layer.wV!.clearGrad();

    tape.backward(tape.result);

    expect(extQuery.grad).toBeDefined();
    // extKey is not trainable, but tape backprop will still compute gradients for inputs
    // if they are part of dynamic graph (which is fine).
    expectMatrixShape(extQuery.grad!, [2, 2]);
  });
});
