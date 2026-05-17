import { describe, expect, it } from "vitest";
import { MultiHeadAttention } from "../src/layers/MultiHeadAttention.js";
import { Matrix, engine } from "@oxide-js/core";
import { expectMatrixShape, mat } from "./helpers/matrix.js";

describe("MultiHeadAttention Layer Tests", () => {
  it("should create layer with default config", () => {
    const layer = new MultiHeadAttention({
      name: "mha_custom",
      numHeads: 2,
      keyDim: 4,
      useBias: true,
      trainable: true
    });
    expect(layer.name).toBe("mha_custom");
    expect(layer.numHeads).toBe(2);
    expect(layer.keyDim).toBe(4);
    expect(layer.useBias).toBe(true);
    expect(layer.trainable).toBe(true);
  });

  it("should compute output shape", () => {
    const layer = new MultiHeadAttention({ numHeads: 2, keyDim: 4, inputDim: 3, outputDim: 5, sequenceLength: 3 });
    expect(layer.computeOutputShape([2, 3, 3])).toEqual([6, 5]);
    expect(layer.computeOutputShape([6, 3])).toEqual([6, 5]);
  });

  it("should build parameters correctly", () => {
    const layer = new MultiHeadAttention({ numHeads: 2, keyDim: 4, valueDim: 3, outputDim: 5, sequenceLength: 2, inputDim: 2 });
    layer.build([4, 2]); // batch size = 2

    expect(layer.isBuilt).toBe(true);
    expect(layer.inputShape).toEqual([4, 2]);
    expect(layer.outputShape).toEqual([4, 5]);

    // wQ, wK: [inputDim, numHeads * keyDim] = [2, 8]
    expect(layer.wQ?._shape).toEqual([2, 8]);
    expect(layer.wK?._shape).toEqual([2, 8]);
    // wV: [inputDim, numHeads * valueDim] = [2, 6]
    expect(layer.wV?._shape).toEqual([2, 6]);
    // wO: [numHeads * valueDim, outputDim] = [6, 5]
    expect(layer.wO?._shape).toEqual([6, 5]);

    expect(layer.bQ?._shape).toEqual([8, 1]);
    expect(layer.bK?._shape).toEqual([8, 1]);
    expect(layer.bV?._shape).toEqual([6, 1]);
    expect(layer.bO?._shape).toEqual([5, 1]);
  });

  it("should perform forward pass correctly and backward parity", () => {
    const layer = new MultiHeadAttention({
      numHeads: 2,
      keyDim: 2,
      valueDim: 2,
      outputDim: 2,
      sequenceLength: 3,
      inputDim: 2,
      useBias: true
    });
    layer.build([3, 2]); // batch size = 1

    // Manually set weights for deterministic testing
    layer.wQ!._data.fill(0.1);
    layer.wK!._data.fill(0.05);
    layer.wV!._data.fill(0.15);
    layer.wO!._data.fill(0.2);

    layer.bQ!._data.fill(0.01);
    layer.bK!._data.fill(0.02);
    layer.bV!._data.fill(0.03);
    layer.bO!._data.fill(0.04);

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
    layer.wQ!.clearGrad();
    layer.wK!.clearGrad();
    layer.wV!.clearGrad();
    layer.wO!.clearGrad();
    layer.bQ!.clearGrad();
    layer.bK!.clearGrad();
    layer.bV!.clearGrad();
    layer.bO!.clearGrad();

    tape.backward(tape.result);

    expect(x.grad).toBeDefined();
    expect(layer.wQ!.grad).toBeDefined();
    expect(layer.wK!.grad).toBeDefined();
    expect(layer.wV!.grad).toBeDefined();
    expect(layer.wO!.grad).toBeDefined();
    expect(layer.bQ!.grad).toBeDefined();
    expect(layer.bK!.grad).toBeDefined();
    expect(layer.bV!.grad).toBeDefined();
    expect(layer.bO!.grad).toBeDefined();

    expectMatrixShape(x.grad!, [3, 2]);
    expectMatrixShape(layer.wQ!.grad!, [2, 4]);
    expectMatrixShape(layer.wK!.grad!, [2, 4]);
    expectMatrixShape(layer.wV!.grad!, [2, 4]);
    expectMatrixShape(layer.wO!.grad!, [4, 2]);
    expectMatrixShape(layer.bQ!.grad!, [4, 1]);
    expectMatrixShape(layer.bK!.grad!, [4, 1]);
    expectMatrixShape(layer.bV!.grad!, [4, 1]);
    expectMatrixShape(layer.bO!.grad!, [2, 1]);
  });

  it("should support setExternal to run Cross-Attention with different sequence lengths and custom trainability", () => {
    const layer = new MultiHeadAttention({
      numHeads: 2,
      keyDim: 2,
      valueDim: 2,
      outputDim: 3,
      sequenceLength: 2, // Lq = 2
      inputDim: 2,
      useBias: true
    });
    layer.build([2, 2]); // batch size = 1, Lq = 2

    const extQuery = mat([1.0, 0.5, -0.5, 1.0], [2, 2]); // Lq = 2
    const extKey = mat([0.5, 1.5, 0.0, -1.0, 1.0, 2.0, -0.5, 0.5], [4, 2]); // Lk = 4
    const extValue = mat([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [4, 2]); // Lk = 4

    layer.setExternal({
      query: extQuery,
      key: extKey,
      value: extValue,
      trainableQuery: true,
      trainableKey: false,
      trainableValue: true
    });

    // Verify trainability configurations
    expect(extQuery.requiresGrad).toBe(true);
    expect(extKey.requiresGrad).toBe(false);
    expect(extValue.requiresGrad).toBe(true);

    const trainableParams = layer.getTrainableParameters();
    expect(trainableParams.includes(extQuery)).toBe(true);
    expect(trainableParams.includes(extKey)).toBe(false);
    expect(trainableParams.includes(extValue)).toBe(true);

    // Forward pass
    const dummyInputs = mat([0, 0, 0, 0], [2, 2]);
    const out = layer.forward(dummyInputs);

    // output should have shape [B * Lq, outputDim] = [1 * 2, 3]
    expectMatrixShape(out, [2, 3]);

    const tape = engine.grad(() => {
      return layer.forward(dummyInputs);
    });

    extQuery.clearGrad();
    extValue.clearGrad();
    tape.backward(tape.result);

    expect(extQuery.grad).toBeDefined();
    expect(extValue.grad).toBeDefined();
    expectMatrixShape(extQuery.grad!, [2, 2]);
    expectMatrixShape(extValue.grad!, [4, 2]);
  });
});
