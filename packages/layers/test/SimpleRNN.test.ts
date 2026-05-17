import { describe, expect, it } from "vitest";
import { SimpleRNN } from "../src/layers/SimpleRNN.js";
import { Matrix, engine } from "@oxide-js/core";
import { expectMatrixCloseTo, expectMatrixShape, mat } from "./helpers/matrix.js";

describe("SimpleRNN Layer Tests", () => {
  it("should create layer with default config", () => {
    const layer = new SimpleRNN({
      name: "rnn_custom",
      units: 4,
      activation: "tanh",
      useBias: true,
      returnSequences: true,
      trainable: true
    });
    expect(layer.name).toBe("rnn_custom");
    expect(layer.units).toBe(4);
    expect(layer.activation).toBe("tanh");
    expect(layer.useBias).toBe(true);
    expect(layer.returnSequences).toBe(true);
    expect(layer.trainable).toBe(true);
  });

  it("should compute output shape when returnSequences = false", () => {
    const layer = new SimpleRNN({ units: 5, returnSequences: false, sequenceLength: 3, inputDim: 2 });
    expect(layer.computeOutputShape([2, 3, 2])).toEqual([2, 5]);
    expect(layer.computeOutputShape([6, 2])).toEqual([2, 5]);
  });

  it("should compute output shape when returnSequences = true", () => {
    const layer = new SimpleRNN({ units: 5, returnSequences: true, sequenceLength: 3, inputDim: 2 });
    expect(layer.computeOutputShape([2, 3, 2])).toEqual([6, 5]);
    expect(layer.computeOutputShape([6, 2])).toEqual([6, 5]);
  });

  it("should build parameters correctly", () => {
    const layer = new SimpleRNN({ units: 3, sequenceLength: 4, inputDim: 2 });
    layer.build([8, 2]);

    expect(layer.isBuilt).toBe(true);
    expect(layer.inputShape).toEqual([8, 2]);
    expect(layer.outputShape).toEqual([2, 3]); // returnSequences = false (default)

    expect(layer.kernel?._shape).toEqual([2, 3]);
    expect(layer.recurrentKernel?._shape).toEqual([3, 3]);
    expect(layer.bias?._shape).toEqual([3, 1]);
  });

  it("should perform forward pass correctly in JS and native parity", () => {
    const layer = new SimpleRNN({
      units: 2,
      sequenceLength: 3,
      inputDim: 2,
      activation: "relu",
      useBias: true
    });
    layer.build([3, 2]); // batch size = 1

    // Manually set kernel and recurrentKernel to specific weights for deterministic test
    const kernel = layer.kernel!;
    kernel._data.set([0.5, 0.8, -0.2, 0.4]);
    const recurrentKernel = layer.recurrentKernel!;
    recurrentKernel._data.set([0.1, 0.2, -0.3, 0.4]);
    const bias = layer.bias!;
    bias._data.set([0.1, -0.1]);

    const x = mat([1.0, 2.0, 0.5, -0.5, 0.0, 1.5], [3, 2]);

    // Forward pass
    const y = layer.forward(x);
    expectMatrixShape(y, [1, 2]);

    // We can also test that gradient computation works correctly
    x.requiresGrad = true;
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
    expectMatrixShape(kernel.grad!, [2, 2]);
    expectMatrixShape(recurrentKernel.grad!, [2, 2]);
    expectMatrixShape(bias.grad!, [2, 1]);
  });

  it("should perform forward and backward with returnSequences = true", () => {
    const layer = new SimpleRNN({
      units: 2,
      sequenceLength: 2,
      inputDim: 2,
      activation: "tanh",
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

  it("should support frozen layer status correctly", () => {
    const layer = new SimpleRNN({ units: 2, sequenceLength: 2, inputDim: 2, trainable: false });
    layer.build([2, 2]);

    expect(layer.trainableWeights.length).toBe(0);
    expect(layer.nonTrainableWeights.length).toBeGreaterThan(0);
  });

  it("should generate proper Keras configuration metadata", () => {
    const layer = new SimpleRNN({
      units: 4,
      activation: "sigmoid",
      returnSequences: true
    });
    const config = layer.getKerasConfig();
    expect(config.class_name).toBe("SimpleRNN");
    expect(config.config.units).toBe(4);
    expect(config.config.activation).toBe("sigmoid");
    expect(config.config.returnSequences).toBe(true);
  });
});
