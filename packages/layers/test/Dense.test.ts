import { describe, expect, it } from "vitest";
import { Dense } from "../src/layers/Dense.js";
import { Matrix, engine } from "@oxide-js/core";
import { expectMatrixCloseTo, expectMatrixShape, mat } from "./helpers/matrix.js";

describe("Dense Layer Tests", () => {
  it("should create layer with default config", () => {
    const layer = new Dense({
      name: "dense_custom",
      units: 5,
      useBias: true,
      activation: "relu",
      trainable: true
    });
    expect(layer.name).toBe("dense_custom");
    expect(layer.units).toBe(5);
    expect(layer.useBias).toBe(true);
    expect(layer.activation).toBe("relu");
    expect(layer.trainable).toBe(true);
  });

  it("should compute output shape", () => {
    const layer = new Dense({ units: 10 });
    expect(layer.computeOutputShape([2, 4])).toEqual([2, 10]);
  });

  it("should build weight and bias correctly", () => {
    const layer = new Dense({ units: 3, useBias: true });
    layer.build([2, 4]);

    expect(layer.isBuilt).toBe(true);
    expect(layer.inputShape).toEqual([2, 4]);
    expect(layer.outputShape).toEqual([2, 3]);

    const kernel = layer.getParameter("kernel");
    const bias = layer.getParameter("bias");

    expect(kernel).toBeDefined();
    expect(bias).toBeDefined();
    expect(kernel?._shape).toEqual([4, 3]); // [inFeatures, units]
    expect(bias?._shape).toEqual([3, 1]); // [units, 1]
  });

  it("should perform forward pass with correct output shape and values", () => {
    const layer = new Dense({
      units: 2,
      useBias: true,
      activation: "relu"
    });

    layer.build([1, 3]);

    // Deterministic parameters
    layer.getParameter("kernel")!._data.set([1, 2, 3, 4, 5, 6]); // kernel [3 x 2]
    layer.getParameter("bias")!._data.set([-2, 10]); // bias [2 x 1]

    const x = mat([1, 1, 1], [1, 3]);
    const y = layer.forward(x);

    // [1, 1, 1] * [[1, 2], [3, 4], [5, 6]] = [9, 12]
    // Add bias: [9 - 2, 12 + 10] = [7, 22]
    // ReLU([7, 22]) = [7, 22]
    expectMatrixShape(y, [1, 2]);
    expectMatrixCloseTo(y, [7, 22]);
  });

  it("should perform backward pass via autodiff tape correctly", () => {
    const layer = new Dense({
      units: 2,
      useBias: true,
      activation: "linear"
    });

    layer.build([2, 3]);

    // Deterministic parameters
    layer.getParameter("kernel")!._data.set([1, 2, 3, 4, 5, 6]); // [3 x 2]
    layer.getParameter("bias")!._data.set([0.5, -0.5]); // [2 x 1]

    const x = mat([1, 0, 1, 0, 1, 0], [2, 3]);
    x.requiresGrad = true;

    const kernel = layer.getParameter("kernel")!;
    const bias = layer.getParameter("bias")!;

    const tape = engine.grad(() => {
      return layer.forward(x);
    });

    kernel.clearGrad();
    bias.clearGrad();
    x.clearGrad();

    // Backward pass with output gradient all ones
    const gradOutput = mat([1, 1, 1, 1], [2, 2]);
    tape.backward(tape.result, gradOutput);

    // Verify gradients computed by the autodiff engine
    expect(x.grad).toBeDefined();
    expect(kernel.grad).toBeDefined();
    expect(bias.grad).toBeDefined();

    expectMatrixShape(x.grad!, [2, 3]);
    expectMatrixShape(kernel.grad!, [3, 2]);
    expectMatrixShape(bias.grad!, [2, 1]);

    // Beta grad (bias) should be sum of grad output along rows -> [2.0, 2.0]
    expect(Array.from(bias.grad!._data)).toEqual([2.0, 2.0]);
  });

  it("should return empty trainable weights when frozen", () => {
    const layer = new Dense({
      units: 2,
      trainable: false
    });

    layer.build([2, 3]);

    expect(layer.trainableWeights.length).toBe(0);
    expect(layer.nonTrainableWeights.length).toBeGreaterThan(0);
  });

  it("should reject unknown setWeights entries by default and allow non-strict loading", () => {
    const layer = new Dense({ units: 2 });
    layer.build([1, 3]);

    const unknownWeight = {
      name: "dense/extra",
      shape: [1, 2],
      data: new Float32Array([7, 8])
    };

    expect(() => layer.setWeights([unknownWeight])).toThrow("Parameter 'extra' tidak dikenali");

    layer.setWeights([unknownWeight], { strict: false });
    expect(layer.getParameter("extra")).toBeDefined();
    expect(layer.getParameter("extra")?._shape).toEqual([1, 2]);
  });

  it("should return config", () => {
    const layer = new Dense({
      name: "dense_keras",
      units: 16,
      activation: "tanh",
      useBias: false
    });

    const kConfig = layer.getKerasConfig();
    expect(kConfig.class_name).toBe("Dense");
    expect(kConfig.config.name).toBe("dense_keras");
    expect(kConfig.config.units).toBe(16);
    expect(kConfig.config.activation).toBe("tanh");
    expect(kConfig.config.useBias).toBe(false);
  });

  it("should throw error for invalid input shape mismatch after building", () => {
    const layer = new Dense({ units: 2 });
    layer.build([2, 3]);

    const xInvalid = mat([1, 2, 3, 4], [2, 2]);
    expect(() => layer.forward(xInvalid)).toThrow();
  });
});
