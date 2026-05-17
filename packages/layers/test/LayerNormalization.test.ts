import { describe, expect, it } from "vitest";
import { LayerNormalization } from "../src/layers/LayerNormalization.js";
import { Matrix, engine } from "@oxide-js/core";
import { expectMatrixCloseTo, expectMatrixShape, mat } from "./helpers/matrix.js";

describe("LayerNormalization Layer Tests", () => {
  it("should create layer with default config", () => {
    const layer = new LayerNormalization({
      name: "ln_custom",
      epsilon: 1e-4,
      trainable: true
    });
    expect(layer.name).toBe("ln_custom");
    expect(layer.epsilon).toBe(1e-4);
    expect(layer.trainable).toBe(true);
  });

  it("should compute output shape", () => {
    const layer = new LayerNormalization();
    expect(layer.computeOutputShape([2, 4])).toEqual([2, 4]);
  });

  it("should build gamma and beta parameters correctly", () => {
    const layer = new LayerNormalization();
    layer.build([2, 3]);

    expect(layer.isBuilt).toBe(true);
    expect(layer.inputShape).toEqual([2, 3]);
    expect(layer.outputShape).toEqual([2, 3]);

    expect(layer.getParameter("gamma")?._shape).toEqual([1, 3]);
    expect(layer.getParameter("beta")?._shape).toEqual([1, 3]);
  });

  it("should perform forward pass standardizing the feature dimension (axis 1)", () => {
    const layer = new LayerNormalization({ epsilon: 1e-5 });
    layer.build([1, 3]);

    const x = mat([1.0, 2.0, 3.0], [1, 3]);
    const y = layer.forward(x);

    // Mean = 2, Var = 2/3, Std ≈ 0.8165
    // x1 = -1.2247, x2 = 0, x3 = 1.2247
    expectMatrixShape(y, [1, 3]);
    expectMatrixCloseTo(y, [-1.2247, 0.0, 1.2247], 3);
  });

  it("should perform backward pass via autodiff tape correctly", () => {
    const layer = new LayerNormalization();
    layer.build([2, 2]);

    const x = mat([1, 2, 3, 4], [2, 2]);
    x.requiresGrad = true;

    const gamma = layer.getParameter("gamma")!;
    const beta = layer.getParameter("beta")!;

    const tape = engine.grad(() => {
      return layer.forward(x);
    });

    x.clearGrad();
    gamma.clearGrad();
    beta.clearGrad();

    tape.backward(tape.result);

    // Check that gradients are computed for parameters and inputs
    expect(x.grad).toBeDefined();
    expect(gamma.grad).toBeDefined();
    expect(beta.grad).toBeDefined();

    expectMatrixShape(x.grad!, [2, 2]);
    expectMatrixShape(gamma.grad!, [1, 2]);
    expectMatrixShape(beta.grad!, [1, 2]);

    // Check that beta gradient sums gradOutput along rows -> [2.0, 2.0]
    expect(Array.from(beta.grad!._data)).toEqual([2.0, 2.0]);
  });

  it("should return empty trainable weights when frozen", () => {
    const layer = new LayerNormalization({ trainable: false });
    layer.build([2, 3]);
    expect(layer.trainableWeights.length).toBe(0);
    expect(layer.nonTrainableWeights.length).toBeGreaterThan(0);
  });

  it("should return config", () => {
    const layer = new LayerNormalization({
      name: "ln_keras",
      epsilon: 1e-3
    });
    const kConfig = layer.getKerasConfig();
    expect(kConfig.class_name).toBe("LayerNormalization");
    expect(kConfig.config.name).toBe("ln_keras");
    expect(kConfig.config.epsilon).toBe(1e-3);
  });

  it("should throw error for invalid input shape mismatch after building", () => {
    const layer = new LayerNormalization();
    layer.build([2, 3]);

    const xInvalid = mat([1, 2, 3, 4], [2, 2]);
    expect(() => layer.forward(xInvalid)).toThrow();
  });
});
