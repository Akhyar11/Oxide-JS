import { describe, expect, it } from "vitest";
import { BatchNormalization } from "../src/layers/BatchNormalization.js";
import { Matrix, engine } from "@oxide-js/core";
import { expectMatrixCloseTo, expectMatrixShape, mat } from "./helpers/matrix.js";

describe("BatchNormalization Layer Tests", () => {
  it("should create layer with default config", () => {
    const layer = new BatchNormalization({
      name: "bn_custom",
      epsilon: 1e-4,
      momentum: 0.95
    });
    expect(layer.name).toBe("bn_custom");
    expect(layer.epsilon).toBe(1e-4);
    expect(layer.momentum).toBe(0.95);
    expect(layer.trainable).toBe(true);
  });

  it("should compute output shape", () => {
    const layer = new BatchNormalization();
    expect(layer.computeOutputShape([3, 4])).toEqual([3, 4]);
  });

  it("should build gamma, beta, movingMean, and movingVariance parameters", () => {
    const layer = new BatchNormalization();
    layer.build([4, 3]);

    expect(layer.isBuilt).toBe(true);
    expect(layer.inputShape).toEqual([4, 3]);
    expect(layer.outputShape).toEqual([4, 3]);

    expect(layer.getParameter("gamma")).toBeDefined();
    expect(layer.getParameter("beta")).toBeDefined();
    expect(layer.getParameter("movingMean")).toBeDefined();
    expect(layer.getParameter("movingVariance")).toBeDefined();
  });

  it("should perform forward pass correctly under training and evaluation modes", () => {
    const layer = new BatchNormalization({ epsilon: 1e-5 });
    layer.build([2, 2]);

    // 1. Training mode
    layer.training = true;
    const x = mat([1.0, 3.0, 2.0, 5.0], [2, 2]);
    const yTrain = layer.forward(x);
    expectMatrixShape(yTrain, [2, 2]);
    expectMatrixCloseTo(yTrain, [-0.99998, -0.99999, 0.99998, 0.99999], 4);

    // 2. Evaluation mode using tracked statistics
    layer.training = false;
    layer.getParameter("movingMean")!._data.set([2.0, 4.0]);
    layer.getParameter("movingVariance")!._data.set([4.0, 9.0]);
    const xEval = mat([4.0, 10.0], [1, 2]);
    const yEval = layer.forward(xEval);
    expectMatrixCloseTo(yEval, [0.99999, 1.99999], 4);
  });

  it("should perform backward pass via autodiff tape correctly", () => {
    const layer = new BatchNormalization();
    layer.build([2, 2]);
    layer.training = true;

    const x = mat([1.0, 2.0, 3.0, 4.0], [2, 2]);
    x.requiresGrad = true;

    const beta = layer.getParameter("beta")!;
    const tape = engine.grad(() => {
      return layer.forward(x);
    });

    beta.clearGrad();
    x.clearGrad();

    tape.backward(tape.result);

    // Check that gradients propagated
    expect(x.grad).toBeDefined();
    expect(beta.grad).toBeDefined();
    expect(Array.from(beta.grad!._data)).toEqual([2.0, 2.0]);
  });

  it("should handle trainableWeights and nonTrainableWeights correctly", () => {
    const layer = new BatchNormalization();
    layer.build([2, 2]);

    expect(layer.trainableWeights.length).toBe(2); // gamma, beta
    expect(layer.nonTrainableWeights.length).toBe(2); // movingMean, movingVariance

    // If layer is frozen
    layer.trainable = false;
    expect(layer.trainableWeights.length).toBe(0);
    expect(layer.nonTrainableWeights.length).toBe(4);
  });

  it("should return config", () => {
    const layer = new BatchNormalization({
      name: "bn_keras",
      epsilon: 1e-3,
      momentum: 0.95
    });
    const kConfig = layer.getKerasConfig();
    expect(kConfig.class_name).toBe("BatchNormalization");
    expect(kConfig.config.name).toBe("bn_keras");
    expect(kConfig.config.epsilon).toBe(1e-3);
    expect(kConfig.config.momentum).toBe(0.95);
  });

  it("should throw error for invalid input shape mismatch after building", () => {
    const layer = new BatchNormalization();
    layer.build([2, 3]);

    const xInvalid = mat([1, 2, 3, 4], [2, 2]);
    expect(() => layer.forward(xInvalid)).toThrow();
  });
});
