import { describe, expect, it } from "vitest";
import { Activation } from "../src/layers/Activation.js";
import { Matrix, engine } from "@oxide-js/core";
import { expectMatrixCloseTo, expectMatrixShape, mat } from "./helpers/matrix.js";

describe("Activation Layer Tests", () => {
  it("should create layer with default config", () => {
    const layer = new Activation({
      name: "custom_act",
      activation: "sigmoid"
    });
    expect(layer.name).toBe("custom_act");
    expect(layer.activation).toBe("sigmoid");
    expect(layer.trainable).toBe(true);
  });

  it("should compute output shape", () => {
    const layer = new Activation("relu");
    expect(layer.computeOutputShape([2, 5])).toEqual([2, 5]);
  });

  it("should build correctly", () => {
    const layer = new Activation("tanh");
    layer.build([3, 4]);
    expect(layer.isBuilt).toBe(true);
    expect(layer.inputShape).toEqual([3, 4]);
    expect(layer.outputShape).toEqual([3, 4]);
  });

  it("should perform forward pass for different activation functions", () => {
    // 1. ReLU
    const reluLayer = new Activation("relu");
    const xRelu = mat([-1, 0, 2, 3], [2, 2]);
    const yRelu = reluLayer.forward(xRelu);
    expectMatrixCloseTo(yRelu, [0, 0, 2, 3]);

    // 2. Sigmoid
    const sigmoidLayer = new Activation("sigmoid");
    const xSigmoid = mat([0], [1, 1]);
    const ySigmoid = sigmoidLayer.forward(xSigmoid);
    expectMatrixCloseTo(ySigmoid, [0.5]);

    // 3. Tanh
    const tanhLayer = new Activation("tanh");
    const xTanh = mat([0], [1, 1]);
    const yTanh = tanhLayer.forward(xTanh);
    expectMatrixCloseTo(yTanh, [0]);

    // 4. Linear
    const linearLayer = new Activation("linear");
    const xLinear = mat([1, -2], [1, 2]);
    const yLinear = linearLayer.forward(xLinear);
    expectMatrixCloseTo(yLinear, [1, -2]);

    // 5. Softmax
    const softmaxLayer = new Activation("softmax");
    const xSoftmax = mat([1, 2, 3], [1, 3]);
    const ySoftmax = softmaxLayer.forward(xSoftmax);
    // Total per row should be close to 1
    const sum = ySoftmax._data.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 5);
  });

  it("should perform backward pass via autodiff tape correctly", () => {
    const layer = new Activation("relu");
    const x = mat([-1, 2], [1, 2]);
    x.requiresGrad = true;

    const tape = engine.grad(() => {
      return layer.forward(x);
    });

    x.clearGrad();
    // Simulate gradOutput = [10, 10]
    const gradOutput = mat([10, 10], [1, 2]);
    tape.backward(tape.result, gradOutput);

    // Relu derivative:
    // x = -1 -> derivative = 0 -> gradInput = 10 * 0 = 0
    // x = 2 -> derivative = 1 -> gradInput = 10 * 1 = 10
    expect(x.grad).toBeDefined();
    expectMatrixCloseTo(x.grad!, [0, 10]);
  });

  it("should return empty trainable weights", () => {
    const layer = new Activation("relu");
    layer.build([2, 2]);
    expect(layer.trainableWeights.length).toBe(0);
    expect(layer.nonTrainableWeights.length).toBe(0);
  });

  it("should return config", () => {
    const layer = new Activation({
      name: "act_config",
      activation: "lRelu"
    });
    const kConfig = layer.getKerasConfig();
    expect(kConfig.class_name).toBe("Activation");
    expect(kConfig.config.name).toBe("act_config");
    expect(kConfig.config.activation).toBe("lRelu");
  });

  it("should throw error for invalid activation config", () => {
    expect(() => new Activation("invalid_activation_name" as any)).toThrow();
  });
});
