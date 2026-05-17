import { describe, expect, it } from "vitest";
import { Dropout } from "../src/layers/Dropout.js";
import { Matrix, engine } from "@oxide-js/core";
import { expectMatrixCloseTo, expectMatrixShape, mat } from "./helpers/matrix.js";

describe("Dropout Layer Tests", () => {
  it("should create layer with default config", () => {
    const layer = new Dropout({
      name: "dropout_custom",
      rate: 0.2
    });
    expect(layer.name).toBe("dropout_custom");
    expect(layer.rate).toBe(0.2);
    expect(layer.trainable).toBe(true);
  });

  it("should compute output shape", () => {
    const layer = new Dropout({ rate: 0.5 });
    expect(layer.computeOutputShape([2, 5])).toEqual([2, 5]);
  });

  it("should build correctly", () => {
    const layer = new Dropout({ rate: 0.25 });
    layer.build([3, 4]);
    expect(layer.isBuilt).toBe(true);
    expect(layer.inputShape).toEqual([3, 4]);
    expect(layer.outputShape).toEqual([3, 4]);
  });

  it("should perform forward pass correctly under training and evaluation modes", () => {
    const layer = new Dropout({ rate: 0.5 });
    layer.build([2, 2]);

    // 1. Evaluation mode (eval) - output must match input exactly
    layer.eval();
    const x = mat([1, 2, 3, 4], [2, 2]);
    const yEval = layer.forward(x);
    expectMatrixCloseTo(yEval, [1, 2, 3, 4]);

    // 2. Training mode (train) - elements must be scaled by 1/(1-rate) = 2 or zeroed out
    layer.train();
    const yTrain = layer.forward(x);
    expectMatrixShape(yTrain, [2, 2]);

    for (let i = 0; i < yTrain._data.length; i++) {
      const val = yTrain._data[i];
      const orig = x._data[i];
      if (val !== 0) {
        expect(val).toBeCloseTo(orig * 2, 5);
      }
    }
  });

  it("should perform backward pass via autodiff tape correctly", () => {
    const layer = new Dropout({ rate: 0.5 });
    layer.build([2, 2]);
    layer.train();

    const x = mat([1, 1, 1, 1], [2, 2]);
    x.requiresGrad = true;

    const tape = engine.grad(() => {
      return layer.forward(x);
    });

    x.clearGrad();
    tape.backward(tape.result);

    // D/dx output = mask (which contains either 0 or 2)
    expect(x.grad).toBeDefined();
    expectMatrixShape(x.grad!, [2, 2]);

    for (let i = 0; i < x.grad!._data.length; i++) {
      const g = x.grad!._data[i];
      expect(g === 0 || g === 2).toBe(true);
    }
  });

  it("should return empty trainable weights", () => {
    const layer = new Dropout({ rate: 0.5 });
    layer.build([2, 2]);
    expect(layer.trainableWeights.length).toBe(0);
    expect(layer.nonTrainableWeights.length).toBe(0);
  });

  it("should return config", () => {
    const layer = new Dropout({
      name: "dropout_keras",
      rate: 0.1
    });
    const kConfig = layer.getKerasConfig();
    expect(kConfig.class_name).toBe("Dropout");
    expect(kConfig.config.name).toBe("dropout_keras");
    expect(kConfig.config.rate).toBe(0.1);
  });

  it("should throw error for invalid rate config", () => {
    expect(() => new Dropout({ rate: -0.1 })).toThrow();
    expect(() => new Dropout({ rate: 1.0 })).toThrow();
  });
});
