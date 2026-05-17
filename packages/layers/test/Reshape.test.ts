import { describe, expect, it } from "vitest";
import { Reshape } from "../src/layers/Reshape.js";
import { Matrix, engine } from "@oxide-js/core";
import { expectMatrixCloseTo, expectMatrixShape, mat } from "./helpers/matrix.js";

describe("Reshape Layer Tests", () => {
  it("should create layer with default config", () => {
    const layer = new Reshape({
      name: "reshape_custom",
      targetShape: [6]
    });
    expect(layer.name).toBe("reshape_custom");
    expect(layer.targetShape).toEqual([6]);
    expect(layer.trainable).toBe(true);
  });

  it("should compute output shape", () => {
    const layer = new Reshape({ targetShape: [6] });
    expect(layer.computeOutputShape([2, 6])).toEqual([2, 6]);

    const inferredLayer = new Reshape({ targetShape: [2, -1] });
    expect(inferredLayer.computeOutputShape([2, 6])).toEqual([2, 2, 3]);
  });

  it("should build correctly", () => {
    const layer = new Reshape({ targetShape: [6] });
    layer.build([2, 6]);
    expect(layer.isBuilt).toBe(true);
    expect(layer.inputShape).toEqual([2, 6]);
    expect(layer.outputShape).toEqual([2, 6]);
  });

  it("should reshape matrix correctly in forward pass", () => {
    const layer = new Reshape({ targetShape: [6] });
    const x = mat([1, 2, 3, 4, 5, 6], [1, 6]);
    const y = layer.forward(x);

    expectMatrixShape(y, [1, 6]);
    expectMatrixCloseTo(y, [1, 2, 3, 4, 5, 6]);
  });

  it("should perform backward pass via autodiff tape correctly", () => {
    const layer = new Reshape({ targetShape: [4, 1] });
    layer.build([1, 4]);

    const x = mat([10, 20, 30, 40], [1, 4]);
    x.requiresGrad = true;

    const tape = engine.grad(() => {
      return layer.forward(x);
    });

    x.clearGrad();
    const gradOutput = mat([10, 10, 10, 10], [1, 4]);
    tape.backward(tape.result, gradOutput);

    expect(x.grad).toBeDefined();
    expectMatrixShape(x.grad!, [1, 4]);
    expectMatrixCloseTo(x.grad!, [10, 10, 10, 10]);
  });

  it("should return empty trainable weights", () => {
    const layer = new Reshape({ targetShape: [2, 2] });
    layer.build([2, 4]);
    expect(layer.trainableWeights.length).toBe(0);
    expect(layer.nonTrainableWeights.length).toBe(0);
  });

  it("should return config", () => {
    const layer = new Reshape({
      name: "reshape_keras",
      targetShape: [2, 5]
    });
    const kConfig = layer.getKerasConfig();
    expect(kConfig.class_name).toBe("Reshape");
    expect(kConfig.config.name).toBe("reshape_keras");
    expect(kConfig.config.targetShape).toEqual([2, 5]);
  });

  it("should throw error when total size does not match target shape", () => {
    const layer = new Reshape({ targetShape: [5] });
    expect(() => {
      layer.computeOutputShape([2, 3]); // 3 elements !== 5 elements
    }).toThrow();
  });
});
