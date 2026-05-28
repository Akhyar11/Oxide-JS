import { describe, expect, it } from "vitest";
import { Flatten } from "../src/layers/Flatten.js";
import { Matrix, engine } from "@oxide-js/core";
import { expectMatrixCloseTo, expectMatrixShape, mat } from "./helpers/matrix.js";

describe("Flatten Layer Tests", () => {
  it("should create layer with default config", () => {
    const layer = new Flatten({
      name: "flatten_custom",
      trainable: false
    });
    expect(layer.name).toBe("flatten_custom");
    expect(layer.trainable).toBe(false);
  });

  it("should compute output shape", () => {
    const layer = new Flatten();
    expect(layer.computeOutputShape([2, 3, 4])).toEqual([1, 24]);
    expect(layer.computeOutputShape([5, 2, 2, 3])).toEqual([1, 60]);
  });

  it("should build correctly", () => {
    const layer = new Flatten();
    layer.build([2, 3, 4]);
    expect(layer.isBuilt).toBe(true);
    expect(layer.inputShape).toEqual([2, 3, 4]);
    expect(layer.outputShape).toEqual([1, 24]);
  });

  it("should perform forward pass reshaping to 2D but keeping identical data values", () => {
    const layer = new Flatten();
    const x = mat([1, 2, 3, 4, 5, 6], [2, 3]); // logically 2D
    const y = layer.forward(x);

    expectMatrixShape(y, [1, 6]);
    expectMatrixCloseTo(y, [1, 2, 3, 4, 5, 6]);
  });

  it("should perform backward pass via autodiff tape correctly", () => {
    const layer = new Flatten();
    layer.build([2, 3]);

    const x = mat([1, 2, 3, 4, 5, 6], [2, 3]);
    x.requiresGrad = true;

    const tape = engine.grad(() => {
      return layer.forward(x);
    });

    x.clearGrad();
    const gradOutput = mat([10, 10, 10, 10, 10, 10], [1, 6]);
    tape.backward(tape.result, gradOutput);

    expect(x.grad).toBeDefined();
    expectMatrixShape(x.grad!, [2, 3]);
    expectMatrixCloseTo(x.grad!, [10, 10, 10, 10, 10, 10]);
  });

  it("should return empty trainable weights", () => {
    const layer = new Flatten();
    layer.build([2, 3]);
    expect(layer.trainableWeights.length).toBe(0);
    expect(layer.nonTrainableWeights.length).toBe(0);
  });

  it("should return config", () => {
    const layer = new Flatten({ name: "flatten_keras" });
    const kConfig = layer.getKerasConfig();
    expect(kConfig.class_name).toBe("Flatten");
    expect(kConfig.config.name).toBe("flatten_keras");
  });

  it("should throw error for invalid input shape mismatch after building", () => {
    const layer = new Flatten();
    layer.build([2, 3, 4]); // expected features total 12

    const xInvalid = mat([1, 2, 3, 4], [2, 2]); // features total 2
    expect(() => layer.forward(xInvalid)).toThrow();
  });
});
