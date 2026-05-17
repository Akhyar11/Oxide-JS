import { describe, expect, it } from "vitest";
import { AveragePooling1D } from "../src/layers/AveragePooling1D.js";
import { Matrix, engine } from "@oxide-js/core";
import { expectMatrixCloseTo, expectMatrixShape, mat } from "./helpers/matrix.js";

describe("AveragePooling1D Layer Tests", () => {
  it("should create layer with custom config", () => {
    const layer = new AveragePooling1D({
      name: "avgpool1d_custom",
      poolSize: 3,
      strides: 2,
      padding: "same"
    });
    expect(layer.name).toBe("avgpool1d_custom");
    expect(layer.poolSize).toBe(3);
    expect(layer.strides).toBe(2);
    expect(layer.padding).toBe("same");
  });

  it("should compute output shape correctly for valid padding", () => {
    const layer = new AveragePooling1D({ poolSize: 2, strides: 2, padding: "valid" });
    // [batch, sequenceLength, inputDim] -> [batch * L_out, inputDim]
    expect(layer.computeOutputShape([2, 5, 3])).toEqual([4, 3]); // L_out = Math.floor((5-2)/2) + 1 = 2. rows = 2 * 2 = 4.
  });

  it("should compute output shape correctly for same padding", () => {
    const layer = new AveragePooling1D({ poolSize: 3, strides: 2, padding: "same" });
    // [batch, sequenceLength, inputDim] -> [batch * L_out, inputDim]
    expect(layer.computeOutputShape([2, 5, 4])).toEqual([6, 4]); // L_out = Math.ceil(5/2) = 3. rows = 2 * 3 = 6.
  });

  it("should perform forward pass mathematically correctly", () => {
    const layer = new AveragePooling1D({
      poolSize: 2,
      strides: 1,
      padding: "valid"
    });
    layer.build([1, 4, 2]); // B=1, L=4, C=2

    // Inputs: B=1, L=4, C=2
    // step 0: [1, 5]
    // step 1: [3, 2]
    // step 2: [2, 8]
    // step 3: [4, 6]
    const x = mat([
      1, 5,
      3, 2,
      2, 8,
      4, 6
    ], [4, 2]);

    const y = layer.forward(x);

    // L_out = 4 - 2 + 1 = 3. outputShape = [3, 2]
    expectMatrixShape(y, [3, 2]);

    // Output step 0 (window step 0 & 1):
    // Avg of [1, 3] = 2. Avg of [5, 2] = 3.5. -> [2, 3.5]
    // Output step 1 (window step 1 & 2):
    // Avg of [3, 2] = 2.5. Avg of [2, 8] = 5. -> [2.5, 5]
    // Output step 2 (window step 2 & 3):
    // Avg of [2, 4] = 3. Avg of [8, 6] = 7. -> [3, 7]

    expectMatrixCloseTo(y, [
      2, 3.5,
      2.5, 5,
      3, 7
    ]);
  });

  it("should calculate backward gradients correctly via tape", () => {
    const layer = new AveragePooling1D({
      poolSize: 2,
      strides: 1,
      padding: "valid"
    });
    layer.build([1, 4, 2]);

    const x = mat([
      1, 5,
      3, 2,
      2, 8,
      4, 6
    ], [4, 2]);
    x.requiresGrad = true;

    const tape = engine.grad(() => {
      return layer.forward(x);
    });

    x.clearGrad();

    // Upstream gradient shape [3, 2]
    const gradOutput = mat([
      1.0, 2.0,
      3.0, 4.0,
      5.0, 6.0
    ], [3, 2]);

    tape.backward(tape.result, gradOutput);

    // Average pooling backward trace:
    // c=0 (column 0):
    // - step 0 (window 0&1, count=2): gradOutput[0,0] = 1.0 -> adds 1.0/2 = 0.5 to gradX[0,0] and gradX[1,0]
    // - step 1 (window 1&2, count=2): gradOutput[1,0] = 3.0 -> adds 3.0/2 = 1.5 to gradX[1,0] and gradX[2,0]
    // - step 2 (window 2&3, count=2): gradOutput[2,0] = 5.0 -> adds 5.0/2 = 2.5 to gradX[2,0] and gradX[3,0]
    // Total gradX for c=0:
    // gradX[0,0] = 0.5
    // gradX[1,0] = 0.5 + 1.5 = 2.0
    // gradX[2,0] = 1.5 + 2.5 = 4.0
    // gradX[3,0] = 2.5
    //
    // c=1 (column 1):
    // - step 0 (window 0&1, count=2): gradOutput[0,1] = 2.0 -> adds 2.0/2 = 1.0 to gradX[0,1] and gradX[1,1]
    // - step 1 (window 1&2, count=2): gradOutput[1,1] = 4.0 -> adds 4.0/2 = 2.0 to gradX[1,1] and gradX[2,1]
    // - step 2 (window 2&3, count=2): gradOutput[2,1] = 6.0 -> adds 6.0/2 = 3.0 to gradX[2,1] and gradX[3,1]
    // Total gradX for c=1:
    // gradX[0,1] = 1.0
    // gradX[1,1] = 1.0 + 2.0 = 3.0
    // gradX[2,1] = 2.0 + 3.0 = 5.0
    // gradX[3,1] = 3.0

    expect(x.grad).toBeDefined();
    expectMatrixShape(x.grad!, [4, 2]);
    expectMatrixCloseTo(x.grad!, [
      0.5, 1.0,
      2.0, 3.0,
      4.0, 5.0,
      2.5, 3.0
    ]);
  });

  it("should return config correctly", () => {
    const layer = new AveragePooling1D({
      name: "avgpool1d_test",
      poolSize: 3,
      strides: 2,
      padding: "same",
      sequenceLength: 10,
      inputDim: 4
    });

    const kConfig = layer.getKerasConfig();
    expect(kConfig.class_name).toBe("AveragePooling1D");
    expect(kConfig.config.name).toBe("avgpool1d_test");
    expect(kConfig.config.poolSize).toBe(3);
    expect(kConfig.config.strides).toBe(2);
    expect(kConfig.config.padding).toBe("same");
    expect(kConfig.config.sequenceLength).toBe(10);
    expect(kConfig.config.inputDim).toBe(4);
  });
});
