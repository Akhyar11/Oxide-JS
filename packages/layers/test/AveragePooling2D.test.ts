import { describe, expect, it } from "vitest";
import { AveragePooling2D } from "../src/layers/AveragePooling2D.js";
import { Matrix, engine } from "@oxide-js/core";
import { expectMatrixCloseTo, expectMatrixShape, mat } from "./helpers/matrix.js";

describe("AveragePooling2D Layer Tests", () => {
  it("should create layer with custom config", () => {
    const layer = new AveragePooling2D({
      name: "avgpool2d_custom",
      poolSize: [2, 3],
      strides: [1, 2],
      padding: "same"
    });
    expect(layer.name).toBe("avgpool2d_custom");
    expect(layer.poolSize).toEqual([2, 3]);
    expect(layer.strides).toEqual([1, 2]);
    expect(layer.padding).toBe("same");
  });

  it("should compute output shape correctly for valid padding", () => {
    const layer = new AveragePooling2D({ poolSize: 2, strides: 2, padding: "valid" });
    // [batch, height, width, channels] -> [batch * H_out * W_out, channels]
    // H_out = Math.floor((6-2)/2) + 1 = 3. W_out = 3. rows = 2 * 3 * 3 = 18.
    expect(layer.computeOutputShape([2, 6, 6, 3])).toEqual([18, 3]);
  });

  it("should compute output shape correctly for same padding", () => {
    const layer = new AveragePooling2D({ poolSize: 3, strides: 2, padding: "same" });
    // [batch, height, width, channels] -> [batch * H_out * W_out, channels]
    // H_out = Math.ceil(6/2) = 3. W_out = 3. rows = 2 * 3 * 3 = 18.
    expect(layer.computeOutputShape([2, 6, 6, 4])).toEqual([18, 4]);
  });

  it("should perform forward pass mathematically correctly", () => {
    const layer = new AveragePooling2D({
      poolSize: 2,
      strides: 1,
      padding: "valid"
    });
    layer.build([1, 3, 3, 1]); // B=1, H=3, W=3, C=1 (grayscale image)

    // Input image: shape [9, 1]
    // 1 5 2
    // 3 2 8
    // 4 6 7
    const x = mat([
      1, 5, 2,
      3, 2, 8,
      4, 6, 7
    ], [9, 1]);

    const y = layer.forward(x);

    // H_out = 3 - 2 + 1 = 2. W_out = 2. outputShape = [4, 1]
    expectMatrixShape(y, [4, 1]);

    // Output at row 0 (window at (0,0)):
    // 1 5
    // 3 2
    // Avg = (1+5+3+2)/4 = 11/4 = 2.75
    //
    // Output at row 1 (window at (0,1)):
    // 5 2
    // 2 8
    // Avg = (5+2+2+8)/4 = 17/4 = 4.25
    //
    // Output at row 2 (window at (1,0)):
    // 3 2
    // 4 6
    // Avg = (3+2+4+6)/4 = 15/4 = 3.75
    //
    // Output at row 3 (window at (1,1)):
    // 2 8
    // 6 7
    // Avg = (2+8+6+7)/4 = 23/4 = 5.75

    expectMatrixCloseTo(y, [
      2.75,
      4.25,
      3.75,
      5.75
    ]);
  });

  it("should calculate backward gradients correctly via tape", () => {
    const layer = new AveragePooling2D({
      poolSize: 2,
      strides: 1,
      padding: "valid"
    });
    layer.build([1, 3, 3, 1]);

    const x = mat([
      1, 5, 2,
      3, 2, 8,
      4, 6, 7
    ], [9, 1]);
    x.requiresGrad = true;

    const tape = engine.grad(() => {
      return layer.forward(x);
    });

    x.clearGrad();

    // Upstream gradient shape [4, 1]
    const gradOutput = mat([
      1.0,
      2.0,
      3.0,
      4.0
    ], [4, 1]);

    tape.backward(tape.result, gradOutput);

    // Average pooling backward trace:
    // For each window, we add gradOut / 4 to each pixel gradient.
    //
    // - (0,0) window [index 0, 1, 3, 4]: gradOutput[0] = 1.0 -> adds 1.0/4 = 0.25 to index 0, 1, 3, 4
    // - (0,1) window [index 1, 2, 4, 5]: gradOutput[1] = 2.0 -> adds 2.0/4 = 0.50 to index 1, 2, 4, 5
    // - (1,0) window [index 3, 4, 6, 7]: gradOutput[2] = 3.0 -> adds 3.0/4 = 0.75 to index 3, 4, 6, 7
    // - (1,1) window [index 4, 5, 7, 8]: gradOutput[3] = 4.0 -> adds 4.0/4 = 1.00 to index 4, 5, 7, 8
    //
    // Total gradX:
    // gradX[0] = 0.25
    // gradX[1] = 0.25 + 0.50 = 0.75
    // gradX[2] = 0.50
    // gradX[3] = 0.25 + 0.75 = 1.00
    // gradX[4] = 0.25 + 0.50 + 0.75 + 1.00 = 2.50
    // gradX[5] = 0.50 + 1.00 = 1.50
    // gradX[6] = 0.75
    // gradX[7] = 0.75 + 1.00 = 1.75
    // gradX[8] = 1.00

    expect(x.grad).toBeDefined();
    expectMatrixShape(x.grad!, [9, 1]);
    expectMatrixCloseTo(x.grad!, [
      0.25, 0.75, 0.50,
      1.00, 2.50, 1.50,
      0.75, 1.75, 1.00
    ]);
  });

  it("should return config correctly", () => {
    const layer = new AveragePooling2D({
      name: "avgpool2d_test",
      poolSize: [2, 2],
      strides: [2, 2],
      padding: "same",
      imageShape: [28, 28],
      inputDim: 3
    });

    const kConfig = layer.getKerasConfig();
    expect(kConfig.class_name).toBe("AveragePooling2D");
    expect(kConfig.config.name).toBe("avgpool2d_test");
    expect(kConfig.config.poolSize).toEqual([2, 2]);
    expect(kConfig.config.strides).toEqual([2, 2]);
    expect(kConfig.config.padding).toBe("same");
    expect(kConfig.config.imageShape).toEqual([28, 28]);
    expect(kConfig.config.inputDim).toBe(3);
  });
});
