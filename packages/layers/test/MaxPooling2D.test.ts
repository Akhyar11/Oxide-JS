import { describe, expect, it } from "vitest";
import { MaxPooling2D } from "../src/layers/MaxPooling2D.js";
import { Matrix, engine } from "@oxide-js/core";
import { expectMatrixCloseTo, expectMatrixShape, mat } from "./helpers/matrix.js";

describe("MaxPooling2D Layer Tests", () => {
  it("should create layer with custom config", () => {
    const layer = new MaxPooling2D({
      name: "pool2d_custom",
      poolSize: [2, 3],
      strides: [1, 2],
      padding: "same"
    });
    expect(layer.name).toBe("pool2d_custom");
    expect(layer.poolSize).toEqual([2, 3]);
    expect(layer.strides).toEqual([1, 2]);
    expect(layer.padding).toBe("same");
  });

  it("should compute output shape correctly for valid padding", () => {
    const layer = new MaxPooling2D({ poolSize: 2, strides: 2, padding: "valid" });
    // [batch, height, width, channels] -> [batch * H_out * W_out, channels]
    // H_out = Math.floor((6-2)/2) + 1 = 3. W_out = 3. rows = 2 * 3 * 3 = 18.
    expect(layer.computeOutputShape([2, 6, 6, 3])).toEqual([18, 3]);
  });

  it("should compute output shape correctly for same padding", () => {
    const layer = new MaxPooling2D({ poolSize: 3, strides: 2, padding: "same" });
    // [batch, height, width, channels] -> [batch * H_out * W_out, channels]
    // H_out = Math.ceil(6/2) = 3. W_out = 3. rows = 2 * 3 * 3 = 18.
    expect(layer.computeOutputShape([2, 6, 6, 4])).toEqual([18, 4]);
  });

  it("should perform forward pass mathematically correctly", () => {
    const layer = new MaxPooling2D({
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
    // Max is 5.
    //
    // Output at row 1 (window at (0,1)):
    // 5 2
    // 2 8
    // Max is 8.
    //
    // Output at row 2 (window at (1,0)):
    // 3 2
    // 4 6
    // Max is 6.
    //
    // Output at row 3 (window at (1,1)):
    // 2 8
    // 6 7
    // Max is 8.

    expectMatrixCloseTo(y, [
      5,
      8,
      6,
      8
    ]);
  });

  it("should calculate backward gradients correctly via tape", () => {
    const layer = new MaxPooling2D({
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

    // Max pooling indices trace:
    // - (0,0) window [1, 5, 3, 2] -> max is 5 (index 1). gradOutput[0]=1.0 adds to gradX[1]
    // - (0,1) window [5, 2, 2, 8] -> max is 8 (index 5). gradOutput[1]=2.0 adds to gradX[5]
    // - (1,0) window [3, 2, 4, 6] -> max is 6 (index 7). gradOutput[2]=3.0 adds to gradX[7]
    // - (1,1) window [2, 8, 6, 7] -> max is 8 (index 5). gradOutput[3]=4.0 adds to gradX[5]
    // Total gradX:
    // gradX[1] = 1.0
    // gradX[5] = 2.0 + 4.0 = 6.0
    // gradX[7] = 3.0
    // all others 0.

    expect(x.grad).toBeDefined();
    expectMatrixShape(x.grad!, [9, 1]);
    expectMatrixCloseTo(x.grad!, [
      0, 1.0, 0,
      0, 0, 6.0,
      0, 3.0, 0
    ]);
  });

  it("should return config correctly", () => {
    const layer = new MaxPooling2D({
      name: "pool2d_test",
      poolSize: [2, 2],
      strides: [2, 2],
      padding: "same",
      imageShape: [28, 28],
      inputDim: 3
    });

    const kConfig = layer.getKerasConfig();
    expect(kConfig.class_name).toBe("MaxPooling2D");
    expect(kConfig.config.name).toBe("pool2d_test");
    expect(kConfig.config.poolSize).toEqual([2, 2]);
    expect(kConfig.config.strides).toEqual([2, 2]);
    expect(kConfig.config.padding).toBe("same");
    expect(kConfig.config.imageShape).toEqual([28, 28]);
    expect(kConfig.config.inputDim).toBe(3);
  });
});
