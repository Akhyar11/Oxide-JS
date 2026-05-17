import { describe, expect, it } from "vitest";
import { Conv2D } from "../src/layers/Conv2D.js";
import { Matrix, engine } from "@oxide-js/core";
import { expectMatrixCloseTo, expectMatrixShape, mat } from "./helpers/matrix.js";

describe("Conv2D Layer Tests", () => {
  it("should create layer with custom config", () => {
    const layer = new Conv2D({
      name: "conv2d_custom",
      filters: 8,
      kernelSize: [3, 3],
      strides: [2, 2],
      padding: "same",
      activation: "relu",
      useBias: true,
      kernelInitializer: "random",
      biasInitializer: "zeros"
    });
    expect(layer.name).toBe("conv2d_custom");
    expect(layer.filters).toBe(8);
    expect(layer.kernelSize).toEqual([3, 3]);
    expect(layer.strides).toEqual([2, 2]);
    expect(layer.padding).toBe("same");
    expect(layer.activation).toBe("relu");
    expect(layer.useBias).toBe(true);
    expect(layer.kernelInitializer).toBe("random");
    expect(layer.biasInitializer).toBe("zeros");
  });

  it("should compute output shape correctly for valid padding", () => {
    const layer = new Conv2D({ filters: 4, kernelSize: 3, padding: "valid" });
    // [batch, height, width, channels] -> [batch * H_out * W_out, filters]
    // H_out = 6 - 3 + 1 = 4. W_out = 6 - 3 + 1 = 4. rows = 2 * 4 * 4 = 32.
    expect(layer.computeOutputShape([2, 6, 6, 2])).toEqual([32, 4]);
  });

  it("should compute output shape correctly for same padding", () => {
    const layer = new Conv2D({ filters: 8, kernelSize: [3, 3], padding: "same" });
    // [batch, height, width, channels] -> [batch * H_out * W_out, filters]
    // H_out = 6. W_out = 6. rows = 2 * 6 * 6 = 72.
    expect(layer.computeOutputShape([2, 6, 6, 2])).toEqual([72, 8]);
  });

  it("should build weights correctly", () => {
    const layer = new Conv2D({
      filters: 4,
      kernelSize: [3, 3],
      useBias: true,
      kernelInitializer: "random"
    });
    layer.build([2, 6, 6, 2]); // B=2, H=6, W=6, C=2

    expect(layer.isBuilt).toBe(true);
    expect(layer.imageShape).toEqual([6, 6]);
    expect(layer.inputDim).toBe(2);

    const kernel = layer.kernel;
    expect(kernel).toBeDefined();
    expect(kernel?._shape).toEqual([18, 4]); // [kernelRows * kernelCols * inputDim, filters] = [3 * 3 * 2, 4] = [18, 4]
    expect(kernel?.requiresGrad).toBe(true);

    const bias = layer.bias;
    expect(bias).toBeDefined();
    expect(bias?._shape).toEqual([4, 1]); // [filters, 1]
    expect(bias?.requiresGrad).toBe(true);
  });

  it("should perform forward pass mathematically correctly for stride 1, valid padding", () => {
    const layer = new Conv2D({
      filters: 2,
      kernelSize: 2,
      useBias: true,
      padding: "valid",
      activation: "linear",
      kernelInitializer: "zeros",
      biasInitializer: "zeros"
    });
    layer.build([1, 3, 3, 1]); // B=1, H=3, W=3, C=1 (grayscale image)

    // Set custom kernel weights: shape [4, 2] (2 * 2 * 1, 2)
    // Filter 0 weights: [1, 2, 3, 4]
    // Filter 1 weights: [0.5, 0.5, 0.5, 0.5]
    layer.kernel!._data.set([
      1,   0.5,
      2,   0.5,
      3,   0.5,
      4,   0.5
    ]);

    // Set custom bias: shape [2, 1]
    layer.bias!._data.set([0.1, 0.2]);

    // Input image: shape [3*3, 1]
    // 1 2 3
    // 4 5 6
    // 7 8 9
    const x = mat([
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9
    ], [9, 1]);

    const y = layer.forward(x);

    // H_out = 3 - 2 + 1 = 2. W_out = 3 - 2 + 1 = 2. outputShape = [4, 2]
    expectMatrixShape(y, [4, 2]);

    // Output at row 0 (window at (0,0)):
    // 1 2
    // 4 5
    // Flattened window: [1, 2, 4, 5]
    // Filter 0: 1*1 + 2*2 + 4*3 + 5*4 + bias 0.1 = 1 + 4 + 12 + 20 + 0.1 = 37.1
    // Filter 1: 1*0.5 + 2*0.5 + 4*0.5 + 5*0.5 + bias 0.2 = 0.5 + 1.0 + 2.0 + 2.5 + 0.2 = 6.2
    //
    // Output at row 1 (window at (0,1)):
    // 2 3
    // 5 6
    // Flattened window: [2, 3, 5, 6]
    // Filter 0: 2*1 + 3*2 + 5*3 + 6*4 + bias 0.1 = 2 + 6 + 15 + 24 + 0.1 = 47.1
    // Filter 1: 2*0.5 + 3*0.5 + 5*0.5 + 6*0.5 + bias 0.2 = 1.0 + 1.5 + 2.5 + 3.0 + 0.2 = 8.2
    //
    // Output at row 2 (window at (1,0)):
    // 4 5
    // 7 8
    // Flattened window: [4, 5, 7, 8]
    // Filter 0: 4*1 + 5*2 + 7*3 + 8*4 + bias 0.1 = 4 + 10 + 21 + 32 + 0.1 = 67.1
    // Filter 1: 4*0.5 + 5*0.5 + 7*0.5 + 8*0.5 + bias 0.2 = 2.0 + 2.5 + 3.5 + 4.0 + 0.2 = 12.2
    //
    // Output at row 3 (window at (1,1)):
    // 5 6
    // 8 9
    // Flattened window: [5, 6, 8, 9]
    // Filter 0: 5*1 + 6*2 + 8*3 + 9*4 + bias 0.1 = 5 + 12 + 24 + 36 + 0.1 = 77.1
    // Filter 1: 5*0.5 + 6*0.5 + 8*0.5 + 9*0.5 + bias 0.2 = 2.5 + 3.0 + 4.0 + 4.5 + 0.2 = 14.2

    expectMatrixCloseTo(y, [
      37.1, 6.2,
      47.1, 8.2,
      67.1, 12.2,
      77.1, 14.2
    ]);
  });

  it("should perform forward pass mathematically correctly for same padding", () => {
    const layer = new Conv2D({
      filters: 1,
      kernelSize: 3,
      useBias: false,
      padding: "same",
      activation: "linear",
      kernelInitializer: "zeros"
    });
    layer.build([1, 3, 3, 1]); // B=1, H=3, W=3, C=1

    // Kernel shape: [9, 1]
    // 0 1 0
    // 1 0 1
    // 0 1 0
    layer.kernel!._data.set([
      0, 1, 0,
      1, 0, 1,
      0, 1, 0
    ]);

    // Input image: shape [9, 1]
    // 0 2 0
    // 1 0 3
    // 0 4 0
    const x = mat([
      0, 2, 0,
      1, 0, 3,
      0, 4, 0
    ], [9, 1]);

    const y = layer.forward(x);

    // H_out = 3, W_out = 3. Same padding maps center pixels and includes zero boundaries.
    // Center pixel at (1,1): value should be 2*1 + 1*1 + 3*1 + 4*1 = 10.
    expectMatrixShape(y, [9, 1]);
    expect(y._data[4]).toBeCloseTo(10);
  });

  it("should calculate backward gradients correctly via tape", () => {
    const layer = new Conv2D({
      filters: 2,
      kernelSize: 2,
      useBias: true,
      padding: "valid",
      activation: "linear",
      kernelInitializer: "zeros",
      biasInitializer: "zeros"
    });
    layer.build([1, 3, 3, 1]); // B=1, H=3, W=3, C=1

    layer.kernel!._data.set([
      1,   0.5,
      2,   0.5,
      3,   0.5,
      4,   0.5
    ]);
    layer.bias!._data.set([0.1, 0.2]);

    const x = mat([
      1, 2, 3,
      4, 5, 6,
      7, 8, 9
    ], [9, 1]);
    x.requiresGrad = true;

    const kernel = layer.kernel!;
    const bias = layer.bias!;

    const tape = engine.grad(() => {
      return layer.forward(x);
    });

    x.clearGrad();
    kernel.clearGrad();
    bias.clearGrad();

    // Upstream gradient shape [4, 2] (H_out * W_out = 4, filters = 2)
    const gradOutput = mat([
      1.0, 2.0, // (0,0) window
      3.0, 4.0, // (0,1) window
      5.0, 6.0, // (1,0) window
      7.0, 8.0  // (1,1) window
    ], [4, 2]);

    tape.backward(tape.result, gradOutput);

    // Verify gradients computed on the inputs
    // Input grid indices:
    // 0 1 2
    // 3 4 5
    // 6 7 8
    //
    // For input[0] (only used in (0,0) window as kernel[0]):
    // gradX[0] = gradOutput[0,0]*kernel[0] + gradOutput[0,1]*kernel[1] = 1.0*1 + 2.0*0.5 = 2.0
    //
    // For input[4] (center, used in all 4 windows):
    // - (0,0) window as kernel[3]: gradOutput[0,0]*kernel[6] + gradOutput[0,1]*kernel[7]
    // - (0,1) window as kernel[2]: gradOutput[1,0]*kernel[4] + gradOutput[1,1]*kernel[5]
    // - (1,0) window as kernel[1]: gradOutput[2,0]*kernel[2] + gradOutput[2,1]*kernel[3]
    // - (1,1) window as kernel[0]: gradOutput[3,0]*kernel[0] + gradOutput[3,1]*kernel[1]
    // Wait, let's trace flat kernel weight mappings:
    // kernel is [4, 2], flat index corresponds to:
    // row 0: kernel size step (0,0) -> [1, 0.5]
    // row 1: kernel size step (0,1) -> [2, 0.5]
    // row 2: kernel size step (1,0) -> [3, 0.5]
    // row 3: kernel size step (1,1) -> [4, 0.5]
    // So:
    // - (0,0) window has indices [0, 1, 3, 4]. input[4] is at kernel step (1,1) (row 3).
    //   Contribution = gradOutput[0,0]*4 + gradOutput[0,1]*0.5 = 1.0*4 + 2.0*0.5 = 5.0
    // - (0,1) window has indices [1, 2, 4, 5]. input[4] is at kernel step (1,0) (row 2).
    //   Contribution = gradOutput[1,0]*3 + gradOutput[1,1]*0.5 = 3.0*3 + 4.0*0.5 = 11.0
    // - (1,0) window has indices [3, 4, 6, 7]. input[4] is at kernel step (0,1) (row 1).
    //   Contribution = gradOutput[2,0]*2 + gradOutput[2,1]*0.5 = 5.0*2 + 6.0*0.5 = 13.0
    // - (1,1) window has indices [4, 5, 7, 8]. input[4] is at kernel step (0,0) (row 0).
    //   Contribution = gradOutput[3,0]*1 + gradOutput[3,1]*0.5 = 7.0*1 + 8.0*0.5 = 11.0
    // Total gradX[4] = 5.0 + 11.0 + 13.0 + 11.0 = 40.0

    expect(x.grad).toBeDefined();
    expect(x.grad!._data[0]).toBeCloseTo(2.0);
    expect(x.grad!._data[4]).toBeCloseTo(40.0);
  });

  it("should return config correctly", () => {
    const layer = new Conv2D({
      name: "conv2d_test",
      filters: 16,
      kernelSize: [3, 5],
      strides: [1, 2],
      padding: "same",
      activation: "relu",
      imageShape: [28, 28],
      inputDim: 3
    });

    const kConfig = layer.getKerasConfig();
    expect(kConfig.class_name).toBe("Conv2D");
    expect(kConfig.config.name).toBe("conv2d_test");
    expect(kConfig.config.filters).toBe(16);
    expect(kConfig.config.kernelSize).toEqual([3, 5]);
    expect(kConfig.config.strides).toEqual([1, 2]);
    expect(kConfig.config.padding).toBe("same");
    expect(kConfig.config.activation).toBe("relu");
    expect(kConfig.config.imageShape).toEqual([28, 28]);
    expect(kConfig.config.inputDim).toBe(3);
  });
});
