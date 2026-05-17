import { describe, expect, it } from "vitest";
import { Conv1D } from "../src/layers/Conv1D.js";
import { Matrix, engine } from "@oxide-js/core";
import { expectMatrixCloseTo, expectMatrixShape, mat } from "./helpers/matrix.js";

describe("Conv1D Layer Tests", () => {
  it("should create layer with custom config", () => {
    const layer = new Conv1D({
      name: "conv1d_custom",
      filters: 4,
      kernelSize: 3,
      strides: 2,
      padding: "same",
      activation: "relu",
      useBias: true,
      kernelInitializer: "random",
      biasInitializer: "zeros"
    });
    expect(layer.name).toBe("conv1d_custom");
    expect(layer.filters).toBe(4);
    expect(layer.kernelSize).toBe(3);
    expect(layer.strides).toBe(2);
    expect(layer.padding).toBe("same");
    expect(layer.activation).toBe("relu");
    expect(layer.useBias).toBe(true);
    expect(layer.kernelInitializer).toBe("random");
    expect(layer.biasInitializer).toBe("zeros");
  });

  it("should compute output shape correctly for valid padding", () => {
    const layer = new Conv1D({ filters: 4, kernelSize: 3, padding: "valid" });
    // [batch, sequenceLength, inputDim] -> [batch * L_out, filters]
    expect(layer.computeOutputShape([2, 5, 2])).toEqual([6, 4]); // L_out = 5 - 3 + 1 = 3. rows = 2 * 3 = 6.
  });

  it("should compute output shape correctly for same padding", () => {
    const layer = new Conv1D({ filters: 8, kernelSize: 3, padding: "same" });
    // [batch, sequenceLength, inputDim] -> [batch * L_out, filters]
    expect(layer.computeOutputShape([2, 5, 2])).toEqual([10, 8]); // L_out = Math.ceil(5 / 1) = 5. rows = 2 * 5 = 10.
  });

  it("should build weights correctly", () => {
    const layer = new Conv1D({
      filters: 4,
      kernelSize: 3,
      useBias: true,
      kernelInitializer: "random"
    });
    layer.build([2, 5, 2]); // B=2, L=5, C=2

    expect(layer.isBuilt).toBe(true);
    expect(layer.sequenceLength).toBe(5);
    expect(layer.inputDim).toBe(2);

    const kernel = layer.kernel;
    expect(kernel).toBeDefined();
    expect(kernel?._shape).toEqual([6, 4]); // [kernelSize * inputDim, filters] = [3 * 2, 4] = [6, 4]
    expect(kernel?.requiresGrad).toBe(true);

    const bias = layer.bias;
    expect(bias).toBeDefined();
    expect(bias?._shape).toEqual([4, 1]); // [filters, 1]
    expect(bias?.requiresGrad).toBe(true);
  });

  it("should perform forward pass mathematically correctly for stride 1, valid padding", () => {
    const layer = new Conv1D({
      filters: 2,
      kernelSize: 2,
      useBias: true,
      padding: "valid",
      activation: "linear",
      kernelInitializer: "zeros",
      biasInitializer: "zeros"
    });
    layer.build([1, 3, 2]); // B=1, L=3, C=2

    // Set custom kernel weights: shape [4, 2] (kernelSize * inputDim, filters)
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

    // Inputs: B=1, L=3, C=2
    // step 0: [1, 2]
    // step 1: [3, 4]
    // step 2: [5, 6]
    const x = mat([
      1, 2,
      3, 4,
      5, 6
    ], [3, 2]);

    const y = layer.forward(x);

    // L_out = 3 - 2 + 1 = 2. shape: [2, 2]
    expectMatrixShape(y, [2, 2]);

    // Output at step 0 (window step 0 & 1):
    // Flattened window: [1, 2, 3, 4]
    // Filter 0: 1*1 + 2*2 + 3*3 + 4*4 + bias 0.1 = 1 + 4 + 9 + 16 + 0.1 = 30.1
    // Filter 1: 1*0.5 + 2*0.5 + 3*0.5 + 4*0.5 + bias 0.2 = 0.5 + 1.0 + 1.5 + 2.0 + 0.2 = 5.2
    // Output at step 1 (window step 1 & 2):
    // Flattened window: [3, 4, 5, 6]
    // Filter 0: 3*1 + 4*2 + 5*3 + 6*4 + bias 0.1 = 3 + 8 + 15 + 24 + 0.1 = 50.1
    // Filter 1: 3*0.5 + 4*0.5 + 5*0.5 + 6*0.5 + bias 0.2 = 1.5 + 2.0 + 2.5 + 3.0 + 0.2 = 9.2

    expectMatrixCloseTo(y, [
      30.1, 5.2,
      50.1, 9.2
    ]);
  });

  it("should perform forward pass mathematically correctly for same padding", () => {
    const layer = new Conv1D({
      filters: 1,
      kernelSize: 3,
      useBias: false,
      padding: "same",
      activation: "linear",
      kernelInitializer: "zeros"
    });
    layer.build([1, 3, 1]); // B=1, L=3, C=1

    // Kernel shape: [3, 1]
    layer.kernel!._data.set([1, 2, 3]);

    // Inputs: L=3, C=1
    // step 0: [1]
    // step 1: [2]
    // step 2: [3]
    const x = mat([1, 2, 3], [3, 1]);

    const y = layer.forward(x);

    // For L=3, kernelSize=3, strides=1, same padding:
    // padLeft = Math.floor((3 - 3) / 2) = 1.
    // L_out = 3
    // step 0 (tStart = -1): [0, 1, 2] * [1, 2, 3] = 0*1 + 1*2 + 2*3 = 8
    // step 1 (tStart = 0): [1, 2, 3] * [1, 2, 3] = 1*1 + 2*2 + 3*3 = 14
    // step 2 (tStart = 1): [2, 3, 0] * [1, 2, 3] = 2*1 + 3*2 + 0*3 = 8

    expectMatrixShape(y, [3, 1]);
    expectMatrixCloseTo(y, [8, 14, 8]);
  });

  it("should calculate backward gradients correctly via tape", () => {
    const layer = new Conv1D({
      filters: 2,
      kernelSize: 2,
      useBias: true,
      padding: "valid",
      activation: "linear",
      kernelInitializer: "zeros",
      biasInitializer: "zeros"
    });
    layer.build([1, 3, 2]); // B=1, L=3, C=2

    layer.kernel!._data.set([
      1,   0.5,
      2,   0.5,
      3,   0.5,
      4,   0.5
    ]);
    layer.bias!._data.set([0.1, 0.2]);

    const x = mat([
      1, 2,
      3, 4,
      5, 6
    ], [3, 2]);
    x.requiresGrad = true;

    const kernel = layer.kernel!;
    const bias = layer.bias!;

    const tape = engine.grad(() => {
      return layer.forward(x);
    });

    x.clearGrad();
    kernel.clearGrad();
    bias.clearGrad();

    // Upstream gradient shape [2, 2] (L_out = 2, filters = 2)
    const gradOutput = mat([
      1.0, 2.0,
      3.0, 4.0
    ], [2, 2]);

    tape.backward(tape.result, gradOutput);

    // Verify gradients computed on the inputs
    // seq2col output mapping:
    // row 0: window [x[0], x[1]]. gradOutput[0] = [1.0, 2.0]
    // row 1: window [x[1], x[2]]. gradOutput[1] = [3.0, 4.0]
    //
    // For inputs:
    // gradX[0] (step 0) = gradOutput[0,0] * kernel[0] + gradOutput[0,1] * kernel[1]
    //                  = 1.0 * [1, 2] + 2.0 * [0.5, 0.5] = [1, 2] + [1, 1] = [2, 3]
    //
    // gradX[1] (step 1) = (gradOutput[0,0] * kernel[2] + gradOutput[0,1] * kernel[3]) // from row 0
    //                   + (gradOutput[1,0] * kernel[0] + gradOutput[1,1] * kernel[1]) // from row 1
    //                  = (1.0 * [3, 4] + 2.0 * [0.5, 0.5]) + (3.0 * [1, 2] + 4.0 * [0.5, 0.5])
    //                  = ([3, 4] + [1, 1]) + ([3, 6] + [2, 2])
    //                  = [4, 5] + [5, 8] = [9, 13]
    //
    // gradX[2] (step 2) = gradOutput[1,0] * kernel[2] + gradOutput[1,1] * kernel[3]
    //                  = 3.0 * [3, 4] + 4.0 * [0.5, 0.5] = [9, 12] + [2, 2] = [11, 14]

    expect(x.grad).toBeDefined();
    expectMatrixShape(x.grad!, [3, 2]);
    expectMatrixCloseTo(x.grad!, [
      2, 3,
      9, 13,
      11, 14
    ]);

    // Verify bias gradients: shape [2, 1]
    // Bias gradient is sum of gradOutput across rows:
    // row 0: gradOutput[0,0] + gradOutput[1,0] = 1.0 + 3.0 = 4.0
    // row 1: gradOutput[0,1] + gradOutput[1,1] = 2.0 + 4.0 = 6.0
    expect(bias.grad).toBeDefined();
    expectMatrixCloseTo(bias.grad!, [4.0, 6.0]);
  });

  it("should return config correctly", () => {
    const layer = new Conv1D({
      name: "conv_test",
      filters: 16,
      kernelSize: 5,
      strides: 2,
      padding: "same",
      activation: "relu",
      sequenceLength: 10,
      inputDim: 4
    });

    const kConfig = layer.getKerasConfig();
    expect(kConfig.class_name).toBe("Conv1D");
    expect(kConfig.config.name).toBe("conv_test");
    expect(kConfig.config.filters).toBe(16);
    expect(kConfig.config.kernelSize).toBe(5);
    expect(kConfig.config.strides).toBe(2);
    expect(kConfig.config.padding).toBe("same");
    expect(kConfig.config.activation).toBe("relu");
    expect(kConfig.config.sequenceLength).toBe(10);
    expect(kConfig.config.inputDim).toBe(4);
  });
});
