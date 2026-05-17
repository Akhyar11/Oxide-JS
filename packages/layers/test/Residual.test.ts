import { describe, expect, it } from "vitest";
import { Residual } from "../src/layers/Residual.js";
import { Dense } from "../src/layers/Dense.js";
import { Matrix, engine } from "@oxide-js/core";
import { expectMatrixShape, mat } from "./helpers/matrix.js";

describe("Residual Layer Tests", () => {
  it("should build and compute shape correctly without shortcut if input matches output", () => {
    const subLayer = new Dense({ units: 3 });
    const layer = new Residual({ layer: subLayer });

    layer.build([4, 3]); // input shape is [4, 3], output shape is [4, 3]

    expect(layer.isBuilt).toBe(true);
    expect(layer.inputShape).toEqual([4, 3]);
    expect(layer.outputShape).toEqual([4, 3]);
  });

  it("should throw error during build if shapes mismatch and no shortcut is provided", () => {
    const subLayer = new Dense({ units: 5 });
    const layer = new Residual({ layer: subLayer });

    expect(() => {
      layer.build([4, 3]); // input shape [4, 3] != output shape [4, 5]
    }).toThrow();
  });

  it("should build and compute shape correctly with shortcut projection if shapes mismatch", () => {
    const subLayer = new Dense({ units: 5 });
    const shortcutLayer = new Dense({ units: 5 });
    const layer = new Residual({
      layer: subLayer,
      shortcut: shortcutLayer
    });

    layer.build([4, 3]); // input [4, 3] -> projects to [4, 5] in both paths

    expect(layer.isBuilt).toBe(true);
    expect(layer.inputShape).toEqual([4, 3]);
    expect(layer.outputShape).toEqual([4, 5]);
  });

  it("should gather nested parameters correctly", () => {
    const subLayer = new Dense({ units: 5, useBias: true });
    const shortcutLayer = new Dense({ units: 5, useBias: true });
    const layer = new Residual({
      layer: subLayer,
      shortcut: shortcutLayer
    });

    layer.build([4, 3]);

    const allParams = layer.getTrainableParameters();
    // 2 (weights + bias for subLayer) + 2 (weights + bias for shortcutLayer) = 4
    expect(allParams.length).toBe(4);
    expect(layer.countParams()).toBe((3 * 5 + 5) + (3 * 5 + 5)); // 40 params
  });

  it("should perform forward pass and backward pass with correct autodiff gradient flows", () => {
    const subLayer = new Dense({ units: 3, useBias: false });
    const layer = new Residual({ layer: subLayer });

    layer.build([2, 3]);

    // Set weights to deterministic values
    subLayer.kernel!._data.fill(0.5);

    const x = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    x.requiresGrad = true;

    // Forward pass
    // fx = x * W = [2, 3] * [3, 3]
    // y = fx + x
    const y = layer.forward(x);
    expectMatrixShape(y, [2, 3]);

    // Backward pass
    const tape = engine.grad(() => {
      return layer.forward(x);
    });

    x.clearGrad();
    subLayer.kernel!.clearGrad();
    tape.backward(tape.result);

    expect(x.grad).toBeDefined();
    expect(subLayer.kernel!.grad).toBeDefined();
    expectMatrixShape(x.grad!, [2, 3]);
    expectMatrixShape(subLayer.kernel!.grad!, [3, 3]);
  });
});
