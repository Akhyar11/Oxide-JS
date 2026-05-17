import { describe, expect, it } from "vitest";
import { Dense, Activation, Dropout, LayerNormalization, Flatten } from "../src/index.js";
import { Matrix, engine } from "@oxide-js/core";
import { expectMatrixShape, mat } from "./helpers/matrix.js";

describe("Layer Integration and Pipeline Tests", () => {
  it("should perform forward pass sequentially through Dense -> ReLU -> Dense -> LayerNormalization", () => {
    const dense1 = new Dense({ units: 4, name: "dense1" });
    const relu = new Activation({ activation: "relu", name: "relu" });
    const dense2 = new Dense({ units: 2, name: "dense2" });
    const ln = new LayerNormalization({ name: "ln" });

    const x = mat([1, 2, 3, 4, 5, 6], [2, 3]);

    const y1 = dense1.forward(x);
    expectMatrixShape(y1, [2, 4]);

    const y2 = relu.forward(y1);
    expectMatrixShape(y2, [2, 4]);

    const y3 = dense2.forward(y2);
    expectMatrixShape(y3, [2, 2]);

    const y4 = ln.forward(y3);
    expectMatrixShape(y4, [2, 2]);
  });

  it("should backward propagate gradients via Autodiff correctly through the entire pipeline", () => {
    const dense1 = new Dense({ units: 4, name: "dense1" });
    const relu = new Activation({ activation: "relu", name: "relu" });
    const dense2 = new Dense({ units: 2, name: "dense2" });

    const x = mat([1, 2, 3, 4, 5, 6], [2, 3]);
    x.requiresGrad = true;

    // Run training pipeline inside engine.grad to register dependencies on the Tape
    const tape = engine.grad(() => {
      const y1 = dense1.forward(x);
      const y2 = relu.forward(y1);
      const y3 = dense2.forward(y2);
      return y3;
    });

    x.clearGrad();
    dense1.clearGradients();
    dense2.clearGradients();

    // Perform backward pass using initial gradient output of all ones
    const gradOutput = mat([1, 1, 1, 1], [2, 2]);
    tape.backward(tape.result, gradOutput);

    // Verify gradients propagate back to inputs and trainable parameters
    expect(x.grad).toBeDefined();
    expectMatrixShape(x.grad!, [2, 3]);

    expect(dense1.getParameter("kernel")?.grad).toBeDefined();
    expect(dense2.getParameter("kernel")?.grad).toBeDefined();
  });
});
