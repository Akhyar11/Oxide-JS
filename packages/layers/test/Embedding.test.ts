import { describe, expect, it } from "vitest";
import { Embedding } from "../src/layers/Embedding.js";
import { Matrix, engine } from "@oxide-js/core";
import { expectMatrixCloseTo, expectMatrixShape, mat } from "./helpers/matrix.js";

describe("Embedding Layer Tests", () => {
  it("should create layer with default config", () => {
    const layer = new Embedding({
      name: "embedding_custom",
      inputDim: 10,
      outputDim: 4,
      embeddingsInitializer: "random",
      trainable: true
    });
    expect(layer.name).toBe("embedding_custom");
    expect(layer.inputDim).toBe(10);
    expect(layer.outputDim).toBe(4);
    expect(layer.embeddingsInitializer).toBe("random");
    expect(layer.trainable).toBe(true);
  });

  it("should compute output shape correctly", () => {
    const layer = new Embedding({ inputDim: 10, outputDim: 4 });
    // [batchSize, sequenceLength] -> [batchSize * sequenceLength, outputDim]
    expect(layer.computeOutputShape([2, 5])).toEqual([10, 4]);
    expect(layer.computeOutputShape([1, 1])).toEqual([1, 4]);
  });

  it("should build weight parameters correctly", () => {
    const layer = new Embedding({ inputDim: 10, outputDim: 4 });
    layer.build([2, 3]);

    expect(layer.isBuilt).toBe(true);
    expect(layer.inputShape).toEqual([2, 3]);
    expect(layer.outputShape).toEqual([6, 4]);

    const embeddings = layer.embeddings;
    expect(embeddings).toBeDefined();
    expect(embeddings?._shape).toEqual([10, 4]); // [vocabSize, embeddingDim]
    expect(embeddings?.requiresGrad).toBe(true);
  });

  it("should perform forward pass mapping token IDs to vectors", () => {
    const layer = new Embedding({
      inputDim: 3,
      outputDim: 2
    });
    layer.build([2, 2]);

    // Force deterministic embeddings
    // vocab index 0: [0.1, 0.2]
    // vocab index 1: [0.3, 0.4]
    // vocab index 2: [0.5, 0.6]
    layer.embeddings!._data.set([
      0.1, 0.2,
      0.3, 0.4,
      0.5, 0.6
    ]);

    // Inputs: batch size 2, seq len 2
    const x = mat([0, 2, 1, 0], [2, 2]);
    const y = layer.forward(x);

    expectMatrixShape(y, [4, 2]);
    expectMatrixCloseTo(y, [
      0.1, 0.2, // vocab index 0
      0.5, 0.6, // vocab index 2
      0.3, 0.4, // vocab index 1
      0.1, 0.2  // vocab index 0
    ]);
  });

  it("should perform backward pass via autodiff tape correctly", () => {
    const layer = new Embedding({
      inputDim: 3,
      outputDim: 2
    });
    layer.build([2, 2]);

    layer.embeddings!._data.set([
      0.1, 0.2,
      0.3, 0.4,
      0.5, 0.6
    ]);

    const x = mat([0, 2, 1, 0], [2, 2]);
    // Inputs (indices) do not need gradient
    x.requiresGrad = false;

    const embeddings = layer.embeddings!;

    const tape = engine.grad(() => {
      return layer.forward(x);
    });

    embeddings.clearGrad();

    // Upstream gradient has shape [4, 2]
    const gradOutput = mat([
      1.0, 2.0, // for token at index 0 (row 0 of inputs)
      3.0, 4.0, // for token at index 2 (row 1 of inputs)
      5.0, 6.0, // for token at index 1 (row 2 of inputs)
      7.0, 8.0  // for token at index 0 (row 3 of inputs)
    ], [4, 2]);

    tape.backward(tape.result, gradOutput);

    // Verify gradients computed on the embeddings parameter
    expect(embeddings.grad).toBeDefined();
    expectMatrixShape(embeddings.grad!, [3, 2]);

    // Token index 0 was looked up twice (at index 0 and 3).
    // Gradients should accumulate:
    // row 0: gradOutput[0] + gradOutput[3] = [1.0 + 7.0, 2.0 + 8.0] = [8.0, 10.0]
    // Token index 1 was looked up once (at index 2):
    // row 1: gradOutput[2] = [5.0, 6.0]
    // Token index 2 was looked up once (at index 1):
    // row 2: gradOutput[1] = [3.0, 4.0]
    expectMatrixCloseTo(embeddings.grad!, [
      8.0, 10.0,
      5.0, 6.0,
      3.0, 4.0
    ]);

    // x (indices) should not have gradients
    expect(x.grad).toBeNull();
  });

  it("should return empty trainable weights when frozen", () => {
    const layer = new Embedding({
      inputDim: 5,
      outputDim: 3,
      trainable: false
    });
    layer.build([2, 2]);

    expect(layer.trainableWeights.length).toBe(0);
    expect(layer.nonTrainableWeights.length).toBeGreaterThan(0);
  });

  it("should return config correctly", () => {
    const layer = new Embedding({
      name: "emb_keras",
      inputDim: 100,
      outputDim: 16,
      embeddingsInitializer: "random"
    });

    const kConfig = layer.getKerasConfig();
    expect(kConfig.class_name).toBe("Embedding");
    expect(kConfig.config.name).toBe("emb_keras");
    expect(kConfig.config.inputDim).toBe(100);
    expect(kConfig.config.outputDim).toBe(16);
    expect(kConfig.config.embeddingsInitializer).toBe("random");
  });

  it("should throw error for out of bounds token indices", () => {
    const layer = new Embedding({ inputDim: 3, outputDim: 2 });
    layer.build([1, 2]);
    layer.embeddings!._data.set([1, 2, 3, 4, 5, 6]);

    // Token index 4 is out of vocab size 3
    const xInvalid = mat([0, 4], [1, 2]);
    expect(() => layer.forward(xInvalid)).toThrow();
  });

  it("should throw error for invalid constructor config", () => {
    expect(() => new Embedding({ inputDim: -1, outputDim: 10 })).toThrow();
    expect(() => new Embedding({ inputDim: 10, outputDim: 0 })).toThrow();
  });
});
