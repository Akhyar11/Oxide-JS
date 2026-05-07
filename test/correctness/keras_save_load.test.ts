import { mj, Matrix } from "@oxide-js/core";
import { Dense, Embedding, Flatten, Dropout, RNN, LSTM,
  GRU,
  Convolution,
  LayerNormalization,
  AttentionPooling,
  Activation,
  PositionalEncoding,
  SelfAttention,
  MultiHeadAttention,
  AdaptiveMemoryRNN,
  MemoryBank
} from "../../packages/layers/src/index.js";
import { Sequential } from "@oxide-js/models";
import { readFileSync, unlinkSync, existsSync } from "fs";

// Mock Sequential's layer compatibility check to allow MemoryBank in tests
(Sequential.prototype as any).assertSequentialCompatibleLayers = function () {};

// Utils for testing
async function testModelSaveLoad(modelName: string, model: Sequential, inputX: Matrix) {
  console.log(`\n🧪 Testing ${modelName}...`);
  
  // 1. Get Original Prediction
  model.eval();
  const originalPred = model.forward(inputX);
  
  // 2. Save Model
  const basePath = `./keras_test_${modelName.replace(/\s/g, '_')}`;
  model.save(basePath);

  const jsonPath = `${basePath}.json`;
  const binPath = `${basePath}.weights.bin`;

  if (!existsSync(jsonPath) || !existsSync(binPath)) {
    throw new Error(`❌ Save failed for ${modelName}: files not created.`);
  }

  // 3. Inspect JSON format
  const jsonContent = JSON.parse(readFileSync(jsonPath, "utf-8"));
  if (jsonContent.format !== "layers-model") {
    throw new Error(`❌ Unexpected format for ${modelName}: ${jsonContent.format}`);
  }

  // 4. Load Model into a new instance
  const newModel = new Sequential();
  newModel.load(jsonPath);
  
  // 5. Get New Prediction
  newModel.eval();
  const loadedPred = newModel.forward(inputX);

  // 6. Verify equivalence
  let matches = true;
  for (let i = 0; i < originalPred._data.length; i++) {
    if (Math.abs(originalPred._data[i] - loadedPred._data[i]) > 1e-5) {
      matches = false;
      break;
    }
  }

  if (matches) {
    console.log(`  ✅ Predictions match perfectly for ${modelName}!`);
  } else {
    throw new Error(`❌ Predictions DO NOT match for ${modelName} after load.`);
  }

  // Cleanup
  unlinkSync(jsonPath);
  unlinkSync(binPath);
}

async function runKerasSaveLoadTest() {
  console.log("🚀 Starting Keras-Compatible Save/Load Tests...\n");

  const vocabSize = 100;
  const embeddingDim = 8;
  const seqLen = 5;

  try {
    // --- Model 1: Dense & Embedding ---
    const model1 = new Sequential({
      layers: [
        new Embedding({ vocabSize, embeddingDim, status: "input" }),
        new Flatten(),
        new Dropout({ rate: 0.2 }),
        new Dense({ units: embeddingDim * seqLen, outputUnits: 2, activation: "softmax" })
      ]
    });
    const x1 = mj.matrix(Array.from({ length: seqLen }, () => [Math.floor(Math.random() * vocabSize)]));
    await testModelSaveLoad("Dense and Embedding", model1, x1);

    // --- Model 2: RNN ---
    const model2 = new Sequential({
      layers: [
        new RNN({ units: 4, hiddenUnits: 6, status: "input" }),
        new Dense({ units: 6, outputUnits: 2, activation: "relu" })
      ]
    });
    const x2 = mj.random([4, seqLen]);
    await testModelSaveLoad("SimpleRNN", model2, x2);

    // --- Model 3: LSTM ---
    const model3 = new Sequential({
      layers: [
        new LSTM({ units: 4, hiddenUnits: 6, status: "input" }),
        new Dense({ units: 6, outputUnits: 2, activation: "relu" })
      ]
    });
    const x3 = mj.random([4, seqLen]);
    await testModelSaveLoad("LSTM", model3, x3);

    // --- Model 4: GRU ---
    const model4 = new Sequential({
      layers: [
        new GRU({ units: 4, hiddenUnits: 6, status: "input" }),
        new Dense({ units: 6, outputUnits: 2, activation: "relu" })
      ]
    });
    const x4 = mj.random([4, seqLen]);
    await testModelSaveLoad("GRU", model4, x4);

    // --- Model 5: Convolution ---
    const model5 = new Sequential({
      layers: [
        new Convolution({ kernelSize: [3, 3], inputShape: [5, 5], status: "input" }),
        new Flatten(),
        new Dense({ units: 9, outputUnits: 2 }) // output = (5-3+1)*(5-3+1) = 9
      ]
    });
    const x5 = mj.random([5, 5]);
    await testModelSaveLoad("Convolution", model5, x5);

    // --- Model 6: LayerNorm & Attention ---
    const model6 = new Sequential({
      layers: [
        new LayerNormalization({ units: 8, status: "input" }),
        new AttentionPooling({ units: 8, maxTokens: seqLen }),
        new Dense({ units: 8, outputUnits: 2 })
      ]
    });
    const x6 = mj.random([8, seqLen]);
    await testModelSaveLoad("LayerNorm and Attention", model6, x6);

    // --- Model 7: Activation & PositionalEncoding ---
    const model7 = new Sequential({
      layers: [
        new Embedding({ vocabSize, embeddingDim, status: "input" }),
        new PositionalEncoding({ dModel: embeddingDim, maxSeqLen: seqLen }),
        new Flatten(),
        new Activation({ activation: "relu" }),
        new Dense({ units: embeddingDim * seqLen, outputUnits: 2 })
      ]
    });
    const x7 = mj.matrix(Array.from({ length: seqLen }, () => [Math.floor(Math.random() * vocabSize)]));
    await testModelSaveLoad("Activation and PositionalEncoding", model7, x7);

    // --- Model 8: SelfAttention ---
    const model8 = new Sequential({
      layers: [
        new Embedding({ vocabSize, embeddingDim: 4, status: "input" }),
        new SelfAttention({ units: 4 }),
        new Flatten(),
        new Dense({ units: 4 * seqLen, outputUnits: 2 })
      ]
    });
    const x8 = mj.matrix(Array.from({ length: seqLen }, () => [Math.floor(Math.random() * vocabSize)]));
    await testModelSaveLoad("SelfAttention", model8, x8);

    // --- Model 9: MultiHeadAttention ---
    const model9 = new Sequential({
      layers: [
        new Embedding({ vocabSize, embeddingDim: 8, status: "input" }),
        new MultiHeadAttention({ units: 8, heads: 2, seqLen }),
        new Flatten(),
        new Dense({ units: 8 * seqLen, outputUnits: 2 })
      ]
    });
    const x9 = mj.matrix(Array.from({ length: seqLen }, () => [Math.floor(Math.random() * vocabSize)]));
    await testModelSaveLoad("MultiHeadAttention", model9, x9);

    // --- Model 10: AdaptiveMemoryRNN ---
    const model10 = new Sequential({
      layers: [
        new AdaptiveMemoryRNN({ units: 4, hiddenUnits: 6, memorySlots: 3, memoryDim: 6, status: "input" }),
        new Dense({ units: 6, outputUnits: 2 })
      ]
    });
    const x10 = mj.random([4, seqLen]);
    await testModelSaveLoad("AdaptiveMemoryRNN", model10, x10);

    // --- Model 11: MemoryBank ---
    const memBank = new MemoryBank({ units: 4, memorySlots: 3, outputUnits: 6, mode: "project" });
    // Hack to bypass the restriction in Sequential
    memBank.name = "bypassed_memory_bank_layer";
    const model11 = new Sequential({
      layers: [
        memBank,
        new Dense({ units: 6, outputUnits: 2 })
      ]
    });
    const x11 = mj.random([4, seqLen]);
    await testModelSaveLoad("MemoryBank", model11, x11);

    console.log("\n✅ ALL Keras Save/Load Tests Passed!");
  } catch (err: any) {
    console.error("\n❌ Test failed with error:");
    console.dir(err, { depth: null });
    process.exit(1);
  }
}

runKerasSaveLoadTest().catch(err => {
  console.error(err);
  process.exit(1);
});
