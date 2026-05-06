import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import { mj } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { Embedding } from "@oxide-js/layers";
import { RecurrentModel, Sequential, Transformers } from "@oxide-js/models";
import { setLayers } from "@oxide-js/layers";

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}

function assertMatrixEqual(actual: Matrix, expected: number[][], message: string): void {
  const actualValue = actual._value;
  assert(actualValue.length === expected.length, `${message}: row mismatch`);
  for (let i = 0; i < expected.length; i++) {
    assert(actualValue[i].length === expected[i].length, `${message}: col mismatch pada row ${i}`);
    for (let j = 0; j < expected[i].length; j++) {
      if (actualValue[i][j] !== expected[i][j]) {
        throw new Error(`${message}: mismatch pada [${i}, ${j}] expected ${expected[i][j]} got ${actualValue[i][j]}`);
      }
    }
  }
}

function assertMatrixUnchanged(actual: Matrix, before: Matrix, message: string): void {
  assert(actual._data.length === before._data.length, `${message}: length mismatch`);
  for (let i = 0; i < before._data.length; i++) {
    if (actual._data[i] !== before._data[i]) {
      throw new Error(`${message}: nilai berubah pada flat index ${i}`);
    }
  }
}

function assertThrows(fn: () => void, expectedMessage: string, message: string): void {
  let threw = false;
  try {
    fn();
  } catch (error: any) {
    threw = true;
    assert(
      typeof error?.message === "string" && error.message.includes(expectedMessage),
      `${message}: expected error containing '${expectedMessage}', got '${error?.message}'`
    );
  }

  assert(threw, `${message}: expected function to throw`);
}

function createEmbedding(config?: Partial<ConstructorParameters<typeof Embedding>[0]>): Embedding {
  const layer = new Embedding({
    vocabSize: 4,
    embeddingDim: 3,
    alpha: 0.1,
    optimizer: "sgd",
    status: "input",
    padTokenId: null,
    trainable: true,
    ...config,
  });

  layer.fillWeight([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
  ]);

  return layer;
}

function runTrainableFalseFreezeTest(): void {
  const embedding = createEmbedding({ trainable: false });
  const before = embedding.weight.clone();
  const x = mj.matrix([[1], [2], [1]]);
  const out = embedding.forward(x);
  const zeroGrad = embedding.backward(out, mj.ones(out._shape));

  assertMatrixUnchanged(embedding.weight, before, "Embedding trainable=false harus membekukan weight");
  assert(zeroGrad._shape[0] === x._shape[0] && zeroGrad._shape[1] === x._shape[1], "Gradient input nol harus punya shape input");
  assert(zeroGrad._data.every((value) => value === 0), "Gradient input saat frozen harus seluruhnya nol");
}

function runTrainableTrueUpdateTest(): void {
  const embedding = createEmbedding({ trainable: true });
  const before = embedding.weight.clone();
  const x = mj.matrix([[1], [2], [1]]);
  const out = embedding.forward(x);
  embedding.backward(out, mj.ones(out._shape));

  assert(embedding.weight.get(0, 1) !== before.get(0, 1), "Token yang muncul harus berubah saat trainable=true");
  assert(embedding.weight.get(1, 2) !== before.get(1, 2), "Token kedua yang muncul harus berubah saat trainable=true");
  assert(embedding.weight.get(2, 3) === before.get(2, 3), "Token yang tidak muncul tidak boleh berubah pada sparse update");
}

function runFillWeightPreservesTrainableFromLayerJsonTest(): void {
  const tempDir = mkdtempSync(join(tmpdir(), "mlv1-embedding-"));
  const filePath = join(tempDir, "embedding-layer.json");
  const incomingWeight = [
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120],
  ];

  try {
    writeFileSync(filePath, JSON.stringify({
      name: "embedding layer",
      vocabSize: 4,
      embeddingDim: 3,
      trainable: true,
      weight: incomingWeight,
    }));

    const embedding = createEmbedding({ trainable: false });
    embedding.fillWeight(filePath);

    assertMatrixEqual(embedding.weight, incomingWeight, "fillWeight harus memuat weight dari layer JSON");
    assert(embedding.trainable === false, "fillWeight tidak boleh mengubah trainable");
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

function runFillWeightFromModelArrayJsonTest(): void {
  const tempDir = mkdtempSync(join(tmpdir(), "mlv1-embedding-"));
  const filePath = join(tempDir, "model-array.json");
  const incomingWeight = [
    [2, 4, 6, 8],
    [1, 3, 5, 7],
    [9, 11, 13, 15],
  ];

  try {
    writeFileSync(filePath, JSON.stringify([
      {
        name: "embedding layer",
        vocabSize: 4,
        embeddingDim: 3,
        weight: incomingWeight,
      },
      {
        name: "rnn layer",
        units: 3,
      },
    ]));

    const embedding = createEmbedding();
    embedding.fillWeight(filePath);
    assertMatrixEqual(embedding.weight, incomingWeight, "fillWeight harus menerima model JSON array");
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

function runFillWeightRejectsNonEmbeddingFirstLayerTest(): void {
  const tempDir = mkdtempSync(join(tmpdir(), "mlv1-embedding-"));
  const filePath = join(tempDir, "invalid-model-array.json");

  try {
    writeFileSync(filePath, JSON.stringify([
      {
        name: "dense layer",
        weight: [[1]],
      },
      {
        name: "embedding layer",
        vocabSize: 4,
        embeddingDim: 3,
        weight: [
          [1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
        ],
      },
    ]));

    const embedding = createEmbedding();
    assertThrows(
      () => embedding.fillWeight(filePath),
      "Embedding.fillWeight: JSON pretrained weight harus berasal dari Embedding layer atau model dengan layer pertama Embedding.",
      "Model JSON dengan layer pertama non-embedding harus ditolak"
    );
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

function runFillWeightDimensionMismatchTest(): void {
  const embedding = createEmbedding();
  assertThrows(
    () => embedding.fillWeight([[1, 2, 3, 4], [5, 6, 7, 8]]),
    "Embedding.fillWeight: dimensi weight tidak cocok. Expected [3, 4], got [2, 4].",
    "Mismatch jumlah row harus throw"
  );
  assertThrows(
    () => embedding.fillWeight([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]),
    "Embedding.fillWeight: dimensi weight tidak cocok. Expected [3, 4], got [3, 5].",
    "Mismatch jumlah col harus throw"
  );
}

function runFillWeightPreservesConfigTest(): void {
  const embedding = createEmbedding({
    trainable: false,
    alpha: 0.25,
    optimizer: "adam",
    status: "train",
    padTokenId: 0,
  });

  const before = {
    trainable: embedding.trainable,
    alpha: embedding.alpha,
    optimizerName: embedding.optimizerName,
    status: embedding.status,
    padTokenId: embedding.padTokenId,
    vocabSize: embedding.vocabSize,
    embeddingDim: embedding.embeddingDim,
    params: embedding.params,
  };

  embedding.fillWeight([
    [3, 3, 3, 3],
    [4, 4, 4, 4],
    [5, 5, 5, 5],
  ]);

  assert(embedding.trainable === before.trainable, "fillWeight tidak boleh mengubah trainable");
  assert(embedding.alpha === before.alpha, "fillWeight tidak boleh mengubah alpha");
  assert(embedding.optimizerName === before.optimizerName, "fillWeight tidak boleh mengubah optimizer");
  assert(embedding.status === before.status, "fillWeight tidak boleh mengubah status");
  assert(embedding.padTokenId === before.padTokenId, "fillWeight tidak boleh mengubah padTokenId");
  assert(embedding.vocabSize === before.vocabSize, "fillWeight tidak boleh mengubah vocabSize");
  assert(embedding.embeddingDim === before.embeddingDim, "fillWeight tidak boleh mengubah embeddingDim");
  assert(embedding.params === before.params, "fillWeight tidak boleh mengubah params");
}

function runSaveLoadPreservesTrainableTest(): void {
  const embedding = createEmbedding({ trainable: false, optimizer: "adam", alpha: 0.05, padTokenId: 0 });
  const saved = embedding.save();
  const restored = setLayers([saved])[0] as Embedding;
  const legacyRestored = setLayers([{
    ...saved,
    trainable: undefined,
  }])[0] as Embedding;

  assert(restored.trainable === false, "setLayers/save/load harus preserve trainable=false");
  assert(legacyRestored.trainable === true, "Artefak lama tanpa trainable harus default ke true");
}

function runSequentialFillEmbeddingWeightModelApiTest(): void {
  const model = new Sequential({
    layers: [
      createEmbedding({ trainable: false }),
    ],
  });

  model.fillEmbeddingWeight([
    [7, 7, 7, 7],
    [8, 8, 8, 8],
    [9, 9, 9, 9],
  ]);

  const embedding = model.layers[0] as Embedding;
  assert(embedding.trainable === false, "Sequential.fillEmbeddingWeight tidak boleh mengubah trainable embedding");
  assertMatrixEqual(embedding.weight, [
    [7, 7, 7, 7],
    [8, 8, 8, 8],
    [9, 9, 9, 9],
  ], "Sequential.fillEmbeddingWeight harus mengalir ke layer embedding");
}

function runTransformersModelLevelEmbeddingSupportTest(): void {
  const tempDir = mkdtempSync(join(tmpdir(), "mlv1-transformers-"));
  const filePath = join(tempDir, "transformers.json");

  try {
    const model = new Transformers({
      units: 3,
      seqLen: 4,
      vocabSize: 4,
      heads: 1,
      numBlocks: 1,
      alpha: 0.01,
      padTokenId: 0,
      embeddingTrainable: false,
      dropoutRate: 0,
    });

    model.fillEmbeddingWeight([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ]);
    model.save(filePath);

    const restored = new Transformers({
      units: 3,
      seqLen: 4,
      vocabSize: 4,
      heads: 1,
      numBlocks: 1,
      alpha: 0.01,
      padTokenId: 0,
      embeddingTrainable: true,
      dropoutRate: 0,
    });
    restored.load(filePath);

    const embedding = restored.layers[0] as Embedding;
    assert(embedding.trainable === false, "Transformers.load harus restore trainable embedding yang disimpan");
    assertMatrixEqual(embedding.weight, [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ], "Transformers.load harus restore weight embedding");
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

function runRecurrentModelEmbeddingTrainableConfigTest(): void {
  const model = new RecurrentModel({
    kind: "rnn",
    vocabSize: 4,
    embeddingDim: 3,
    embeddingTrainable: false,
    hiddenSize: 5,
    outputSize: 2,
    seqLen: 4,
    mode: "many-to-one",
    optimizer: "sgd",
    alpha: 0.01,
  });

  const embedding = model.layers[0] as Embedding;
  assert(embedding.trainable === false, "RecurrentModel harus meneruskan embeddingTrainable ke Embedding internal");

  model.fillEmbeddingWeight([
    [4, 3, 2, 1],
    [8, 7, 6, 5],
    [12, 11, 10, 9],
  ]);
  assert(embedding.trainable === false, "RecurrentModel.fillEmbeddingWeight tidak boleh mengubah trainable");
}

export function runEmbeddingTrainableCorrectnessSuite(): void {
  runTrainableFalseFreezeTest();
  runTrainableTrueUpdateTest();
  runFillWeightPreservesTrainableFromLayerJsonTest();
  runFillWeightFromModelArrayJsonTest();
  runFillWeightRejectsNonEmbeddingFirstLayerTest();
  runFillWeightDimensionMismatchTest();
  runFillWeightPreservesConfigTest();
  runSaveLoadPreservesTrainableTest();
  runSequentialFillEmbeddingWeightModelApiTest();
  runTransformersModelLevelEmbeddingSupportTest();
  runRecurrentModelEmbeddingTrainableConfigTest();

  console.log("=== Embedding Trainable Correctness ===");
  console.table([
    { check: "trainable=false freeze", status: "pass" },
    { check: "trainable=true sparse update", status: "pass" },
    { check: "fillWeight JSON + config preservation", status: "pass" },
    { check: "save/load trainable compatibility", status: "pass" },
    { check: "model-level embedding support", status: "pass" },
  ]);
}

