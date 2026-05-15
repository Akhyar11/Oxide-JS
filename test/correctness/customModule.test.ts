import { mj, Matrix } from "@oxide-js/core";
import {
  Convolution,
  Dense,
  Dropout,
  Embedding,
  Flatten,
  LayerNormalization,
  MultiHeadAttention,
  PositionalEncoding,
  RNN,
} from "@oxide-js/layers";
import { Module, Trainer } from "@oxide-js/models";
import { fileURLToPath } from "url";

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}

class ResidualDenseBlock extends Module {
  readonly left = new Dense({ units: 6, outputUnits: 6, activation: "relu", status: "train", optimizer: "adam", alpha: 0.01 });
  readonly right = new Dense({ units: 6, outputUnits: 6, activation: "relu", status: "train", optimizer: "adam", alpha: 0.01 });

  forward(x: any) {
    const a = this.left.forward(x);
    const b = this.right.forward(a);
    return mj.add(x, b);
  }
}

class ResidualXorModule extends Module {
  readonly input = new Dense({ units: 2, outputUnits: 6, activation: "relu", status: "input", optimizer: "adam", alpha: 0.01 });
  readonly blocks = [new ResidualDenseBlock(), new ResidualDenseBlock()];
  readonly output = new Dense({ units: 6, outputUnits: 1, activation: "linear", status: "output", optimizer: "adam", alpha: 0.01, loss: "mse" });

  forward(x: any) {
    let hidden = this.input.forward(x);
    for (const block of this.blocks) {
      hidden = block.forward(hidden);
    }
    return this.output.forward(hidden);
  }
}

class TokenAttentionModule extends Module {
  readonly embedding = new Embedding({ vocabSize: 12, embeddingDim: 4, optimizer: "adam", alpha: 0.01, trainable: true });
  readonly positional = new PositionalEncoding({ dModel: 4, maxSeqLen: 4 });
  readonly attention = new MultiHeadAttention({ units: 4, heads: 2, seqLen: 4, alpha: 0.01, status: "input", clipGradient: 5.0 });
  readonly norm = new LayerNormalization({ units: 4, alpha: 0.01, optimizer: "adam" });
  readonly dropout = new Dropout({ rate: 0.2, status: "train" });
  readonly flatten = new Flatten("train");
  readonly output = new Dense({ units: 16, outputUnits: 1, activation: "linear", status: "output", optimizer: "adam", alpha: 0.01, loss: "mse" });

  forward(x: any) {
    const embedded = this.embedding.forward(x);
    const positioned = this.positional.forward(embedded, 0, x._shape[0]);
    const attended = this.attention.forward(positioned);
    const residual = mj.add(positioned, attended);
    const normalized = this.norm.forward(residual);
    const dropped = this.dropout.forward(normalized);
    const flat = this.flatten.forward(dropped);
    return this.output.forward(flat);
  }
}

class ConvClassifierModule extends Module {
  readonly conv = new Convolution({
    kernelSize: [2, 2],
    inputShape: [3, 3],
    activation: "relu",
    status: "input",
    optimizer: "adam",
    alpha: 0.01,
    loss: "mse",
  });
  readonly flatten = new Flatten("train");
  readonly output = new Dense({ units: 4, outputUnits: 1, activation: "linear", status: "output", optimizer: "adam", alpha: 0.01, loss: "mse" });

  forward(x: any) {
    const features = this.conv.forward(x);
    const flat = this.flatten.forward(features);
    return this.output.forward(flat);
  }
}

class RecurrentSequenceModule extends Module {
  readonly rnn = new RNN({
    units: 2,
    hiddenUnits: 4,
    returnSequences: false,
    status: "input",
    optimizer: "adam",
    alpha: 0.01,
    loss: "mse",
  });
  readonly output = new Dense({ units: 4, outputUnits: 1, activation: "linear", status: "output", optimizer: "adam", alpha: 0.01, loss: "mse" });

  forward(x: any) {
    const hidden = this.rnn.forward(x);
    return this.output.forward(hidden);
  }
}

function assertAllFinite(values: number[], message: string): void {
  if (!values.every((value) => Number.isFinite(value))) {
    throw new Error(message);
  }
}

function assertParamsCleared(module: Module, message: string): void {
  module.zeroGrad();
  assert(module.parameters().every((param) => param.grad === null), message);
}

export function runCustomModuleCorrectnessSuite(): void {
  console.log("  - Checking custom Module + Trainer architecture...");

  const model = new ResidualXorModule();
  model.compile({ alpha: 0.01, optimizer: "adam", error: "mse" });

  const params = model.parameters();
  assert(params.length === 12, `nested custom module should expose 12 trainable matrices, got ${params.length}`);

  const trainer = new Trainer(model, "mse");
  const X = [
    mj.matrix([[0], [0]]),
    mj.matrix([[0], [1]]),
    mj.matrix([[1], [0]]),
    mj.matrix([[1], [1]]),
  ];
  const Y = [
    mj.matrix([[0]]),
    mj.matrix([[1]]),
    mj.matrix([[1]]),
    mj.matrix([[0]]),
  ];

  const result = trainer.fit(X, Y, 4, {
    batchSize: 2,
    shuffle: false,
    verbose: false,
  });

  assert(result.history.loss.length === 4, `custom module fit should record 4 losses, got ${result.history.loss.length}`);
  assertAllFinite(result.history.loss, "custom module fit losses must be finite");
  const pred = model.predict<Matrix>(X[0]);
  assert(pred._shape[0] === 1 && pred._shape[1] === 1, `custom module predict returned unexpected shape [${pred._shape[0]}, ${pred._shape[1]}]`);
  assertParamsCleared(model, "custom module zeroGrad should clear all parameter gradients");

  const tokenModel = new TokenAttentionModule();
  tokenModel.compile({ alpha: 0.01, optimizer: "adam", error: "mse" });
  assert(tokenModel.parameters().length === 10, `token attention module should expose 10 trainable matrices, got ${tokenModel.parameters().length}`);
  tokenModel.train();
  assert(tokenModel.dropout.isTraining(), "Module.train should propagate training mode to Dropout");
  tokenModel.eval();
  assert(!tokenModel.dropout.isTraining(), "Module.eval should propagate eval mode to Dropout");
  const tokenTrainer = new Trainer(tokenModel, "mse");
  const tokenBatch = tokenTrainer.trainBatch(mj.matrix([[1], [2], [3], [4]]), mj.matrix([[1]]), 0.01);
  assert(Number.isFinite(tokenBatch.loss), "token attention module trainBatch loss must be finite");
  const tokenPred = tokenModel.predict<Matrix>(mj.matrix([[1], [2], [3], [0]]));
  assert(tokenPred._shape[0] === 1 && tokenPred._shape[1] === 1, `token attention predict returned unexpected shape [${tokenPred._shape[0]}, ${tokenPred._shape[1]}]`);
  assertParamsCleared(tokenModel, "token attention module zeroGrad should clear all parameter gradients");

  const convModel = new ConvClassifierModule();
  convModel.compile({ alpha: 0.01, optimizer: "adam", error: "mse" });
  assert(convModel.parameters().length === 4, `conv custom module should expose 4 trainable matrices, got ${convModel.parameters().length}`);
  const convTrainer = new Trainer(convModel, "mse");
  const convHistory = convTrainer.fit(
    [
      mj.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
      mj.matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
    ],
    [
      mj.matrix([[1]]),
      mj.matrix([[0]]),
    ],
    2,
    { batchSize: 1, shuffle: false, verbose: false }
  );
  assertAllFinite(convHistory.history.loss, "conv custom module losses must be finite");
  const convPred = convModel.predict<Matrix>(mj.matrix([[1, 1, 0], [0, 1, 0], [0, 0, 1]]));
  assert(convPred._shape[0] === 1 && convPred._shape[1] === 1, `conv custom module predict returned unexpected shape [${convPred._shape[0]}, ${convPred._shape[1]}]`);
  assertParamsCleared(convModel, "conv custom module zeroGrad should clear all parameter gradients");

  const recurrentModel = new RecurrentSequenceModule();
  recurrentModel.compile({ alpha: 0.01, optimizer: "adam", error: "mse" });
  assert(recurrentModel.parameters().length === 5, `recurrent custom module should expose 5 trainable matrices, got ${recurrentModel.parameters().length}`);
  const recurrentTrainer = new Trainer(recurrentModel, "mse");
  const recurrentBatch = recurrentTrainer.trainBatch(
    mj.matrix([[0, 1, 1], [1, 0, 1]]),
    mj.matrix([[1]]),
    0.01
  );
  assert(Number.isFinite(recurrentBatch.loss), "recurrent custom module trainBatch loss must be finite");
  const recurrentPred = recurrentModel.predict<Matrix>(mj.matrix([[1, 0, 1], [0, 1, 0]]));
  assert(recurrentPred._shape[0] === 1 && recurrentPred._shape[1] === 1, `recurrent custom module predict returned unexpected shape [${recurrentPred._shape[0]}, ${recurrentPred._shape[1]}]`);
  assertParamsCleared(recurrentModel, "recurrent custom module zeroGrad should clear all parameter gradients");

  console.log("    ✅ Custom Module + Trainer supports nested blocks, arrays, token/attention, convolution, and recurrent custom architectures.");
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];

if (isMain) {
  runCustomModuleCorrectnessSuite();
}
