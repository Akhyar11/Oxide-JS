import { mj, setLoss } from "@oxide-js/core";
import {
  Activation,
  AttentionPooling,
  Dense,
  Dropout,
  Embedding,
  GRU,
  LayerNormalization,
  LSTM,
  SelfAttention,
} from "@oxide-js/layers";
import { Module, ModuleList, SequentialBlock, Trainer } from "@oxide-js/models";
import { fileURLToPath } from "url";

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}

function assertAllFinite(values: number[], message: string): void {
  if (!values.every((value) => Number.isFinite(value))) {
    throw new Error(message);
  }
}

function assertMatrixShape(
  value: { _shape: [number, number] },
  expected: [number, number],
  message: string
): void {
  assert(
    value._shape[0] === expected[0] && value._shape[1] === expected[1],
    `${message}. expected [${expected[0]}, ${expected[1]}], got [${value._shape[0]}, ${value._shape[1]}]`
  );
}

class ModuleListResidualClassifier extends Module {
  readonly input = new Dense({ units: 2, outputUnits: 6, activation: "linear", status: "input", optimizer: "adam", alpha: 0.01 });
  readonly stem = new SequentialBlock([
    new Activation({ activation: "relu", status: "train" }),
    new LayerNormalization({ units: 6, alpha: 0.01, optimizer: "adam" }),
    new Dropout({ rate: 0.1, status: "train" }),
  ]);
  readonly blocks = new ModuleList([
    new Dense({ units: 6, outputUnits: 6, activation: "relu", status: "train", optimizer: "adam", alpha: 0.01 }),
    new Dense({ units: 6, outputUnits: 6, activation: "relu", status: "train", optimizer: "adam", alpha: 0.01 }),
  ]);
  readonly output = new Dense({ units: 6, outputUnits: 1, activation: "linear", status: "output", optimizer: "adam", alpha: 0.01, loss: "mse" });

  forward(x: any) {
    let hidden = this.stem.forward(this.input.forward(x));
    for (const block of this.blocks) {
      hidden = mj.add(hidden, block.forward(hidden));
    }
    return this.output.forward(hidden);
  }
}

class DualInputDualOutputModule extends Module {
  readonly left = new Dense({ units: 2, outputUnits: 4, activation: "relu", status: "input", optimizer: "adam", alpha: 0.01 });
  readonly right = new Dense({ units: 2, outputUnits: 4, activation: "relu", status: "input", optimizer: "adam", alpha: 0.01 });
  readonly shared = new SequentialBlock([
    new Dense({ units: 4, outputUnits: 4, activation: "relu", status: "train", optimizer: "adam", alpha: 0.01 }),
    new LayerNormalization({ units: 4, alpha: 0.01, optimizer: "adam" }),
  ]);
  readonly scoreHead = new Dense({ units: 4, outputUnits: 1, activation: "linear", status: "output", optimizer: "adam", alpha: 0.01, loss: "mse" });
  readonly auxHead = new Dense({ units: 4, outputUnits: 1, activation: "linear", status: "output", optimizer: "adam", alpha: 0.01, loss: "mse" });

  forward(input: { left: any; right: any }) {
    const left = this.left.forward(input.left);
    const right = this.right.forward(input.right);
    const mixed = this.shared.forward(mj.add(left, right));
    return {
      score: this.scoreHead.forward(mixed),
      aux: this.auxHead.forward(mj.sub(left, right)),
    };
  }
}

class EmbeddingAttentionPoolingModule extends Module {
  readonly embedding = new Embedding({ vocabSize: 10, embeddingDim: 3, optimizer: "adam", alpha: 0.01, trainable: true });
  readonly selfAttention = new SelfAttention({ units: 3, outputUnits: 3, seqLen: 4, alpha: 0.01, status: "train" });
  readonly pooling = new AttentionPooling({ units: 3, maxTokens: 4, alpha: 0.01, optimizer: "adam", status: "train" });
  readonly output = new Dense({ units: 3, outputUnits: 1, activation: "linear", status: "output", optimizer: "adam", alpha: 0.01, loss: "mse" });

  forward(x: any) {
    const embedded = this.embedding.forward(x);
    const attended = this.selfAttention.forward(embedded);
    this.pooling.setValidLength(x._shape[0]);
    const pooled = this.pooling.forward(attended);
    return this.output.forward(pooled);
  }
}

class LstmModule extends Module {
  readonly lstm = new LSTM({
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
    return this.output.forward(this.lstm.forward(x));
  }
}

class GruModule extends Module {
  readonly gru = new GRU({
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
    return this.output.forward(this.gru.forward(x));
  }
}

export function runModuleLayerCompatibilitySuite(): void {
  console.log("  - Checking Module layer compatibility...");

  const residualModel = new ModuleListResidualClassifier();
  residualModel.compile({ alpha: 0.01, optimizer: "adam", error: "mse" });
  assert(residualModel.blocks.length === 2, `ModuleList should expose 2 blocks, got ${residualModel.blocks.length}`);
  residualModel.blocks.push(new Dense({ units: 6, outputUnits: 6, activation: "relu", status: "train", optimizer: "adam", alpha: 0.01 }));
  assert(residualModel.blocks.length === 3, `ModuleList.push should append blocks, got ${residualModel.blocks.length}`);

  const residualTrainer = new Trainer(residualModel, "mse");
  const residualResult = residualTrainer.fit(
    [
      mj.matrix([[0], [0]]),
      mj.matrix([[0], [1]]),
      mj.matrix([[1], [0]]),
      mj.matrix([[1], [1]]),
    ],
    [
      mj.matrix([[0]]),
      mj.matrix([[1]]),
      mj.matrix([[1]]),
      mj.matrix([[0]]),
    ],
    3,
    { batchSize: 2, shuffle: false, verbose: false }
  );
  assertAllFinite(residualResult.history.loss, "ModuleList + SequentialBlock losses must be finite");
  assert(residualModel.blocks.at(0) instanceof Dense, "ModuleList.at should return stored layers");
  residualModel.train();
  assert(residualModel.stem.layers[2] instanceof Dropout && residualModel.stem.layers[2].isTraining(), "Module.train should propagate into SequentialBlock");
  residualModel.eval();
  assert(residualModel.stem.layers[2] instanceof Dropout && !residualModel.stem.layers[2].isTraining(), "Module.eval should propagate into SequentialBlock");

  const dualModel = new DualInputDualOutputModule();
  dualModel.compile({ alpha: 0.01, optimizer: "adam", error: "mse" });
  const dualTrainer = new Trainer(dualModel, (target, prediction) => {
    const yTrue = target as { score: any; aux: any };
    const yPred = prediction as { score: any; aux: any };
    const [scoreLoss, scoreGrad] = setLoss("mse")(yTrue.score, yPred.score);
    const [auxLoss, auxGrad] = setLoss("mse")(yTrue.aux, yPred.aux);
    return {
      loss: scoreLoss + auxLoss * 0.25,
      grads: {
        score: scoreGrad,
        aux: mj.mul(auxGrad, 0.25),
      },
    };
  });

  const dualX = [
    { left: mj.matrix([[0], [0]]), right: mj.matrix([[0], [1]]) },
    { left: mj.matrix([[1], [0]]), right: mj.matrix([[0], [1]]) },
    { left: mj.matrix([[1], [1]]), right: mj.matrix([[1], [0]]) },
    { left: mj.matrix([[0], [1]]), right: mj.matrix([[1], [1]]) },
  ];
  const dualY = [
    { score: mj.matrix([[0]]), aux: mj.matrix([[0]]) },
    { score: mj.matrix([[1]]), aux: mj.matrix([[1]]) },
    { score: mj.matrix([[1]]), aux: mj.matrix([[0]]) },
    { score: mj.matrix([[0]]), aux: mj.matrix([[-1]]) },
  ];
  const dualResult = dualTrainer.fit(dualX, dualY, 3, { batchSize: 2, shuffle: false, verbose: false });
  assertAllFinite(dualResult.history.loss, "multi-input/multi-output Trainer losses must be finite");
  const dualPred = dualModel.predict<{ score: any; aux: any }>({ left: dualX[0].left, right: dualX[0].right });
  assertMatrixShape(dualPred.score, [1, 1], "dual-output score head shape mismatch");
  assertMatrixShape(dualPred.aux, [1, 1], "dual-output aux head shape mismatch");

  const attentionModel = new EmbeddingAttentionPoolingModule();
  attentionModel.compile({ alpha: 0.01, optimizer: "adam", error: "mse" });
  const attentionTrainer = new Trainer(attentionModel, "mse");
  const attentionBatch = attentionTrainer.trainBatch(
    mj.matrix([[1], [2], [3], [0]]),
    mj.matrix([[1]]),
    0.01
  );
  assert(Number.isFinite(attentionBatch.loss), "embedding/self-attention/pooling module loss must be finite");
  const attentionPred = attentionModel.predict(mj.matrix([[1], [2], [0], [0]]));
  assertMatrixShape(attentionPred, [1, 1], "embedding/self-attention/pooling predict shape mismatch");

  const lstmModel = new LstmModule();
  lstmModel.compile({ alpha: 0.01, optimizer: "adam", error: "mse" });
  const lstmTrainer = new Trainer(lstmModel, "mse");
  const lstmBatch = lstmTrainer.trainBatch(mj.matrix([[0, 1, 1], [1, 0, 1]]), mj.matrix([[1]]), 0.01);
  assert(Number.isFinite(lstmBatch.loss), "LSTM custom module loss must be finite");
  assertMatrixShape(lstmModel.predict(mj.matrix([[1, 0, 1], [0, 1, 0]])), [1, 1], "LSTM custom module predict shape mismatch");

  const gruModel = new GruModule();
  gruModel.compile({ alpha: 0.01, optimizer: "adam", error: "mse" });
  const gruTrainer = new Trainer(gruModel, "mse");
  const gruBatch = gruTrainer.trainBatch(mj.matrix([[1, 0, 1], [0, 1, 1]]), mj.matrix([[0]]), 0.01);
  assert(Number.isFinite(gruBatch.loss), "GRU custom module loss must be finite");
  assertMatrixShape(gruModel.predict(mj.matrix([[1, 1, 0], [0, 1, 0]])), [1, 1], "GRU custom module predict shape mismatch");

  console.log("    ✅ Module supports SequentialBlock, ModuleList, structured Trainer I/O, Activation, SelfAttention, AttentionPooling, LSTM, and GRU.");
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];

if (isMain) {
  runModuleLayerCompatibilitySuite();
}
