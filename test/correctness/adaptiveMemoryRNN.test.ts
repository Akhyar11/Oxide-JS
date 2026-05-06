import { AdaptiveMemoryRNN } from "@oxide-js/layers";
import { mj } from "@oxide-js/core";
import { isNativeAvailable } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { Sequential } from "@oxide-js/models";
import { Dense } from "@oxide-js/layers";

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
}

function assertShape(matrix: Matrix, rows: number, cols: number, message: string): void {
  assert(
    matrix._shape[0] === rows && matrix._shape[1] === cols,
    `${message}: expected [${rows},${cols}], got [${matrix._shape[0]},${matrix._shape[1]}]`
  );
}

function assertFinite(matrix: Matrix, message: string): void {
  for (const value of matrix._data) {
    assert(Number.isFinite(value), `${message}: found non-finite value ${value}`);
  }
}

function assertChanged(before: Matrix, after: Matrix, message: string): void {
  let changed = false;
  for (let i = 0; i < before._data.length; i++) {
    if (Math.abs(before._data[i] - after._data[i]) > 1e-7) {
      changed = true;
      break;
    }
  }
  assert(changed, message);
}

function assertMatrixClose(actual: Matrix, expected: Matrix, tolerance: number, message: string): void {
  assert(actual._data.length === expected._data.length, `${message}: length mismatch`);
  for (let i = 0; i < actual._data.length; i++) {
    if (Math.abs(actual._data[i] - expected._data[i]) > tolerance) {
      throw new Error(`${message}: mismatch at flat index ${i}, expected ${expected._data[i]}, got ${actual._data[i]}`);
    }
  }
}

function sampleInput(): Matrix {
  return mj.matrix([
    [0.2, 0.4, 0.6],
    [1.0, 0.0, 1.0],
    [0.5, 0.25, 0.75],
  ]);
}

export function runAdaptiveMemoryRNNCorrectnessSuite(): void {
  const layer = new AdaptiveMemoryRNN({ units: 3, hiddenUnits: 5, memorySlots: 4, memoryDim: 6 });
  assert(layer.units === 3 && layer.hiddenUnits === 5, "constructor minimal should set units and hiddenUnits");

  const outLast = layer.forward(sampleInput());
  assertShape(outLast, 5, 1, "forward returnSequences=false");
  assertFinite(outLast, "forward returnSequences=false");

  const sequenceLayer = new AdaptiveMemoryRNN({
    units: 3,
    hiddenUnits: 5,
    memorySlots: 4,
    memoryDim: 6,
    returnSequences: true,
  });
  const outSeq = sequenceLayer.forward(sampleInput());
  assertShape(outSeq, 5, 3, "forward returnSequences=true");
  assertFinite(outSeq, "forward returnSequences=true");

  let threw = false;
  try {
    layer.forward(mj.matrix([[1, 2, 3]]));
  } catch {
    threw = true;
  }
  assert(threw, "invalid input rows should throw");

  const memoryLayer = new AdaptiveMemoryRNN({ units: 3, hiddenUnits: 4, memorySlots: 3, memoryDim: 4 });
  memoryLayer.forward(sampleInput());
  assert(memoryLayer.memoryUsage.some((value) => value > 0), "forward should increment memoryUsage");
  assert(memoryLayer.memoryUsage.filter((value) => value > 0).length > 1, "forward should allocate across empty memory slots");
  assert(memoryLayer.memoryValues._data.some((value) => value !== 0), "forward should update memoryValues");

  const statefulLayer = new AdaptiveMemoryRNN({
    units: 3,
    hiddenUnits: 4,
    memorySlots: 3,
    memoryDim: 4,
    stateful: true,
  });
  statefulLayer.forward(sampleInput());
  const state = statefulLayer.getState();
  assert(state.h._data.some((value) => value !== 0), "stateful forward should update hidden state");
  assert(state.memoryUsage.some((value) => value > 0), "stateful forward should update memory state");
  statefulLayer.resetState();
  const resetState = statefulLayer.getState();
  assert(resetState.h._data.every((value) => value === 0), "resetState should clear hidden state");
  assert(resetState.memoryValues._data.every((value) => value === 0), "resetState should clear memoryValues");
  assert(resetState.memoryUsage.every((value) => value === 0), "resetState should clear memoryUsage");

  const saved = sequenceLayer.save();
  const loaded = new AdaptiveMemoryRNN({ units: 3, hiddenUnits: 5 });
  loaded.load(saved);
  assertShape(loaded.Wxh, 5, 9, "load should restore Wxh shape");
  assertShape(loaded.memoryValues, 6, 4, "load should restore memoryValues shape");
  assertShape(loaded.forward(sampleInput()), 5, 3, "loaded layer should forward with restored returnSequences");

  const backwardLayer = new AdaptiveMemoryRNN({ units: 3, hiddenUnits: 5, memorySlots: 4, memoryDim: 6 });
  backwardLayer.forward(sampleInput());
  const beforeWxh = backwardLayer.Wxh.clone();
  const beforeWhh = backwardLayer.Whh.clone();
  const beforeBh = backwardLayer.bh.clone();
  const beforeWq = backwardLayer.Wq.clone();
  const beforeWm = backwardLayer.Wm.clone();
  const beforeWg = backwardLayer.Wg.clone();
  const beforeBg = backwardLayer.bg.clone();
  const dx = backwardLayer.backward(mj.matrix([[0]]), mj.matrix([[0.1], [0.2], [0.3], [0.4], [0.5]]));
  assertShape(dx, 3, 3, "backward should return dx for input shape");
  assertFinite(dx, "backward dx");
  assert(dx._data.some((value) => Math.abs(value) > 1e-8), "backward dx should be non-zero for non-zero err");
  assertChanged(beforeWxh, backwardLayer.Wxh, "backward should update Wxh");
  assertChanged(beforeWhh, backwardLayer.Whh, "backward should update Whh");
  assertChanged(beforeBh, backwardLayer.bh, "backward should update bh");
  assertChanged(beforeWq, backwardLayer.Wq, "backward should update Wq");
  assertChanged(beforeWm, backwardLayer.Wm, "backward should update Wm");
  assertChanged(beforeWg, backwardLayer.Wg, "backward should update Wg");
  assertChanged(beforeBg, backwardLayer.bg, "backward should update bg");

  const batchLayer = new AdaptiveMemoryRNN({
    units: 3,
    hiddenUnits: 5,
    memorySlots: 4,
    memoryDim: 6,
    returnSequences: false,
  });
  const batchInput = mj.matrix([
    [0.2, 0.8, 0.4, 0.6, 0.1, 0.9],
    [1.0, 0.0, 0.0, 1.0, 0.5, 0.5],
    [0.5, 0.7, 0.25, 0.35, 0.75, 0.85],
  ]);
  const batchOut = batchLayer.forwardBatch(batchInput, 2);
  assertShape(batchOut, 5, 2, "forwardBatch returnSequences=false");
  assertFinite(batchOut, "forwardBatch output");
  const batchBeforeWq = batchLayer.Wq.clone();
  const batchBeforeWm = batchLayer.Wm.clone();
  const batchBeforeWg = batchLayer.Wg.clone();
  const batchBeforeBg = batchLayer.bg.clone();
  const batchDx = batchLayer.backwardBatch(
    mj.matrix([[0, 1]]),
    mj.matrix([
      [0.1, -0.1],
      [0.2, -0.2],
      [0.3, -0.3],
      [0.4, -0.4],
      [0.5, -0.5],
    ]),
    2
  );
  assertShape(batchDx, 3, 6, "backwardBatch should return dx for batched input shape");
  assertFinite(batchDx, "backwardBatch dx");
  assert(batchDx._data.some((value) => Math.abs(value) > 1e-8), "backwardBatch dx should be non-zero");
  assertChanged(batchBeforeWq, batchLayer.Wq, "backwardBatch should update Wq");
  assertChanged(batchBeforeWm, batchLayer.Wm, "backwardBatch should update Wm");
  assertChanged(batchBeforeWg, batchLayer.Wg, "backwardBatch should update Wg");
  assertChanged(batchBeforeBg, batchLayer.bg, "backwardBatch should update bg");

  let nativeParityChecked = "skipped";
  if (isNativeAvailable()) {
    const nativeLayer = new AdaptiveMemoryRNN({ units: 3, hiddenUnits: 4, memorySlots: 3, memoryDim: 4 });
    const jsLayer = new AdaptiveMemoryRNN({ units: 3, hiddenUnits: 4, memorySlots: 3, memoryDim: 4 });
    jsLayer.load(nativeLayer.save());
    nativeLayer.forward(sampleInput());
    jsLayer.forward(sampleInput());

    const errMatrix = mj.matrix([[0.1], [0.2], [0.3], [0.4]]);
    const nativeAny = nativeLayer as any;
    const jsAny = jsLayer as any;
    const seqLen = sampleInput()._shape[1];
    const externalError = jsAny.resolveError(mj.matrix([[0]]), errMatrix, seqLen) as Float32Array[];
    const perSampleErrors = [externalError];

    const nativeDWxh = mj.zeros(nativeLayer.Wxh._shape);
    const nativeDWhh = mj.zeros(nativeLayer.Whh._shape);
    const nativeDBh = mj.zeros(nativeLayer.bh._shape);
    const nativeDWq = mj.zeros(nativeLayer.Wq._shape);
    const nativeDWm = mj.zeros(nativeLayer.Wm._shape);
    const nativeDWg = mj.zeros(nativeLayer.Wg._shape);
    const nativeDBg = mj.zeros(nativeLayer.bg._shape);
    const nativeDx = mj.zeros([nativeLayer.units, seqLen]);

    const jsDWxh = mj.zeros(jsLayer.Wxh._shape);
    const jsDWhh = mj.zeros(jsLayer.Whh._shape);
    const jsDBh = mj.zeros(jsLayer.bh._shape);
    const jsDWq = mj.zeros(jsLayer.Wq._shape);
    const jsDWm = mj.zeros(jsLayer.Wm._shape);
    const jsDWg = mj.zeros(jsLayer.Wg._shape);
    const jsDBg = mj.zeros(jsLayer.bg._shape);
    const jsDx = mj.zeros([jsLayer.units, seqLen]);

    const nativeOk = nativeAny.runNativeBackward(
      [nativeAny.stepCaches],
      perSampleErrors,
      nativeDWxh,
      nativeDWhh,
      nativeDBh,
      nativeDWq,
      nativeDWm,
      nativeDWg,
      nativeDBg,
      nativeDx._data,
      seqLen,
      1
    );
    assert(nativeOk, "native parity test expected native backward to run");

    jsAny.backwardThroughStepCaches(
      jsAny.stepCaches,
      externalError,
      jsDWxh,
      jsDWhh,
      jsDBh,
      jsDWq,
      jsDWm,
      jsDWg,
      jsDBg,
      jsDx._data,
      seqLen,
      0,
      1
    );

    assertMatrixClose(nativeDWxh, jsDWxh, 1e-5, "native parity dWxh");
    assertMatrixClose(nativeDWhh, jsDWhh, 1e-5, "native parity dWhh");
    assertMatrixClose(nativeDBh, jsDBh, 1e-5, "native parity dBh");
    assertMatrixClose(nativeDWq, jsDWq, 1e-5, "native parity dWq");
    assertMatrixClose(nativeDWm, jsDWm, 1e-5, "native parity dWm");
    assertMatrixClose(nativeDWg, jsDWg, 1e-5, "native parity dWg");
    assertMatrixClose(nativeDBg, jsDBg, 1e-5, "native parity dBg");
    assertMatrixClose(nativeDx, jsDx, 1e-5, "native parity dx");
    nativeParityChecked = "pass";
  }

  const trainModel = new Sequential({
    layers: [
      new AdaptiveMemoryRNN({
        units: 2,
        hiddenUnits: 4,
        memorySlots: 2,
        memoryDim: 3,
        alpha: 0.05,
        optimizer: "sgd",
        status: "input",
      }),
      new Dense({
        units: 4,
        outputUnits: 2,
        activation: "linear",
        status: "output",
        loss: "softmaxCrossEntropy",
        alpha: 0.05,
        optimizer: "sgd",
      }),
    ],
  });

  const trainX = [
    mj.matrix([[1, 0, 1], [0, 1, 0]]),
    mj.matrix([[0, 1, 0], [1, 0, 1]]),
    mj.matrix([[1, 1, 0], [0, 0, 1]]),
    mj.matrix([[0, 0, 1], [1, 1, 0]]),
  ];
  const trainY = [
    mj.matrix([[0]]),
    mj.matrix([[1]]),
    mj.matrix([[0]]),
    mj.matrix([[1]]),
  ];
  const history: number[] = [];
  trainModel.fit(trainX, trainY, 12, {
    batchSize: 1,
    shuffle: false,
    verbose: false,
    onEpochEnd: (_epoch: number, loss: number) => history.push(loss),
  });
  assert(history[history.length - 1] < history[0], "tiny AdaptiveMemoryRNN model should reduce loss on synthetic data");

  console.log("=== AdaptiveMemoryRNN Correctness ===");
  console.table([
    { check: "constructor and forward shapes", status: "pass" },
    { check: "stateful reset and memory update", status: "pass" },
    { check: "save/load and backward core", status: "pass" },
    { check: "memory/query/write params update", status: "pass" },
    { check: "batch backward updates memory path", status: "pass" },
    { check: "native backward parity", status: nativeParityChecked },
    { check: "tiny synthetic overfit", status: "pass" },
  ]);
}

