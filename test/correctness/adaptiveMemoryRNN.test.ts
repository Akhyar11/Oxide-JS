import { AdaptiveMemoryRNN } from "../../src/layers";
import mj from "../../src/math";
import Matrix from "../../src/matrix";

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
  const dx = backwardLayer.backward(mj.matrix([[0]]), mj.matrix([[0.1], [0.2], [0.3], [0.4], [0.5]]));
  assertShape(dx, 3, 3, "backward should return dx for input shape");
  assertFinite(dx, "backward dx");

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

  console.log("=== AdaptiveMemoryRNN Correctness ===");
  console.table([
    { check: "constructor and forward shapes", status: "pass" },
    { check: "stateful reset and memory update", status: "pass" },
    { check: "save/load and backward core", status: "pass" },
  ]);
}

if (require.main === module) {
  runAdaptiveMemoryRNNCorrectnessSuite();
}
