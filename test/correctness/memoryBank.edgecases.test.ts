import { MemoryBank } from "@oxide-js/layers";
import { mj } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
}

function assertFiniteMatrix(matrix: Matrix, message: string): void {
  for (let i = 0; i < matrix._data.length; i++) {
    if (!Number.isFinite(matrix._data[i])) {
      throw new Error(`${message}: non-finite value at flat index ${i}`);
    }
  }
}

function countFilled(layer: MemoryBank): number {
  return layer.getMemoryState().memoryFilled.reduce((sum, value) => sum + value, 0);
}

export function runMemoryBankEdgeCaseSuite(): void {
  // 1) Empty memory in project mode should produce finite output and zero reads.
  {
    const layer = new MemoryBank({
      units: 3,
      memorySlots: 2,
      outputUnits: 3,
      mode: "project",
      similarity: "cosine",
      readTopK: 2,
      writeEnabled: false,
      clipGradient: false,
    });

    const out = layer.forward(mj.matrix([[1], [-2], [0.5]]));
    const trace = layer.getDebugTrace();

    assert(out._shape[0] === 3 && out._shape[1] === 1, "empty-memory forward should preserve [outputUnits, cols]");
    assertFiniteMatrix(out, "empty-memory forward output");
    assert(trace.length === 1, "empty-memory trace should have one entry");
    assert(trace[0].readSlots.length === 0, "empty-memory trace should have no read slots");
    assert(trace[0].readNorm === 0, "empty-memory read vector should stay zero");
  }

  // 2) Cosine mode must remain finite with a zero-norm stored key and backward pass.
  {
    const layer = new MemoryBank({
      units: 2,
      memorySlots: 2,
      outputUnits: 2,
      mode: "project",
      similarity: "cosine",
      readTopK: 2,
      writeEnabled: false,
      optimizer: "sgd",
      alpha: 0.01,
      clipGradient: false,
    });

    layer.forward(mj.matrix([[0], [0]]));
    layer.setMemoryState({
      memoryKeys: [
        [0, 1],
        [0, 0],
      ],
      memoryValues: [
        [1, 0],
        [0, 1],
      ],
      memoryFilled: [1, 1],
      memoryUsage: [1, 1],
      memoryAge: [1, 2],
      memoryStep: 2,
      units: 2,
      memoryDim: 2,
      memorySlots: 2,
    });

    const out = layer.forward(mj.matrix([[1], [0]]));
    const dx = layer.backward(mj.matrix([[]]), mj.matrix([[0.25], [-0.75]]));

    assertFiniteMatrix(out, "zero-key cosine forward output");
    assertFiniteMatrix(dx, "zero-key cosine backward dx");
  }

  // 3) Multi-column forward with writes active should let later columns read earlier writes.
  {
    const layer = new MemoryBank({
      units: 2,
      memorySlots: 3,
      outputUnits: 2,
      mode: "project",
      similarity: "dot",
      readTopK: 1,
      optimizer: "sgd",
      alpha: 0.01,
      clipGradient: false,
    });

    const out = layer.forward(mj.matrix([
      [1, 1],
      [0, 1],
    ]));
    const trace = layer.getDebugTrace();

    assertFiniteMatrix(out, "multi-column write-active output");
    assert(trace.length === 2, "multi-column write-active trace should have two entries");
    assert(trace[0].column === 0 && trace[1].column === 1, "trace columns should preserve forward order");
    assert(trace[0].writeCommitted === true, "first column should commit a write");
    assert(trace[1].readSlots.length > 0, "second column should be able to read the first write");
    assert(countFilled(layer) === 2, "two columns with writes should fill two memory slots");
  }

  // 4) Sequence detach must clear history without clearing runtime memory.
  {
    const layer = new MemoryBank({
      units: 2,
      memorySlots: 2,
      outputUnits: 2,
      mode: "project",
      similarity: "dot",
      readTopK: 1,
    });

    layer.beginSequence({ maxHistorySteps: 4 });
    layer.forward(mj.matrix([[1], [0]]));
    layer.forward(mj.matrix([[0], [1]]));
    const filledBeforeDetach = countFilled(layer);

    assert(layer.getSequenceLength() === 2, "sequence history should track two steps before detach");
    layer.detachSequence();
    assert(layer.getSequenceLength() === 0, "detachSequence should clear history");
    assert(countFilled(layer) === filledBeforeDetach, "detachSequence should not clear runtime memory");
    layer.endSequence();
  }
}

