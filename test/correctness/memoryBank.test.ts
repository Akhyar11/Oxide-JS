import { unlinkSync } from "fs";
import { MemoryBank } from "@oxide-js/layers";
import { mj } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { setLayers } from "@oxide-js/layers";

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
}

function assertClose(actual: number, expected: number, tol: number, message: string): void {
  if (Math.abs(actual - expected) > tol) {
    throw new Error(`${message}: expected ${expected}, got ${actual}`);
  }
}

function assertMatrixClose(a: Matrix, b: Matrix, tol: number, message: string): void {
  assert(a._shape[0] === b._shape[0] && a._shape[1] === b._shape[1], `${message}: shape mismatch`);
  for (let i = 0; i < a._data.length; i++) {
    if (Math.abs(a._data[i] - b._data[i]) > tol) {
      throw new Error(`${message}: mismatch at index ${i}, ${a._data[i]} vs ${b._data[i]}`);
    }
  }
}

function setIdentity(m: Matrix): void {
  m._data.fill(0);
  const [rows, cols] = m._shape;
  for (let i = 0; i < Math.min(rows, cols); i++) m._data[i * cols + i] = 1;
}

function cloneState(state: any): any {
  return JSON.parse(JSON.stringify(state));
}

function makeLayer(extra: Partial<ConstructorParameters<typeof MemoryBank>[0]> = {}): MemoryBank {
  const units = extra.units ?? 3;
  const outputUnits = extra.outputUnits ?? 3;
  const layer = new MemoryBank({
    units,
    memorySlots: 3,
    outputUnits,
    mode: "project",
    similarity: "cosine",
    readTopK: 1,
    optimizer: "sgd",
    alpha: 0.1,
    ...extra,
  });
  layer.forward(mj.zeros([units, 1]));
  setIdentity((layer as any).queryKernel);
  (layer as any).writeGateKernel._data.fill(0);
  (layer as any).writeGateBias._data[0] = 20;
  (layer as any).writeQueryKernel._data.fill(0);
  for (let i = 0; i < units; i++) (layer as any).writeQueryKernel._data[i * (units + units) + i] = 1;
  if ((layer as any).needKernel) (layer as any).needKernel._data.fill(0);
  if ((layer as any).outputKernel) {
    (layer as any).outputKernel._data.fill(0);
    for (let i = 0; i < 3; i++) (layer as any).outputKernel._data[i * 6 + 3 + i] = 1;
  }
  if ((layer as any).outputBias) (layer as any).outputBias._data.fill(0);
  layer.resetMemory();
  layer.eval();
  return layer;
}

export function runMemoryBankCorrectnessSuite(): void {
  // 1) backward updates differentiable params but not runtime state
  {
    const layer = makeLayer({ readTopK: 2 });
    layer.setMemoryState({
      memoryKeys: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
      ],
      memoryValues: [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
      ],
      memoryFilled: [1, 1, 0],
      memoryUsage: [1, 1, 0],
      memoryAge: [1, 2, 0],
      memoryStep: 2,
      units: 3,
      memoryDim: 3,
      memorySlots: 3,
    });

    const q0 = (layer as any).queryKernel.clone();
    const wg0 = (layer as any).writeGateKernel.clone();
    const wgb0 = (layer as any).writeGateBias.clone();
    const n0 = (layer as any).needKernel.clone();
    const o0 = (layer as any).outputKernel.clone();
    const b0 = (layer as any).outputBias.clone();

    const out = layer.forward(mj.matrix([[1], [0], [0]]));
    const stateAfterForward = cloneState(layer.getMemoryState());
    const dx = layer.backward(mj.matrix([[]]), mj.matrix([[1], [0], [-1]]));

    assert(dx._shape[0] === 3 && dx._shape[1] === 1, "backward dx shape should match input");
    assert(JSON.stringify(stateAfterForward) === JSON.stringify(layer.getMemoryState()), "backward must not mutate runtime memory");
    assert(out._shape[0] === 3 && out._shape[1] === 1, "project output shape should be [outputUnits, cols]");

    let queryChanged = false;
    let needChanged = false;
    let outputChanged = false;
    let biasChanged = false;
    for (let i = 0; i < q0._data.length; i++) if (Math.abs(q0._data[i] - (layer as any).queryKernel._data[i]) > 1e-12) queryChanged = true;
    for (let i = 0; i < n0._data.length; i++) if (Math.abs(n0._data[i] - (layer as any).needKernel._data[i]) > 1e-12) needChanged = true;
    for (let i = 0; i < o0._data.length; i++) if (Math.abs(o0._data[i] - (layer as any).outputKernel._data[i]) > 1e-12) outputChanged = true;
    for (let i = 0; i < b0._data.length; i++) if (Math.abs(b0._data[i] - (layer as any).outputBias._data[i]) > 1e-12) biasChanged = true;

    assert(queryChanged, "backward should update queryKernel");
    assert(needChanged, "backward should update needKernel");
    assert(outputChanged, "backward should update outputKernel");
    assert(biasChanged, "backward should update outputBias");
  }

  // 2) write stores input directly and least-relevant replacement works
  {
    const layer = makeLayer();
    layer.forward(mj.matrix([[1], [0], [0]]));
    layer.forward(mj.matrix([[0], [1], [0]]));
    layer.forward(mj.matrix([[0], [0], [1]]));
    const before = layer.getMemoryState();
    assert(before.memoryFilled.every((v) => v === 1), "all slots should be filled after 3 writes");

    layer.forward(mj.matrix([[0], [0], [-1]]));
    const after = layer.getMemoryState();
    let replaced = 0;
    for (let slot = 0; slot < 3; slot++) {
      if (Math.abs(after.memoryValues[2][slot] + 1) < 1e-6) replaced++;
    }
    assert(replaced === 1, "full memory should replace exactly one least-relevant slot");
  }

  // 3) concat mode exposes [x; read]
  {
    const layer = new MemoryBank({
      units: 2,
      memorySlots: 2,
      outputUnits: 2,
      mode: "concat",
      similarity: "cosine",
      readTopK: 1,
      writeEnabled: false,
    });
    layer.forward(mj.matrix([[0], [0]]));
    layer.eval();
    setIdentity((layer as any).queryKernel);
    layer.setMemoryState({
      memoryKeys: [
        [1, 0],
        [0, 0],
      ],
      memoryValues: [
        [0, 0],
        [1, 0],
      ],
      memoryFilled: [1, 0],
      memoryUsage: [1, 0],
      memoryAge: [1, 0],
      memoryStep: 1,
      units: 2,
      memoryDim: 2,
      memorySlots: 2,
    });

    const out = layer.forward(mj.matrix([[1], [0]]));
    assert(out._shape[0] === 4 && out._shape[1] === 1, "concat output shape should be [2*units, cols]");
    assertClose(out._data[0], 1, 1e-6, "concat should keep x dim0");
    assertClose(out._data[1], 0, 1e-6, "concat should keep x dim1");
    assertClose(out._data[2], 0, 1e-6, "concat should append read dim0");
    assertClose(out._data[3], 1, 1e-6, "concat should append read dim1");
  }

  // 3b) overwriteThreshold controls overwrite vs allocate even when empty slots remain
  {
    const makeOverwriteLayer = (overwriteThreshold: number): MemoryBank => {
      const layer = new MemoryBank({
        units: 2,
        memorySlots: 3,
        outputUnits: 2,
        mode: "project",
        similarity: "dot",
        readTopK: 1,
        writeEnabled: true,
        overwriteThreshold,
        optimizer: "sgd",
        alpha: 0.1,
      });
      layer.forward(mj.zeros([2, 1]));
      setIdentity((layer as any).queryKernel);
      (layer as any).writeGateKernel._data.fill(0);
      (layer as any).writeGateBias._data[0] = 20;
      (layer as any).writeQueryKernel._data.fill(0);
      (layer as any).writeQueryKernel._data[0 * 4 + 0] = 1;
      (layer as any).writeQueryKernel._data[1 * 4 + 1] = 1;
      layer.resetMemory();
      layer.eval();
      return layer;
    };

    const overwriteLayer = makeOverwriteLayer(0.1);
    overwriteLayer.forward(mj.matrix([[1], [0]]));
    overwriteLayer.forward(mj.matrix([[1], [0]]));
    const overwriteState = overwriteLayer.getMemoryState();
    assert(overwriteState.memoryFilled.filter((v) => v === 1).length === 1, "similar write harus overwrite slot lama meski masih ada slot kosong");

    const allocateLayer = makeOverwriteLayer(2.0);
    allocateLayer.forward(mj.matrix([[1], [0]]));
    allocateLayer.forward(mj.matrix([[1], [0]]));
    const allocateState = allocateLayer.getMemoryState();
    assert(allocateState.memoryFilled.filter((v) => v === 1).length === 2, "threshold tinggi harus memaksa allocate slot baru");
  }

  // 4) sequence mode enables BPTT across separate forward calls
  {
    const layer = makeLayer();
    assert(layer.isSequenceActive() === false, "sequence mode should start inactive");
    layer.beginSequence({ maxHistorySteps: 8 });
    layer.forward(mj.matrix([[1], [0], [0]]));
    layer.forward(mj.matrix([[0], [1], [0]]));
    layer.forward(mj.matrix([[1], [0], [0]]));
    assert(layer.getSequenceLength() === 3, "sequence history should collect 3 steps");

    const o0 = (layer as any).outputKernel.clone();
    const wg0 = (layer as any).writeGateKernel.clone();
    const err = mj.zeros([3, 3]);
    err._data[0 * 3 + 2] = 1;
    err._data[1 * 3 + 2] = -1;
    const dx = layer.backwardSequence(err);
    assert(dx._shape[0] === 3 && dx._shape[1] === 3, "backwardSequence should return dx for the full active sequence");

    let outputChanged = false;
    let writeGateChanged = false;
    for (let i = 0; i < o0._data.length; i++) if (Math.abs(o0._data[i] - (layer as any).outputKernel._data[i]) > 1e-12) outputChanged = true;
    for (let i = 0; i < wg0._data.length; i++) if (Math.abs(wg0._data[i] - (layer as any).writeGateKernel._data[i]) > 1e-12) writeGateChanged = true;
    assert(outputChanged, "backwardSequence should update trainable sequence-path parameters");
    assert(writeGateChanged, "backwardSequence should update writeGateKernel when writes affect future reads");

    layer.detachSequence();
    assert(layer.getSequenceLength() === 0, "detachSequence should clear active history");
    layer.endSequence();
    assert(layer.isSequenceActive() === false, "endSequence should deactivate sequence mode");
  }

  // 5) save/load roundtrip preserves config, state, and output
  {
    const layer = makeLayer({ memorySlots: 2 });
    layer.forward(mj.matrix([[1], [0], [0]]));
    layer.forward(mj.matrix([[0], [1], [0]]));
    layer.freezeWrites();

    const probe = mj.matrix([[1], [0], [0]]);
    const expected = layer.forward(probe);
    const saved = layer.save();
    const [loaded] = setLayers([saved]) as MemoryBank[];
    loaded.freezeWrites();
    const loadedState = cloneState(loaded.getMemoryState());
    const actual = loaded.forward(probe);

    assert(saved.config.mode === "project", "save should preserve simplified mode");
    assert(JSON.stringify(saved.memoryState) === JSON.stringify(loadedState), "save/load should preserve runtime state");
    assertMatrixClose(expected, actual, 1e-6, "save/load should preserve output");
  }

  // 6) saveMemory/loadMemory roundtrip works
  {
    const path = "/tmp/ml-v1-memory-bank-state.json";
    const layer = makeLayer({ memorySlots: 2 });
    layer.forward(mj.matrix([[1], [0], [0]]));
    layer.saveMemory(path);

    const restored = makeLayer({ memorySlots: 2 });
    restored.loadMemory(path);
    assert(JSON.stringify(layer.getMemoryState()) === JSON.stringify(restored.getMemoryState()), "saveMemory/loadMemory should roundtrip runtime state");

    try {
      unlinkSync(path);
    } catch {}
  }

  // 7) load() must fully reconfigure an already-initialized instance
  {
    const source = new MemoryBank({
      units: 2,
      memorySlots: 2,
      outputUnits: 2,
      mode: "concat",
      similarity: "cosine",
      readTopK: 1,
      writeEnabled: false,
    });
    source.forward(mj.matrix([[0], [0]]));
    source.eval();
    setIdentity((source as any).queryKernel);
    source.setMemoryState({
      memoryKeys: [
        [1, 0],
        [0, 0],
      ],
      memoryValues: [
        [0, 0],
        [1, 0],
      ],
      memoryFilled: [1, 0],
      memoryUsage: [1, 0],
      memoryAge: [1, 0],
      memoryStep: 1,
      units: 2,
      memoryDim: 2,
      memorySlots: 2,
    });

    const saved = source.save();
    const reused = makeLayer({ units: 3, memorySlots: 3, outputUnits: 3, mode: "project" });
    reused.load(saved);
    reused.freezeWrites();

    const actual = reused.forward(mj.matrix([[1], [0]]));
    assert(reused.mode === "concat", "load() should replace previous mode on initialized instances");
    assert(actual._shape[0] === 4 && actual._shape[1] === 1, "loaded concat layer should output [2*units, cols]");
    assertClose(actual._data[0], 1, 1e-6, "loaded concat layer should keep x dim0");
    assertClose(actual._data[1], 0, 1e-6, "loaded concat layer should keep x dim1");
    assertClose(actual._data[2], 0, 1e-6, "loaded concat layer should append read dim0");
    assertClose(actual._data[3], 1, 1e-6, "loaded concat layer should append read dim1");
  }

  // 8) enableWrites()/disableWrites() must control writeEnabled as named
  {
    const layer = new MemoryBank({
      units: 2,
      memorySlots: 2,
      outputUnits: 2,
      mode: "project",
      writeEnabled: false,
    });
    layer.forward(mj.matrix([[0], [0]]));
    layer.eval();
    layer.resetMemory();

    layer.enableWrites();
    layer.forward(mj.matrix([[1], [0]]));
    assert(layer.getLastWriteInfo() !== null, "enableWrites() should re-enable writes when writeEnabled was false");

    layer.disableWrites();
    layer.forward(mj.matrix([[0], [1]]));
    assert(layer.getLastWriteInfo() === null, "disableWrites() should disable writes, not only freeze them");
  }

  // 9) load() must reset transient runtime flags that are not serialized
  {
    const source = new MemoryBank({
      units: 2,
      memorySlots: 2,
      outputUnits: 2,
      mode: "project",
    });
    source.forward(mj.matrix([[1], [0]]));
    source.eval();
    const saved = source.save();

    const reused = new MemoryBank({
      units: 2,
      memorySlots: 2,
      outputUnits: 2,
      mode: "project",
    });
    reused.beginSequence({ maxHistorySteps: 4 });
    reused.freezeWrites();
    reused.load(saved);
    reused.eval();

    assert(reused.isSequenceActive() === false, "load() should clear prior sequence-active state");
    reused.forward(mj.matrix([[0], [1]]));
    assert(reused.getSequenceLength() === 0, "load() should not append history from stale sequence mode");
    assert(reused.getLastWriteInfo() !== null, "load() should clear stale write freeze state");
  }

  // 10) load() must reject invalid serialized config
  {
    const layer = makeLayer({ memorySlots: 2 });
    const saved = layer.save();
    saved.readTopK = 3 as any;
    saved.config.readTopK = 3 as any;

    let threw = false;
    try {
      const fresh = new MemoryBank({ memorySlots: 2 });
      fresh.load(saved);
    } catch {
      threw = true;
    }
    assert(threw, "load() should reject readTopK > memorySlots");
  }

  // 11) writeGate should preserve prior memory when near zero and overwrite when near one
  {
    const layer = makeLayer({ units: 2, memorySlots: 1, outputUnits: 2, similarity: "dot" });
    (layer as any).queryKernel._data.fill(0);
    (layer as any).queryKernel._data[0] = 1;
    (layer as any).queryKernel._data[3] = 1;

    layer.setMemoryState({
      memoryKeys: [
        [1],
        [0],
      ],
      memoryValues: [
        [0.25],
        [0.75],
      ],
      memoryFilled: [1],
      memoryUsage: [1],
      memoryAge: [1],
      memoryStep: 1,
      units: 2,
      memoryDim: 2,
      memorySlots: 1,
    });

    (layer as any).writeGateKernel._data.fill(0);
    (layer as any).writeGateBias._data[0] = -20;
    layer.forward(mj.matrix([[1], [0]]));
    const lowGateInfo = layer.getLastWriteInfo();
    const lowGateTrace = layer.getDebugTrace().at(-1);
    const lowGateState = layer.getMemoryState();
    assert(lowGateInfo === null, "low gate should skip write and not produce lastWriteInfo");
    assert(lowGateTrace !== undefined && lowGateTrace.writeGate < 1e-6, "low gate should be near zero");
    assertClose(lowGateState.memoryValues[0][0], 0.25, 1e-6, "low gate should preserve previous value dim0");
    assertClose(lowGateState.memoryValues[1][0], 0.75, 1e-6, "low gate should preserve previous value dim1");

    (layer as any).writeGateBias._data[0] = 20;
    layer.forward(mj.matrix([[1], [0]]));
    const highGateInfo = layer.getLastWriteInfo();
    const highGateState = layer.getMemoryState();
    assert(highGateInfo !== null && highGateInfo.writeGate > 0.999999, "high gate should be near one");
    assertClose(highGateState.memoryValues[0][0], 1, 1e-6, "high gate should overwrite value dim0");
    assertClose(highGateState.memoryValues[1][0], 0, 1e-6, "high gate should overwrite value dim1");
  }
}

