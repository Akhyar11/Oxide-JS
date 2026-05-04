import { unlinkSync } from "fs";
import { MemoryBank } from "../../src/layers";
import mj from "../../src/math";
import Matrix from "../../src/matrix";
import setLayers from "../../src/utils/setLayers";

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
  const rows = m._shape[0];
  const cols = m._shape[1];
  for (let i = 0; i < Math.min(rows, cols); i++) {
    m._data[i * cols + i] = 1;
  }
}

function cloneState(state: any): any {
  return JSON.parse(JSON.stringify(state));
}

function topSlot(layer: MemoryBank, x: Matrix): number {
  layer.forward(x);
  const trace = layer.getDebugTrace();
  return trace[0].readSlots[0]?.slot ?? -1;
}

function argmaxCol(m: Matrix, col = 0): number {
  let best = 0;
  let bestValue = -Infinity;
  for (let i = 0; i < m._shape[0]; i++) {
    const value = m._data[i * m._shape[1] + col];
    if (value > bestValue) {
      bestValue = value;
      best = i;
    }
  }
  return best;
}

function makeDeterministicLayer(extra: Partial<ConstructorParameters<typeof MemoryBank>[0]> = {}): MemoryBank {
  const layer = new MemoryBank({
    units: 3,
    memorySlots: 3,
    memoryDim: 3,
    outputUnits: 3,
    mode: "read-project",
    similarity: "cosine",
    readTopK: 1,
    updateMode: "replace",
    writePolicy: "empty-first",
    writeThreshold: 0.5,
    writeEnabled: true,
    forceNeedGate: 1,
    valueMode: "identity",
    writeKeyMode: "shared-query",
    writeGateMode: "always",
    optimizer: "sgd",
    alpha: 0.1,
    ...extra,
  });
  layer.forward(mj.matrix([[0], [0], [0]]));
  setIdentity((layer as any).queryKernel);
  setIdentity((layer as any).outputKernel);
  (layer as any).outputBias._data.fill(0);
  layer.resetMemory();
  return layer;
}

export function runMemoryBankCorrectnessSuite(): void {
  // 1) backward updates differentiable read/output params but not runtime state
  {
    const layer = new MemoryBank({
      units: 3,
      memorySlots: 2,
      memoryDim: 3,
      outputUnits: 3,
      mode: "project",
      similarity: "cosine",
      readTopK: 2,
      updateMode: "replace",
      writeThreshold: 2,
      optimizer: "sgd",
      alpha: 0.1,
    });

    layer.setMemoryState({
      memoryKeys: [
        [1, 0],
        [0, 1],
        [0, 0],
      ],
      memoryValues: [
        [0, 1],
        [1, 0],
        [0, 0],
      ],
      memoryFilled: [1, 1],
      memoryUsage: [1, 1],
      memoryAge: [1, 2],
      memoryStep: 2,
      units: 3,
      memoryDim: 3,
      memorySlots: 2,
    });

    const q0 = (layer as any).queryKernel.clone();
    const n0 = (layer as any).needKernel.clone();
    const o0 = (layer as any).outputKernel.clone();
    const b0 = (layer as any).outputBias.clone();

    const out = layer.forward(mj.matrix([[1], [0], [0]]));
    const stateAfterForward = cloneState(layer.getMemoryState());
    const dx = layer.backward(mj.matrix([[]]), mj.matrix([[1], [0], [-1]]));

    assert(dx._shape[0] === 3 && dx._shape[1] === 1, "backward dx shape should match input");
    assert(JSON.stringify(stateAfterForward) === JSON.stringify(layer.getMemoryState()), "optimizer step must not mutate memory state");

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
    assert(out._shape[0] === 3 && out._shape[1] === 1, "project mode output shape should be [outputUnits, cols]");
  }

  // 2) empty slot write is full replace even under gated-merge
  {
    const layer = new MemoryBank({
      units: 2,
      memorySlots: 1,
      memoryDim: 2,
      outputUnits: 2,
      mode: "read-project",
      updateMode: "gated-merge",
      writeThreshold: 0.2,
      valueMode: "identity",
      writeKeyMode: "shared-query",
      writeGateMode: "learned",
      forceNeedGate: 0.3,
    });
    layer.forward(mj.matrix([[0], [0]]));
    setIdentity((layer as any).queryKernel);
    setIdentity((layer as any).outputKernel);
    (layer as any).outputBias._data.fill(0);
    (layer as any).writeGateKernel._data.fill(0);
    (layer as any).writeGateKernel._data[0] = 1;
    layer.resetMemory();

    layer.forward(mj.matrix([[1], [0]]));
    const info = layer.getLastWriteInfo();
    const state = layer.getMemoryState();

    assert(info !== null && info.writeGate > 0.2 && info.writeGate < 1, "learned gate should be above threshold but below 1");
    assertClose(state.memoryValues[0][0], 1, 1e-6, "empty slot write should store full new value on dim0");
    assertClose(state.memoryValues[1][0], 0, 1e-6, "empty slot write should store full new value on dim1");
  }

  // 3) shared-key deterministic write/read
  {
    const layer = makeDeterministicLayer();
    const items = [
      { x: mj.matrix([[1], [0], [0]]), slot: 0, value: [1, 0, 0] },
      { x: mj.matrix([[0], [1], [0]]), slot: 1, value: [0, 1, 0] },
      { x: mj.matrix([[0], [0], [1]]), slot: 2, value: [0, 0, 1] },
    ];

    for (const item of items) layer.forward(item.x);
    layer.freezeWrites();

    for (const item of items) {
      const slot = topSlot(layer, item.x);
      assert(slot === item.slot, `shared key write/read should retrieve slot ${item.slot}, got ${slot}`);
      const out = layer.forward(item.x);
      const predicted = Array.from(out.getCol(0));
      for (let i = 0; i < item.value.length; i++) {
        assertClose(predicted[i], item.value[i], 1e-6, `shared key read should reconstruct stored value at dim ${i}`);
      }
    }
  }

  // 4) active writes provide causal gain over frozen writes
  {
    const keys = [
      mj.matrix([[1], [0], [0]]),
      mj.matrix([[0], [1], [0]]),
      mj.matrix([[0], [0], [1]]),
    ];
    const labels = [0, 1, 2];

    const evaluate = (freezeWrites: boolean): number => {
      const layer = makeDeterministicLayer();
      if (freezeWrites) layer.freezeWrites();
      for (const key of keys) layer.forward(key);
      layer.freezeWrites();

      let correct = 0;
      for (let i = 0; i < keys.length; i++) {
        const pred = argmaxCol(layer.forward(keys[i]));
        if (pred === labels[i]) correct += 1;
      }
      return correct / keys.length;
    };

    const activeAcc = evaluate(false);
    const frozenAcc = evaluate(true);

    assert(activeAcc >= 0.99, `active writes should solve episodic task, got ${activeAcc}`);
    assert(frozenAcc <= 0.34, `frozen writes should stay near random, got ${frozenAcc}`);
    assert(activeAcc - frozenAcc >= 0.5, `active writes should beat frozen writes by a meaningful margin, got ${activeAcc - frozenAcc}`);
  }

  // 5) output path really uses memory values
  {
    const layer = makeDeterministicLayer();
    layer.setMemoryState({
      memoryKeys: [
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ],
      memoryValues: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      memoryFilled: [1, 0, 0],
      memoryUsage: [1, 0, 0],
      memoryAge: [1, 0, 0],
      memoryStep: 1,
      units: 3,
      memoryDim: 3,
      memorySlots: 3,
    });
    layer.freezeWrites();

    const query = mj.matrix([[1], [0], [0]]);
    const outA = layer.forward(query);

    layer.setMemoryState({
      memoryKeys: [
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ],
      memoryValues: [
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
      ],
      memoryFilled: [1, 0, 0],
      memoryUsage: [1, 0, 0],
      memoryAge: [1, 0, 0],
      memoryStep: 1,
      units: 3,
      memoryDim: 3,
      memorySlots: 3,
    });
    const outB = layer.forward(query);

    assert(argmaxCol(outA) !== argmaxCol(outB), "changing memoryValues should change prediction when mode='read-project'");
  }

  // 6) save/load roundtrip preserves config, state, and output
  {
    const layer = makeDeterministicLayer({
      memorySlots: 2,
      writeThreshold: 0.25,
      writeGateMode: "threshold",
    });
    layer.forward(mj.matrix([[1], [0], [0]]));
    layer.forward(mj.matrix([[0], [1], [0]]));
    layer.freezeWrites();

    const probe = mj.matrix([[1], [0], [0]]);
    const expected = layer.forward(probe);
    const saved = layer.save();

    const [loaded] = setLayers([saved]) as MemoryBank[];
    loaded.freezeWrites();
    const loadedState = loaded.getMemoryState();
    const actual = loaded.forward(probe);
    const savedState = layer.getMemoryState();
    const loadedSaved = loaded.save();

    assert(saved.config.forceNeedGate === loadedSaved.config.forceNeedGate, "save/load should preserve forceNeedGate");
    assert(saved.config.valueMode === loadedSaved.config.valueMode, "save/load should preserve valueMode");
    assert(saved.config.writeKeyMode === loadedSaved.config.writeKeyMode, "save/load should preserve writeKeyMode");
    assert(saved.config.writeGateMode === loadedSaved.config.writeGateMode, "save/load should preserve writeGateMode");
    assert(JSON.stringify(savedState) === JSON.stringify(loadedState), "save/load should preserve runtime memory state");
    assertMatrixClose(expected, actual, 1e-6, "save/load should preserve query output");
  }

  // 7) saveMemory/loadMemory remains functional with lazy fs access
  {
    const path = "/tmp/ml-v1-memory-bank-state.json";
    const layer = makeDeterministicLayer({ memorySlots: 2 });
    layer.forward(mj.matrix([[1], [0], [0]]));
    layer.saveMemory(path);

    const restored = makeDeterministicLayer({ memorySlots: 2 });
    restored.loadMemory(path);
    assert(JSON.stringify(layer.getMemoryState()) === JSON.stringify(restored.getMemoryState()), "saveMemory/loadMemory should roundtrip runtime state");

    try {
      unlinkSync(path);
    } catch {}
  }

  // 8) separate-project auxiliary key training improves alignment
  {
    const layer = new MemoryBank({
      units: 2,
      memorySlots: 2,
      memoryDim: 2,
      outputUnits: 2,
      mode: "read-project",
      similarity: "cosine",
      readTopK: 1,
      updateMode: "replace",
      writePolicy: "empty-first",
      writeThreshold: 0.5,
      writeEnabled: true,
      forceNeedGate: 1,
      valueMode: "identity",
      writeKeyMode: "separate-project",
      writeGateMode: "always",
      optimizer: "sgd",
      alpha: 0.3,
    });
    layer.forward(mj.matrix([[0], [0]]));
    setIdentity((layer as any).queryKernel);
    setIdentity((layer as any).outputKernel);
    (layer as any).outputBias._data.fill(0);
    (layer as any).writeKeyKernel._data.set([0, 1, 1, 0]);
    layer.resetMemory();

    const dataset = [
      { x: mj.matrix([[1], [0]]), expectedSlot: 0 },
      { x: mj.matrix([[0], [1]]), expectedSlot: 1 },
    ];

    const accuracy = (): number => {
      layer.resetMemory();
      for (const item of dataset) layer.forward(item.x);
      layer.freezeWrites();
      let correct = 0;
      for (const item of dataset) {
        if (topSlot(layer, item.x) === item.expectedSlot) correct += 1;
      }
      layer.unfreezeWrites();
      return correct / dataset.length;
    };

    const before = accuracy();
    for (let epoch = 0; epoch < 8; epoch++) {
      layer.resetMemory();
      for (const item of dataset) {
        layer.forward(item.x);
        const targetKey = layer.getQueryVectorForInput(item.x, true);
        const loss = layer.trainLastWriteKey(targetKey);
        assert(loss !== null, "trainLastWriteKey should be available in separate-project mode");
      }
    }
    const after = accuracy();

    assert(before <= 0.5, `separate-project alignment should start poor, got ${before}`);
    assert(after >= 0.99, `trainLastWriteKey should improve retrieval alignment, got ${after}`);
  }

  // 9) write-side kernels behave consistently across all output modes
  {
    const modes: Array<"project" | "read-project" | "concat" | "add"> = ["project", "read-project", "concat", "add"];

    for (const mode of modes) {
      const layer = new MemoryBank({
        units: 2,
        memorySlots: 1,
        memoryDim: 2,
        outputUnits: 2,
        mode,
        similarity: "dot",
        readTopK: 1,
        updateMode: "replace",
        writePolicy: "empty-first",
        writeThreshold: 0.6,
        writeEnabled: true,
        forceNeedGate: 0.75,
        valueMode: "project",
        writeKeyMode: "separate-project",
        writeGateMode: "learned",
        optimizer: "sgd",
        alpha: 0.2,
      });

      layer.forward(mj.matrix([[0], [0]]));
      (layer as any).queryKernel._data.set([1, 0, 0, 1]);
      (layer as any).writeKeyKernel._data.set([2, 0, 0, 3]);
      (layer as any).writeValueKernel._data.set([4, 0, 0, 5]);
      (layer as any).writeGateKernel._data.fill(0);
      (layer as any).writeGateKernel._data[0] = 2;
      if ((layer as any).outputKernel) setIdentity((layer as any).outputKernel);
      if ((layer as any).outputBias) (layer as any).outputBias._data.fill(0);
      layer.resetMemory();

      const x = mj.matrix([[1], [1]]);
      layer.forward(x);
      const info = layer.getLastWriteInfo();
      const state = layer.getMemoryState();

      assert(info !== null, `mode=${mode}: learned gate should allow a committed write`);
      assertClose(info!.writeGate, 1 / (1 + Math.exp(-2)), 1e-6, `mode=${mode}: writeGate should come from writeGateKernel`);
      assertClose(info!.newKey[0], 2, 1e-6, `mode=${mode}: writeKey dim0 should come from writeKeyKernel`);
      assertClose(info!.newKey[1], 3, 1e-6, `mode=${mode}: writeKey dim1 should come from writeKeyKernel`);
      assertClose(info!.newValue[0], 4, 1e-6, `mode=${mode}: writeValue dim0 should come from writeValueKernel`);
      assertClose(info!.newValue[1], 5, 1e-6, `mode=${mode}: writeValue dim1 should come from writeValueKernel`);
      assertClose(state.memoryKeys[0][0], 2, 1e-6, `mode=${mode}: stored key dim0 should match writeKeyKernel output`);
      assertClose(state.memoryKeys[1][0], 3, 1e-6, `mode=${mode}: stored key dim1 should match writeKeyKernel output`);
      assertClose(state.memoryValues[0][0], 4, 1e-6, `mode=${mode}: stored value dim0 should match writeValueKernel output`);
      assertClose(state.memoryValues[1][0], 5, 1e-6, `mode=${mode}: stored value dim1 should match writeValueKernel output`);

      const keyLossBefore = layer.trainLastWriteKey([1, 0])!;
      const valueLossBefore = layer.trainLastWriteValue([0, 1])!;
      const gateLossBefore = layer.trainLastWriteGate(1)!;
      layer.resetMemory();
      layer.forward(x);
      const after = layer.getLastWriteInfo()!;

      const keyLossAfter = 0.5 * ((after.newKey[0] - 1) ** 2 + (after.newKey[1] - 0) ** 2) / 2;
      const valueLossAfter = 0.5 * ((after.newValue[0] - 0) ** 2 + (after.newValue[1] - 1) ** 2) / 2;
      const gateLossAfter = 0.5 * (after.writeGate - 1) * (after.writeGate - 1);

      assert(keyLossAfter < keyLossBefore, `mode=${mode}: trainLastWriteKey should improve key alignment`);
      assert(valueLossAfter < valueLossBefore, `mode=${mode}: trainLastWriteValue should improve value alignment`);
      assert(gateLossAfter < gateLossBefore, `mode=${mode}: trainLastWriteGate should improve gate target fit`);
    }
  }

  // 10) main backward performs within-sequence BPTT into write-side kernels across modes
  {
    const modes: Array<"project" | "read-project" | "concat" | "add"> = ["project", "read-project", "concat", "add"];

    for (const mode of modes) {
      const layer = new MemoryBank({
        units: 2,
        memorySlots: 2,
        memoryDim: 2,
        outputUnits: 2,
        mode,
        similarity: "dot",
        readTopK: 2,
        updateMode: "gated-merge",
        writePolicy: "empty-first",
        writeThreshold: 0,
        writeEnabled: true,
        forceNeedGate: 1,
        valueMode: "project",
        writeKeyMode: "separate-project",
        writeGateMode: "learned",
        optimizer: "sgd",
        alpha: 0.1,
      });

      layer.forward(mj.matrix([[0], [0]]));
      setIdentity((layer as any).queryKernel);
      setIdentity((layer as any).writeKeyKernel);
      setIdentity((layer as any).writeValueKernel);
      (layer as any).writeGateKernel._data.fill(0);
      if (mode === "project") {
        (layer as any).outputKernel._data.set([
          0, 0, 1, 0,
          0, 0, 0, 1,
        ]);
      } else if (mode === "read-project") {
        setIdentity((layer as any).outputKernel);
      }
      if ((layer as any).outputBias) (layer as any).outputBias._data.fill(0);
      layer.resetMemory();

      const x = mj.matrix([
        [1, 0, 1, 1],
        [0, 1, 1, 0],
      ]);

      const wk0 = (layer as any).writeKeyKernel.clone();
      const wv0 = (layer as any).writeValueKernel.clone();
      const wg0 = (layer as any).writeGateKernel.clone();

      const out = layer.forward(x);
      const err = mj.zeros(out._shape);
      const lastCol = out._shape[1] - 1;
      for (let r = 0; r < out._shape[0]; r++) {
        err._data[r * out._shape[1] + lastCol] = 1;
      }
      layer.backward(mj.matrix([[]]), err);

      let keyChanged = false;
      let valueChanged = false;
      let gateChanged = false;
      for (let i = 0; i < wk0._data.length; i++) if (Math.abs(wk0._data[i] - (layer as any).writeKeyKernel._data[i]) > 1e-12) keyChanged = true;
      for (let i = 0; i < wv0._data.length; i++) if (Math.abs(wv0._data[i] - (layer as any).writeValueKernel._data[i]) > 1e-12) valueChanged = true;
      for (let i = 0; i < wg0._data.length; i++) if (Math.abs(wg0._data[i] - (layer as any).writeGateKernel._data[i]) > 1e-12) gateChanged = true;

      assert(keyChanged, `mode=${mode}: main backward should update writeKeyKernel through sequence history`);
      assert(valueChanged, `mode=${mode}: main backward should update writeValueKernel through sequence history`);
      assert(gateChanged, `mode=${mode}: main backward should update writeGateKernel through sequence history`);
    }
  }

  // 11) sequence mode enables BPTT across separate forward() calls
  {
    const layer = new MemoryBank({
      units: 2,
      memorySlots: 2,
      memoryDim: 2,
      outputUnits: 2,
      mode: "read-project",
      similarity: "dot",
      readTopK: 2,
      updateMode: "gated-merge",
      writePolicy: "empty-first",
      writeThreshold: 0,
      writeEnabled: true,
      forceNeedGate: 1,
      valueMode: "project",
      writeKeyMode: "separate-project",
      writeGateMode: "learned",
      optimizer: "sgd",
      alpha: 0.1,
    });

    layer.forward(mj.matrix([[0], [0]]));
    setIdentity((layer as any).queryKernel);
    setIdentity((layer as any).writeKeyKernel);
    setIdentity((layer as any).writeValueKernel);
    setIdentity((layer as any).outputKernel);
    (layer as any).outputBias._data.fill(0);
    (layer as any).writeGateKernel._data.fill(0);
    layer.resetMemory();

    assert(layer.isSequenceActive() === false, "sequence mode should start inactive");
    layer.beginSequence({ maxHistorySteps: 8 });
    assert(layer.isSequenceActive() === true, "beginSequence should activate sequence mode");

    const wk0 = (layer as any).writeKeyKernel.clone();
    const wv0 = (layer as any).writeValueKernel.clone();
    const wg0 = (layer as any).writeGateKernel.clone();

    layer.forward(mj.matrix([[1], [0]])); // write slot 0
    layer.forward(mj.matrix([[0], [1]])); // write slot 1
    layer.forward(mj.matrix([[1], [1]])); // update one occupied slot through gated-merge
    layer.forward(mj.matrix([[1], [1]])); // query future state
    assert(layer.getSequenceLength() === 4, `sequence history should collect 4 steps, got ${layer.getSequenceLength()}`);

    const err = mj.zeros([2, 4]);
    err._data[0 * 4 + 3] = 1;
    err._data[1 * 4 + 3] = -1;
    const dx = layer.backwardSequence(err);
    assert(dx._shape[0] === 2 && dx._shape[1] === 4, "backwardSequence should return dx for the full active sequence");

    let keyChanged = false;
    let valueChanged = false;
    let gateChanged = false;
    for (let i = 0; i < wk0._data.length; i++) if (Math.abs(wk0._data[i] - (layer as any).writeKeyKernel._data[i]) > 1e-12) keyChanged = true;
    for (let i = 0; i < wv0._data.length; i++) if (Math.abs(wv0._data[i] - (layer as any).writeValueKernel._data[i]) > 1e-12) valueChanged = true;
    for (let i = 0; i < wg0._data.length; i++) if (Math.abs(wg0._data[i] - (layer as any).writeGateKernel._data[i]) > 1e-12) gateChanged = true;

    assert(keyChanged, "backwardSequence should update writeKeyKernel across separate forward calls");
    assert(valueChanged, "backwardSequence should update writeValueKernel across separate forward calls");
    assert(gateChanged, "backwardSequence should update writeGateKernel across separate forward calls");

    layer.detachSequence();
    assert(layer.getSequenceLength() === 0, "detachSequence should clear active history but keep memory state");

    layer.forward(mj.matrix([[1], [0]]));
    assert(layer.getSequenceLength() === 1, "history should continue collecting after detach");
    const shortErr = mj.zeros([2, 1]);
    shortErr._data[0] = 1;
    const shortDx = layer.backwardSequence(shortErr);
    assert(shortDx._shape[0] === 2 && shortDx._shape[1] === 1, "backwardSequence should still run on a fresh post-detach history");

    layer.endSequence();
    assert(layer.isSequenceActive() === false, "endSequence should deactivate sequence mode");
    assert(layer.getSequenceLength() === 0, "endSequence should clear sequence history");
  }
}

if (require.main === module) {
  runMemoryBankCorrectnessSuite();
  console.log("[PASS] memoryBank.test: all tests passed");
}
