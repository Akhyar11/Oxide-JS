import { MemoryBank } from "@oxide-js/layers";
import { mj } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
}

function assertFiniteMatrix(matrix: Matrix, message: string): void {
  for (let i = 0; i < matrix._data.length; i++) {
    if (!Number.isFinite(matrix._data[i])) {
      throw new Error(`${message}: non-finite at flat index ${i}`);
    }
  }
}

function createRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function randInt(rng: () => number, min: number, max: number): number {
  return min + Math.floor(rng() * (max - min + 1));
}

function randScalar(rng: () => number): number {
  return (rng() * 2 - 1) * 1.5;
}

function randomMatrix(rng: () => number, rows: number, cols: number): Matrix {
  const array = new Array(rows);
  for (let r = 0; r < rows; r++) {
    const row = new Array(cols);
    for (let c = 0; c < cols; c++) row[c] = randScalar(rng);
    array[r] = row;
  }
  return mj.matrix(array);
}

function runCase(rng: () => number, caseId: number): void {
  const units = randInt(rng, 2, 4);
  const memorySlots = randInt(rng, 1, 4);
  const cols = randInt(rng, 1, 3);
  const mode = rng() < 0.5 ? "project" as const : "concat" as const;
  const similarity = rng() < 0.5 ? "cosine" as const : "dot" as const;
  const writeEnabled = rng() < 0.7;
  const useSequence = cols > 1 && rng() < 0.5;
  const readTopK = randInt(rng, 1, memorySlots);

  const layer = new MemoryBank({
    units,
    memorySlots,
    outputUnits: units,
    mode,
    similarity,
    readTopK,
    writeEnabled,
    optimizer: "sgd",
    alpha: 0.05,
    clipGradient: false,
  });

  if (useSequence) layer.beginSequence({ maxHistorySteps: cols });

  const x = randomMatrix(rng, units, cols);
  const out = layer.forward(x);
  assertFiniteMatrix(out, `stress case ${caseId}: forward output`);
  assert(layer.getDebugTrace().length === cols, `stress case ${caseId}: trace length should equal cols`);

  const expectedRows = mode === "project" ? units : units * 2;
  assert(out._shape[0] === expectedRows && out._shape[1] === cols, `stress case ${caseId}: output shape mismatch`);

  const err = randomMatrix(rng, expectedRows, cols);
  let dx: Matrix;
  if (useSequence) {
    dx = layer.backwardSequence(err);
    assert(layer.getSequenceLength() <= cols, `stress case ${caseId}: sequence history should honor maxHistorySteps`);
    layer.endSequence();
  } else {
    dx = layer.backward(mj.matrix([[]]), err);
  }

  assertFiniteMatrix(dx, `stress case ${caseId}: backward dx`);
  assert(dx._shape[0] === units && dx._shape[1] === cols, `stress case ${caseId}: dx shape mismatch`);

  const state = layer.getMemoryState();
  assert(state.memoryFilled.length === memorySlots, `stress case ${caseId}: memoryFilled length mismatch`);
  for (const filled of state.memoryFilled) {
    assert(filled === 0 || filled === 1, `stress case ${caseId}: memoryFilled must stay binary`);
  }
  assert(state.memoryStep === cols, `stress case ${caseId}: memoryStep should advance once per column`);

  const saved = layer.save();
  const restored = new MemoryBank({ memorySlots });
  restored.load(saved);
  restored.freezeWrites();
  layer.freezeWrites();

  const probe = randomMatrix(rng, units, cols);
  const expected = layer.forward(probe);
  const actual = restored.forward(probe);
  assert(expected._shape[0] === actual._shape[0] && expected._shape[1] === actual._shape[1], `stress case ${caseId}: restored output shape mismatch`);
  for (let i = 0; i < expected._data.length; i++) {
    if (Math.abs(expected._data[i] - actual._data[i]) > 1e-5) {
      throw new Error(`stress case ${caseId}: save/load output mismatch at flat index ${i}`);
    }
  }
}

export function runMemoryBankStressSuite(): void {
  const rng = createRng(0xC0FFEE);
  for (let i = 0; i < 50; i++) runCase(rng, i);
}

