import { MemoryBank } from "../../src/layers";
import mj from "../../src/math";
import Matrix from "../../src/matrix";
import Sequential from "../../src/models/sequential";
import { writeFileSync, unlinkSync } from "fs";

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
}

function assertShape(matrix: Matrix, rows: number, cols: number, message: string): void {
  assert(
    matrix._shape[0] === rows && matrix._shape[1] === cols,
    `${message}: expected [${rows},${cols}], got [${matrix._shape[0]},${matrix._shape[1]}]`
  );
}

export function runMemoryBankCorrectnessSuite(): void {
  // Test lazy init
  const layer = new MemoryBank({ memorySlots: 4 });
  assert(!layer['initialized'], "should not initialized before forward");
  const x = mj.matrix([[1, 2], [3, 4], [5, 6]]);
  const out = layer.forward(x);
  assert(layer['initialized'], "should be initialized after forward");
  assertShape(out, layer['outputShape'][0], 2, "forward output shape");

  // explicit init
  const layer2 = new MemoryBank({ units: 5, memorySlots: 4, memoryDim: 3, outputUnits: 6 });
  const in2 = mj.matrix([[1,2],[3,4],[5,6],[7,8],[9,10]]);
  const o2 = layer2.forward(in2);
  assertShape(o2, 6, 2, "explicit init forward shape");

  // memory fills
  const layer3 = new MemoryBank({ units: 3, memorySlots: 4, writeThreshold: 0.0 });
  assert(!layer3.hasMemory(), "starts empty");
  layer3.forward(x);
  assert(layer3.hasMemory(), "should have memory after forward with writeThreshold 0");
  const state = layer3.getMemoryState();
  assert(state.memoryFilled.some((v: number) => v === 1), "some slots filled");

  // resetMemory
  layer3.resetMemory();
  const state2 = layer3.getMemoryState();
  assert(state2.memoryFilled.every((v: number) => v === 0), "reset clears filled");

  // save/load memory
  layer3.forward(x);
  const tmp = "/tmp/memstate_test.json";
  layer3.saveMemory(tmp);
  layer3.resetMemory();
  layer3.loadMemory(tmp);
  unlinkSync(tmp);

  // backward basic shape
  const layer4 = new MemoryBank({ units: 3, memorySlots: 3 });
  const out4 = layer4.forward(x);
  const before = layer4['queryKernel'].clone();
  const err = mj.zeros(out4._shape);
  const dx = layer4.backward(mj.matrix([[]]), err);
  assertShape(dx, 3, 2, "backward dx shape");
  const after = layer4['queryKernel'];
  // queryKernel should have changed slightly if trainablePolicy true
  let changed = false;
  for (let i=0;i<before._data.length;i++) if (Math.abs(before._data[i]-after._data[i])>1e-12) { changed=true; break }
  assert(changed, "queryKernel should change on backward when trainablePolicy=true");

  // freezeWrites
  const layer5 = new MemoryBank({ units:3, memorySlots:3, writeThreshold:0 });
  layer5.freezeWrites();
  layer5.resetMemory();
  layer5.forward(x);
  assert(!layer5.hasMemory(), "freezeWrites prevents memory filling");
  layer5.enableWrites();
  layer5.resetMemory();
  layer5.forward(x);
  assert(layer5.hasMemory(), "enableWrites allows memory filling");

  // Sequential helpers
  const model = new Sequential({ layers: [new MemoryBank({ units:3, memorySlots:2 })] });
  model.resetMemory();
  model.freezeMemoryWrites();
  model.enableMemoryWrites();
}

if (require.main === module) runMemoryBankCorrectnessSuite();
