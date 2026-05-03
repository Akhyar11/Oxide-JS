/**
 * PART 5 — Manual Memory Read Test
 *
 * Proves that the MemoryBank read path (query → similarity → softmax → slot selection)
 * works correctly when memory state is pre-loaded manually.
 *
 * If this test fails, fix read/similarity/query/output path first.
 * Only proceed to write-path debugging AFTER this passes.
 */

import { MemoryBank } from "../../src/layers";
import mj from "../../src/math";
import Matrix from "../../src/matrix";

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
}

export function runMemoryBankRetrievalSuite(): void {
  // -----------------------------------------------------------------------
  // Test 1: Identity queryKernel, 4-slot memory, 4-dim
  //   slot 0 key=[1,0,0,0], value=[0,1,0,0]
  //   slot 1 key=[0,1,0,0], value=[0,0,1,0]
  //   Query x=[1,0,0,0] => top slot must be 0
  //   Query x=[0,1,0,0] => top slot must be 1
  // -----------------------------------------------------------------------
  {
    const layer = new MemoryBank({
      units: 4,
      memorySlots: 4,
      memoryDim: 4,
      outputUnits: 4,
      mode: "project",
      similarity: "cosine",
      readTopK: 4,
      writeThreshold: 2.0, // disable writes (gate never reaches 2.0)
      writeEnabled: true,
    });

    // Trigger initialization with a dummy forward so weights exist
    layer.forward(mj.matrix([[0], [0], [0], [0]]));
    layer.resetMemory();

    // Set queryKernel to identity [4x4] so q = x directly
    (layer as any).queryKernel = mj.zeros([4, 4]);
    for (let i = 0; i < 4; i++) {
      (layer as any).queryKernel._data[i * 4 + i] = 1;
    }

    // Manually set memory state:
    //   slot 0: key=[1,0,0,0], value=[0,1,0,0]
    //   slot 1: key=[0,1,0,0], value=[0,0,1,0]
    //   slots 2,3: unfilled
    layer.setMemoryState({
      memoryKeys: [
        [1, 0, 0, 0], // row 0 (dim 0) across 4 slots
        [0, 1, 0, 0], // row 1 (dim 1) across 4 slots
        [0, 0, 0, 0], // row 2
        [0, 0, 0, 0], // row 3
      ],
      memoryValues: [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
      ],
      memoryFilled: [1, 1, 0, 0],
      memoryUsage: [1, 1, 0, 0],
      memoryAge: [1, 2, 0, 0],
      memoryStep: 3,
      units: 4,
      memoryDim: 4,
      memorySlots: 4,
    });

    // Query 1: x = [1,0,0,0] => q = [1,0,0,0] => cos sim with slot0=[1,0,0,0] max
    (layer as any).writeFrozen = true;
    layer.forward(mj.matrix([[1], [0], [0], [0]]));
    const trace1 = layer.getDebugTrace();

    assert(trace1.length === 1, "retrieval test 1: should have 1 trace entry for 1 column");
    assert(trace1[0].readSlots.length > 0, "retrieval test 1: should have at least 1 readSlot");
    const topSlot1 = trace1[0].readSlots[0].slot;
    assert(
      topSlot1 === 0,
      `retrieval test 1: query x=[1,0,0,0] should read slot 0, got slot ${topSlot1}. ` +
        `readSlots=${JSON.stringify(trace1[0].readSlots)}`
    );

    // Query 2: x = [0,1,0,0] => q = [0,1,0,0] => cos sim with slot1=[0,1,0,0] max
    layer.forward(mj.matrix([[0], [1], [0], [0]]));
    const trace2 = layer.getDebugTrace();

    assert(trace2.length === 1, "retrieval test 2: should have 1 trace entry for 1 column");
    assert(trace2[0].readSlots.length > 0, "retrieval test 2: should have at least 1 readSlot");
    const topSlot2 = trace2[0].readSlots[0].slot;
    assert(
      topSlot2 === 1,
      `retrieval test 2: query x=[0,1,0,0] should read slot 1, got slot ${topSlot2}. ` +
        `readSlots=${JSON.stringify(trace2[0].readSlots)}`
    );
  }

  // -----------------------------------------------------------------------
  // Test 2: clearDebugTrace resets correctly
  // -----------------------------------------------------------------------
  {
    const layer = new MemoryBank({
      units: 2,
      memorySlots: 2,
      memoryDim: 2,
      outputUnits: 2,
      mode: "project",
      writeThreshold: 2.0,
    });

    layer.forward(mj.matrix([[1], [0]]));
    assert(layer.getDebugTrace().length === 1, "clearDebugTrace test: should have 1 trace before clear");

    layer.clearDebugTrace();
    assert(layer.getDebugTrace().length === 0, "clearDebugTrace test: should have 0 after clear");

    layer.forward(mj.matrix([[1], [0]]));
    assert(layer.getDebugTrace().length === 1, "clearDebugTrace test: should have 1 after next forward");
  }

  // -----------------------------------------------------------------------
  // Test 3: writeMemoryForDebug sets memory correctly
  // -----------------------------------------------------------------------
  {
    const layer = new MemoryBank({
      units: 3,
      memorySlots: 3,
      memoryDim: 3,
      outputUnits: 3,
      mode: "project",
      writeThreshold: 2.0,
    });

    layer.forward(mj.matrix([[0], [0], [0]]));
    layer.resetMemory();

    layer.writeMemoryForDebug([1, 0, 0], [0, 1, 0], 0);
    layer.writeMemoryForDebug([0, 1, 0], [0, 0, 1], 1);

    const state = layer.getMemoryState();
    assert(state.memoryFilled[0] === 1, "writeMemoryForDebug: slot 0 should be filled");
    assert(state.memoryFilled[1] === 1, "writeMemoryForDebug: slot 1 should be filled");
    assert(state.memoryFilled[2] === 0, "writeMemoryForDebug: slot 2 should be empty");

    // Verify key for slot 0 (column 0 = dim 0 row of keys matrix)
    assert(
      Math.abs(state.memoryKeys[0][0] - 1) < 1e-6,
      `writeMemoryForDebug: slot0 key dim0 should be 1, got ${state.memoryKeys[0][0]}`
    );
    assert(
      Math.abs(state.memoryKeys[1][0]) < 1e-6,
      `writeMemoryForDebug: slot0 key dim1 should be 0, got ${state.memoryKeys[1][0]}`
    );

    // Verify value for slot 1
    assert(
      Math.abs(state.memoryValues[2][1] - 1) < 1e-6,
      `writeMemoryForDebug: slot1 value dim2 should be 1, got ${state.memoryValues[2][1]}`
    );
  }

  // -----------------------------------------------------------------------
  // Test 4: getLastWriteInfo returns null when no write
  // -----------------------------------------------------------------------
  {
    const layer = new MemoryBank({
      units: 2,
      memorySlots: 2,
      memoryDim: 2,
      outputUnits: 2,
      mode: "project",
      writeThreshold: 2.0, // ensure gate never fires
    });

    layer.forward(mj.matrix([[1], [0]]));
    const info = layer.getLastWriteInfo();
    assert(info === null, "getLastWriteInfo: should be null when writeGate < writeThreshold");
    const valMat = layer.getLastWriteValueMatrix();
    assert(valMat === null, "getLastWriteValueMatrix: should be null when no write committed");
  }

  // -----------------------------------------------------------------------
  // Test 5: getLastWriteInfo returns data when write fires
  // -----------------------------------------------------------------------
  {
    const layer = new MemoryBank({
      units: 2,
      memorySlots: 2,
      memoryDim: 2,
      outputUnits: 2,
      mode: "project",
      writeThreshold: 0.0, // always write
    });

    layer.forward(mj.matrix([[1], [0]]));
    const info = layer.getLastWriteInfo();
    assert(info !== null, "getLastWriteInfo: should be non-null when writes always fire");
    assert(info!.committed === true, "getLastWriteInfo: committed should be true");
    assert(info!.slot >= 0 && info!.slot < 2, "getLastWriteInfo: slot should be in range");
    assert(info!.newValue.length === 2, "getLastWriteInfo: newValue length should match memoryDim");

    const valMat = layer.getLastWriteValueMatrix();
    assert(valMat !== null, "getLastWriteValueMatrix: should not be null after write");
    assert(valMat!._shape[0] === 2 && valMat!._shape[1] === 1, "getLastWriteValueMatrix: shape should be [memoryDim, 1]");
  }

  // -----------------------------------------------------------------------
  // Test 6: multi-column forward trace has correct column indices
  // -----------------------------------------------------------------------
  {
    const layer = new MemoryBank({
      units: 2,
      memorySlots: 2,
      memoryDim: 2,
      outputUnits: 2,
      mode: "project",
      writeThreshold: 2.0,
    });

    layer.forward(mj.matrix([[1, 0], [0, 1]])); // 2 columns
    const trace = layer.getDebugTrace();
    assert(trace.length === 2, `multi-col trace: expected 2 entries, got ${trace.length}`);
    assert(trace[0].column === 0, `multi-col trace: first entry column should be 0`);
    assert(trace[1].column === 1, `multi-col trace: second entry column should be 1`);
  }
}

if (require.main === module) {
  runMemoryBankRetrievalSuite();
  console.log("[PASS] memoryBank.retrieval.test: all tests passed");
}
