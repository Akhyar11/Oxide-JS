import { mj, engine } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { LSTM } from "@oxide-js/layers";

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
}

function assertGradientClose(actual: Float32Array, expected: Float32Array, tol: number, label: string): void {
  assert(actual.length === expected.length, `${label}: gradient length mismatch`);

  let maxAbsErr = 0;
  for (let i = 0; i < actual.length; i++) {
    const absErr = Math.abs(actual[i] - expected[i]);
    if (absErr > maxAbsErr) {
      maxAbsErr = absErr;
    }
  }

  if (maxAbsErr > tol) {
    throw new Error(`${label}: max abs error ${maxAbsErr} exceeds tolerance ${tol}`);
  }
}

/**
 * Gradient check untuk LSTM layer menggunakan Tape
 */
function checkLSTMGradients(): void {
  console.log("  - Checking LSTM layer gradients (Auto-Diff)...");
  
  const units = 2;
  const hiddenUnits = 3;
  const seqLen = 2;
  const batchSize = 1;
  const epsilon = 1e-4;
  const tol = 1e-2;

  const x = mj.zeros([units, seqLen * batchSize]);
  for(let i=0; i<x._data.length; i++) x._data[i] = Math.random() - 0.5;

  const layer = new LSTM({ units, hiddenUnits, optimizer: "sgd", alpha: 0.1 });
  
  // 1. Hitung gradien analitik menggunakan Tape
  engine.startTape();
  const tape = engine.tape!;
  
  const z = layer.forwardBatch(x, batchSize);
  
  // Loss = sum(z)
  tape.backward(z); 
  engine.endTape();
  
  const params = layer.getParams();
  const analyticGrads = params.map(p => new Float32Array(p.grad!._data));

  // 2. Hitung gradien numerik untuk setiap parameter
  const getLoss = () => {
    const out = layer.forwardBatch(x, batchSize);
    let sum = 0;
    for(let i=0; i<out._data.length; i++) sum += out._data[i];
    return sum;
  };

  for (let pIdx = 0; pIdx < params.length; pIdx++) {
    const p = params[pIdx];
    const numGrad = new Float32Array(p._data.length);
    for (let i = 0; i < p._data.length; i++) {
      const original = p._data[i];
      p._data[i] = original + epsilon;
      const lP = getLoss();
      p._data[i] = original - epsilon;
      const lM = getLoss();
      p._data[i] = original;
      numGrad[i] = (lP - lM) / (2 * epsilon);
    }
    assertGradientClose(analyticGrads[pIdx], numGrad, tol, `LSTM Param [${p.name}]`);
  }

  console.log("    ✅ LSTM layer gradients are correct.");
}

export function runLSTMAutoDiffSuite(): void {
  console.log("\n🚀 Running LSTM Auto-Diff Testing...");
  checkLSTMGradients();
  console.log("✅ LSTM Auto-Diff Testing Passed!");
}

const isMain = process.argv[1] && process.argv[1].includes("lstm.autodiff.test.ts");
if (isMain) {
  runLSTMAutoDiffSuite();
}
