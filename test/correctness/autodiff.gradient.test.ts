import { mj, engine, softmax } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { Dense, MultiHeadAttention } from "@oxide-js/layers";

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
}

function assertGradientClose(actual: Float32Array, expected: Float32Array, tol: number, label: string): void {
  assert(actual.length === expected.length, `${label}: gradient length mismatch`);

  let maxAbsErr = 0;
  let worstIndex = -1;
  for (let i = 0; i < actual.length; i++) {
    const absErr = Math.abs(actual[i] - expected[i]);
    if (absErr > maxAbsErr) {
      maxAbsErr = absErr;
      worstIndex = i;
    }
  }

  if (maxAbsErr > tol) {
    throw new Error(
      `${label}: max abs error ${maxAbsErr} exceeds tolerance ${tol} at flat index ${worstIndex} \n` +
      `  (analytic=${actual[worstIndex].toFixed(8)}, numeric=${expected[worstIndex].toFixed(8)})`
    );
  }
}

/**
 * Gradient check untuk operasi dotProduct + addBias + activation (Dense layer)
 */
function checkDenseGradients(): void {
  console.log("  - Checking Dense layer gradients (Auto-Diff)...");
  
  const inputUnits = 3;
  const outputUnits = 2;
  const epsilon = 1e-5;
  const tol = 1e-2;

  const x = mj.matrix([[0.1], [0.5], [-0.3]]);

  const activations = ["linear", "relu", "sigmoid", "tanh", "lRelu"] as const;

  for (const act of activations) {
    const layer = new Dense({ units: inputUnits, outputUnits, activation: act, optimizer: "sgd", alpha: 0 });
    
    // 1. Hitung gradien analitik menggunakan Tape
    const tape = engine.startTape();
    
    const z = layer.forward(x);
    tape.backward(z); 
    engine.endTape();
    
    const finalAnalyticW = new Float32Array(layer.weight.grad!._data);
    const finalAnalyticB = new Float32Array(layer.bias.grad!._data);

    // 2. Hitung gradien numerik untuk Weight
    const getLoss = (m: Matrix) => m._data.reduce((a, b) => a + b, 0);
    
    const numW = new Float32Array(layer.weight._data.length);
    for (let i = 0; i < layer.weight._data.length; i++) {
      const v = layer.weight._data[i];
      layer.weight._data[i] = v + epsilon;
      const lP = getLoss(layer.forward(x));
      layer.weight._data[i] = v - epsilon;
      const lM = getLoss(layer.forward(x));
      layer.weight._data[i] = v;
      numW[i] = (lP - lM) / (2 * epsilon);
    }

    const numB = new Float32Array(layer.bias._data.length);
    for (let i = 0; i < layer.bias._data.length; i++) {
      const v = layer.bias._data[i];
      layer.bias._data[i] = v + epsilon;
      const lP = getLoss(layer.forward(x));
      layer.bias._data[i] = v - epsilon;
      const lM = getLoss(layer.forward(x));
      layer.bias._data[i] = v;
      numB[i] = (lP - lM) / (2 * epsilon);
    }

    assertGradientClose(finalAnalyticW, numW, tol, `Dense weight [${act}]`);
    assertGradientClose(finalAnalyticB, numB, tol, `Dense bias [${act}]`);
  }
  console.log("    ✅ Dense layer gradients are correct.");
}

/**
 * Gradient check untuk Softmax (Jacobian-vector product)
 */
function checkSoftmaxGradients(): void {
  console.log("  - Checking Softmax gradients (Auto-Diff)...");
  const epsilon = 1e-5;
  const tol = 1e-2;
  const x = mj.matrix([[1.0], [2.0], [-1.0], [0.5]]); 
  
  const tape = engine.startTape();
  const [p] = softmax(x, true);
  tape.backward(p);
  const analyticGradX = x.grad!._data;

  // Numerik (Loss = sum(p))
  const numX = new Float32Array(x._data.length);
  for (let i = 0; i < x._data.length; i++) {
    const v = x._data[i];
    x._data[i] = v + epsilon;
    const lP = softmax(x, true)[0]._data.reduce((a, b) => a + b, 0);
    x._data[i] = v - epsilon;
    const lM = softmax(x, true)[0]._data.reduce((a, b) => a + b, 0);
    x._data[i] = v;
    numX[i] = (lP - lM) / (2 * epsilon);
  }

  // Karena sum(softmax) = 1, gradiennya memang harus ~0
  assertGradientClose(analyticGradX, numX, tol, "Softmax sum gradient");
  
  engine.endTape();
  console.log("    ✅ Softmax gradients (sum path) are correct.");
}

/**
 * Gradient check untuk MultiHeadAttention
 */
function checkMultiHeadAttentionGradients(): void {
  console.log("  - Checking MultiHeadAttention gradients (Auto-Diff)...");
  
  const units = 4;
  const heads = 2;
  const seqLen = 3;
  const epsilon = 1e-5;
  const tol = 1e-1; // Sedikit lebih longgar karena kompleksitas MHA

  const mha = new MultiHeadAttention({ units, heads, seqLen, alpha: 0 });
  const x = mj.xavier([units, seqLen]);

  // 1. Hitung gradien analitik menggunakan Tape
  const tape = engine.startTape();
  const out = mha.forward(x);
  
  // Loss = sum(out)
  tape.backward(out);
  engine.endTape();

  const analyticGradQ = new Float32Array(mha.q.grad!._data);
  const analyticGradK = new Float32Array(mha.k.grad!._data);
  const analyticGradV = new Float32Array(mha.v.grad!._data);
  const analyticGradWo = new Float32Array(mha.wo.weight.grad!._data);

  // 2. Hitung gradien numerik
  const getLoss = () => {
    const output = mha.forward(x);
    return output._data.reduce((a, b) => a + b, 0);
  };

  const checkNumerical = (p: Matrix, analytic: Float32Array, label: string) => {
    const num = new Float32Array(p._data.length);
    for (let i = 0; i < p._data.length; i++) {
      const v = p._data[i];
      p._data[i] = v + epsilon;
      const lP = getLoss();
      p._data[i] = v - epsilon;
      const lM = getLoss();
      p._data[i] = v;
      num[i] = (lP - lM) / (2 * epsilon);
    }
    assertGradientClose(analytic, num, tol, label);
  };

  checkNumerical(mha.q, analyticGradQ, "MHA Q");
  checkNumerical(mha.k, analyticGradK, "MHA K");
  checkNumerical(mha.v, analyticGradV, "MHA V");
  checkNumerical(mha.wo.weight, analyticGradWo, "MHA Wo");

  console.log("    ✅ MultiHeadAttention gradients are correct.");
}

export function runAutoDiffGradientSuite(): void {
  console.log("\n🚀 Running Auto-Diff Gradient Checking...");
  checkDenseGradients();
  checkSoftmaxGradients();
  checkMultiHeadAttentionGradients();
  console.log("✅ Auto-Diff Gradient Checking Passed!");
}

const isMain = process.argv[1] && process.argv[1].includes("autodiff.gradient.test.ts");
if (isMain) {
  runAutoDiffGradientSuite();
}
