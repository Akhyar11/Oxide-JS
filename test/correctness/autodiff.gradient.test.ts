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

  // 1. Hitung gradien analitik melalui backward layer langsung
  const out = mha.forward(x);
  const gradOut = mj.ones(out._shape);
  mha.backward(mj.matrix([[]]), gradOut, true);

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

function checkMultiHeadAttentionExternalInputs(): void {
  console.log("  - Checking MultiHeadAttention external query/key/value inputs...");

  const mha = new MultiHeadAttention({ units: 4, heads: 2, seqLen: 2, alpha: 0 });
  const query = mj.matrix([
    [1, 0],
    [0, 1],
    [0, 0],
    [0, 0],
  ]);
  const key = mj.matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ]);
  const value = mj.matrix([
    [2, 0, 1, 0],
    [0, 3, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ]);

  mha.setAttentionInputs({
    query,
    key,
    value,
    queryProjected: true,
    keyProjected: true,
    valueProjected: true,
    querySeqLen: 2,
    keySeqLen: 4,
    causal: false,
  });
  const out = mha.forward();
  assert(out._shape[0] === 4 && out._shape[1] === 2, "MHA external inputs should produce [units, qCols] output");

  const grad = mj.matrix([
    [1, -1],
    [0.5, -0.5],
    [0, 0],
    [0, 0],
  ]);
  const dx = mha.backward(mj.matrix([[]]), grad, true);
  const inputGrads = mha.getLastInputGradients();

  assert(dx._shape[0] === 4 && dx._shape[1] === 2, "MHA backward should return query-source gradient shape");
  assert(inputGrads.query !== null && inputGrads.key !== null && inputGrads.value !== null, "MHA should expose query/key/value gradients");

  const gradMagnitude = (m: Matrix | null): number => m ? m._data.reduce((sum, value) => sum + Math.abs(value), 0) : 0;
  assert(gradMagnitude(inputGrads.query) > 0, "MHA external query should receive non-zero gradient");
  assert(gradMagnitude(inputGrads.key) > 0, "MHA external key should receive non-zero gradient");
  assert(gradMagnitude(inputGrads.value) > 0, "MHA external value should receive non-zero gradient");

  console.log("    ✅ MultiHeadAttention external inputs are correct.");
}

function checkScalarOpGradients(): void {
  console.log("  - Checking scalar elementwise op gradients...");

  {
    const x = mj.matrix([[2], [4]]);
    const tape = engine.startTape();
    const y = mj.add(x, 3);
    const loss = mj.mean(y);
    tape.backward(loss);
    engine.endTape();
    assertGradientClose(x.grad!._data, new Float32Array([0.5, 0.5]), 1e-6, "add(matrix, scalar)");
  }

  {
    const x = mj.matrix([[2], [4]]);
    const tape = engine.startTape();
    const y = mj.sub(10, x);
    const loss = mj.mean(y);
    tape.backward(loss);
    engine.endTape();
    assertGradientClose(x.grad!._data, new Float32Array([-0.5, -0.5]), 1e-6, "sub(scalar, matrix)");
  }

  {
    const x = mj.matrix([[2], [4]]);
    const tape = engine.startTape();
    const y = mj.mul(x, 0.5);
    const loss = mj.mean(y);
    tape.backward(loss);
    engine.endTape();
    assertGradientClose(x.grad!._data, new Float32Array([0.25, 0.25]), 1e-6, "mul(matrix, scalar)");
  }

  {
    const x = mj.matrix([[2], [4]]);
    const tape = engine.startTape();
    const y = mj.div(x, 2);
    const loss = mj.mean(y);
    tape.backward(loss);
    engine.endTape();
    assertGradientClose(x.grad!._data, new Float32Array([0.25, 0.25]), 1e-6, "div(matrix, scalar)");
  }

  {
    const x = mj.matrix([[2], [4]]);
    const tape = engine.startTape();
    const y = mj.div(2, x);
    const loss = mj.mean(y);
    tape.backward(loss);
    engine.endTape();
    assertGradientClose(x.grad!._data, new Float32Array([-0.25, -0.0625]), 1e-6, "div(scalar, matrix)");
  }

  console.log("    ✅ Scalar elementwise op gradients are correct.");
}

function checkTapeRestoresShapeSnapshots(): void {
  console.log("  - Checking tape shape snapshot restore...");

  const x = mj.matrix([
    [1, 2],
    [3, 4],
  ]);
  const tape = engine.startTape();
  const y = mj.add(x, x);
  const expectedShape: [number, number] = [2, 2];

  tape.record([x], [y], (_grad) => {
    assert(
      y._shape[0] === expectedShape[0] && y._shape[1] === expectedShape[1],
      `tape should restore output shape snapshot, got [${y._shape[0]}, ${y._shape[1]}]`
    );
  });

  y.reshape([4, 1]);
  tape.backward(y);
  engine.endTape();

  console.log("    ✅ Tape restores shape snapshots.");
}

function checkMultiOutputTapeBackward(): void {
  console.log("  - Checking multi-output tape backward support...");

  const x = mj.matrix([[1], [2]]);
  const tape = engine.startTape();
  const y1 = mj.matrix([[3], [4]]);
  const y2 = mj.matrix([[5], [6]]);

  tape.record([x], [y1, y2], (_grad, outputGrads) => {
    assert(outputGrads !== undefined && outputGrads.length === 2, "tape should provide grads for every recorded output");
    const g1 = outputGrads![0];
    const g2 = outputGrads![1];
    assert(g1 !== null && g2 !== null, "both output gradients should be visible to backward callback");
    const total = mj.add(g1!, g2!);
    if (x.grad) x.grad.addInPlace(total);
    else x.grad = total;
  });

  y2.grad = mj.ones(y2._shape);
  tape.backward(y1);
  engine.endTape();

  assertGradientClose(x.grad!._data, new Float32Array([2, 2]), 1e-6, "multi-output tape backward");
  console.log("    ✅ Multi-output tape backward is correct.");
}

function checkEngineGradLifecycle(): void {
  console.log("  - Checking engine.grad lifecycle cleanup...");

  const tape = engine.grad(() => {
    const x = mj.matrix([[1], [2]]);
    return mj.mean(mj.mul(x, 2));
  });

  assert(engine.tape === null, "engine.grad should clear active tape after callback");
  assert(tape.isActive() === false, "engine.grad should stop the returned tape");
  assert(tape.result._shape[0] === 1 && tape.result._shape[1] === 1, "engine.grad should expose callback result on returned tape");
  console.log("    ✅ engine.grad lifecycle cleanup is correct.");
}

export function runAutoDiffGradientSuite(): void {
  console.log("\n🚀 Running Auto-Diff Gradient Checking...");
  checkDenseGradients();
  checkSoftmaxGradients();
  checkMultiHeadAttentionGradients();
  checkMultiHeadAttentionExternalInputs();
  checkScalarOpGradients();
  checkTapeRestoresShapeSnapshots();
  checkMultiOutputTapeBackward();
  checkEngineGradLifecycle();
  console.log("✅ Auto-Diff Gradient Checking Passed!");
}

const isMain = process.argv[1] && process.argv[1].includes("autodiff.gradient.test.ts");
if (isMain) {
  runAutoDiffGradientSuite();
}
