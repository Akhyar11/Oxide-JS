/**
 * Test ML_V2 Library
 * Jalankan: npx ts-node test/test.ts
 *
 * Mencakup:
 * 1. Unit test math operations
 * 2. Forward + Backward 1 epoch
 * 3. Validasi loss turun per epoch (convergence test)
 * 4. Test optimizer Adam
 * 5. Test CrossEntropy
 */

import mj from "../src/math";
import { Dense } from "../src/layers";
import { Sequential, Transformers } from "../src/models";

let passed = 0;
let failed = 0;

function assert(condition: boolean, name: string) {
  if (condition) {
    console.log(`  ✅ PASS: ${name}`);
    passed++;
  } else {
    console.error(`  ❌ FAIL: ${name}`);
    failed++;
  }
}

function assertClose(a: number, b: number, tol = 1e-6, name: string = "") {
  assert(Math.abs(a - b) < tol, `${name} (got ${a.toFixed(8)}, expected ${b.toFixed(8)})`);
}

// ============================================================
// 1. UNIT TEST MATH
// ============================================================
console.log("\n=== 1. Math Operations ===");

const a = mj.matrix([[1, 2], [3, 4]]);
const b = mj.matrix([[5, 6], [7, 8]]);

// add
const addResult = mj.add(a, b);
assert(addResult._value[0][0] === 6 && addResult._value[1][1] === 12, "add matrix");
const addOut = mj.zeros([2, 2]);
const addIntoResult = mj.addInto(a, b, addOut);
assert(addIntoResult === addOut, "addInto return output buffer");
assert(addOut._value[0][0] === 6 && addOut._value[1][1] === 12, "addInto matrix correctness");
const addOutReuseRef = addOut._data;
const addOutReuse = mj.add(a, b, addOut);
assert(addOutReuse === addOut && addOut._data === addOutReuseRef, "add(..., out) reuse output buffer");

// sub
const subResult = mj.sub(b, a);
assert(subResult._value[0][0] === 4 && subResult._value[1][1] === 4, "sub matrix");
const subOut = mj.zeros([2, 2]);
const subIntoResult = mj.subInto(b, a, subOut);
assert(subIntoResult === subOut, "subInto return output buffer");
assert(subOut._value[0][0] === 4 && subOut._value[1][1] === 4, "subInto matrix correctness");
const subOutReuseRef = subOut._data;
const subOutReuse = mj.sub(b, a, subOut);
assert(subOutReuse === subOut && subOut._data === subOutReuseRef, "sub(..., out) reuse output buffer");

// addInto/subInto alias guard (out tidak boleh alias ke input matrix)
let addAliasGuardThrew = false;
try {
  mj.addInto(a, b, a);
} catch (_) {
  addAliasGuardThrew = true;
}
assert(addAliasGuardThrew, "addInto reject aliasing out===a");

let subAliasGuardThrew = false;
try {
  mj.subInto(b, a, b);
} catch (_) {
  subAliasGuardThrew = true;
}
assert(subAliasGuardThrew, "subInto reject aliasing out===a");

// mul (element-wise)
const mulResult = mj.mul(a, b);
assert(mulResult._value[0][0] === 5 && mulResult._value[0][1] === 12, "mul element-wise");

// dotProduct
const x = mj.matrix([[1, 0], [0, 1]]);
const dotResult = mj.dotProduct(a, x);
assert(dotResult._value[0][0] === 1 && dotResult._value[1][1] === 4, "dotProduct dengan identity");

// transpose
const t = mj.transpose(a);
assert(t._value[0][1] === 3 && t._value[1][0] === 2, "transpose");

// add scalar
const addScalar = mj.add(a, 10);
assert(addScalar._value[0][0] === 11 && addScalar._value[1][1] === 14, "add scalar");

// mul scalar
const mulScalar = mj.mul(a, 2);
assert(mulScalar._value[0][0] === 2 && mulScalar._value[1][1] === 8, "mul scalar");

// zeros & ones shape
const z = mj.zeros([3, 2]);
assert(z._shape[0] === 3 && z._shape[1] === 2, "zeros shape");
assert(z._value[0][0] === 0, "zeros value");

const o = mj.ones([2, 3]);
assert(o._value[0][0] === 1 && o._value[1][2] === 1, "ones value");

// mean
const m1 = mj.matrix([[1, 2], [3, 4]]);
assertClose(mj.mean(m1), 2.5, 1e-9, "mean");

// norm
const v = mj.matrix([[3, 4]]);
assertClose(mj.norm(v), 5.0, 1e-9, "norm (3-4-5 triangle)");

// flatten
const flat = mj.flatten(a);
assert(flat._shape[0] === 4 && flat._shape[1] === 1, "flatten shape");
assert(flat._value[0][0] === 1 && flat._value[3][0] === 4, "flatten value");

// reshape
const reshaped = mj.reshape(a, [1, 4]);
assert(reshaped._shape[0] === 1 && reshaped._shape[1] === 4, "reshape");

// logm (tidak ada NaN/Infinity)
const logInput = mj.matrix([[1, Math.E]]);
const logResult = mj.logm(logInput);
assertClose(logResult._value[0][0], 0.0, 1e-9, "log(1)=0");
assertClose(logResult._value[0][1], 1.0, 1e-9, "log(e)=1");

// logm clamp (input=0 jangan NaN)
const logZero = mj.logm(mj.matrix([[0]]));
assert(!isNaN(logZero._value[0][0]) && isFinite(logZero._value[0][0]), "logm guard: log(0) tidak NaN");

const addLegacy = mj.add(a, b);
const addOptimized = mj.add(a, b, mj.zeros([2, 2]));
assert(addLegacy._value[0][1] === addOptimized._value[0][1] && addLegacy._value[1][0] === addOptimized._value[1][0], "add legacy vs add with out konsisten");
const subLegacy = mj.sub(b, a);
const subOptimized = mj.sub(b, a, mj.zeros([2, 2]));
assert(subLegacy._value[0][1] === subOptimized._value[0][1] && subLegacy._value[1][0] === subOptimized._value[1][0], "sub legacy vs sub with out konsisten");

// ============================================================
// 2. FORWARD 1 EPOCH
// ============================================================
console.log("\n=== 2. Forward Pass ===");

const inputLayer = new Dense({ units: 2, outputUnits: 3, activation: "relu", optimizer: "sgd", alpha: 0.1, status: "input" });
const outputLayer = new Dense({ units: 3, outputUnits: 1, activation: "sigmoid", optimizer: "sgd", alpha: 0.1, status: "output", loss: "mse" });

const testInput = mj.matrix([[0.5], [0.3]]);
const testLabel = mj.matrix([[1.0]]);

const h = inputLayer.forward(testInput);
assert(h._shape[0] === 3 && h._shape[1] === 1, "hidden layer output shape [3,1]");

const out = outputLayer.forward(h);
assert(out._shape[0] === 1 && out._shape[1] === 1, "output layer shape [1,1]");
assert(out._value[0][0] >= 0 && out._value[0][0] <= 1, "sigmoid output dalam [0,1]");

// ============================================================
// 2B. TRANSFORMER RESIDUAL/ERROR PATH
// ============================================================
console.log("\n=== 2B. Transformer Residual Path ===");

const transformer = new Transformers({
  units: 4,
  seqLen: 3,
  vocabSize: 16,
  heads: 2,
  dropoutRate: 0,
  alpha: 0.01,
  padTokenId: 0,
});
const transformerInput = mj.matrix([[1, 2], [3, 0], [4, 5]]);
const transformerTarget = mj.matrix([[4, 3]]);
const transformerOut = transformer.forward(transformerInput);
assert(transformerOut._shape[0] === 16 && transformerOut._shape[1] === 2, "transformer forward output shape benar");
let transformerFinite = true;
for (let i = 0; i < transformerOut._data.length; i++) {
  if (!Number.isFinite(transformerOut._data[i])) {
    transformerFinite = false;
    break;
  }
}
assert(transformerFinite, "transformer forward output finite");
transformer.backward(transformerTarget);
assert(Number.isFinite(transformer.loss), "transformer backward loss finite");

// ============================================================
// 3. BACKWARD + CONVERGENCE TEST
// ============================================================
console.log("\n=== 3. Backward + Convergence Test ===");

const model = new Sequential();
model.add(new Dense({ units: 2, outputUnits: 4, activation: "relu", status: "input" }));
model.add(new Dense({ units: 4, outputUnits: 1, activation: "sigmoid", status: "output", loss: "mse" }));
model.compile({ alpha: 0.5, optimizer: "adam" });

// XOR problem
const X = [
  mj.matrix([[0], [0]]),
  mj.matrix([[0], [1]]),
  mj.matrix([[1], [0]]),
  mj.matrix([[1], [1]]),
];
const Y = [
  mj.matrix([[0]]),
  mj.matrix([[1]]),
  mj.matrix([[1]]),
  mj.matrix([[0]]),
];

const losses: number[] = [];
model.fit(X, Y, 200, (loss) => {
  losses.push(loss);
});

const firstLoss = losses[0];
const lastLoss = losses[losses.length - 1];
assert(lastLoss < firstLoss, `Loss turun: ${firstLoss.toFixed(4)} → ${lastLoss.toFixed(4)}`);
console.log(`  📊 Final loss: ${lastLoss.toFixed(6)}`);

// ============================================================
// 4. ADAM OPTIMIZER
// ============================================================
console.log("\n=== 4. Adam Optimizer ===");

import Adam from "../src/optimizer/adam";
const adam = new Adam([2, 2]);
const g1 = mj.matrix([[0.1, -0.2], [0.3, -0.4]]);
const u1 = adam.calculate(g1, 0.001);
assert(u1._shape[0] === 2 && u1._shape[1] === 2, "Adam output shape benar");
assert(u1._value[0][0] > 0 && u1._value[0][1] < 0, "Adam update sign benar (sesuai sign gradient)");

// tidak boleh ada NaN
let hasNaN = false;
for (let i = 0; i < u1._shape[0]; i++)
  for (let j = 0; j < u1._shape[1]; j++)
    if (isNaN(u1._value[i][j])) hasNaN = true;
assert(!hasNaN, "Adam output tidak ada NaN");

// ============================================================
// 5. CROSS ENTROPY
// ============================================================
console.log("\n=== 5. CrossEntropy Loss ===");

import { BinaryCrossEntropy } from "../src/cost/crossEntropy";
import CategoricalCrossEntropy from "../src/cost/crossEntropy";
import SoftmaxCrossEntropy from "../src/cost/softmaxCrossEntropy";

// Binary CE
const yTrue = mj.matrix([[1], [0], [1]]);
const yPred = mj.matrix([[0.9], [0.1], [0.8]]);
const [bceLoss, bceGrad] = BinaryCrossEntropy(yTrue, yPred);
assert(bceLoss > 0 && bceLoss < 0.5, `BCE loss reasonable: ${bceLoss.toFixed(4)}`);
assert(bceGrad._value[0][0] < 0, "BCE gradient negatif saat y=1 dan pred tinggi (mengarah ke update yang benar)");

// Categorical CE
const ytCat = mj.matrix([[0, 1, 0]]);
const ypCat = mj.matrix([[0.1, 0.8, 0.1]]);
const [cceLoss, cceGrad] = CategoricalCrossEntropy(ytCat, ypCat);
assert(cceLoss > 0, `CCE loss > 0: ${cceLoss.toFixed(4)}`);

// Sparse Softmax CE
const sparseTarget = mj.matrix([[1]]);
const logits = mj.matrix([[1.0], [3.0], [0.5]]);
const [sceLoss, sceGrad] = SoftmaxCrossEntropy(sparseTarget, logits);
assert(sceLoss > 0, `Sparse SCE loss > 0: ${sceLoss.toFixed(4)}`);
assert(sceGrad._shape[0] === 3 && sceGrad._shape[1] === 1, "Sparse SCE gradient shape benar");
assert(sceGrad._value[1][0] < 0, "Sparse SCE gradient negatif untuk kelas target");

// ============================================================
// SUMMARY
// ============================================================
console.log(`\n${"=".repeat(40)}`);
console.log(`✅ PASSED: ${passed}  ❌ FAILED: ${failed}`);
console.log(`${"=".repeat(40)}`);
if (failed > 0) process.exit(1);
