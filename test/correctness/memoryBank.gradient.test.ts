import { MemoryBank } from "@oxide-js/layers";
import { mj } from "@oxide-js/core";

type ParamName = "queryKernel" | "writeGateKernel" | "writeGateBias" | "writeQueryKernel" | "needKernel" | "outputKernel" | "outputBias";

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
}

function deepClone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value));
}

function makeLayerFromSave(saved: any): MemoryBank {
  const memorySlots = saved?.dimensions?.memorySlots ?? saved?.memorySlots ?? 1;
  const layer = new MemoryBank({ memorySlots });
  layer.load(deepClone(saved));
  return layer;
}

function scalarLoss(layer: MemoryBank, x: number[][], err: number[][]): number {
  const out = layer.forward(mj.matrix(x));
  let sum = 0;
  for (let i = 0; i < out._data.length; i++) sum += out._data[i] * err[Math.floor(i / out._shape[1])][i % out._shape[1]];
  return sum;
}

function analyticGradient(saved: any, x: number[][], err: number[][], paramName: ParamName): Float32Array {
  const layer = makeLayerFromSave(saved);
  const before = ((layer as any)[paramName] as { _data: Float32Array })._data.slice();
  layer.forward(mj.matrix(x));
  layer.backward(mj.matrix([[]]), mj.matrix(err));
  const after = ((layer as any)[paramName] as { _data: Float32Array })._data;
  const grad = new Float32Array(before.length);
  for (let i = 0; i < before.length; i++) grad[i] = before[i] - after[i];
  return grad;
}

function numericGradient(saved: any, x: number[][], err: number[][], paramName: ParamName, epsilon: number): Float32Array {
  const base = deepClone(saved);
  const param = base.trainableParams?.[paramName] ?? base[paramName];
  assert(param, `numericGradient: missing parameter '${paramName}' in save payload`);

  const flat: Array<{ row: number; col: number }> = [];
  for (let row = 0; row < param.length; row++) {
    for (let col = 0; col < param[row].length; col++) flat.push({ row, col });
  }

  const grad = new Float32Array(flat.length);
  for (let idx = 0; idx < flat.length; idx++) {
    const { row, col } = flat[idx];

    const plus = deepClone(base);
    const minus = deepClone(base);
    (plus.trainableParams?.[paramName] ?? plus[paramName])[row][col] += epsilon;
    (minus.trainableParams?.[paramName] ?? minus[paramName])[row][col] -= epsilon;

    const lossPlus = scalarLoss(makeLayerFromSave(plus), x, err);
    const lossMinus = scalarLoss(makeLayerFromSave(minus), x, err);
    grad[idx] = (lossPlus - lossMinus) / (2 * epsilon);
  }

  return grad;
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
      `${label}: max abs error ${maxAbsErr} exceeds tolerance ${tol} at flat index ${worstIndex} ` +
      `(analytic=${actual[worstIndex]}, numeric=${expected[worstIndex]})`
    );
  }
}

export function runMemoryBankGradientSuite(): void {
  // 1) Gradient-check for project mode read path with fixed memory.
  {
    const layer = new MemoryBank({
      units: 2,
      memorySlots: 2,
      outputUnits: 2,
      mode: "project",
      similarity: "dot",
      readTopK: 2,
      writeEnabled: false,
      optimizer: "sgd",
      alpha: 1,
      clipGradient: false,
    });

    layer.forward(mj.matrix([[0], [0]]));
    layer.setMemoryState({
      memoryKeys: [
        [1, 0.2],
        [0, 1],
      ],
      memoryValues: [
        [0.25, 1],
        [1, -0.5],
      ],
      memoryFilled: [1, 1],
      memoryUsage: [1, 1],
      memoryAge: [1, 2],
      memoryStep: 2,
      units: 2,
      memoryDim: 2,
      memorySlots: 2,
    });

    const saved = layer.save();
    const x = [[0.6], [-0.8]];
    const err = [[0.35], [-0.2]];
    const epsilon = 1e-3;
    const tol = 3e-2;

    for (const paramName of ["queryKernel", "needKernel", "outputKernel", "outputBias"] as ParamName[]) {
      const analytic = analyticGradient(saved, x, err, paramName);
      const numeric = numericGradient(saved, x, err, paramName, epsilon);
      assertGradientClose(analytic, numeric, tol, `fixed-memory ${paramName}`);
    }
  }

  // 2) Gradient-check for queryKernel through write-then-read temporal path.
  {
    const layer = new MemoryBank({
      units: 2,
      memorySlots: 2,
      outputUnits: 2,
      mode: "project",
      similarity: "dot",
      readTopK: 1,
      writeEnabled: true,
      optimizer: "sgd",
      alpha: 1,
      clipGradient: false,
    });

    const saved = layer.save();
    const x = [
      [1, 1],
      [0, 1],
    ];
    const err = [
      [0, 0.4],
      [0, -0.3],
    ];
    const epsilon = 1e-3;
    const tol = 4e-2;

    for (const paramName of ["queryKernel", "writeGateKernel", "writeGateBias", "writeQueryKernel"] as ParamName[]) {
      const analytic = analyticGradient(saved, x, err, paramName);
      const numeric = numericGradient(saved, x, err, paramName, epsilon);
      assertGradientClose(analytic, numeric, tol, `write-read ${paramName}`);
    }
  }
}

