import mj from "../src/math";
import Matrix from "../src/matrix";

const SCALE = Number(process.env.BENCH_SCALE ?? "1");
const EXTRA_ITERS = Number(process.env.BENCH_ITERS ?? "1");

let sink = 0;

type BenchCase = {
  name: string;
  warmup: number;
  iterations: number;
  runOptimized: () => number;
  runBaseline: () => number;
  check: () => void;
};

function makeMatrix(rows: number, cols: number, offset = 0, positiveOnly = false): Matrix {
  const arr: number[][] = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const row = new Array(cols);
    for (let j = 0; j < cols; j++) {
      const raw = ((i * 131 + j * 17 + offset) % 1000) / 100;
      row[j] = positiveOnly ? raw + 1 : raw - 5;
    }
    arr[i] = row;
  }
  return mj.matrix(arr);
}

function assertCloseMatrix(actual: Matrix, expected: Matrix, name: string, tol = 1e-9) {
  if (actual._shape[0] !== expected._shape[0] || actual._shape[1] !== expected._shape[1]) {
    throw new Error(`${name}: shape mismatch ${actual._shape} != ${expected._shape}`);
  }
  for (let i = 0; i < actual._data.length; i++) {
    if (Math.abs(actual._data[i] - expected._data[i]) > tol) {
      throw new Error(`${name}: value mismatch at flat index ${i}`);
    }
  }
}

function baselineAdd(a: Matrix, b: Matrix): Matrix {
  const av = a._value;
  const bv = b._value;
  const out: number[][] = [];
  for (let i = 0; i < av.length; i++) {
    out[i] = [];
    for (let j = 0; j < av[0].length; j++) out[i][j] = av[i][j] + bv[i][j];
  }
  return mj.matrix(out);
}

function baselineMul(a: Matrix, b: Matrix): Matrix {
  const av = a._value;
  const bv = b._value;
  const out: number[][] = [];
  for (let i = 0; i < av.length; i++) {
    out[i] = [];
    for (let j = 0; j < av[0].length; j++) out[i][j] = av[i][j] * bv[i][j];
  }
  return mj.matrix(out);
}

function baselineDiv(a: Matrix, b: Matrix): Matrix {
  const av = a._value;
  const bv = b._value;
  const out: number[][] = [];
  for (let i = 0; i < av.length; i++) {
    out[i] = [];
    for (let j = 0; j < av[0].length; j++) out[i][j] = av[i][j] / bv[i][j];
  }
  return mj.matrix(out);
}

function baselineTranspose(a: Matrix): Matrix {
  const av = a._value;
  const rows = av.length;
  const cols = av[0].length;
  const out: number[][] = new Array(cols);
  for (let j = 0; j < cols; j++) {
    out[j] = new Array(rows);
    for (let i = 0; i < rows; i++) out[j][i] = av[i][j];
  }
  return mj.matrix(out);
}

function baselineDotProduct(a: Matrix, b: Matrix): Matrix {
  const av = a._value;
  const bv = b._value;
  const rows = av.length;
  const shared = av[0].length;
  const cols = bv[0].length;
  const out: number[][] = new Array(rows);
  for (let i = 0; i < rows; i++) {
    out[i] = new Array(cols).fill(0);
    for (let k = 0; k < shared; k++) {
      for (let j = 0; j < cols; j++) out[i][j] += av[i][k] * bv[k][j];
    }
  }
  return mj.matrix(out);
}

function baselineFlatten(a: Matrix): Matrix {
  const av = a._value;
  const out: number[][] = [];
  for (let i = 0; i < av.length; i++) {
    for (let j = 0; j < av[0].length; j++) out.push([av[i][j]]);
  }
  return mj.matrix(out);
}

function baselineReshape(a: Matrix, rows: number, cols: number): Matrix {
  const flat = baselineFlatten(a)._value.map((row) => row[0]);
  const out: number[][] = new Array(rows);
  let idx = 0;
  for (let i = 0; i < rows; i++) {
    out[i] = new Array(cols);
    for (let j = 0; j < cols; j++) out[i][j] = flat[idx++];
  }
  return mj.matrix(out);
}

function measure(label: string, iterations: number, fn: () => number): { label: string; totalMs: number; avgMs: number } {
  const start = process.hrtime.bigint();
  for (let i = 0; i < iterations; i++) sink += fn();
  const totalMs = Number(process.hrtime.bigint() - start) / 1e6;
  return { label, totalMs, avgMs: totalMs / iterations };
}

function printResult(name: string, optimized: { totalMs: number; avgMs: number }, baseline: { totalMs: number; avgMs: number }) {
  const speedup = baseline.avgMs / optimized.avgMs;
  const verdict = speedup >= 1 ? "lebih cepat" : "lebih lambat";
  console.log(
    `${name.padEnd(14)} | optimized ${optimized.avgMs.toFixed(3)} ms | baseline ${baseline.avgMs.toFixed(3)} ms | ${speedup.toFixed(2)}x ${verdict}`
  );
}

const wideA = makeMatrix(256 * SCALE, 256 * SCALE, 11);
const wideB = makeMatrix(256 * SCALE, 256 * SCALE, 29);
const divA = makeMatrix(256 * SCALE, 256 * SCALE, 7, true);
const divB = makeMatrix(256 * SCALE, 256 * SCALE, 13, true);
const dotA = makeMatrix(192 * SCALE, 256 * SCALE, 5);
const dotB = makeMatrix(256 * SCALE, 128 * SCALE, 19);
const flatSource = makeMatrix(512 * SCALE, 256 * SCALE, 23);

const cases: BenchCase[] = [
  {
    name: "add",
    warmup: 3,
    iterations: 20 * EXTRA_ITERS,
    runOptimized: () => mj.add(wideA, wideB).get(0, 0),
    runBaseline: () => baselineAdd(wideA, wideB).get(0, 0),
    check: () => assertCloseMatrix(mj.add(wideA, wideB), baselineAdd(wideA, wideB), "add"),
  },
  {
    name: "mul",
    warmup: 3,
    iterations: 20 * EXTRA_ITERS,
    runOptimized: () => mj.mul(wideA, wideB).get(0, 0),
    runBaseline: () => baselineMul(wideA, wideB).get(0, 0),
    check: () => assertCloseMatrix(mj.mul(wideA, wideB), baselineMul(wideA, wideB), "mul"),
  },
  {
    name: "div",
    warmup: 3,
    iterations: 20 * EXTRA_ITERS,
    runOptimized: () => mj.div(divA, divB).get(0, 0),
    runBaseline: () => baselineDiv(divA, divB).get(0, 0),
    check: () => assertCloseMatrix(mj.div(divA, divB), baselineDiv(divA, divB), "div"),
  },
  {
    name: "transpose",
    warmup: 3,
    iterations: 12 * EXTRA_ITERS,
    runOptimized: () => mj.transpose(flatSource).get(0, 0),
    runBaseline: () => baselineTranspose(flatSource).get(0, 0),
    check: () => assertCloseMatrix(mj.transpose(flatSource), baselineTranspose(flatSource), "transpose"),
  },
  {
    name: "dotProduct",
    warmup: 2,
    iterations: 6 * EXTRA_ITERS,
    runOptimized: () => mj.dotProduct(dotA, dotB).get(0, 0),
    runBaseline: () => baselineDotProduct(dotA, dotB).get(0, 0),
    check: () => assertCloseMatrix(mj.dotProduct(dotA, dotB), baselineDotProduct(dotA, dotB), "dotProduct", 1e-7),
  },
  {
    name: "flatten",
    warmup: 5,
    iterations: 100 * EXTRA_ITERS,
    runOptimized: () => mj.flatten(flatSource).get(0, 0),
    runBaseline: () => baselineFlatten(flatSource).get(0, 0),
    check: () => assertCloseMatrix(mj.flatten(flatSource), baselineFlatten(flatSource), "flatten"),
  },
  {
    name: "reshape",
    warmup: 5,
    iterations: 100 * EXTRA_ITERS,
    runOptimized: () => mj.reshape(flatSource, [256 * SCALE, 512 * SCALE]).get(0, 0),
    runBaseline: () => baselineReshape(flatSource, 256 * SCALE, 512 * SCALE).get(0, 0),
    check: () => assertCloseMatrix(
      mj.reshape(flatSource, [256 * SCALE, 512 * SCALE]),
      baselineReshape(flatSource, 256 * SCALE, 512 * SCALE),
      "reshape"
    ),
  },
];

console.log("=== Math Performance Benchmark ===");
console.log(`scale=${SCALE} extraIters=${EXTRA_ITERS}`);
console.log("Angka lebih kecil lebih cepat. Perbandingan terhadap baseline naif number[][].\n");

for (const testCase of cases) {
  testCase.check();
  for (let i = 0; i < testCase.warmup; i++) {
    sink += testCase.runOptimized();
    sink += testCase.runBaseline();
  }

  const optimized = measure(`${testCase.name}-optimized`, testCase.iterations, testCase.runOptimized);
  const baseline = measure(`${testCase.name}-baseline`, testCase.iterations, testCase.runBaseline);
  printResult(testCase.name, optimized, baseline);
}

console.log(`\nbenchmark sink=${sink.toFixed(4)}`);
