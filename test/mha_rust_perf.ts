import mj from "../src/math";
import MultiHeadAttention from "../src/layers/multiHeadAttention";
import { setForceDisableNative } from "../src/math/rust_backend";

function runPass(layer: MultiHeadAttention, xData: ReturnType<typeof mj.matrix>, errData: ReturnType<typeof mj.matrix>, rounds: number): number {
  const t0 = Date.now();
  for (let i = 0; i < rounds; i++) {
    layer.forward(xData);
    layer.backward(mj.matrix([[]]), errData);
  }
  return Date.now() - t0;
}

function makeLayer(units: number, heads: number, seqLen: number): MultiHeadAttention {
  const layer = new MultiHeadAttention({ units, heads, seqLen, alpha: 0.01, status: "train" });
  layer.compile({ alpha: 0.01, optimizer: "sgd" });
  return layer;
}

const units = 32;
const heads = 4;
const seqLen = 16;
const batchSize = 4;
const cols = seqLen * batchSize;

const xRows: number[][] = [];
const errRows: number[][] = [];
for (let r = 0; r < units; r++) {
  const xRow: number[] = [];
  const eRow: number[] = [];
  for (let c = 0; c < cols; c++) {
    xRow.push(Math.sin((r + 1) * (c + 1) * 0.01));
    eRow.push(Math.cos((r + 1) * (c + 1) * 0.015) * 0.1);
  }
  xRows.push(xRow);
  errRows.push(eRow);
}

const x = mj.matrix(xRows);
const err = mj.matrix(errRows);
const rounds = 30;

setForceDisableNative(true);
const jsLayer = makeLayer(units, heads, seqLen);
const jsMs = runPass(jsLayer, x, err, rounds);

setForceDisableNative(false);
const nativeLayer = makeLayer(units, heads, seqLen);
const nativeMs = runPass(nativeLayer, x, err, rounds);

console.log(`mha_rust_perf: rounds=${rounds} js=${jsMs}ms native=${nativeMs}ms speedup=${(jsMs / Math.max(nativeMs, 1)).toFixed(2)}x`);
