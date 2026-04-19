import mj from "../src/math";
import LayerNormalization from "../src/layers/layerNormalization";
import { setForceDisableNative } from "../src/math/rust_backend";

function makeLayer(units: number): LayerNormalization {
  const layer = new LayerNormalization({
    units,
    status: "train",
    alpha: 0.001,
    optimizer: "sgd",
  });
  layer.compile({ alpha: 0.001, optimizer: "sgd" });
  return layer;
}

function runPass(layer: LayerNormalization, x: ReturnType<typeof mj.matrix>, err: ReturnType<typeof mj.matrix>, rounds: number): number {
  const t0 = Date.now();
  for (let i = 0; i < rounds; i++) {
    layer.forward(x);
    layer.backward(mj.matrix([[]]), err);
  }
  return Date.now() - t0;
}

const units = 128;
const seqLen = 128;
const batchSize = 8;
const cols = seqLen * batchSize;
const rounds = 40;

const xRows: number[][] = [];
const errRows: number[][] = [];
for (let r = 0; r < units; r++) {
  const xRow: number[] = [];
  const eRow: number[] = [];
  for (let c = 0; c < cols; c++) {
    xRow.push(Math.sin((r + 1) * (c + 1) * 0.002));
    eRow.push(Math.cos((r + 1) * (c + 1) * 0.003) * 0.1);
  }
  xRows.push(xRow);
  errRows.push(eRow);
}

const x = mj.matrix(xRows);
const err = mj.matrix(errRows);

setForceDisableNative(true);
const fallbackLayer = makeLayer(units);
const jsMs = runPass(fallbackLayer, x, err, rounds);

setForceDisableNative(false);
const nativeLayer = makeLayer(units);
const nativeMs = runPass(nativeLayer, x, err, rounds);

console.log(`layernorm_rust_perf: rounds=${rounds} js=${jsMs}ms native=${nativeMs}ms speedup=${(jsMs / Math.max(nativeMs, 1)).toFixed(2)}x`);
