import { LSTM } from "@oxide-js/layers";
import { setLayers } from "@oxide-js/layers";

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
}

function assertAllClose(values: Float32Array, expected: number, message: string): void {
  for (const value of values) {
    if (Math.abs(value - expected) > 1e-6) {
      throw new Error(`${message}: expected ${expected}, got ${value}`);
    }
  }
}

function runDefaultForgetBiasTest(): void {
  const layer = new LSTM({ units: 3, hiddenUnits: 4 });
  assertAllClose(layer.bf._data, 1, "LSTM default forget bias harus 1");
}

function runCustomForgetBiasTest(): void {
  const layer = new LSTM({ units: 3, hiddenUnits: 4, forgetBias: 0 });
  assertAllClose(layer.bf._data, 0, "LSTM forgetBias custom harus dipakai saat init");
}

function runLoadDoesNotOverwriteSerializedBfTest(): void {
  const original = new LSTM({ units: 3, hiddenUnits: 4, forgetBias: 1 });
  original.bf._data.fill(0.25);
  const saved = original.save();

  const restored = new LSTM({ units: 3, hiddenUnits: 4 });
  restored.load(saved as any);

  assertAllClose(restored.bf._data, 0.25, "LSTM.load tidak boleh overwrite bf serialized");
}

function runSaveLoadPreservesForgetBiasTest(): void {
  const layer = new LSTM({ units: 3, hiddenUnits: 4, forgetBias: 0.5 });
  const saved = layer.save();
  const restored = setLayers([saved])[0] as LSTM;

  assert(restored.forgetBias === 0.5, "setLayers/save/load harus preserve field forgetBias");
  assertAllClose(restored.bf._data, 0.5, "save/load harus preserve nilai bf");
}

export function runLSTMForgetBiasCorrectnessSuite(): void {
  runDefaultForgetBiasTest();
  runCustomForgetBiasTest();
  runLoadDoesNotOverwriteSerializedBfTest();
  runSaveLoadPreservesForgetBiasTest();

  console.log("=== LSTM Forget Bias Correctness ===");
  console.table([
    { check: "default forget bias = 1", status: "pass" },
    { check: "custom forget bias init", status: "pass" },
    { check: "load preserves serialized bf", status: "pass" },
    { check: "save/load preserves forgetBias", status: "pass" },
  ]);
}

