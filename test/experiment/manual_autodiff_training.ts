import { mj, engine, Adam } from "@oxide-js/core";
import { Dense } from "@oxide-js/layers";

async function main() {
  console.log("🚀 Starting Manual Auto-Diff Training Experiment...");

  const inputDim = 4;
  const outputDim = 1;
  const learningRate = 0.03;
  const steps = 80;
  
  // 1. Setup Model & Optimizer
  const model = new Dense({ units: inputDim, outputUnits: outputDim, activation: "linear" });
  model.weight._data.fill(0);
  model.bias._data.fill(0);
  const params = model.getParams();
  const optimizers = new Map(params.map((p) => [p, new Adam(p._shape)]));

  // 2. Single-sample linear regression target.
  // The goal is not model quality; it proves Tape can drive an external optimizer loop.
  const x = mj.matrix([[0.25], [-0.75], [0.5], [1.0]]);
  const target = mj.matrix([[1.25]]);
  const half = mj.matrix([[0.5]]);

  let initialLoss = Number.NaN;
  let finalLoss = Number.NaN;

  for (let step = 0; step < steps; step++) {
    for (const p of params) p.clearGrad();

    const tape = engine.startTape();
    const pred = model.forward(x);
    const diff = mj.sub(pred, target);
    const squaredError = mj.mul(diff, diff);
    const loss = mj.mul(squaredError, half);

    tape.backward(loss);
    engine.endTape();

    const lossValue = loss._data[0];
    if (step === 0) initialLoss = lossValue;
    finalLoss = lossValue;

    for (const p of params) {
      if (p.grad) {
        optimizers.get(p)!.apply(p, learningRate);
      }
    }
  }

  console.log(`  Initial loss: ${initialLoss.toFixed(8)}`);
  console.log(`  Final loss:   ${finalLoss.toFixed(8)}`);

  if (!Number.isFinite(finalLoss) || finalLoss >= initialLoss * 0.2) {
    throw new Error(
      `Manual Auto-Diff training did not converge enough: initial=${initialLoss}, final=${finalLoss}`
    );
  }

  console.log("✅ Manual Auto-Diff Training Experiment Finished!");
}

main().catch(err => {
  console.error("CAUGHT ERROR:", err);
});
