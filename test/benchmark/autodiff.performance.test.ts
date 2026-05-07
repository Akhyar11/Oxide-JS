import { performance } from "perf_hooks";
import { engine, mj, Matrix } from "@oxide-js/core";
import { Transformers } from "@oxide-js/models";
import { writeFileSync } from "fs";

/**
 * Benchmark untuk mengukur overhead Gradient Tape pada throughput pelatihan CPU.
 */
export async function runAutoDiffPerformanceBenchmark(): Promise<void> {
  console.log("\n🚀 Running Auto-Diff Performance Benchmark...");

  const config = {
    epochs: 2,
    sampleCount: 32,
    seqLen: 128,    // Sequence length cukup panjang untuk memberi beban pada tape
    vocabSize: 1000,
    units: 128,     // d_model
    heads: 4,
    numBlocks: 2,
    batchSize: 4,
  };

  // 1. Generate Synthetic Data
  const X = Array.from({ length: config.sampleCount }, () => {
    const data = new Float32Array(config.seqLen);
    for (let i = 0; i < config.seqLen; i++) {
      data[i] = Math.floor(Math.random() * config.vocabSize);
    }
    return Matrix.fromFlat(data, [config.seqLen, 1]);
  });
  const y = Array.from({ length: config.sampleCount }, () => {
    const data = new Float32Array(config.seqLen);
    for (let i = 0; i < config.seqLen; i++) {
      data[i] = Math.floor(Math.random() * config.vocabSize);
    }
    return Matrix.fromFlat(data, [config.seqLen, 1]);
  });

  const runBench = async (useTape: boolean) => {
    const model = new Transformers({
      units: config.units,
      seqLen: config.seqLen,
      vocabSize: config.vocabSize,
      heads: config.heads,
      numBlocks: config.numBlocks,
      alpha: 0.001,
    });

    const start = performance.now();
    for (let epoch = 0; epoch < config.epochs; epoch++) {
      for (let i = 0; i < config.sampleCount; i += config.batchSize) {
        process.stdout.write(`.`);
        const batchX = mj.zeros([config.seqLen, config.batchSize]);
        const batchY = mj.zeros([config.seqLen, config.batchSize]);
        for (let j = 0; j < config.batchSize; j++) {
           batchX.setCol(j, X[i + j]._data);
           batchY.setCol(j, y[i + j]._data);
        }

        try {
          if (useTape) {
            engine.startTape();
            model.forward(batchX);
            model.backward(batchY);
            engine.endTape();
          } else {
            model.forward(batchX);
            model.backward(batchY);
          }
        } catch (e: any) {
          writeFileSync("benchmark_error.log", e.stack || String(e));
          console.error("\n❌ Error during benchmark execution. See benchmark_error.log");
          process.exit(1);
        }
      }
    }
    console.log(" done.");
    return performance.now() - start;
  };

  console.log(`  Configuration: seqLen=${config.seqLen}, units=${config.units}, blocks=${config.numBlocks}, batchSize=${config.batchSize}`);
  
  // Warmup
  await runBench(false);
  await runBench(true);

  const timeNoTape = await runBench(false);
  const timeWithTape = await runBench(true);

  const overhead = ((timeWithTape - timeNoTape) / timeNoTape) * 100;

  console.table([
    { Mode: "Manual Backward (Tape OFF)", TotalMs: timeNoTape.toFixed(2), SamplesPerSec: ((config.sampleCount * config.epochs) / (timeNoTape / 1000)).toFixed(2) },
    { Mode: "Manual Backward (Tape ON - Forward Recording)", TotalMs: timeWithTape.toFixed(2), SamplesPerSec: ((config.sampleCount * config.epochs) / (timeWithTape / 1000)).toFixed(2) }
  ]);

  console.log(`\n  Estimated Tape Recording Overhead: ${overhead.toFixed(2)}%`);
  
  if (overhead > 20) {
    console.warn("  ⚠️ Warning: Tape overhead is quite high (> 20%). Consider optimizing Tape.record and snapshotting.");
  } else {
    console.log("  ✅ Tape overhead is within acceptable range.");
  }
}

const isMain = process.argv[1] && process.argv[1].includes("autodiff.performance.test.ts");
if (isMain) {
  runAutoDiffPerformanceBenchmark();
}
