import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import { Worker } from "worker_threads";
import { Transformers } from "../../src/models";
import { BPETokenizer } from "../../src/tokenizer";
import Matrix from "../../src/matrix";
import { TransformerPipeline } from "../../src/pipeline/transformer-pipeline";

interface TrainingConfig {
  numEpochs: number;
  microBatchSize: number;
  learningRate: number;
  numPipelineStages: number;
  usePipeline: boolean;
  contextLen: number;
  shuffle: boolean;
}

interface TrainingSample {
  input: Matrix;
  target: Matrix; // class index [1, 1]
}

interface TrainingMetrics {
  epoch: number;
  loss: number;
  throughput: number;
  timeElapsedSec: number;
}

const DEFAULT_CONFIG: TrainingConfig = {
  numEpochs: 5,
  microBatchSize: 32,
  learningRate: 0.0001,
  numPipelineStages: 4,
  usePipeline: true,
  contextLen: 32,
  shuffle: true,
};

class AdvancedTrainer {
  private model: Transformers;
  private tokenizer: BPETokenizer;
  private config: TrainingConfig;
  private pipeline?: TransformerPipeline;
  private metrics: TrainingMetrics[] = [];

  constructor(model: Transformers, tokenizer: BPETokenizer, config: Partial<TrainingConfig> = {}) {
    this.model = model;
    this.tokenizer = tokenizer;
    this.config = { ...DEFAULT_CONFIG, ...config };

    if (this.config.usePipeline) {
      this.pipeline = new TransformerPipeline(model, this.config.numPipelineStages, this.config.microBatchSize);
    }
  }

  async train(samples: TrainingSample[]): Promise<void> {
    if (samples.length === 0) {
      throw new Error("Tidak ada sample training.");
    }

    const trainStart = Date.now();
    for (let epoch = 0; epoch < this.config.numEpochs; epoch++) {
      const epochStart = Date.now();
      if (this.config.shuffle) {
        this.shuffleInPlace(samples);
      }
      const loss = this.config.usePipeline && this.pipeline
        ? await this.trainWithPipeline(samples)
        : await this.trainNaive(samples);

      const epochElapsedSec = (Date.now() - epochStart) / 1000;
      const throughput = samples.length / Math.max(epochElapsedSec, 1e-9);
      const timeElapsedSec = (Date.now() - trainStart) / 1000;

      this.metrics.push({
        epoch: epoch + 1,
        loss,
        throughput,
        timeElapsedSec,
      });

      console.log(
        `Epoch ${epoch + 1}/${this.config.numEpochs} - ` +
        `Loss: ${loss.toFixed(6)} - ` +
        `Throughput: ${throughput.toFixed(0)} sampel/s - ` +
        `Elapsed: ${timeElapsedSec.toFixed(2)}s`
      );
    }
  }

  private async trainNaive(samples: TrainingSample[]): Promise<number> {
    let lossSum = 0;

    for (const sample of samples) {
      this.model.forward(sample.input);
      this.model.backward(sample.target);
      lossSum += this.model.loss;
    }

    return lossSum / samples.length;
  }

  private async trainWithPipeline(samples: TrainingSample[]): Promise<number> {
    let lossSum = 0;

    for (let i = 0; i < samples.length; i += this.config.microBatchSize) {
      const microBatch = samples.slice(i, i + this.config.microBatchSize);
      // Forward + backward per sample, dijadwalkan via API pipeline (tanpa double-forward).
      for (const sample of microBatch) {
        await this.pipeline!.forwardPipeline(sample.input);
        this.model.backward(sample.target);
        lossSum += this.model.loss;
      }
    }

    return lossSum / samples.length;
  }

  async loadTrainingData(dataPath: string): Promise<TrainingSample[]> {
    if (!fs.existsSync(dataPath)) {
      console.warn(`Data file tidak ditemukan: ${dataPath}`);
      return [];
    }

    const rawData = JSON.parse(fs.readFileSync(dataPath, "utf-8"));
    if (!Array.isArray(rawData)) {
      throw new Error("Format training_data.json harus berupa array objek { text: string }.");
    }

    const samples: TrainingSample[] = [];
    const padId = this.tokenizer.getPadId();

    for (const item of rawData) {
      if (!item || typeof item.text !== "string") continue;
      const tokens = this.tokenizer.encode(item.text.toLowerCase());
      for (let idx = 0; idx < tokens.length - 1; idx++) {
        const start = Math.max(0, idx - this.config.contextLen + 1);
        const ctxLen = idx - start + 1;

        const x = new Float64Array(this.config.contextLen);
        x.fill(padId);
        const offset = this.config.contextLen - ctxLen;
        for (let j = 0; j < ctxLen; j++) {
          x[offset + j] = tokens[start + j];
        }

        const input = Matrix.fromFlat(x, [this.config.contextLen, 1]);
        const target = Matrix.fromFlat(new Float64Array([tokens[idx + 1]]), [1, 1]);
        samples.push({ input, target });
      }
    }

    return samples;
  }

  saveMetrics(outputPath: string): void {
    fs.writeFileSync(outputPath, JSON.stringify(this.metrics, null, 2));
    console.log(`📊 Metrics disimpan ke ${outputPath}`);
  }

  getMetrics(): TrainingMetrics[] {
    return [...this.metrics];
  }

  async cleanup(): Promise<void> {
    if (this.pipeline) {
      await this.pipeline.shutdown();
    }
  }

  private shuffleInPlace<T>(arr: T[]): void {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }
}

async function main() {
  console.log("🚀 Advanced Transformer Training with Pipeline\n");

  const modelCandidates = [
    path.join(__dirname, "dataset", "generative_model.json"),
    path.join(__dirname, "dataset", "finetuned_model.json"),
  ];
  const vocabCandidates = [
    path.join(__dirname, "dataset", "generative_vocab.json"),
    path.join(__dirname, "dataset", "finetuned_vocab.json"),
  ];
  const modelPath = modelCandidates.find((p) => fs.existsSync(p));
  const vocabPath = vocabCandidates.find((p) => fs.existsSync(p));

  if (!modelPath || !vocabPath) {
    console.error("❌ Model / vocab belum tersedia. Jalankan project/generative-bot/main.ts dulu.");
    process.exit(1);
  }

  const tokenizer = BPETokenizer.load(vocabPath);
  const vocabSize = tokenizer.getVocabSize();
  const contextLen = DEFAULT_CONFIG.contextLen;

  const model = new Transformers({
    units: 32,
    seqLen: contextLen,
    vocabSize,
    heads: 4,
    alpha: DEFAULT_CONFIG.learningRate,
    padTokenId: tokenizer.getPadId(),
  });

  try {
    model.load(modelPath);
  } catch {
    console.warn(`⚠️ Gagal load model dari ${modelPath}. Lanjut dari bobot baru.`);
  }
  model.compile({ alpha: DEFAULT_CONFIG.learningRate, optimizer: "adam", error: "softmaxCrossEntropy" });

  const trainer = new AdvancedTrainer(model, tokenizer, DEFAULT_CONFIG);

  try {
    const dataPath = process.env.TRAINING_DATA_PATH
      ? path.resolve(process.env.TRAINING_DATA_PATH)
      : path.join(__dirname, "dataset", "training_data.json");
    let trainingSamples = await trainer.loadTrainingData(dataPath);

    if (trainingSamples.length === 0) {
      console.log("ℹ️ training_data.json tidak ada / kosong, fallback ke dataset/cerita_rakyat.txt");
      const ceritaPath = path.join(__dirname, "..", "..", "dataset", "cerita_rakyat.txt");
      const lines = fs.readFileSync(ceritaPath, "utf-8").toLowerCase().split("\n").filter((l) => l.trim().length > 0);
      const tempDataPath = path.join(__dirname, "dataset", `_tmp_training_data_${process.pid}_${Date.now()}.json`);
      fs.writeFileSync(tempDataPath, JSON.stringify(lines.map((text) => ({ text }))));
      trainingSamples = await trainer.loadTrainingData(tempDataPath);
      if (fs.existsSync(tempDataPath)) {
        fs.unlinkSync(tempDataPath);
      }
    }

    console.log(`Total samples: ${trainingSamples.length}`);

    if (process.env.BENCHMARK_PIPELINE === "1") {
      await runPipelineBenchmark(trainingSamples, tokenizer, modelPath);
      return;
    }

    await trainer.train(trainingSamples);

    trainer.saveMetrics(path.join(__dirname, "dataset", "advanced_metrics.json"));
    model.save(modelPath);

    console.log("\n✅ Training pipeline selesai.");
  } finally {
    await trainer.cleanup();
  }
}

async function runPipelineBenchmark(
  trainingSamples: TrainingSample[],
  tokenizer: BPETokenizer,
  modelPath: string,
): Promise<void> {
  const limitFromEnv = Number(process.env.BENCHMARK_SAMPLES ?? "256");
  const benchmarkLimit = Number.isFinite(limitFromEnv) && limitFromEnv > 0 ? Math.floor(limitFromEnv) : 256;
  const baseSamples = trainingSamples.slice(0, Math.min(trainingSamples.length, benchmarkLimit));
  if (baseSamples.length === 0) {
    throw new Error("Tidak ada sample untuk benchmark.");
  }
  const benchmarkSamples: TrainingSample[] = [];
  for (let i = 0; i < benchmarkLimit; i++) {
    benchmarkSamples.push(baseSamples[i % baseSamples.length]);
  }
  const benchConfigBase: Partial<TrainingConfig> = {
    numEpochs: 1,
    microBatchSize: 32,
    learningRate: DEFAULT_CONFIG.learningRate,
    contextLen: DEFAULT_CONFIG.contextLen,
    shuffle: false,
  };

  const createModel = () => {
    const model = new Transformers({
      units: 32,
      seqLen: DEFAULT_CONFIG.contextLen,
      vocabSize: tokenizer.getVocabSize(),
      heads: 4,
      alpha: DEFAULT_CONFIG.learningRate,
      padTokenId: tokenizer.getPadId(),
    });
    try {
      model.load(modelPath);
    } catch {
      // fallback: gunakan bobot inisialisasi baru
    }
    model.compile({ alpha: DEFAULT_CONFIG.learningRate, optimizer: "adam", error: "softmaxCrossEntropy" });
    return model;
  };

  const sampleVectors = benchmarkSamples.map((sample) => Array.from(sample.input._data));

  const naiveModel = createModel();
  const naiveMatrices = sampleVectors.map((v) => Matrix.fromFlat(Float64Array.from(v), [DEFAULT_CONFIG.contextLen, 1]));

  const pipelineModel = createModel();
  const pipeline = new TransformerPipeline(
    pipelineModel,
    DEFAULT_CONFIG.numPipelineStages,
    benchConfigBase.microBatchSize ?? 32,
  );

  const naiveStart = Date.now();
  for (const sample of naiveMatrices) {
    naiveModel.forward(sample);
  }
  const naiveMs = Date.now() - naiveStart;

  const pipelineStart = Date.now();
  await pipeline.forwardMicroBatches(naiveMatrices);
  const pipelineMs = Date.now() - pipelineStart;

  const threadCount = Math.max(1, Math.min(Number(process.env.BENCHMARK_WORKERS ?? "4"), os.cpus().length));
  const multiThreadMs = await runWorkerForwardBenchmark({
    samples: sampleVectors,
    modelPath,
    vocabSize: tokenizer.getVocabSize(),
    padTokenId: tokenizer.getPadId(),
    contextLen: DEFAULT_CONFIG.contextLen,
    workerCount: threadCount,
    modelConfig: {
      units: 32,
      heads: 4,
      alpha: DEFAULT_CONFIG.learningRate,
    },
  });

  const speedup = naiveMs / Math.max(pipelineMs, 1);
  const mtSpeedup = naiveMs / Math.max(multiThreadMs, 1);
  console.log("\n=== PIPELINE BENCHMARK ===");
  console.log(`Mode    : Forward-only`);
  console.log(`Samples : ${benchmarkSamples.length}`);
  console.log(`Naive   : ${naiveMs} ms`);
  console.log(`Pipeline: ${pipelineMs} ms`);
  console.log(`Workers : ${multiThreadMs} ms (${threadCount} thread)`);
  console.log(`Speedup : ${speedup.toFixed(3)}x`);
  console.log(`MT Gain : ${mtSpeedup.toFixed(3)}x`);
  await pipeline.shutdown();
}

interface WorkerBenchmarkConfig {
  samples: number[][];
  modelPath: string;
  vocabSize: number;
  padTokenId: number;
  contextLen: number;
  workerCount: number;
  modelConfig: {
    units: number;
    heads: number;
    alpha: number;
  };
}

async function runWorkerForwardBenchmark(config: WorkerBenchmarkConfig): Promise<number> {
  const { samples, workerCount } = config;
  if (samples.length === 0 || workerCount <= 1) {
    return 0;
  }

  const chunkSize = Math.ceil(samples.length / workerCount);
  const chunks: number[][][] = [];
  for (let i = 0; i < samples.length; i += chunkSize) {
    chunks.push(samples.slice(i, i + chunkSize));
  }

  const workerPath = path.join(__dirname, "pipeline-benchmark-worker.ts");
  const workerTimes = await Promise.all(
    chunks.map(
      (chunk) =>
        new Promise<number>((resolve, reject) => {
          const worker = new Worker(workerPath, {
            workerData: { ...config, samples: chunk },
            execArgv: ["-r", "ts-node/register"],
            env: { ...process.env, ML_DISABLE_NATIVE: "1" },
          });
          worker.once("message", (msg: { elapsedMs?: number }) => resolve(msg?.elapsedMs ?? 0));
          worker.once("error", reject);
          worker.once("exit", (code) => {
            if (code !== 0) {
              reject(new Error(`Worker exit ${code}`));
            }
          });
        }),
    ),
  );
  return Math.max(...workerTimes, 0);
}

main().catch((err) => {
  console.error("Training advanced gagal:", err);
  process.exit(1);
});

export { AdvancedTrainer, TrainingConfig, TrainingMetrics, TrainingSample };
