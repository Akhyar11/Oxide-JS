import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import { dirname } from "path";
import { BPETokenizer, Matrix, mj } from "@oxide-js/core";
import { Dense, Embedding, GRU, LSTM, RNN } from "@oxide-js/layers";
import { Sequential } from "@oxide-js/models";;
import { getDatasetTrain, getDatasetValid } from "./get_dataset.js";
import { runEvaluation } from "./f1score.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// --- Logging Setup ---
const logDir = path.join(__dirname, "../log");
if (!fs.existsSync(logDir)) {
    fs.mkdirSync(logDir, { recursive: true });
}

const now = new Date();
const timestamp =
    now.getFullYear().toString() +
    (now.getMonth() + 1).toString().padStart(2, "0") +
    now.getDate().toString().padStart(2, "0") + "_" +
    now.getHours().toString().padStart(2, "0") +
    now.getMinutes().toString().padStart(2, "0") +
    now.getSeconds().toString().padStart(2, "0");

const logFilePath = path.join(logDir, `log_${timestamp}.txt`);
const logStream = fs.createWriteStream(logFilePath, { flags: "a" });

const originalStdoutWrite = process.stdout.write.bind(process.stdout);
const originalStderrWrite = process.stderr.write.bind(process.stderr);

// @ts-ignore
process.stdout.write = (chunk: any, encoding: any, callback: any) => {
    // Avoid writing frequent progress bar \r updates to the file stream to prevent Node queue memory leaks
    if (typeof chunk === "string" && chunk.startsWith("\r")) {
        return originalStdoutWrite(chunk, encoding, callback);
    }
    logStream.write(chunk, encoding);
    return originalStdoutWrite(chunk, encoding, callback);
};

// @ts-ignore
process.stderr.write = (chunk: any, encoding: any, callback: any) => {
    if (typeof chunk === "string" && chunk.startsWith("\r")) {
        return originalStderrWrite(chunk, encoding, callback);
    }
    logStream.write(chunk, encoding);
    return originalStderrWrite(chunk, encoding, callback);
};

console.log(`Log file created at: ${logFilePath}`);

type Sample = {
    x: Matrix;
    y: Matrix;
};

const MAX_SEQ_LEN = 128;
const BATCHSIZE = 16
const EMBEDDING_DIM = 128;
const HIDDEN_UNITS = 32;
const OUTPUT_CLASSES = 3;
const EPOCHS = 30;
const ALPHA = 0.001;

// Validation control
const VAL_SAMPLES_PER_EPOCH = 500; // ubah ke 1000 jika RAM aman
const FULL_VAL_EVERY = 5;          // full validation tiap 5 epoch
const ENABLE_GC_AFTER_VAL = true;

function forceGC() {
    // Jalankan node dengan: node --expose-gc ...
    // @ts-ignore
    if (ENABLE_GC_AFTER_VAL && global.gc) {
        // @ts-ignore
        global.gc();
    }
}

function logMemory(prefix: string) {
    const mem = process.memoryUsage();
    console.log(
        `${prefix} | rss=${(mem.rss / 1024 / 1024).toFixed(1)}MB ` +
        `heapUsed=${(mem.heapUsed / 1024 / 1024).toFixed(1)}MB ` +
        `external=${(mem.external / 1024 / 1024).toFixed(1)}MB`
    );
}

function argmax(matrix: Matrix): number {
    let maxIndex = 0;
    let maxValue = matrix._data[0]!;

    for (let i = 1; i < matrix._data.length; i++) {
        if (matrix._data[i]! > maxValue) {
            maxValue = matrix._data[i]!;
            maxIndex = i;
        }
    }

    return maxIndex;
}

function shuffleArray<T>(array: T[]): T[] {
    const copied = [...array];

    for (let i = copied.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [copied[i], copied[j]] = [copied[j]!, copied[i]!];
    }

    return copied;
}

function sampleValidationIndices(total: number, limit: number, epoch: number): number[] {
    if (total <= 0) return [];

    if (limit >= total) {
        return Array.from({ length: total }, (_, i) => i);
    }

    const indices: number[] = [];
    const start = (epoch * limit) % total;

    for (let i = 0; i < limit; i++) {
        indices.push((start + i) % total);
    }

    return indices;
}

function validateModel(
    model: Sequential,
    valX: Matrix[],
    valY: Matrix[],
    epoch: number,
    full: boolean
): number {
    const total = valX.length;
    if (total === 0) return 0;

    const limit = full ? total : Math.min(VAL_SAMPLES_PER_EPOCH, total);
    const indices = sampleValidationIndices(total, limit, epoch);

    let correct = 0;

    model.eval();

    for (const idx of indices) {
        // Penting: pakai forward(), bukan predict().
        // predict() akan toggle eval/train di tiap sample.
        const logits = model.forward(valX[idx]!);
        const pred = argmax(logits);
        const target = valY[idx]!._data[0]!;

        if (pred === target) correct++;
    }

    model.train();
    forceGC();

    return correct / indices.length;
}

// 1. Data Preparation
const tokenizer = BPETokenizer.load(
    path.join(__dirname, "../tokenizer.json")
);

const VOCAB_SIZE = tokenizer.getVocabSize ? tokenizer.getVocabSize() : 2685;
const PAD_ID = tokenizer.getPadId ? tokenizer.getPadId() : 0;

console.log("--- Dataset Information ---");
console.log("VOCAB_SIZE:", VOCAB_SIZE);
console.log("PAD_ID:", PAD_ID);

let rawDatasetTrain: any[] | null = getDatasetTrain();
const rawDatasetValid: any[] = getDatasetValid();

const trainSize = rawDatasetTrain.length;
const validSize = rawDatasetValid.length;

console.log("Raw Dataset Train size:", trainSize);
console.log("Raw Dataset Valid size:", validSize);
console.log("NOTE: Tokenizer is NOT updated during testing/evaluation.");

const samples: Sample[] = rawDatasetTrain
    .map((item) => {
        const text = String(item.text ?? "").trim();
        const label = String(item.label ?? "").trim();

        if (!text) return null;
        if (label !== "positive" && label !== "negative" && label !== "neutral") return null;

        const tokenIds = tokenizer.encode(text);
        if (tokenIds.length === 0) return null;

        // Penyeragaman panjang sequence dengan LEFT-PADDING
        // Agar hidden state terakhir RNN merepresentasikan kata asli, bukan padding.

        const slicedTokens = tokenIds.slice(0, MAX_SEQ_LEN);
        const paddedTokenIds = new Array(MAX_SEQ_LEN - slicedTokens.length).fill(PAD_ID).concat(slicedTokens);

        const x = mj.matrix(paddedTokenIds.map((id: number) => [id]));
        const classIndex = label === "positive" ? 1 : label === "negative" ? 0 : 2;
        const y = mj.matrix([[classIndex]]);

        return { x, y };
    })
    .filter((item): item is Sample => item !== null);

rawDatasetTrain = null;
forceGC();

const shuffled = shuffleArray(samples);
const validationRatio = 0.2;
const validationSize = Math.floor(shuffled.length * validationRatio);

const validationSamples = shuffled.slice(0, validationSize);
const trainSamples = shuffled.slice(validationSize);

const trainX = trainSamples.map((s) => s.x);
const trainY = trainSamples.map((s) => s.y);
const valX = validationSamples.map((s) => s.x);
const valY = validationSamples.map((s) => s.y);

console.log("Processed Train samples:", trainX.length);
console.log("Processed Inner-Validation samples:", valX.length);
console.log("---------------------------\n");

logMemory("After dataset preparation");

// 2. Training Loop for 3 Models
const modelTypes = ["RNN", "LSTM", "GRU"] as const;
type ModelType = (typeof modelTypes)[number];

const STABILITY_REPS = 10;

type RunResultByRep = {
    run: number;
    finalLoss: number;
    lossHistory: number[];
    bestValAcc: number;
    f1: number;
};

type StabilityResult = {
    params: number;
    runs: RunResultByRep[];
    avgLoss: number;
    avgBestValAcc: number;
    avgF1: number;
};

const results: Record<ModelType, StabilityResult> = {
    RNN: { params: 0, runs: [], avgLoss: 0, avgBestValAcc: 0, avgF1: 0 },
    LSTM: { params: 0, runs: [], avgLoss: 0, avgBestValAcc: 0, avgF1: 0 },
    GRU: { params: 0, runs: [], avgLoss: 0, avgBestValAcc: 0, avgF1: 0 },
};

for (const type of modelTypes) {
    console.log(`\n\n===========================================================`);
    console.log(`>>> STARTING STABILITY EXPERIMENT FOR MODEL: ${type} <<<`);
    console.log(`===========================================================\n`);

    for (let rep = 1; rep <= STABILITY_REPS; rep++) {
        console.log(`\n--- [${type}] Run ${rep}/${STABILITY_REPS} ---`);

        const embedding = new Embedding({
            vocabSize: VOCAB_SIZE,
            embeddingDim: EMBEDDING_DIM,
            padTokenId: PAD_ID,
            alpha: ALPHA
        });

        let recurrentLayer: any;

        const commonConfig = {
            units: EMBEDDING_DIM,
            hiddenUnits: HIDDEN_UNITS,
            alpha: ALPHA,
            returnSequences: false,
            stateful: false
        };

        if (type === "RNN") {
            recurrentLayer = new RNN({ ...commonConfig, activation: "tanh" });
        } else if (type === "LSTM") {
            recurrentLayer = new LSTM(commonConfig);
        } else {
            recurrentLayer = new GRU(commonConfig);
        }

        const dense = new Dense({
            units: HIDDEN_UNITS,
            outputUnits: OUTPUT_CLASSES,
            activation: "linear",
            status: "output",
            loss: "softmaxCrossEntropy",
            alpha: ALPHA
        });

        const model = new Sequential({
            layers: [embedding, recurrentLayer, dense]
        });

        model.compile({
            alpha: ALPHA,
            error: "softmaxCrossEntropy",
            optimizer: "adam",
            clipGradient: false
        });

        if (rep === 1) {
            let totalParams = 0;
            model.layers.forEach((layer: any) => {
                if (layer.params) totalParams += layer.params;
            });
            results[type].params = totalParams;
            console.log(`Model ${type} Total Parameters: ${totalParams}`);
            model.summary();
        }

        let bestValAcc = 0;
        let lastLoss = 0;
        const currentLossHistory: number[] = [];

        model.fit(trainX, trainY, EPOCHS, {
            batchSize: BATCHSIZE,
            shuffle: true,
            verbose: true,

            onEpochEnd: (epoch, loss) => {
                const isLastEpoch = epoch === EPOCHS - 1;
                const shouldFullValidate = isLastEpoch || (epoch + 1) % FULL_VAL_EVERY === 0;

                const valAcc = validateModel(
                    model,
                    valX,
                    valY,
                    epoch,
                    shouldFullValidate
                );

                if (valAcc > bestValAcc) bestValAcc = valAcc;
                if (isLastEpoch) lastLoss = loss;
                currentLossHistory.push(loss);

                const valMode = shouldFullValidate
                    ? `full=${valX.length}`
                    : `sample=${Math.min(VAL_SAMPLES_PER_EPOCH, valX.length)}`;

                console.log(
                    `[${type} R${rep}] Epoch ${epoch + 1}/${EPOCHS} | ` +
                    `Loss: ${loss.toFixed(4)} | ` +
                    `Val Acc (${valMode}): ${(valAcc * 100).toFixed(2)}% | ` +
                    `Best: ${(bestValAcc * 100).toFixed(2)}%`
                );

                logMemory(`After ${type} R${rep} epoch ${epoch + 1}`);
            }
        });

        console.log(`Evaluating F1 Score for ${type} Run ${rep}...`);
        model.eval();
        const evalResult = runEvaluation(model, tokenizer, rawDatasetValid);
        const f1 = evalResult.weightedF1;
        model.train();

        results[type].runs.push({
            run: rep,
            finalLoss: lastLoss,
            lossHistory: currentLossHistory,
            bestValAcc: bestValAcc,
            f1: f1
        });

        // Always save the model of each run with a unique name
        const savePath = path.join(logDir, `model_${type.toLowerCase()}_run${rep}_${timestamp}.json`);
        model.save(savePath);

        model.dispose();
        forceGC();
        logMemory(`After ${type} Run ${rep} cleanup`);
    }

    // Calculate averages
    const n = results[type].runs.length;
    results[type].avgLoss = results[type].runs.reduce((s, r) => s + r.finalLoss, 0) / n;
    results[type].avgBestValAcc = results[type].runs.reduce((s, r) => s + r.bestValAcc, 0) / n;
    results[type].avgF1 = results[type].runs.reduce((s, r) => s + r.f1, 0) / n;
}

// 3. Save Experiment Metadata
const metadataPath = path.join(logDir, `stability_experiment_${timestamp}.json`);
fs.writeFileSync(metadataPath, JSON.stringify({
    timestamp,
    config: {
        epochs: EPOCHS,
        alpha: ALPHA,
        embeddingDim: EMBEDDING_DIM,
        hiddenUnits: HIDDEN_UNITS,
        repetitions: STABILITY_REPS
    },
    results
}, null, 2));

console.log(`\nStability experiment metadata saved to: ${metadataPath}`);

// 4. Final Summary
console.log("\n" + "=".repeat(60));
console.log("FINAL MULTI-MODEL REPORT");
console.log("=".repeat(60));
console.log(`Dataset Size (Train):      ${trainSize}`);
console.log(`Dataset Size (Valid):      ${validSize}`);
console.log(`Tokenizer Status:          Fixed (Not updated during test)`);
console.log("-".repeat(60));
console.log("Model Type | Avg Loss | Avg Val Acc | Avg Weighted F1 | Params");
console.log("-".repeat(75));

for (const type of modelTypes) {
    const res = results[type];
    const loss = res.avgLoss.toFixed(4).padEnd(10);
    const acc = (res.avgBestValAcc * 100).toFixed(2).padStart(10) + "%";
    const f1 = (res.avgF1 * 100).toFixed(2).padStart(14) + "%";
    const params = String(res.params).padStart(10);

    console.log(`${type.padEnd(10)} | ${loss} | ${acc} | ${f1} | ${params}`);
}

console.log("=".repeat(75));

logStream.end();