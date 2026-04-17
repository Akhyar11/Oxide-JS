import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";
import { Transformers } from "../../src/models";
import { BPETokenizer } from "../../src/tokenizer";
import { softmax } from "../../src/activation";
import mj from "../../src/math";
import Matrix from "../../src/matrix";
import { Profiler } from "../../src/utils/profiler";

// ================================================================
// GENERATIVE CHATBOT — Latih pada cerita rakyat (Next-Word Prediction)
// ================================================================

const DEFAULT_CONTEXT_LEN = 128; 
const DEFAULT_EMBEDDING_DIM = 64;
const DEFAULT_HEADS = 8;
const LEARNING_RATE = 0.00001;
const EPOCHS = 100;
const TEMPERATURE = 0.7;
const TOKENIZER_VOCAB_SIZE = 20000;
const CHECKPOINT_EVERY = 10;
const BATCH_SIZE = 64;
const RESET_TRAINING = process.env.RESET_TRAINING === "1";
const TRAINING_MODE = process.env.TRAINING_MODE?.toLowerCase();

const botDatasetDir = path.join(__dirname, "dataset");
const generativeModelPath = path.join(botDatasetDir, "generative_model.json");
const generativeVocabPath = path.join(botDatasetDir, "generative_vocab.json");

interface TrainPair {
    xData: Float32Array;
    target: number;
}

interface ModelConfig {
    units: number;
    seqLen: number;
    heads: number;
    padTokenId: number;
    vocabSize: number;
}

type TrainingMode = "resume" | "reset";

function setDropoutLayersStatus(layers: Array<{ name: string; status?: string }>, status: "train" | "test"): void {
    for (const layer of layers) {
        if (layer.name === "dropout layer") layer.status = status;
    }
}

function askQuestion(question: string): Promise<string> {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    return new Promise(resolve => {
        rl.question(question, answer => {
            rl.close();
            resolve(answer.trim());
        });
    });
}

async function selectTrainingMode(hasCheckpoint: boolean): Promise<TrainingMode> {
    if (RESET_TRAINING || TRAINING_MODE === "reset") {
        console.log("\n=== 2. Training Mode ===\n");
        console.log("Mode dipilih dari environment: reset training dari nol.");
        return "reset";
    }

    if (TRAINING_MODE === "resume") {
        console.log("\n=== 2. Training Mode ===\n");
        if (hasCheckpoint) {
            console.log("Mode dipilih dari environment: fine-tuning / lanjut model lama.");
            return "resume";
        }
        console.log("Checkpoint belum ada, fallback ke reset training dari nol.");
        return "reset";
    }

    if (!process.stdin.isTTY) {
        return hasCheckpoint ? "resume" : "reset";
    }

    console.log("\n=== 2. Pilih Mode Training ===\n");
    console.log("1. Fine-tuning / lanjut model lama");
    console.log("2. Reset training dari nol");
    if (!hasCheckpoint) {
        console.log("Catatan: checkpoint belum ada, jadi opsi 1 akan otomatis fallback ke reset.");
    }

    while (true) {
        const answer = await askQuestion(`Pilihan (1/2, default ${hasCheckpoint ? "1" : "2"}): `);
        if (answer === "") {
            return hasCheckpoint ? "resume" : "reset";
        }
        if (answer === "1") {
            if (hasCheckpoint) return "resume";
            console.log("Checkpoint belum ada, lanjut dengan reset training dari nol.");
            return "reset";
        }
        if (answer === "2") {
            return "reset";
        }
        console.log("Pilihan tidak valid. Masukkan 1 atau 2.");
    }
}

function readModelConfig(modelPath: string): ModelConfig {
    const layers = JSON.parse(fs.readFileSync(modelPath, "utf-8"));
    const embedding = layers.find((layer: any) => layer.name === "embedding layer");
    const pe = layers.find((layer: any) => layer.name === "positional encoding");
    const mha = layers.find((layer: any) => layer.name === "multi head attention layer");

    if (!embedding) {
        throw new Error(`Cannot infer model config from ${modelPath}: embedding layer not found`);
    }

    return {
        units: embedding.embeddingDim,
        seqLen: pe?.maxSeqLen ?? mha?.seqLen ?? DEFAULT_CONTEXT_LEN,
        heads: mha?.heads ?? DEFAULT_HEADS,
        padTokenId: embedding.padTokenId ?? 0,
        vocabSize: embedding.vocabSize ?? 0,
    };
}

function formatDuration(ms: number): string {
    const totalSeconds = Math.max(0, Math.round(ms / 1000));
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;

    if (hours > 0) return `${hours}j ${minutes}m ${seconds}d`;
    if (minutes > 0) return `${minutes}m ${seconds}d`;
    return `${seconds}d`;
}

function shuffleInPlace<T>(items: T[]): void {
    for (let i = items.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [items[i], items[j]] = [items[j], items[i]];
    }
}

// 5. Training
async function main() {
    // 1. Load Dataset
    console.log("=== 1. Loading Text Dataset (cerita_rakyat.txt) ===\n");
    const dataPath = path.join(__dirname, "..", "..", "dataset", "cerita_rakyat.txt");
    const corpus = fs.readFileSync(dataPath, "utf-8").toLowerCase();
    const lines = corpus.split("\n").filter(l => l.trim().length > 0);
    console.log(`Loaded ${lines.length} lines.`);

    // 2. Pilih mode training
    fs.mkdirSync(botDatasetDir, { recursive: true });
    const hasVocab = fs.existsSync(generativeVocabPath);
    const hasModel = fs.existsSync(generativeModelPath);
    const hasCheckpoint = hasVocab && hasModel;
    
    // Jika ada vocab tapi tidak ada model, kita tetap bisa "resume" vocab-nya
    const trainingMode = await selectTrainingMode(hasCheckpoint || hasVocab);
    const shouldResume = trainingMode === "resume";

    // 3. Load / Train Tokenizer
    let tokenizer: BPETokenizer;
    let runtimeConfig: ModelConfig;
    let savedModelVocabSize = 0;

    if (shouldResume && hasVocab) {
        console.log("\n=== 3. Loading Existing Tokenizer ===\n");
        tokenizer = BPETokenizer.load(generativeVocabPath);
    } else if (hasVocab) {
        // Jika pilih Reset tapi vocab sudah ada, tetap pakai vocab yang lama agar tidak kerja dua kali
        console.log("\n=== 3. Reusing Existing Tokenizer (Resetting Model Only) ===\n");
        tokenizer = BPETokenizer.load(generativeVocabPath);
    } else {
        console.log("\n=== 3. Training New Tokenizer ===\n");
        tokenizer = new BPETokenizer({
            vocabSize: TOKENIZER_VOCAB_SIZE,
            minFrequency: 1,
            specialTokens: ["<SEP>", ...Array.from({ length: 5000 }, (_, i) => `<RESERVED_${i}>`)]
        });
        tokenizer.train(lines);
        tokenizer.save(generativeVocabPath);
        console.log(`Tokenizer saved to: ${generativeVocabPath}`);
    }

    // Common setup for both modes
    savedModelVocabSize = tokenizer.getVocabSize();
    console.log(`[BPE] Vocab saat ini: ${savedModelVocabSize}. Mencoba update...`);
    tokenizer.update(lines, TOKENIZER_VOCAB_SIZE);
    tokenizer.save(generativeVocabPath);

    if (shouldResume && hasModel) {
        console.log(`[Model] Memuat konfigurasi dari model lama...`);
        runtimeConfig = readModelConfig(generativeModelPath);
        runtimeConfig.padTokenId = tokenizer.getPadId();
    } else {
        console.log(`[Model] Menyiapkan konfigurasi model baru dari nol.`);
        runtimeConfig = {
            units: DEFAULT_EMBEDDING_DIM,
            seqLen: DEFAULT_CONTEXT_LEN,
            heads: DEFAULT_HEADS,
            padTokenId: tokenizer.getPadId(),
            vocabSize: tokenizer.getVocabSize(),
        };
    }

    const VOCAB_SIZE = tokenizer.getVocabSize();
    const PAD_ID = tokenizer.getPadId();
    runtimeConfig.vocabSize = VOCAB_SIZE;
    runtimeConfig.padTokenId = PAD_ID;
    console.log(
        `Model config: context=${runtimeConfig.seqLen}, dim=${runtimeConfig.units}, ` +
        `heads=${runtimeConfig.heads}, vocab=${runtimeConfig.vocabSize}`
    );

    // 4. Prepare Data
    const trainPairs: TrainPair[] = [];
    for (const text of lines) {
        const tokens = tokenizer.encode(text);
        for (let i = 0; i < tokens.length - 1; i++) {
            const start = Math.max(0, i - runtimeConfig.seqLen + 1);
            const ctxLen = i - start + 1;
            const xData = new Float32Array(runtimeConfig.seqLen);
            xData.fill(PAD_ID);
            const offset = runtimeConfig.seqLen - ctxLen;
            for (let j = 0; j < ctxLen; j++) {
                xData[offset + j] = tokens[start + j];
            }

            trainPairs.push({
                xData,
                target: tokens[i + 1],
            });
        }
    }
    console.log(`Training pairs: ${trainPairs.length}`);

    // 5. Build Model
    const currentVocabSize = tokenizer.getVocabSize();
    const model = new Transformers({
        units: runtimeConfig.units,
        seqLen: runtimeConfig.seqLen,
        vocabSize: currentVocabSize,
        heads: runtimeConfig.heads,
        alpha: LEARNING_RATE,
        padTokenId: PAD_ID
    });

    if (shouldResume && hasModel) {
        console.log(`\n=== 3.5 Loading Model Weights ===`);
        model.load(generativeModelPath);
        
        // Resize jika vocab bertambah
        if (currentVocabSize > model.vocabSize) {
            console.log(`Expanding model vocabulary: ${model.vocabSize} -> ${currentVocabSize}`);
            model.resizeVocab(currentVocabSize);
        }
    }
    model.summary()
    model.compile({ alpha: LEARNING_RATE, optimizer: "adam", error: "softmaxCrossEntropy" });

    function saveArtifacts(epoch?: number) {
        model.save(generativeModelPath);

        if (epoch !== undefined) {
            console.log(`Checkpoint saved at epoch ${epoch}:`);
        } else {
            console.log("Final model saved:");
        }
        console.log(`  Model: ${generativeModelPath}`);
    }

    console.log(`\n=== 4. ${shouldResume ? "Resuming" : "Starting"} Training (${EPOCHS} Epochs) ===\n`);

    // Pastikan semua layer (terutama Dropout) masuk ke mode training
    setDropoutLayersStatus(model.layers as Array<{ name: string; status?: string }>, "train");
    for (const l of model.layers) {
        // Hanya update alpha, jangan reset optimizer/loss ke default
        if ((l as any).compile) (l as any).compile({ alpha: LEARNING_RATE });
    }

    const trainingStart = performance.now();
    const epochTimes: number[] = [];
    
    // Matriks buffer untuk batch
    const batchX = mj.zeros([runtimeConfig.seqLen, BATCH_SIZE]);
    const batchY = mj.zeros([1, BATCH_SIZE]);

    let bestLoss = Infinity; // Melacak loss terbaik

    for (let ep = 0; ep < EPOCHS; ep++) {
        for (const l of model.layers) if (typeof (l as any).resetLoss === "function") (l as any).resetLoss();

        const epochStart = performance.now();
        shuffleInPlace(trainPairs);
        
        for (let i = 0; i < trainPairs.length; i += BATCH_SIZE) {
            const actualBatchSize = Math.min(BATCH_SIZE, trainPairs.length - i);
            
            // Resize buffer jika batch terakhir lebih kecil
            const currentBatchX = actualBatchSize === BATCH_SIZE ? batchX : mj.zeros([runtimeConfig.seqLen, actualBatchSize]);
            const currentBatchY = actualBatchSize === BATCH_SIZE ? batchY : mj.zeros([1, actualBatchSize]);
            
            const xData = currentBatchX._data;
            const yData = currentBatchY._data;

            for (let b = 0; b < actualBatchSize; b++) {
                const pair = trainPairs[i + b];
                // Copy xData per kolom
                for (let row = 0; row < runtimeConfig.seqLen; row++) {
                    xData[row * actualBatchSize + b] = pair.xData[row];
                }
                yData[b] = pair.target;
            }

            Profiler.start("Forward");
            model.forward(currentBatchX);
            Profiler.end("Forward");

            Profiler.start("Backward");
            model.backward(currentBatchY);
            Profiler.end("Backward");

            if ((i / BATCH_SIZE) % 10 === 0 && i > 0) {
                Profiler.report();
            }
        }

        const epochElapsed = performance.now() - epochStart;
        epochTimes.push(epochElapsed);
        const avgEpochMs = epochTimes.reduce((sum, time) => sum + time, 0) / epochTimes.length;
        const recentEpochs = epochTimes.slice(-5);
        const recentAvgMs = recentEpochs.reduce((sum, time) => sum + time, 0) / recentEpochs.length;
        const elapsedMs = performance.now() - trainingStart;
        const remainingEpochs = EPOCHS - (ep + 1);
        const etaMs = recentAvgMs * remainingEpochs;

        console.log(
            `Epoch ${ep + 1}/${EPOCHS} - Loss: ${model.loss.toFixed(6)} | ` +
            `Epoch: ${formatDuration(epochElapsed)} | ` +
            `Avg: ${formatDuration(avgEpochMs)} | ` +
            `Elapsed: ${formatDuration(elapsedMs)} | ` +
            `ETA: ${formatDuration(etaMs)}`
        );

        // Simpan hanya jika loss lebih baik (Best Model Saving)
        if (model.loss < bestLoss) {
            bestLoss = model.loss;
            saveArtifacts(ep + 1);
            console.log(`  ---> Rekor baru! Loss terbaik: ${bestLoss.toFixed(6)}`);
        }
    }
    saveArtifacts();

    // 6. Generate Function
    function generate(seed: string, maxTokens: number = 30): string {
        let tokens = tokenizer.encode(seed.toLowerCase());
        const gen: number[] = [];
        for (let i = 0; i < maxTokens; i++) {
            let win = tokens.slice(-runtimeConfig.seqLen);
            while (win.length < runtimeConfig.seqLen) win.unshift(PAD_ID);
            const logits = model.forward(mj.matrix(win.map(token => [token])));

            const lastLogits = new Float32Array(VOCAB_SIZE);
            for (let v = 0; v < VOCAB_SIZE; v++) {
                lastLogits[v] = logits._data[v] / TEMPERATURE;
            }

            const [probs] = softmax(Matrix.fromFlat(lastLogits, [VOCAB_SIZE, 1]));
            const r = Math.random();
            let cum = 0, nextToken = 0;
            for (let v = 0; v < VOCAB_SIZE; v++) {
                cum += probs._data[v];
                if (r <= cum) { nextToken = v; break; }
            }
            if (nextToken === PAD_ID) break;
            gen.push(nextToken);
            tokens.push(nextToken);
            if (tokenizer.decode([nextToken]).includes(".")) break;
        }
        return tokenizer.decode(gen);
    }

    // 7. Interactive
    console.log("\n=== 5. Chat Selesai di-Latih! ===");
    console.log("Ketik 'exit' untuk keluar.\n");

    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    const ask = () => {
        rl.question("You: ", (input) => {
            if (input.toLowerCase() === "exit") { rl.close(); return; }
            console.log(`Bot: ...${generate(input)}\n`);
            ask();
        });
    };
    ask();
}

main().catch(error => {
    console.error("Training failed:", error);
    process.exit(1);
});
