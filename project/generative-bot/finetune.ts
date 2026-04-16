import * as fs from "fs";
import * as path from "path";
import { Transformers } from "../../src/models";
import { BPETokenizer } from "../../src/tokenizer";
import mj from "../../src/math";
import Matrix from "../../src/matrix";

// ================================================================
// FINE-TUNING SCRIPT
// Adapt pre-trained model to specific new knowledge/personality
// ================================================================

const NEW_LEARNING_RATE = 0.001; // Lebih kecil dari training awal
const EPOCHS = 500;

const baseModelPath = path.join(__dirname, "dataset", "generative_model.json");
const finetuneModelPath = path.join(__dirname, "dataset", "finetuned_model.json");
const baseVocabPath = path.join(__dirname, "dataset", "generative_vocab.json");
const finetuneVocabPath = path.join(__dirname, "dataset", "finetuned_vocab.json");
const finetuneDataPath = path.join(__dirname, "dataset", "conversations.json");

function readModelConfig(modelPath: string) {
    if (!fs.existsSync(modelPath)) return null;
    const layers = JSON.parse(fs.readFileSync(modelPath, "utf-8"));
    const embedding = layers.find((layer: any) => layer.name === "embedding layer");
    const pe = layers.find((layer: any) => layer.name === "positional encoding");
    const mha = layers.find((layer: any) => layer.name === "multi head attention layer");

    if (!embedding) {
        throw new Error(`Cannot infer model config from ${modelPath}: embedding layer not found`);
    }

    return {
        units: embedding.embeddingDim,
        seqLen: pe?.maxSeqLen ?? mha?.seqLen ?? 16,
        heads: mha?.heads ?? 8,
        padTokenId: embedding.padTokenId ?? null,
    };
}

// 1. Create default finetune data if not exists
if (!fs.existsSync(finetuneDataPath)) {
    const sampleFinetune = [
        { "input": "siapa penciptamu", "output": "saya diciptakan oleh akhyar untuk membantu tugas sehari-hari." },
        { "input": "apa motto hidupmu", "output": "teruslah belajar karena dunia tidak pernah berhenti mengajar." }
    ];
    fs.writeFileSync(finetuneDataPath, JSON.stringify(sampleFinetune, null, 2));
    console.log("Created sample finetune.json");
}

// 2. Load and Update Tokenizer
console.log("=== 1. Loading & Updating Tokenizer ===");
const conversations: { input: string; output: string }[] = JSON.parse(fs.readFileSync(finetuneDataPath, "utf-8"));

// Cari vocab yang ada (prioritaskan finetuned, fallback ke base)
const currentVocabPath = fs.existsSync(finetuneVocabPath) ? finetuneVocabPath : baseVocabPath;
if (!fs.existsSync(currentVocabPath)) {
    console.error("error: Base vocabulary not found. Run main.ts first!");
    process.exit(1);
}

const tokenizer = BPETokenizer.load(currentVocabPath);

// Scan for new words/tokens dari data percakapan
const finetuneTexts = conversations.flatMap(c => [c.input.toLowerCase(), c.output.toLowerCase()]);
const oldVocabSize = tokenizer.getVocabSize();
tokenizer.update(finetuneTexts);
const newVocabSize = tokenizer.getVocabSize();

const PAD_ID = tokenizer.getPadId();
const SEP_ID = tokenizer.getTokenId("<SEP>") || tokenizer.getTokenId("<UNK>")!; // Fallback safe
const BOS_ID = tokenizer.getTokenId("<BOS>")!;
const EOS_ID = tokenizer.getTokenId("<EOS>")!;

// 3. Initialize & Load Model
console.log("\n=== 2. Loading Pre-trained Model ===");
const currentModelPath = fs.existsSync(finetuneModelPath) ? finetuneModelPath : baseModelPath;

if (!fs.existsSync(currentModelPath)) {
    console.error("error: Pre-trained model not found at " + currentModelPath + ". Train it with main.ts first!");
    process.exit(1);
}

const modelConfig = readModelConfig(currentModelPath)!;
const model = new Transformers({
    units: modelConfig.units,
    seqLen: modelConfig.seqLen,
    vocabSize: oldVocabSize, // Mulai dari ukuran vocab lama
    heads: modelConfig.heads,
    padTokenId: modelConfig.padTokenId ?? PAD_ID
});

model.load(currentModelPath);
console.log(`Loaded weights from: ${path.basename(currentModelPath)}`);

// Resize model if vocabulary grew (untuk kata-kata baru di percakapan)
if (newVocabSize > oldVocabSize) {
    console.log(`Expanding model vocabulary: ${oldVocabSize} -> ${newVocabSize}`);
    model.resizeVocab(newVocabSize);
}

// Re-compile dengan learning rate baru (lebih rendah agar tidak lupa pengetahuan lama)
model.compile({ alpha: NEW_LEARNING_RATE, optimizer: "adam", error: "softmaxCrossEntropy" });
console.log("Model initialized with vocab size: " + newVocabSize);

// Pastikan mode training aktif
for (const l of model.layers) {
    if (l.name === "dropout layer") l.status = "train";
}

// 4. Prepare Fine-tuning Data
console.log("\n=== 3. Preparing Fine-tune Data ===");
const trainPairs: { x: Matrix, y: Matrix }[] = [];

for (const conv of conversations) {
    const inTok = tokenizer.encode(conv.input.toLowerCase());
    const outTok = tokenizer.encode(conv.output.toLowerCase());
    const seq = [BOS_ID, ...inTok, SEP_ID, ...outTok, EOS_ID];
    const sepIdx = seq.indexOf(SEP_ID);

    for (let i = sepIdx; i < seq.length - 1; i++) {
        const start = Math.max(0, i - modelConfig.seqLen + 1);
        const ctx = seq.slice(start, i + 1);
        while (ctx.length < modelConfig.seqLen) ctx.unshift(PAD_ID);
        const target = seq[i + 1];
        trainPairs.push({ x: mj.matrix(ctx.map(t => [t])), y: mj.matrix([[target]]) });
    }
}
console.log(`Training on ${trainPairs.length} small samples`);

// 5. Training Loop
console.log("\n=== 4. Starting Fine-tuning ===");
for (let ep = 0; ep < EPOCHS; ep++) {
    for (const l of model.layers) if ((l as any).resetLoss) (l as any).resetLoss();

    // Shuffle
    const sh = [...trainPairs].sort(() => Math.random() - 0.5);
    for (const p of sh) {
        model.forward(p.x);
        model.backward(p.y);
    }

    if (ep % 10 === 0 || ep === EPOCHS - 1) {
        console.log(`Epoch ${ep + 1}/${EPOCHS} - Fine-tune Loss: ${model.loss.toFixed(6)}`);
        const targetModelPath = path.join(__dirname, "dataset", "finetuned_model.json");
        model.save(targetModelPath);
    }
    if (model.loss < 0.0001) break;
}

// 6. Save Fine-tuned Model & Updated Vocab
const targetVocabPath = path.join(__dirname, "dataset", "finetuned_vocab.json");
tokenizer.save(targetVocabPath);
console.log("Fine-tuning complete!");
// console.log("Model saved to: " + targetModelPath);
// console.log("Vocab saved to: " + targetVocabPath);
