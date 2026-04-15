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

const NEW_LEARNING_RATE = 0.00005; // Lebih kecil dari training awal
const EPOCHS = 100;
const CONTEXT_LEN = 16;
const EMBEDDING_DIM = 16;

const modelPath = path.join(__dirname, "dataset", "finetuned_model.json");
const vocabPath = path.join(__dirname, "dataset", "finetuned_vocab.json");
const finetuneDataPath = path.join(__dirname, "dataset", "conversations.json");

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
const tokenizer = BPETokenizer.load(vocabPath);

// Scan for new words/tokens
const finetuneTexts = conversations.flatMap(c => [c.input.toLowerCase(), c.output.toLowerCase()]);
const oldVocabSize = tokenizer.getVocabSize();
tokenizer.update(finetuneTexts);
const newVocabSize = tokenizer.getVocabSize();

const PAD_ID = tokenizer.getPadId();
const SEP_ID = tokenizer.getTokenId("<SEP>")!;
const BOS_ID = tokenizer.getTokenId("<BOS>")!;
const EOS_ID = tokenizer.getTokenId("<EOS>")!;

// 3. Initialize & Load Model
console.log("\n=== 2. Loading Pre-trained Model ===");
const model = new Transformers({
    units: EMBEDDING_DIM,
    seqLen: CONTEXT_LEN,
    vocabSize: oldVocabSize, // Start with old size
    padTokenId: PAD_ID
});

if (fs.existsSync(modelPath)) {
    model.load(modelPath);
    console.log("Loaded weights from generative_model.json");

    // Resize model if vocabulary grew
    if (newVocabSize > oldVocabSize) {
        console.log(`Expanding model vocabulary: ${oldVocabSize} -> ${newVocabSize}`);
        model.resizeVocab(newVocabSize);
    }
} else {
    console.error("error: pre-trained model not found at " + modelPath);
    process.exit(1);
}

// Re-compile dengan learning rate baru (lebih rendah)
model.compile({ alpha: NEW_LEARNING_RATE, optimizer: "adam", error: "softmaxCrossEntropy" });
console.log("Model initialized with vocab size: " + newVocabSize);

// 4. Prepare Fine-tuning Data
console.log("\n=== 3. Preparing Fine-tune Data ===");
const trainPairs: { x: Matrix, y: Matrix }[] = [];

for (const conv of conversations) {
    const inTok = tokenizer.encode(conv.input.toLowerCase());
    const outTok = tokenizer.encode(conv.output.toLowerCase());
    const seq = [BOS_ID, ...inTok, SEP_ID, ...outTok, EOS_ID];
    const sepIdx = seq.indexOf(SEP_ID);

    for (let i = sepIdx; i < seq.length - 1; i++) {
        const start = Math.max(0, i - CONTEXT_LEN + 1);
        const ctx = seq.slice(start, i + 1);
        while (ctx.length < CONTEXT_LEN) ctx.unshift(PAD_ID);
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
