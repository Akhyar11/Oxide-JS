import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";
import { Transformers } from "../../src/models";
import { BPETokenizer } from "../../src/tokenizer";
import { softmax } from "../../src/activation";
import mj from "../../src/math";
import Matrix from "../../src/matrix";

// ================================================================
// INTERACTIVE REAL-TIME CHAT
// Selection: Base Model vs Fine-tuned Model
// ================================================================

const TEMPERATURE = 0.7;
const TOP_K = 20;
const REPETITION_PENALTY = 1.25;
const MIN_RESPONSE_TOKENS = 5;

const baseModelPath = path.join(__dirname, "dataset", "generative_model.json");
const finetuneModelPath = path.join(__dirname, "dataset", "finetuned_model.json");
const vocabPath = path.join(__dirname, "dataset", "generative_vocab.json");
const finetuneVocabPath = path.join(__dirname, "dataset", "finetuned_vocab.json");

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

function readModelConfig(modelPath: string) {
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
        padTokenId: embedding.padTokenId ?? 0,
        vocabSize: embedding.vocabSize ?? null,
    };
}

async function delay(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// 1. Helper for streaming printing
function printDelta(currentFullText: string, lastPrintedText: string): string {
    const delta = currentFullText.slice(lastPrintedText.length);
    process.stdout.write(delta);
    return currentFullText;
}

// 2. Sample Token (copied from main.ts for independence)
function sampleToken(logits: Matrix, temp: number, generated: number[], ids: any): number {
    const adjusted = new Float32Array(logits._data);
    const seen = new Set(generated);
    for (let i = 0; i < adjusted.length; i++) {
        if (seen.has(i)) {
            if (adjusted[i] > 0) adjusted[i] /= REPETITION_PENALTY;
            else adjusted[i] *= REPETITION_PENALTY;
        }
    }

    if (generated.length < MIN_RESPONSE_TOKENS) {
        adjusted[ids.EOS_ID] = -Infinity;
        adjusted[ids.PAD_ID] = -Infinity;
    }

    if (temp <= 0.01) {
        let maxI = 0, maxV = -Infinity;
        const [p] = softmax(Matrix.fromFlat(adjusted, [logits._shape[0], logits._shape[1]]), false);
        for (let i = 0; i < p._data.length; i++) {
            if (p._data[i] > maxV) { maxV = p._data[i]; maxI = i; }
        }
        return maxI;
    }

    const scaledData = new Float32Array(adjusted.length);
    for (let i = 0; i < adjusted.length; i++) scaledData[i] = adjusted[i] / temp;
    const [sp] = softmax(Matrix.fromFlat(scaledData, [logits._shape[0], logits._shape[1]]), false);

    const topIndices = Array.from({ length: sp._data.length }, (_, i) => i)
        .sort((a, b) => sp._data[b] - sp._data[a])
        .slice(0, Math.min(TOP_K, sp._data.length));

    let totalTopProb = 0;
    for (const idx of topIndices) totalTopProb += sp._data[idx];

    const r = Math.random();
    let cum = 0;
    for (const idx of topIndices) {
        cum += sp._data[idx] / totalTopProb;
        if (r <= cum) return idx;
    }
    return topIndices[0] ?? 0;
}

async function start() {
    console.clear();
    console.log("====================================================");
    console.log("       🤖 GENERATIVE TRANSFORMER CHATBOT 🤖         ");
    console.log("====================================================\n");

    console.log("Select model to use:");
    console.log("1. Base Pre-trained Model (generative_model.json)");
    console.log("2. Fine-tuned Model (finetuned_model.json)");
    const choice = await new Promise<string>(res => rl.question("\nChoice (1/2): ", res));
    const selectedPath = choice === "2" ? finetuneModelPath : baseModelPath;

    // 1. Load Tokenizer
    if (!fs.existsSync(vocabPath)) {
        console.error("error: Vocab file not found at " + vocabPath);
        process.exit(1);
    }
    const tokenizer = BPETokenizer.load(choice === "2" ? finetuneVocabPath : vocabPath);
    const ids = {
        VOCAB_SIZE: tokenizer.getVocabSize(),
        PAD_ID: tokenizer.getPadId(),
        SEP_ID: tokenizer.getTokenId("<SEP>")!,
        BOS_ID: tokenizer.getTokenId("<BOS>")!,
        EOS_ID: tokenizer.getTokenId("<EOS>")!
    };

    // 2. Select Model

    if (!fs.existsSync(selectedPath)) {
        console.error(`error: Model file not found at ${selectedPath}`);
        process.exit(1);
    }

    const modelConfig = readModelConfig(selectedPath);
    console.log(`\nLoading ${choice === "2" ? "Fine-tuned" : "Base"} model...`);
    const model = new Transformers({
        units: modelConfig.units,
        seqLen: modelConfig.seqLen,
        vocabSize: ids.VOCAB_SIZE,
        heads: modelConfig.heads,
        padTokenId: modelConfig.padTokenId ?? ids.PAD_ID
    });
    model.load(selectedPath);
    
    // Matikan dropout untuk respon yang lebih stabil dan kuat
    for (const l of model.layers) {
        if (l.name === "dropout layer") l.status = "test";
    }

    console.log("✅ Model loaded successfully!\n");
    console.log("Ketik 'exit' untuk keluar. Selamat mengobrol!\n");

    const chatLoop = async () => {
        rl.question("You: ", async (input) => {
            if (input.toLowerCase() === "exit" || input.toLowerCase() === "quit") {
                console.log("\nBot: Sampai jumpa lagi! 👋");
                rl.close();
                return;
            }

            if (!input.trim()) { chatLoop(); return; }

            process.stdout.write("Bot: ");

            const inTok = tokenizer.encode(input.toLowerCase());
            let ctx = [ids.BOS_ID, ...inTok, ids.SEP_ID];
            const gen: number[] = [];
            let lastText = "";

            for (let s = 0; s < 50; s++) { // Max 50 tokens
                let win = ctx.slice(-modelConfig.seqLen);
                while (win.length < modelConfig.seqLen) win.unshift(ids.PAD_ID);

                const logits = model.forward(mj.matrix(win.map(t => [t])));
                const next = sampleToken(logits, TEMPERATURE, gen, ids);

                if (next === ids.EOS_ID || next === ids.PAD_ID) break;

                gen.push(next);
                ctx.push(next);

                // Streaming effect
                const currentText = tokenizer.decode(gen);
                lastText = printDelta(currentText, lastText);

                await delay(25 + Math.random() * 50); // Simulasikan kecepatan mengetik mahluk hidup
            }

            console.log("\n");
            chatLoop();
        });
    };

    chatLoop();
}

start();
