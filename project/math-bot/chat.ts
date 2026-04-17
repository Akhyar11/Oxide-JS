import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";
import { Transformers } from "../../src/models";
import { BPETokenizer } from "../../src/tokenizer";
import { softmax } from "../../src/activation";
import mj from "../../src/math";
import Matrix from "../../src/matrix";
import { buildChatPrompt } from "./data";

const TEMPERATURE = 0.7;
const TOP_K = 12;
const MAX_RESPONSE_TOKENS = 48;
const REPETITION_PENALTY = 1.15;

interface ModelConfig {
  units: number;
  seqLen: number;
  heads: number;
  padTokenId: number;
}

function readModelConfig(modelPath: string): ModelConfig {
  const layers = JSON.parse(fs.readFileSync(modelPath, "utf-8")) as Array<Record<string, unknown>>;
  const embedding = layers.find((layer) => layer.name === "embedding layer") as Record<string, unknown> | undefined;
  const pe = layers.find((layer) => layer.name === "positional encoding") as Record<string, unknown> | undefined;
  const mha = layers.find((layer) => layer.name === "multi head attention layer") as Record<string, unknown> | undefined;

  if (!embedding) {
    throw new Error(`Cannot infer model config from ${modelPath}: embedding layer not found`);
  }

  return {
    units: Number(embedding.embeddingDim ?? 64),
    seqLen: Number(pe?.maxSeqLen ?? mha?.seqLen ?? 64),
    heads: Number(mha?.heads ?? 8),
    padTokenId: Number(embedding.padTokenId ?? 0),
  };
}

function sampleToken(logits: Matrix, generated: number[], padId: number): number {
  const adjusted = new Float32Array(logits._data);
  const seen = new Set(generated);

  for (let i = 0; i < adjusted.length; i++) {
    if (i === padId) {
      adjusted[i] = -Infinity;
      continue;
    }

    if (seen.has(i)) {
      if (adjusted[i] > 0) adjusted[i] /= REPETITION_PENALTY;
      else adjusted[i] *= REPETITION_PENALTY;
    }
  }

  const scaled = new Float32Array(adjusted.length);
  for (let i = 0; i < adjusted.length; i++) {
    scaled[i] = adjusted[i] / TEMPERATURE;
  }

  const [probabilities] = softmax(Matrix.fromFlat(scaled, [scaled.length, 1]), false);
  const topIndices = Array.from({ length: probabilities._data.length }, (_, i) => i)
    .sort((a, b) => probabilities._data[b] - probabilities._data[a])
    .slice(0, Math.min(TOP_K, probabilities._data.length));

  let total = 0;
  for (const idx of topIndices) {
    total += probabilities._data[idx];
  }

  const randomValue = Math.random();
  let cumulative = 0;

  for (const idx of topIndices) {
    cumulative += probabilities._data[idx] / total;
    if (randomValue <= cumulative) {
      return idx;
    }
  }

  return topIndices[0] ?? 0;
}

function generateAnswer(model: Transformers, tokenizer: BPETokenizer, modelConfig: ModelConfig, question: string): string {
  const prompt = buildChatPrompt(question);
  const promptTokenIds = tokenizer.encode(prompt);
  const generatedIds: number[] = [];
  const padId = tokenizer.getPadId();

  for (let step = 0; step < MAX_RESPONSE_TOKENS; step++) {
    const context = promptTokenIds.concat(generatedIds).slice(-modelConfig.seqLen);
    while (context.length < modelConfig.seqLen) {
      context.unshift(padId);
    }

    const logits = model.forward(mj.matrix(context.map((tokenId) => [tokenId])));
    const nextTokenId = sampleToken(logits, generatedIds, padId);

    if (nextTokenId === padId) {
      break;
    }

    generatedIds.push(nextTokenId);
    const decoded = tokenizer.decode(generatedIds).trim();
    if (decoded.endsWith(".") || decoded.endsWith("?") || decoded.endsWith("!")) {
      break;
    }
  }

  return tokenizer.decode(generatedIds).trim();
}

async function main() {
  const outputDir = path.join(__dirname, "dataset");
  const modelPath = path.join(outputDir, "math_model.json");
  const vocabPath = path.join(outputDir, "math_vocab.json");

  if (!fs.existsSync(modelPath) || !fs.existsSync(vocabPath)) {
    throw new Error("Model math-bot belum ada. Jalankan npm run train:math-bot dulu.");
  }

  const tokenizer = BPETokenizer.load(vocabPath);
  const modelConfig = readModelConfig(modelPath);
  const model = new Transformers({
    units: modelConfig.units,
    seqLen: modelConfig.seqLen,
    vocabSize: tokenizer.getVocabSize(),
    heads: modelConfig.heads,
    padTokenId: modelConfig.padTokenId ?? tokenizer.getPadId(),
  });
  model.load(modelPath);

  for (const layer of model.layers) {
    if (layer.name === "dropout layer") {
      layer.status = "test";
    }
  }

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

  console.log("=== Math Bot Chat ===");
  console.log("Ketik pertanyaan matematika. Ketik 'exit' untuk keluar.\n");

  const ask = () => {
    rl.question("You: ", (input) => {
      const trimmed = input.trim();
      if (!trimmed) {
        ask();
        return;
      }

      if (trimmed.toLowerCase() === "exit" || trimmed.toLowerCase() === "quit") {
        rl.close();
        return;
      }

      const answer = generateAnswer(model, tokenizer, modelConfig, trimmed);
      console.log(`Bot: ${answer || "(tidak ada jawaban)"}\n`);
      ask();
    });
  };

  ask();
}

main().catch((error) => {
  console.error("Math bot chat failed:", error);
  process.exit(1);
});
