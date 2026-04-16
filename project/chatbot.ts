import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";
import { Sequential } from "../src/models";
import { Embedding, Dense, SelfAttention, Flatten, PositionalEncoding } from "../src/layers";
import { BPETokenizer } from "../src/tokenizer";
import { softmax } from "../src/activation";
import mj from "../src/math";
import Matrix from "../src/matrix";

// ================================================================
// CHATBOT SEDERHANA MENGGUNAKAN ML_V2
//
// Arsitektur:
//   Input (teks) → BPE Tokenize → Pad → 
//   Embedding → PositionalEncoding → SelfAttention → Flatten → Dense (softmax)
//   → Intent class → Response
// ================================================================

// --- Config ---
const MAX_SEQ_LEN = 15;        // Sedikit lebih besar untuk pattern yang lebih panjang
const EMBEDDING_DIM = 16;      // Proporsi wajar untuk dataset kecil
const LEARNING_RATE = 0.005;
const EPOCHS = 300;

// --- 1. Load Dataset ---
console.log("=== 1. Loading Dataset ===\n");
const datasetPath = path.join(__dirname, "..", "dataset", "chatbot_intents.json");
const intents: {
  intent: string;
  patterns: string[];
  responses: string[];
}[] = JSON.parse(fs.readFileSync(datasetPath, "utf-8"));

const intentNames = intents.map(i => i.intent);
const numClasses = intentNames.length;
console.log(`Loaded ${numClasses} intents: [${intentNames.join(", ")}]`);

// --- 2. Train BPE Tokenizer ---
console.log("\n=== 2. Training BPE Tokenizer ===\n");

const allPatterns: string[] = [];
for (const intent of intents) {
  for (const pattern of intent.patterns) {
    allPatterns.push(pattern.toLowerCase());
  }
}

const tokenizer = new BPETokenizer({ vocabSize: 200, minFrequency: 2 });
tokenizer.train(allPatterns);

const VOCAB_SIZE = tokenizer.getVocabSize();
console.log(`Vocab size: ${VOCAB_SIZE}`);

// --- 3. Prepare Training Data ---
console.log("\n=== 3. Preparing Training Data ===\n");

interface TrainingSample {
  x: Matrix;
  y: Matrix;
  label: string;
}

const samples: TrainingSample[] = [];

for (let classIdx = 0; classIdx < intents.length; classIdx++) {
  const intent = intents[classIdx];
  for (const pattern of intent.patterns) {
    const tokens = tokenizer.encode(pattern.toLowerCase());
    const padded = tokenizer.padSequence(tokens, MAX_SEQ_LEN);

    // Input: [MAX_SEQ_LEN, 1]
    const xVec = mj.matrix(padded.map(t => [t]));

    // Target: One-hot [numClasses, 1]
    const yVec: number[][] = [];
    for (let i = 0; i < numClasses; i++) {
      yVec.push([i === classIdx ? 1 : 0]);
    }

    samples.push({
      x: xVec,
      y: mj.matrix(yVec),
      label: intent.intent,
    });
  }
}

console.log(`Training samples: ${samples.length}`);
console.log(`Input shape: [${MAX_SEQ_LEN}, 1]`);
console.log(`Output shape: [${numClasses}, 1]`);

// Hitung jumlah per intent
for (const intent of intents) {
  console.log(`  ${intent.intent}: ${intent.patterns.length} patterns`);
}

// --- 4. Build Model ---
console.log("\n=== 4. Building Model ===\n");

const attentionOutputDim = Math.floor(EMBEDDING_DIM / 2); // 8
const flattenSize = attentionOutputDim * MAX_SEQ_LEN;      // 8 * 12 = 96

const model = new Sequential();

model.add(new Embedding({
  vocabSize: VOCAB_SIZE,
  embeddingDim: EMBEDDING_DIM,
  status: "input",
}));

model.add(new PositionalEncoding({
  dModel: EMBEDDING_DIM,
  maxSeqLen: MAX_SEQ_LEN,
}));

model.add(new SelfAttention({
  units: EMBEDDING_DIM,
  seqLen: MAX_SEQ_LEN,
  alpha: LEARNING_RATE,
}));

model.add(new Flatten());

// Layer 5: Dense output - linear activation + softmaxCrossEntropy (combined)
// Softmax diterapkan INSIDE loss function, bukan sebagai activation terpisah
// Ini menghasilkan gradient yang benar: (ŷ - y) / N
model.add(new Dense({
  units: flattenSize,
  outputUnits: numClasses,
  activation: "linear",
  status: "output",
  loss: "softmaxCrossEntropy",
}));

model.compile({ alpha: LEARNING_RATE, optimizer: "adam", error: "softmaxCrossEntropy" });
model.summary();

// --- 5. Training dengan SHUFFLE ---
console.log("\n=== 5. Training ===\n");

// Fisher-Yates shuffle
function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

for (let epoch = 0; epoch < EPOCHS; epoch++) {
  // Reset loss di awal epoch
  for (const layer of model.layers) {
    if (typeof (layer as any).resetLoss === "function") {
      (layer as any).resetLoss();
    }
  }

  // Shuffle data setiap epoch agar model tidak bias
  const shuffled = shuffle(samples);

  for (const sample of shuffled) {
    model.forward(sample.x);
    model.backward(sample.y);
  }

  if (epoch % 20 === 0 || epoch === EPOCHS - 1) {
    console.log(`Epoch ${epoch + 1}/${EPOCHS} - Loss: ${model.loss.toFixed(6)}`);
  }

  if (model.loss < 0.001) {
    console.log(`\nEarly stop at epoch ${epoch + 1}, loss: ${model.loss.toFixed(6)}`);
    break;
  }
}

console.log("\n✅ Training selesai!");

// --- 6. Math Evaluator ---
/**
 * Mengevaluasi ekspresi matematika dari teks bahasa Indonesia
 * Contoh: "berapa 5 tambah 10" → 15
 */
function evaluateMath(text: string): string | null {
  const lower = text.toLowerCase();

  // Coba parse pola: [angka] [operator] [angka]
  // Operator yang didukung: tambah/ditambah/+, kurang/dikurangi/-, kali/dikali/x/*, bagi/dibagi//
  const patterns = [
    // tambah
    { regex: /(\d+(?:\.\d+)?)\s*(?:tambah|ditambah|\+)\s*(\d+(?:\.\d+)?)/, op: '+' },
    // kurang
    { regex: /(\d+(?:\.\d+)?)\s*(?:kurang|dikurangi|\-)\s*(\d+(?:\.\d+)?)/, op: '-' },
    // kali
    { regex: /(\d+(?:\.\d+)?)\s*(?:kali|dikali|\*|x)\s*(\d+(?:\.\d+)?)/, op: '*' },
    // bagi
    { regex: /(\d+(?:\.\d+)?)\s*(?:dibagi|bagi|\/)\s*(\d+(?:\.\d+)?)/, op: '/' },
    // pangkat
    { regex: /(\d+(?:\.\d+)?)\s*(?:pangkat|dipangkatkan)\s*(\d+(?:\.\d+)?)/, op: '**' },
    // modulo
    { regex: /(\d+(?:\.\d+)?)\s*(?:modulo|mod)\s*(\d+(?:\.\d+)?)/, op: '%' },
  ];

  for (const { regex, op } of patterns) {
    const match = lower.match(regex);
    if (match) {
      const a = parseFloat(match[1]);
      const b = parseFloat(match[2]);
      let result: number;

      switch (op) {
        case '+': result = a + b; break;
        case '-': result = a - b; break;
        case '*': result = a * b; break;
        case '/':
          if (b === 0) return `${a} ÷ ${b} = tidak bisa dibagi nol!`;
          result = a / b; break;
        case '**': result = Math.pow(a, b); break;
        case '%': result = a % b; break;
        default: return null;
      }

      const opSymbol: Record<string, string> = {
        '+': '+', '-': '-', '*': '×', '/': '÷', '**': '^', '%': 'mod'
      };

      // Format hasil (hilangkan desimal jika bulat)
      const formatted = Number.isInteger(result) ? result.toString() : result.toFixed(4);
      return `${a} ${opSymbol[op]} ${b} = ${formatted}`;
    }
  }

  // Akar kuadrat
  const sqrtMatch = lower.match(/akar\s*(?:dari\s*)?(\d+(?:\.\d+)?)/);
  if (sqrtMatch) {
    const n = parseFloat(sqrtMatch[1]);
    const result = Math.sqrt(n);
    const formatted = Number.isInteger(result) ? result.toString() : result.toFixed(4);
    return `√${n} = ${formatted}`;
  }

  return null;
}

// --- 7. Test Predictions ---
console.log("\n=== 6. Testing ===\n");

function predict(text: string): { intent: string; confidence: number; response: string } {
  const tokens = tokenizer.encode(text.toLowerCase());
  const padded = tokenizer.padSequence(tokens, MAX_SEQ_LEN);
  const input = mj.matrix(padded.map(t => [t]));

  const logits = model.forward(input);
  const [probs] = softmax(logits, false);

  // Argmax
  let maxIdx = 0;
  let maxVal = -Infinity;
  for (let i = 0; i < probs._shape[0]; i++) {
    if (probs._value[i][0] > maxVal) {
      maxVal = probs._value[i][0];
      maxIdx = i;
    }
  }

  const intent = intents[maxIdx];
  let response: string;

  // Jika intent = matematika → evaluasi ekspresi secara dinamis
  if (intent.intent === "matematika") {
    const mathResult = evaluateMath(text);
    if (mathResult) {
      response = `🧮 ${mathResult}`;
    } else {
      response = "Hmm, saya mendeteksi pertanyaan matematika tapi belum bisa memahami formatnya. Coba format: 'berapa 5 tambah 3'";
    }
  } else {
    response = intent.responses[Math.floor(Math.random() * intent.responses.length)];
  }

  return { intent: intent.intent, confidence: maxVal, response };
}

// Test: data dari training
const testInputs = [
  "halo",
  "siapa nama kamu",
  "kamu bisa apa",
  "terima kasih",
  "bye",
  "bagaimana cuaca",
  "jam berapa",
  "siapa pembuat kamu",
  "ceritakan lelucon",
  "tolong bantu saya",
  "saya senang",
  "saya sedih",
  "berapa 5 tambah 10",
  "1 km berapa meter",
  "apa itu bilangan prima",
];

let correct = 0;
const expectedIntents = [
  "salam", "perkenalan", "kemampuan", "terima_kasih", "sampai_jumpa",
  "cuaca", "waktu", "pembuat", "lelucon", "bantuan", "perasaan_baik", "perasaan_buruk",
  "matematika", "konversi", "pengetahuan_math"
];

for (let i = 0; i < testInputs.length; i++) {
  const result = predict(testInputs[i]);
  const isCorrect = result.intent === expectedIntents[i];
  if (isCorrect) correct++;
  console.log(`${isCorrect ? "✅" : "❌"} "${testInputs[i]}" → [${result.intent}] (${(result.confidence * 100).toFixed(1)}%)`);
}
console.log(`\nAkurasi: ${correct}/${testInputs.length} (${(correct / testInputs.length * 100).toFixed(0)}%)`);

// Test: kalimat baru yang TIDAK ada di training
console.log("\n--- Test kalimat baru ---\n");
const novelInputs = [
  "hello",
  "nama kamu apa",
  "makasih banyak",
  "sampai ketemu lagi",
  "lagi happy nih",
  "siapa yang bikin kamu",
  "ada jokes tidak",
  "berapa 99 kali 3",
  "hitung 1000 kurang 250",
  "berapa 2 pangkat 10",
  "berapa akar dari 144",
  "berapa 17 modulo 5",
  "rumus luas lingkaran",
  "1 jam berapa menit",
];

for (const input of novelInputs) {
  const result = predict(input);
  console.log(`"${input}" → [${result.intent}] (${(result.confidence * 100).toFixed(1)}%)`);
  console.log(`  Bot: ${result.response}\n`);
}

// --- 7. Save ---
console.log("=== 7. Saving Model ===\n");
const modelDir = path.join(__dirname, "..", "dataset");
model.save(path.join(modelDir, "chatbot_model.json"));
tokenizer.save(path.join(modelDir, "chatbot_vocab.json"));
console.log("Model dan vocabulary tersimpan!");

// --- 8. Interactive Chat ---
console.log("\n=== 8. Interactive Chat ===");
console.log("Ketik pesan untuk mengobrol (ketik 'quit' untuk keluar)\n");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function chat() {
  rl.question("You: ", (userInput) => {
    if (userInput.toLowerCase() === "quit" || userInput.toLowerCase() === "exit") {
      console.log("Bot: Sampai jumpa! 👋");
      rl.close();
      return;
    }

    if (userInput.trim() === "") {
      chat();
      return;
    }

    const result = predict(userInput);
    console.log(`Bot: ${result.response}`);
    console.log(`     [${result.intent} | ${(result.confidence * 100).toFixed(1)}%]\n`);
    chat();
  });
}

chat();
