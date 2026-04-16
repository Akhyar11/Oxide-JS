import { Sequential } from "../src/models";
import { Embedding, Dense, SelfAttention, Flatten } from "../src/layers";
import mj from "../src/math";
import { Matrix } from "../src/@types/type";

/**
 * SIMULASI PENGGUNAAN MODEL (EMBEDDING -> TRANSFORMER)
 * Tujuannya: Model bisa belajar menerjemahkan string "a+b" (sekuens) ke hasil matematis (regresi).
 */

// 1. Data Preparation
const dataset = [
    { text: "1+2", result: 3 },
    { text: "2+0", result: 2 },
    { text: "3+4", result: 7 },
    { text: "0+1", result: 1 },
];

function tokenize(text: string): number[] {
    const tokens: number[] = [];
    for (const char of text) {
        if (char === '+') tokens.push(10);
        else tokens.push(parseInt(char, 10)); 
    }
    return tokens;
}

const X: Matrix[] = [];
const Y: Matrix[] = [];

for (const data of dataset) {
    const tokenIndices = tokenize(data.text);
    const xVec = mj.matrix(tokenIndices.map(t => [t]));
    X.push(xVec);
    Y.push(mj.matrix([[data.result]]));
}

// 2. Definisi Model
const vocabSize = 11;
const embeddingDim = 8;
const seqLen = 3; // "1+2" selalu 3 karakter
const attentionOutputDim = Math.floor(embeddingDim / 2); // 4
const flattenSize = attentionOutputDim * seqLen; // 4 * 3 = 12
const lrAlpha = 0.05;

console.log("Membangun model Embedding -> Transformers...");

const model = new Sequential();

// Step 1: Merubah angka index menjadi Matrix Densitas [embeddingDim, seqLen] = [8, 3]
model.add(new Embedding({ 
    vocabSize, 
    embeddingDim, 
    status: "input" 
}));

// Step 2: Self-Attention menangkap konteks antar-token → output [4, 3]
model.add(new SelfAttention({
    units: embeddingDim,
    seqLen,
    alpha: lrAlpha
}));

// Step 3: Flatten [4, 3] → [12, 1]
model.add(new Flatten());

// Step 4: Dense [12, 1] → [1, 1] (regresi)
model.add(new Dense({
    units: flattenSize,   // 12, sesuai flatten output
    outputUnits: 1,
    activation: "linear",
    status: "output",
    loss: "mse"
}));

model.compile({ alpha: lrAlpha, optimizer: "adam", error: "mse" });
model.summary();

// 3. Training Loop Process
console.log("= Mulai proses Training ! =");
model.fit(X, Y, 50, (loss) => {
    // Print setiap kali ditraining
    console.log(`Epoch loss: ${loss.toFixed(4)}`);
});

// 4. Output validation Predict
console.log("= Test Prediksi ! =");
const testData = "3+4";
const tX = mj.matrix(tokenize(testData).map(t => [t]));
const tY = model.forward(tX);
console.log(`[Demo] input '${testData}' -> Prediksi: ${tY._value[0][0].toFixed(2)} (Seharusnya 7)`);
