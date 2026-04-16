import * as fs from "fs";
import * as path from "path";
import BPETokenizer from "../src/tokenizer/bpe";
import Embedding from "../src/layers/embedding";
import { Transformers } from "../src/models";
import mj from "../src/math";
import Matrix from "../src/matrix";
import * as readline from "readline";
import { setForceDisableNative } from "../src/math/rust_backend";

// setForceDisableNative(true);

// Fungsi untuk menghitung cosine similarity dari dua vektor (array of numbers)
function cosineSimilarity(vecA: number[], vecB: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Ekstrak vektor embedding untuk token spesifik
function getEmbeddingVector(embeddingLayer: Embedding, tokenId: number): number[] {
    // Forward pass untuk satu token ID
    const input = mj.matrix([[tokenId]]);
    const outMatrix = embeddingLayer.forward(input);

    // outMatrix setelah layer embedding forward akan berukuran [embeddingDim, 1]
    const vec: number[] = [];
    for (let i = 0; i < embeddingLayer.embeddingDim; i++) {
        vec.push(outMatrix._data[i]);
    }
    return vec;
}

// Jalankan test
const corpusPath = path.join(__dirname, "..", "dataset", "cerita_rakyat.txt");
const corpus = fs.readFileSync(corpusPath, "utf-8").split("\n").filter(line => line.trim().length > 0);

console.log("=== Melatih Tokenizer ===");
const vocabSize = 500;
const tokenizer = new BPETokenizer({ vocabSize, minFrequency: 2 });
tokenizer.train(corpus);
console.log(`Vocab Size Tokenizer: ${tokenizer.getVocabSize()}`);

// Siapkan data training sequences (Next-Word Prediction)
const seqLen = 6;
const X_data: Matrix[] = [];
const Y_data: Matrix[] = [];

for (const line of corpus) {
    const tokens = tokenizer.encode(line.toLowerCase());
    for (let i = 0; i < tokens.length - seqLen; i++) {
        const ctx = tokens.slice(i, i + seqLen);
        const target = tokens[i + seqLen];

        // Input: [1, seqLen] indices
        X_data.push(mj.matrix([ctx]));

        // Target: [vocabSize, seqLen] one-hot
        const yMatrix = mj.zeros([vocabSize, seqLen]);
        for (let j = 0; j < seqLen - 1; j++) {
            const nextToken = ctx[j + 1];
            yMatrix.set(nextToken, j, 1);
        }
        // Last position predicts the 'target'
        yMatrix.set(target, seqLen - 1, 1);

        Y_data.push(yMatrix);
    }
}
console.log(`Jumlah Sample Training: ${X_data.length}`);

console.log("\n=== Inisialisasi Model Transformers ===");
const embeddingDim = 64;
const model = new Transformers({
    units: embeddingDim,
    seqLen: seqLen,
    vocabSize: tokenizer.getVocabSize(),
    heads: 8,
    dropoutRate: 0.1,
    padTokenId: tokenizer.getPadId()
});

model.summary()
model.compile({ alpha: 0.0005, optimizer: "adam", error: "softmaxCrossEntropy" });

// Karena `embedding` diset private di dalam Transformers, kita gunakan trik TS untuk mengekstrak referensinya
const embedding = (model as any).embedding as Embedding;

// Test beberapa kata / sequence
const testWords = ["bawang", "merah", "putih", "timun", "mas", "buto", "ijo", "nenek", "raksasa", "sedih", "jahat"];
const testTokens: Record<string, number> = {};

console.log("\n=== Mengambil ID untuk beberapa kata kunci ===");
for (const word of testWords) {
    const ids = tokenizer.encode(word);
    if (ids.length > 0) {
        testTokens[word] = ids[0];
        console.log(`Kata "${word}" -> Token ID: ${ids[0]} ('${tokenizer.getToken(ids[0])}')`);
    }
}

const pairs = [
    ["bawang", "merah"],
    ["bawang", "putih"],
    ["timun", "mas"],
    ["buto", "ijo"],
    ["bawang", "timun"],
    ["nenek", "raksasa"],
    ["sedih", "jahat"]
];

function printSimilarity(title: string) {
    console.log(`\n=== Hubungan (Cosine Similarity) ${title} ===`);
    console.log(`[Penjelasan: 1 (Sangat mirip), 0 (Tidak mirip), -1 (Sangat berlawanan)]`);
    for (const [w1, w2] of pairs) {
        if (testTokens[w1] !== undefined && testTokens[w2] !== undefined) {
            const id1 = testTokens[w1];
            const id2 = testTokens[w2];

            const vec1 = getEmbeddingVector(embedding, id1);
            const vec2 = getEmbeddingVector(embedding, id2);

            const sim = cosineSimilarity(vec1, vec2);
            // Tambahkan padding agar rapi
            const padding = " ".repeat(18 - (w1.length + w2.length));
            console.log(`Similarity "${w1}" & "${w2}" ${padding}: ${sim.toFixed(4)}`);
        }
    }
}

// Cek sebelum di-train
printSimilarity("SEBELUM Training (Acak)");

console.log("\n=== Memulai Training Model (50 Epoch) ===");
const epochs = 50;
const start = performance.now();

model.fit(X_data, Y_data, epochs, (loss) => {
    console.log(`Epoch selesai \t- Loss: ${loss.toFixed(4)}`);
});

const end = performance.now();
console.log(`Training selesai dalam ${((end - start) / 1000).toFixed(2)} detik.`);

// Cek sesudah di-train
printSimilarity("SETELAH Training (Trained)");

console.log("\n=== Kesimpulan Test ===");
console.log("- Sebelum training, nilai cosine similarity terlihat acak karena bobot array baru saja di-generate acak.");
console.log("- Setelah training bahasa (Next-Word Prediction) selama 50 epoch, kata-kata yang sering berdampingan atau memiliki pola konteks yang sama akan mulai menggeser vektornya sehingga nilai similarity-nya berubah secara spesifik.");

// ==========================================
// SESI INTERAKTIF: CARI KATA BERHUBUNGAN
// ==========================================
function findNearestTokens(targetWord: string, topK: number = 5) {
    const targetIds = tokenizer.encode(targetWord.toLowerCase());
    if (targetIds.length === 0 || targetIds[0] === tokenizer.getTokenId("<UNK>")) {
        console.log(`\n❌ Tokenizer gagal mengenali kata "${targetWord}". Coba kata lain yang ada di dataset.`);
        return;
    }

    const targetId = targetIds[0];
    const targetVec = getEmbeddingVector(embedding, targetId);

    // Hitung similarities ke semua token di vocab
    const similarities: { token: string, similarity: number }[] = [];
    const _vocabSize = tokenizer.getVocabSize();

    for (let i = 0; i < _vocabSize; i++) {
        if (i === targetId) continue; // Jangan bandingkan dengan kata itu sendiri

        const tokenStr = tokenizer.getToken(i);
        if (!tokenStr || tokenStr.startsWith("<")) continue; // Skip token special spt <PAD>

        // Bersihkan token dari simbol byte-pair boundary ("▁")
        const cleanToken = tokenStr.replace(/▁/g, "").trim();

        // Hanya ambil token yang setidaknya berupa satu kata atau suku kata yang jelas (>1 huruf)
        if (cleanToken.length > 2) {
            const vec = getEmbeddingVector(embedding, i);
            const sim = cosineSimilarity(targetVec, vec);
            similarities.push({ token: cleanToken, similarity: sim });
        }
    }

    // Urutkan dari Similarity yang paling tinggi ke rendah (Descending)
    similarities.sort((a, b) => b.similarity - a.similarity);

    console.log(`\n🌟 5 Kata Paling Berhubungan dengan "${targetWord}":`);
    const top = similarities.slice(0, topK);

    for (let i = 0; i < top.length; i++) {
        console.log(`  ${i + 1}. ${top[i].token.padEnd(10)} (Similarity: ${top[i].similarity.toFixed(4)})`);
    }
}

// Setup input terminal interaktif
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

const promptUser = () => {
    rl.question('\nMasukkan satu kata yang ingin dicari hubungannya (atau ketik "exit" untuk keluar): ', (answer) => {
        const input = answer.trim();
        if (input.toLowerCase() === 'exit') {
            console.log("Selesai. Dadah!");
            rl.close();
        } else if (input.length > 0) {
            findNearestTokens(input);
            promptUser();
        } else {
            promptUser();
        }
    });
};

promptUser();
