import { Transformers } from "../src/models";
import mj from "../src/math";
import Matrix from "../src/matrix";
import { setForceDisableNative, isNativeAvailable } from "../src/math/rust_backend";

// Konfigurasi Model & Data untuk Benchmark
const CONTEXT_LEN = 16;
const EMBEDDING_DIM = 16; // Naikkan dimensi untuk melihat perbedaan performa yang signifikan
const VOCAB_SIZE = 541;
const SAMPLES = 28893;     // Jumlah sample dalam satu epoch

function createDummyData() {
    const data: { x: Matrix, y: Matrix }[] = [];
    for (let i = 0; i < SAMPLES; i++) {
        const ctx = Array.from({ length: CONTEXT_LEN }, () => Math.floor(Math.random() * VOCAB_SIZE));
        const target = Math.floor(Math.random() * VOCAB_SIZE);
        data.push({
            x: mj.matrix(ctx.map(t => [t])),
            y: mj.matrix([[target]])
        });
    }
    return data;
}

async function runBenchmark() {
    console.log("=== EPOCH PERFORMANCE COMPARISON ===");
    console.log(`Model: Transformers (dim=${EMBEDDING_DIM}, ctx=${CONTEXT_LEN}, vocab=${VOCAB_SIZE})`);
    console.log(`Data: ${SAMPLES} samples\n`);

    const trainData = createDummyData();

    const model = new Transformers({
        units: EMBEDDING_DIM,
        seqLen: CONTEXT_LEN,
        vocabSize: VOCAB_SIZE,
        padTokenId: 0
    });
    model.summary()
    model.compile({ alpha: 0.001, optimizer: "adam", error: "softmaxCrossEntropy" });

    // --- 1. RUST BACKEND ---
    setForceDisableNative(false);
    if (!isNativeAvailable()) {
        console.warn("❌ Rust backend tidak tersedia! Pastikan ml-native.node sudah di-build.");
        process.exit(1);
    }

    console.log("🚀 Running with Rust Backend...");
    // Warmup (agar JIT compiler Node.js siap)
    for (let i = 0; i < 5; i++) {
        model.forward(trainData[i].x);
        model.backward(trainData[i].y);
    }

    const startRust = process.hrtime.bigint();
    for (const p of trainData) {
        model.forward(p.x);
        model.backward(p.y);
    }
    const endRust = process.hrtime.bigint();
    const rustTime = Number(endRust - startRust) / 1e6;
    console.log(`✅ Rust Epoch Time: ${rustTime.toFixed(2)} ms`);


    // --- 2. NATIVE NODE.JS BACKEND ---
    console.log("\n🐌 Running with Native Node.js Backend...");
    setForceDisableNative(true); // Memaksa menggunakan implementasi JS

    // Warmup
    for (let i = 0; i < 5; i++) {
        model.forward(trainData[i].x);
        model.backward(trainData[i].y);
    }

    const startJS = process.hrtime.bigint();
    for (const p of trainData) {
        model.forward(p.x);
        model.backward(p.y);
    }
    const endJS = process.hrtime.bigint();
    const jsTime = Number(endJS - startJS) / 1e6;
    console.log(`✅ Node.js Epoch Time: ${jsTime.toFixed(2)} ms`);

    // --- HASIL PERBANDINGAN ---
    const speedup = jsTime / rustTime;
    console.log("\n" + "=".repeat(30));
    console.log("FINAL RESULT:");
    console.log(`Rust    : ${rustTime.toFixed(2)} ms`);
    console.log(`Node.js : ${jsTime.toFixed(2)} ms`);
    console.log(`Speedup : ${speedup.toFixed(2)}x ${speedup > 1 ? "lebih cepat (Rust menang!)" : "lebih lambat"}`);
    console.log("=".repeat(30));
}

runBenchmark().catch(console.error);
