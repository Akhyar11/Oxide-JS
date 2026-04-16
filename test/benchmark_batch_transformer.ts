import mj from "../src/math";
import Transformers from "../src/models/transformers";
import Matrix from "../src/matrix";
import { performance } from "perf_hooks";

async function benchmark() {
    console.log("=== Benchmarking Transformer Batch Size Increase ===");

    const vocabSize = 20000;
    const units = 64;
    const seqLen = 32;

    const model = new Transformers({
        units,
        seqLen,
        vocabSize,
        heads: 8,
        alpha: 0.0001
    });
    model.compile({ alpha: 0.0001, optimizer: "adam", error: "softmaxCrossEntropy" });
    model.layers.forEach(l => {
        if ((l as any).status && (l as any).status !== "output") (l as any).status = "train";
    });

    const runTest = (batchSize: number, iterations: number) => {
        const x = mj.zeros([seqLen, batchSize]);
        const y = mj.zeros([1, batchSize]);

        // Warmup
        const out = model.forward(x);
        console.log(`Forward output shape: ${out._shape}`);
        model.backward(y);

        const start = performance.now();
        for (let i = 0; i < iterations; i++) {
            model.forward(x);
            model.backward(y);
        }
        const end = performance.now();
        const total = end - start;
        const perStep = total / iterations;
        const perSample = perStep / batchSize;
        return { total, perStep, perSample };
    };

    console.log("\nTesting Batch Size: 1 (Iterasi 50)...");
    const res1 = runTest(1, 50);
    console.log(`  Waktu per step: ${res1.perStep.toFixed(2)} ms`);
    console.log(`  Waktu per sampel: ${res1.perSample.toFixed(2)} ms`);

    console.log("\nTesting Batch Size: 128 (Iterasi 10)...");
    const res128 = runTest(128, 10);
    console.log(`  Waktu per step: ${res128.perStep.toFixed(2)} ms`);
    console.log(`  Waktu per sampel: ${res128.perSample.toFixed(2)} ms`);

    const speedup = (res1.perSample / res128.perSample - 1) * 100;
    console.log(`\n=== KESIMPULAN ===`);
    console.log(`Peningkatan Kecepatan Efektif: ${speedup.toFixed(2)}%`);
}

benchmark().catch(console.error);
