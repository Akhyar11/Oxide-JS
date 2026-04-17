import Matrix from "../src/matrix";
import SelfAttention from "../src/layers/selfAttention";
import LayerNormalization from "../src/layers/layerNormalization";
import { isNativeAvailable } from "../src/math/rust_backend";

const SEQ_LEN = 128;
const EMBED_DIM = 256;
const ITERS = 10;

function makeMatrix(r: number, c: number) {
    const data = new Float32Array(r * c);
    for (let i = 0; i < data.length; i++) data[i] = Math.random();
    return Matrix.fromFlat(data, [r, c]);
}

async function runBenchmark() {
    console.log("=== Transformer Block Benchmark ===");
    console.log(`Native Backend Available: ${isNativeAvailable()}`);
    console.log(`Config: SeqLen=${SEQ_LEN}, EmbedDim=${EMBED_DIM}`);

    const input = makeMatrix(EMBED_DIM, SEQ_LEN);
    
    // 1. Self Attention Benchmark
    const attention = new SelfAttention({ units: EMBED_DIM, seqLen: SEQ_LEN });
    
    console.log("\n--- Self Attention Forward ---");
    // Warmup
    for(let i=0; i<3; i++) attention.forward(input);
    
    const startAttn = performance.now();
    for(let i=0; i<ITERS; i++) {
        attention.forward(input);
    }
    const endAttn = performance.now();
    console.log(`Avg Time: ${(endAttn - startAttn) / ITERS} ms`);

    // 2. Layer Normalization Benchmark
    const ln = new LayerNormalization({ units: EMBED_DIM });
    
    console.log("\n--- Layer Normalization Forward ---");
    // Warmup
    for(let i=0; i<5; i++) ln.forward(input);
    
    const startLn = performance.now();
    for(let i=0; i<ITERS * 10; i++) {
        ln.forward(input);
    }
    const endLn = performance.now();
    console.log(`Avg Time: ${(endLn - startLn) / (ITERS * 10)} ms`);

    if (!isNativeAvailable()) {
        console.log("\n[!] PERINGATAN: Native backend TIDAK aktif. Hasil di atas adalah performa JS murni.");
        console.log("Silakan jalankan 'npm run build:rust' untuk melihat peningkatan performa.");
    }
}

runBenchmark().catch(console.error);
