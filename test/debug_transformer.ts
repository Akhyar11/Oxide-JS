import mj from "../src/math";
import Matrix from "../src/matrix";
import { Transformers } from "../src/models";
import Embedding from "../src/layers/embedding";
import PositionalEncoding from "../src/layers/positionalEncoding";
import LayerNormalization from "../src/layers/layerNormalization";
import MultiHeadAttention from "../src/layers/multiHeadAttention";
import Dropout from "../src/layers/dropout";
import Dense from "../src/layers/dense";

async function debugModel() {
    console.log("=== Debugging Transformer forward/backward ===");
    
    const units = 64;
    const seqLen = 6;
    const vocabSize = 500;
    const heads = 8;
    
    const model = new Transformers({
        units,
        seqLen,
        vocabSize,
        heads,
        dropoutRate: 0.1
    });

    model.compile({ alpha: 0.001, optimizer: "adam", error: "softmaxCrossEntropy" });

    // Mock data: 1 sample
    const x = mj.matrix([[0, 1, 2, 3, 4, 5]]); // [seqLen=6, inputs]
    // Note: Embedding.forward expects a matrix containing token indices
    
    const y = mj.zeros([vocabSize, seqLen]); // [500, 6]
    y.set(10, 5, 1); // target at last position

    console.log("\n--- START FORWARD PASS ---");
    
    // Step by step forward
    // 1. Embedding
    const embedding = (model as any).embedding as Embedding;
    const xEmb = embedding.forward(x);
    printStats("Embedding", xEmb);

    // 2. PE
    const pe = (model as any).pe as PositionalEncoding;
    const xPe = pe.forward(xEmb);
    printStats("PositionalEncoding", xPe);

    // 3. Block 1
    const ln1 = (model as any).ln1 as LayerNormalization;
    const xLn1 = ln1.forward(xPe);
    printStats("LayerNorm 1", xLn1);

    const mha = (model as any).mha as MultiHeadAttention;
    const xMha = mha.forward(xLn1);
    printStats("MultiHeadAttention", xMha);

    const drop1 = (model as any).drop1 as Dropout;
    const xDrop1 = drop1.forward(xMha);
    printStats("Dropout 1", xDrop1);

    // Residual 1
    const xRes1 = xPe.clone();
    xRes1.addInPlace(xDrop1);
    printStats("Residual 1", xRes1);

    // 4. Block 2
    const ln2 = (model as any).ln2 as LayerNormalization;
    const xLn2 = ln2.forward(xRes1);
    printStats("LayerNorm 2", xLn2);

    const ffn1 = (model as any).ffn1 as Dense;
    const xFfn1 = ffn1.forward(xLn2);
    printStats("FFN 1 (ReLU)", xFfn1);

    const ffn2 = (model as any).ffn2 as Dense;
    const xFfn2 = ffn2.forward(xFfn1);
    printStats("FFN 2 (Linear)", xFfn2);

    const dropFfn = (model as any).dropFfn as Dropout;
    const xDropFfn = dropFfn.forward(xFfn2); 
    printStats("Dropout FFN", xDropFfn);

    // Residual 2
    const xRes2 = xRes1.clone();
    xRes2.addInPlace(xDropFfn);
    printStats("Residual 2", xRes2);

    // 5. Output
    const dense = (model as any).dense as Dense;
    const logits = dense.forward(xRes2);
    printStats("Output (Logits)", logits);

    console.log("\n--- START TRAINING (100 iterations) ---");
    for (let iter = 0; iter < 100; iter++) {
        model.forward(x);
        model.backward(y);
        if (Number.isNaN(model.loss)) {
            console.log(`NaN detected at iteration ${iter}`);
            break;
        }
        if (iter % 20 === 0) {
            console.log(`Iteration ${iter} | Loss: ${model.loss.toFixed(4)}`);
        }
    }

    console.log("\nDebug Finished.");
}

function printStats(name: string, m: Matrix) {
    const data = m._data;
    let min = Infinity;
    let max = -Infinity;
    let hasNan = false;
    let hasInf = false;
    let sumSq = 0;
    
    for (let i = 0; i < data.length; i++) {
        const v = data[i];
        if (Number.isNaN(v)) {
            hasNan = true;
            continue;
        }
        if (!Number.isFinite(v)) {
            hasInf = true;
            continue;
        }
        if (v < min) min = v;
        if (v > max) max = v;
        sumSq += v * v;
    }
    
    const norm = Math.sqrt(sumSq);
    console.log(`${name.padEnd(20)} | Norm: ${norm.toFixed(4).padStart(10)} | Min: ${min.toFixed(4).padStart(8)} | Max: ${max.toFixed(4).padStart(8)} | NaN: ${hasNan} | Inf: ${hasInf}`);
}

debugModel();
