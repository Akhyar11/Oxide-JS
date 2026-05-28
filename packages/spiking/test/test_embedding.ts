import { Matrix } from "@oxide-js/core";
import { SpikingEmbedding } from "../src/layers/SpikingEmbedding.js";
import { SpikingDense } from "../src/layers/SpikingDense.js";

// Dataset: Mengajarkan SNN untuk mengenali 4 Token Kosakata
// Token 0 -> Target: Kelas A [1, 0, 0, 0]
// Token 1 -> Target: Kelas B [0, 1, 0, 0]
// Token 2 -> Target: Kelas C [0, 0, 1, 0]
// Token 3 -> Target: Kelas D [0, 0, 0, 1]
const xData = [[0], [1], [2], [3]];
const yData = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1]
];

const vocabSize = 4;
const embedDim = 16; // Ukuran otak embedding
const numClasses = 4;

console.log("Inisialisasi SpikingEmbedding & SpikingDense...");

const embedding = new SpikingEmbedding({
  inputDim: vocabSize,
  outputDim: embedDim,
  beta: 0.9,
  threshold: 1.0,
  embeddingsInitializer: "glorot_normal"
});

const outputLayer = new SpikingDense({
  units: numClasses,
  beta: 0.9,
  threshold: 1.0,
  useBias: true,
  kernelInitializer: "glorot_normal"
});

embedding.build([1, 1]); // input shape [batch=1, num_tokens=1]
outputLayer.build([1, embedDim]);

// Matriks Acak (B) untuk Feedback Alignment dari Output ke Embedding
const bData = new Float32Array(numClasses * embedDim);
for (let i = 0; i < bData.length; i++) bData[i] = (Math.random() * 2) - 1;
const B = Matrix.fromFlat(bData, [numClasses, embedDim]);

const epochs = 200;
const learningRate = 0.05;

console.log("Mulai training SNN Word-to-Class (Feedback Alignment)...");

for (let epoch = 0; epoch < epochs; epoch++) {
  let totalError = 0;

  for (let i = 0; i < xData.length; i++) {
    const x = Matrix.fromFlat(new Float32Array(xData[i]), [1, 1]);
    const y = Matrix.fromFlat(new Float32Array(yData[i]), [1, numClasses]);

    let outSpikes = Matrix.fromFlat(new Float32Array(numClasses), [1, numClasses]);
    let sudahSpike = new Array(numClasses).fill(false);

    embedding.resetState();
    outputLayer.resetState();

    // Berikan SNN waktu 5 timesteps untuk merenung dan menembak
    for (let t = 0; t < 5; t++) {
      const eSpikes = embedding.forward(x) as Matrix;
      outSpikes = outputLayer.forward(eSpikes) as Matrix;

      const actual = outSpikes._data;
      const target = y._data;

      const errData = new Float32Array(numClasses);
      let stepError = 0;

      for (let j = 0; j < numClasses; j++) {
        if (actual[j] === 1) sudahSpike[j] = true;

        if (target[j] === 1) {
          if (!sudahSpike[j]) errData[j] = 1; // Dorong sampai spike
          else errData[j] = 0;                // Sudah spike, diam
        } else {
          errData[j] = 0 - actual[j];         // Kalau salah spike, hukum!
        }
        stepError += Math.abs(errData[j]);
      }

      totalError += stepError;

      if (stepError !== 0) {
        const errorSignal = Matrix.fromFlat(errData, [1, numClasses]);

        // 1. Output Layer Learn
        outputLayer.learnOutput(errorSignal, learningRate);

        // 2. Embedding Layer Learn (lewat Feedback Alignment B)
        embedding.learnEmbedding(errorSignal, B, learningRate);
      }
    }
  }

  if (epoch % 50 === 0 || epoch === epochs - 1) {
    console.log(`Epoch ${epoch} | Total Spiking Error: ${totalError}`);
  }
}

// Uji coba tebak-tebakan kata
console.log("\n--- HASIL PENGUJIAN ---");
for (let i = 0; i < xData.length; i++) {
  const x = Matrix.fromFlat(new Float32Array(xData[i]), [1, 1]);
  embedding.resetState();
  outputLayer.resetState();

  let totalTembakan = new Float32Array(numClasses);
  
  for (let t = 0; t < 5; t++) {
    const eSpikes = embedding.forward(x) as Matrix;
    const outSpikes = outputLayer.forward(eSpikes) as Matrix;
    for(let j=0; j<numClasses; j++) {
        totalTembakan[j] += outSpikes._data[j];
    }
  }

  console.log(`Token Input: [${xData[i][0]}] -> Prediksi Spike Kelas: [${totalTembakan.join(", ")}] | Target Seharusnya: [${yData[i].join(", ")}]`);
}
