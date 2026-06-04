import { Matrix } from "@oxide-js/core";
import { SpikingEmbedding } from "../src/layers/SpikingEmbedding.js";
import { SpikingDense } from "../src/layers/SpikingDense.js";

// Kamus Kosakata (Vocabulary)
// 0: Kucing
// 1: Manusia
// 2: Makan
// 3: Ikan
// 4: Tidur
// 5: Kasur
const vocabSize = 6;
const embedDim = 16;
const numClasses = 2; // Kelas 0: Hewan, Kelas 1: Manusia

// Dataset Kalimat (Urutan indeks kata)
// Kalimat 1: Kucing(0) makan(2) ikan(3) -> Target: Hewan [1, 0]
// Kalimat 2: Manusia(1) makan(2) ikan(3) -> Target: Manusia [0, 1]
// Kalimat 3: Kucing(0) tidur(4) kasur(5) -> Target: Hewan [1, 0]
// Kalimat 4: Manusia(1) tidur(4) kasur(5) -> Target: Manusia [0, 1]

const sentences = [
  [0, 2, 3], // Kucing makan ikan
  [1, 2, 3], // Manusia makan ikan
  [0, 4, 5], // Kucing tidur kasur
  [1, 4, 5]  // Manusia tidur kasur
];

const targets = [
  [1, 0], // Hewan
  [0, 1], // Manusia
  [1, 0], // Hewan
  [0, 1]  // Manusia
];

console.log("Inisialisasi SpikingEmbedding & SpikingDense (Tes Overlap)...");

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

// Batch size 1, num_tokens 1 (Konteks diumpankan secara berurutan dalam timesteps)
embedding.build([1, 1]);
outputLayer.build([1, embedDim]);

// Matriks Feedback Alignment (B) yang bernilai acak namun tetap
const bData = new Float32Array(numClasses * embedDim);
for (let i = 0; i < bData.length; i++) bData[i] = (Math.random() * 2) - 1;
const B = Matrix.fromFlat(bData, [numClasses, embedDim]);

const epochs = 300; // Epoch butuh lebih banyak karena ada tarik-menarik gradien
const learningRate = 0.005;

console.log("Mulai training SNN Word-to-Class dengan overlapping context...");

for (let epoch = 0; epoch < epochs; epoch++) {
  let totalError = 0;

  for (let i = 0; i < sentences.length; i++) {
    const sentence = sentences[i];
    const y = Matrix.fromFlat(new Float32Array(targets[i]), [1, numClasses]);

    let outSpikes = Matrix.fromFlat(new Float32Array(numClasses), [1, numClasses]);
    let sudahSpike = new Array(numClasses).fill(false);

    embedding.resetState();
    outputLayer.resetState();

    // Loop melewati kata-kata di kalimat berulang kali (simulasi durasi waktu SNN membaca)
    // 3 kata x 4 siklus = 12 timestep
    for (let t = 0; t < 12; t++) {
      const tokenIdx = sentence[t % sentence.length]; // Ambil token secara round-robin
      const x = Matrix.fromFlat(new Float32Array([tokenIdx]), [1, 1]);

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
          else errData[j] = 0;                // Sudah spike, biarkan santai
        } else {
          errData[j] = 0 - actual[j];         // Penalti jika salah spike
        }
        stepError += Math.abs(errData[j]);
      }

      totalError += stepError;

      if (stepError !== 0) {
        const errorSignal = Matrix.fromFlat(errData, [1, numClasses]);
        outputLayer.learnOutput(errorSignal, learningRate);
        embedding.learnEmbedding(errorSignal, B, learningRate);
      }
    }
  }

  if (epoch % 50 === 0 || epoch === epochs - 1) {
    console.log(`Epoch ${epoch} | Total Spiking Error: ${totalError}`);
  }
}

// Uji coba inferensi kalimat
console.log("\n--- HASIL PENGUJIAN KALIMAT ---");
for (let i = 0; i < sentences.length; i++) {
  const sentence = sentences[i];
  embedding.resetState();
  outputLayer.resetState();

  let totalTembakan = new Float32Array(numClasses);

  // Baca kalimat selama 12 timestep
  for (let t = 0; t < 12; t++) {
    const tokenIdx = sentence[t % sentence.length];
    const x = Matrix.fromFlat(new Float32Array([tokenIdx]), [1, 1]);

    const eSpikes = embedding.forward(x) as Matrix;
    const outSpikes = outputLayer.forward(eSpikes) as Matrix;

    for (let j = 0; j < numClasses; j++) {
      totalTembakan[j] += outSpikes._data[j];
    }
  }

  const teks = sentence.map(s => {
    switch (s) {
      case 0: return "Kucing";
      case 1: return "Manusia";
      case 2: return "Makan";
      case 3: return "Ikan";
      case 4: return "Tidur";
      case 5: return "Kasur";
      default: return "";
    }
  }).join(" ");

  console.log(`Kalimat: "${teks}"`);
  console.log(` -> Prediksi Spike [Hewan, Manusia]: [${totalTembakan.join(", ")}] | Target Seharusnya: [${targets[i].join(", ")}]`);
}

// Uji coba tebak kata individu (untuk melihat sentimen yang dipelajari embedding layer)
console.log("\n--- ANALISIS SENTIMEN KATA INDIVIDU ---");
const words = ["Kucing (0)", "Manusia (1)", "Makan (2)", "Ikan (3)", "Tidur (4)", "Kasur (5)"];
for (let i = 0; i < vocabSize; i++) {
  embedding.resetState();
  outputLayer.resetState();
  let totalTembakan = new Float32Array(numClasses);

  // Diberi input kata yang sama selama 10 timestep (memaksa SNN memikirkan kata ini saja)
  for (let t = 0; t < 10; t++) {
    const x = Matrix.fromFlat(new Float32Array([i]), [1, 1]);
    const eSpikes = embedding.forward(x) as Matrix;
    const outSpikes = outputLayer.forward(eSpikes) as Matrix;

    for (let j = 0; j < numClasses; j++) {
      totalTembakan[j] += outSpikes._data[j];
    }
  }

  console.log(`Kata ${words[i]} -> Pola Spike [Hewan, Manusia]: [${totalTembakan.join(", ")}]`);
}
