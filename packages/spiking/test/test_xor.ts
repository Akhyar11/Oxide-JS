import { Matrix } from "@oxide-js/core";
import { SpikingDense } from "../src/layers/SpikingDense.js";

// XOR Dataset
const xData = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
];

const yData = [
  [0],
  [1],
  [1],
  [0]
];

// Initialize layers
const hiddenUnits = 8;
const hiddenLayer = new SpikingDense({
  units: hiddenUnits,
  beta: 0.9,
  threshold: 1.0,
  useBias: true,
  kernelInitializer: "glorot_normal"
});

const outputLayer = new SpikingDense({
  units: 1,
  beta: 0.9,
  threshold: 1.0,
  useBias: true,
  kernelInitializer: "glorot_normal"
});

// Build layers
hiddenLayer.build([1, 2]);
outputLayer.build([1, hiddenUnits]);

// Random Matrix B for Feedback Alignment
// The shape should be [outputUnits, hiddenUnits] -> [1, 8]
const bData = new Float32Array(hiddenUnits);
const B = Matrix.fromFlat(bData, [1, hiddenUnits]);
for (let i = 0; i < bData.length; i++) {
  // Random -1 to 1
  bData[i] = (Math.random() * 2) - 1;
}

const epochs = 500;
const learningRate = 0.01;

console.log("Mulai training SNN XOR dengan Feedback Alignment...");

for (let epoch = 0; epoch < epochs; epoch++) {
  let totalError = 0;

  for (let i = 0; i < xData.length; i++) {
    const x = Matrix.fromFlat(new Float32Array(xData[i]), [1, 2]);
    const y = Matrix.fromFlat(new Float32Array(yData[i]), [1, 1]);

    // Berikan waktu (misalnya 5 time-steps) untuk setiap input agar SNN bisa accumulate
    let outSpikes = Matrix.fromFlat(new Float32Array(1), [1, 1]);
    let sudahSpike = false;

    for (let t = 0; t < 5; t++) {
      const hSpikes = hiddenLayer.forward(x) as Matrix;
      outSpikes = outputLayer.forward(hSpikes) as Matrix;

      const actual = outSpikes._data[0];
      const target = y._data[0];

      if (actual === 1) sudahSpike = true;

      let err = 0;
      if (target === 1) {
        if (!sudahSpike) err = 1; // Dorong terus sampai dia spike
        else err = 0;             // Udah spike, biarkan dia istirahat
      } else { // Target === 0
        err = 0 - actual;         // Kalau target 0, dia HARUS 0 terus. Kalau spike, hukum!
      }

      totalError += Math.abs(err);

      if (err !== 0) {
        const errorSignal = Matrix.fromFlat(new Float32Array([err]), [1, 1]);

        // 1. Output Layer Learn (Delta Rule standard)
        outputLayer.learnOutput(errorSignal, learningRate);

        // 2. Hidden Layer Learn (Feedback Alignment broadcast)
        hiddenLayer.learnHidden(errorSignal, B, learningRate);
      }
    }

    // Reset state antar data point
    hiddenLayer.resetState();
    outputLayer.resetState();
  }

  if (epoch % 100 === 0) {
    console.log(`Epoch ${epoch} | Total Spiking Error: ${totalError}`);
  }
}

// Uji coba hasil
console.log("\nHasil Pengujian:");
for (let i = 0; i < xData.length; i++) {
  const x = Matrix.fromFlat(new Float32Array(xData[i]), [1, 2]);
  hiddenLayer.resetState();
  outputLayer.resetState();

  let sumSpikes = 0;
  for (let t = 0; t < 5; t++) {
    const hSpikes = hiddenLayer.forward(x) as Matrix;
    const outSpikes = outputLayer.forward(hSpikes) as Matrix;
    sumSpikes += outSpikes._data[0];
  }
  // Jika dalam 5 timestep dia spike setidaknya 1 kali, kita anggap Prediksi 1
  const pred = sumSpikes >= 1 ? 1 : 0;
  console.log(`Input [${xData[i]}] -> Target: ${yData[i][0]} | Prediksi Spike: ${pred} (Total tembakan: ${sumSpikes})`);
}
