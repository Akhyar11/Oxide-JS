import { SpikingNetwork } from "../src/core/SpikingNetwork.js";
import { STDP } from "../src/learning/STDP.js";
import { PoissonEncoder } from "../src/encoding/PoissonEncoder.js";

/**
 * Simulasi Pengenalan Pola Sederhana menggunakan SNN
 * --------------------------------------------------
 * 1. Kita memiliki 3 neuron Input dan 1 neuron Output (Total 4 Neuron)
 * 2. Input merepresentasikan intensitas sinyal (misalnya: [0.9, 0.1, 0.8])
 * 3. Neuron Input terhubung ke Neuron Output
 * 4. STDP akan mengamati dan memperkuat bobot dari input yang sering "menyala" (0.9 dan 0.8)
 */
async function runDemo() {
  console.log("=== Memulai Simulasi SNN Add-Only ===\n");

  // Konfigurasi
  const NUM_INPUTS = 3;
  const NUM_OUTPUTS = 1;
  const TOTAL_NEURONS = NUM_INPUTS + NUM_OUTPUTS;
  const TIME_STEPS = 100;

  // 1. Inisialisasi Jaringan & Komponen
  // Threshold di-set rendah agar output mudah menembakkan spike
  const net = new SpikingNetwork(TOTAL_NEURONS, 0.8, 2.0);
  const stdp = new STDP(net, {
    learningRate: 0.05,
    aPlus: 1.5,
    aMinus: 0.5, // LTD diturunkan agar sinyal yang sering aktif (berkorelasi kuat) membesar
    wMax: 10.0,
    wMin: 0.0    // Bobot tidak boleh negatif dalam contoh ini
  });
  const encoder = new PoissonEncoder(1.0);

  // Neuron Output berada di indeks ke-3 (karena 0,1,2 adalah input)
  const OUTPUT_NEURON = 3;

  // 2. Hubungkan Input ke Output dengan bobot awal yang seragam (kecil)
  console.log("Membangun koneksi awal:");
  for (let i = 0; i < NUM_INPUTS; i++) {
    net.connect(i, OUTPUT_NEURON, 1.0);
    console.log(`- Input Neuron ${i} -> Output Neuron ${OUTPUT_NEURON} (Bobot Awal: 1.0)`);
  }
  console.log("\n");

  // 3. Menyiapkan Pola Input
  // Neuron 0 (Sinyal Kuat), Neuron 1 (Sinyal Lemah), Neuron 2 (Sinyal Kuat)
  const inputPattern = [0.9, 0.1, 0.8];
  
  let totalOutputSpikes = 0;

  console.log(`Menjalankan simulasi selama ${TIME_STEPS} time-steps...`);
  
  for (let t = 0; t < TIME_STEPS; t++) {
    // A. Encode sinyal input kontinu menjadi Spike
    const currentSpikes = encoder.encodeArray(inputPattern);

    // B. Injeksi spike input secara langsung ke neuron input
    // (Beri potensial yang cukup agar mereka seketika menembakkan spike di step ini)
    for (let i = 0; i < NUM_INPUTS; i++) {
      if (currentSpikes[i] === 1) {
        // Injeksi arus yang jauh di atas threshold (2.0) agar pasti spike
        net.injectCurrent(i, 10.0); 
      }
    }

    // C. Evaluasi Jaringan (Add-Only Propagation)
    net.step();

    // D. Hitung Spike Output
    if (net.spikes[OUTPUT_NEURON] === 1) {
      totalOutputSpikes++;
    }

    // E. Lakukan Proses Pembelajaran (Plasticity)
    stdp.updateWeights();
  }

  // 4. Hasil Simulasi
  console.log("\n=== Hasil Simulasi ===");
  console.log(`Total tembakan (spikes) dari Neuron Output: ${totalOutputSpikes}`);
  console.log("\nPerubahan Bobot Akhir (Setelah proses Belajar STDP):");
  
  for (let i = 0; i < NUM_INPUTS; i++) {
    // Bobot ke-0 dari array koneksi neuron input 'i'
    const finalWeight = net.weights[i][0];
    const initialInput = inputPattern[i];
    
    let status = "";
    if (finalWeight > 1.0) status = "📈 Diperkuat (LTP)";
    else if (finalWeight < 1.0) status = "📉 Diperlemah (LTD)";
    else status = "➖ Tetap";

    console.log(`Neuron ${i} (Intensitas Sinyal: ${initialInput}) -> Bobot Akhir: ${finalWeight.toFixed(4)} ${status}`);
  }

  console.log("\nKesimpulan:");
  console.log("Seperti yang terlihat, SNN secara otomatis mengenali dan memperkuat koneksi dari neuron input yang aktif (Intensitas 0.9 & 0.8),");
  console.log("sementara neuron yang jarang aktif (Intensitas 0.1) bobotnya melemah secara natural melalui aturan STDP.");
}

runDemo().catch(console.error);
