# Tutorial Singkat: Memulai dengan ML-V1

Panduan ini akan membawa Anda melalui dasar-dasar penggunaan ML-V1, mulai dari operasi matriks sederhana hingga melatih model transformer kecil.

## 1. Operasi Matriks Dasar

Semua data dalam ML-V1 direpresentasikan sebagai objek `Matrix`. Gunakan modul `math` (sering dialiaskan sebagai `mj`) untuk melakukan operasi.

```ts
import mj from "./src/math";

// Membuat matriks 2x2
const a = mj.matrix([[1, 2], [3, 4]]);
const b = mj.matrix([[5, 6], [7, 8]]);

// Perkalian Dot Product
const c = mj.dotProduct(a, b);

// Elemen-wise Addition
const d = mj.add(c, 10);

c.print(); // Mencetak isi matriks ke konsol
console.log("Shape:", d._shape);
```

---

## 2. Membangun Model Sederhana

Anda dapat menggunakan kelas `Sequential` untuk menumpuk berbagai layer jaringan saraf.

```ts
import mj from "./src/math";
import { Sequential } from "./src/models";
import { Dense } from "./src/layers";

const model = new Sequential({
  layers: [
    new Dense({ units: 2, outputUnits: 4, activation: "relu", status: "input" }),
    new Dense({ units: 4, outputUnits: 1, activation: "sigmoid", status: "output", loss: "mse" }),
  ],
});

// Compile model dengan optimizer dan learning rate
model.compile({ alpha: 0.01, optimizer: "adam", error: "mse" });
```

---

## 3. Melatih Model (Fit)

Gunakan metode `.fit()` untuk melatih model pada dataset.

```ts
// Data XOR Sederhana
const X = [
  mj.matrix([[0], [0]]), 
  mj.matrix([[0], [1]]), 
  mj.matrix([[1], [0]]), 
  mj.matrix([[1], [1]])
];
const Y = [
  mj.matrix([[0]]), 
  mj.matrix([[1]]), 
  mj.matrix([[1]]), 
  mj.matrix([[0]])
];

// Latih selama 500 epoch
model.fit(X, Y, 500, (loss) => {
  console.log(`Current Loss: ${loss.toFixed(6)}`);
});

// Prediksi
const pred = model.predict(mj.matrix([[1], [0]]));
console.log("Hasil Prediksi [1, 0]:");
pred.print();
```

---

## 4. Menggunakan BPE Tokenizer

Untuk tugas NLP, Anda perlu mengubah teks menjadi urutan angka (token ID).

```ts
import { BPETokenizer } from "./src/tokenizer";

const tokenizer = new BPETokenizer({ vocabSize: 100, minFrequency: 1 });

// Training tokenizer dengan data teks
const corpus = ["saya belajar AI", "AI itu keren", "belajar coding"];
tokenizer.train(corpus);

// Encode teks ke token ID
const ids = tokenizer.encodeWithSpecial("saya belajar coding");
console.log("Token IDs:", ids);

// Decode kembali ke teks
const text = tokenizer.decode(ids);
console.log("Decoded Text:", text);

// Simpan tokenizer untuk digunakan nanti
tokenizer.save("./my-tokenizer.json");
```

---

## 5. Sequence Modeling dengan GRU

Gunakan layer recurrent saat input berupa urutan (shape umum: `[features, seqLen]`).

```ts
import mj from "./src/math";
import { Sequential } from "./src/models";
import { GRU, Dense } from "./src/layers";

const model = new Sequential({
  layers: [
    new GRU({ units: 8, hiddenUnits: 16, returnSequences: false, status: "input" }),
    new Dense({ units: 16, outputUnits: 4, activation: "linear", status: "output", loss: "softmaxCrossEntropy" }),
  ],
});

model.compile({ alpha: 0.001, optimizer: "adam", error: "softmaxCrossEntropy" });
const x = mj.matrix([
  [1, 2, 3],
  [0, 1, 0],
  [1, 0, 1],
  [0, 0, 1],
  [1, 1, 1],
  [0, 1, 1],
  [1, 0, 0],
  [0, 0, 0],
]); // [8, 3]
const y = mj.matrix([[2]]);
model.forward(x);
model.backward(y);
```

---

## 6. Full-Sequence Causal LM dengan Transformers

Untuk `Transformers`, jalur training dan inference sengaja dipisahkan:
- training: logits untuk seluruh posisi token valid
- inference: logits token terakhir saja untuk sampling token berikutnya

```ts
import mj from "./src/math";
import { Transformers } from "./src/models";

const padTokenId = 0;
const model = new Transformers({
  units: 32,
  seqLen: 6,
  vocabSize: 100,
  heads: 4,
  alpha: 0.001,
  padTokenId,
});

const x = mj.matrix([
  [0],
  [11],
  [12],
  [13],
  [14],
  [15],
]);

const y = mj.matrix([
  [0],
  [12],
  [13],
  [14],
  [15],
  [0],
]);

model.train();
const trainLogits = model.forward(x); // [vocabSize, seqLen * batch]
model.backward(y);

model.eval();
const nextTokenLogits = model.predict(x); // [vocabSize, batch]
```

---

## Tips Pengembangan

- **Mode Training vs Evol**: Gunakan `model.train()` saat melatih dan `model.eval()` saat melakukan inferensi (terutama jika menggunakan layer `Dropout`).
- **Dimensi Matriks**: Selalu periksa shape matriks Anda. Sebagian besar layer mengharapkan input dalam bentuk `[features, batch_size]` atau `[sequence_length, batch_size]`.

---

**Langkah Berikutnya:**
Eksplorasi seluruh fungsi yang tersedia di bagian [Referensi Fungsi & API](04-api-functions.md).
