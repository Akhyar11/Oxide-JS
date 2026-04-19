# Tutorial Intermediate

## Tujuan
Menggunakan tokenizer + embedding + attention untuk task sequence sederhana.

## Step 1: latih tokenizer
```ts
import { BPETokenizer } from "../src/tokenizer";

const tokenizer = new BPETokenizer({ vocabSize: 150, minFrequency: 2 });
tokenizer.train([
  "instruksi: hitung 1 + 2 jawaban: 3",
  "instruksi: hitung 2 + 5 jawaban: 7",
]);
```

## Step 2: siapkan sequence
```ts
const seqLen = 12;
const ids = tokenizer.padSequence(tokenizer.encodeWithSpecial("instruksi: hitung 3 + 4 jawaban: 7"), seqLen);
const x = mj.matrix(ids.map((id)=>[id]));
```

## Step 3: model embedding + self-attention
```ts
import { Sequential } from "../src/models";
import { Embedding, PositionalEncoding, SelfAttention, Flatten, Dense } from "../src/layers";

const embeddingDim = 16;
const model = new Sequential();
model.add(new Embedding({ vocabSize: tokenizer.getVocabSize(), embeddingDim, padTokenId: tokenizer.getPadId() }));
model.add(new PositionalEncoding({ dModel: embeddingDim, maxSeqLen: seqLen }));
model.add(new SelfAttention({ units: embeddingDim, seqLen, alpha: 0.001 }));
model.add(new Flatten());
model.add(new Dense({ units: embeddingDim * seqLen, outputUnits: tokenizer.getVocabSize(), activation: "linear", status: "output", loss: "softmaxCrossEntropy" }));

model.compile({ alpha: 0.001, optimizer: "adam", error: "softmaxCrossEntropy" });
```

## Step 4: forward/backward manual
```ts
const y = mj.matrix([[tokenizer.getPadId()]]);
model.forward(x);
model.backward(y);
console.log(model.loss);
```

## Step 5: simpan artefak
```ts
model.save("./seq-attn.json");
tokenizer.save("./bpe.json");
```
