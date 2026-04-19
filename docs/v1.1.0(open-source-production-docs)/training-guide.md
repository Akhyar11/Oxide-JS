# Training Guide

## Menyiapkan data
- Bentuk input ke `Matrix` sesuai ekspektasi layer awal.
- Untuk transformer, input umumnya token index `[seqLen, 1]` per sample.
- Untuk sparse target token, gunakan target `[1, 1]` berisi token id.

## Batching
`Sequential.fit`/`Transformers.fit` saat ini menerima array sample:
- `X: Matrix[]`
- `y: Matrix[]`

Batch manual dapat dilakukan dengan loop luar pada chunk data.

## Compile
```ts
model.compile({ alpha: 0.001, optimizer: "adam", error: "softmaxCrossEntropy" });
```

## Loop training (manual)
```ts
let epochLoss = 0;
for (let i = 0; i < X.length; i++) {
  model.forward(X[i]);
  model.backward(y[i]);
  epochLoss += model.loss;
}
console.log(epochLoss / X.length);
```

## Monitoring loss
- Gunakan callback pada `fit`.
- Untuk detail internal transformer, aktifkan profiling:
```ts
model.enableProfiling(true);
```

## Save checkpoint
```ts
model.save("./checkpoints/model.json");
```

## Resume training
```ts
model.load("./checkpoints/model.json");
model.compile({ alpha: 0.0005, optimizer: "adam", error: "softmaxCrossEntropy" });
```

## Fine-tuning + resize vocab
```ts
model.resizeVocab(newVocabSize);
```
Pastikan tokenizer juga sudah update vocab.

## Best practice training
- Mulai dari learning rate kecil untuk transformer.
- Pastikan `seqLen` preprocessing konsisten dengan model.
- Gunakan `padTokenId` agar embedding/token padding konsisten.
- Cek distribusi target agar loss tidak stagnan.
