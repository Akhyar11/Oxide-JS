# Tutorial Advanced

## Tujuan
Fine-tuning transformer + resize vocab + inference loop + native profiling.

## Step 1: load tokenizer lama lalu update vocab
```ts
const tokenizer = BPETokenizer.load("./bpe.json");
tokenizer.update(["materi baru domain spesifik"], 500);
```

## Step 2: init transformer dan sinkronkan vocab
```ts
const seqLen = 32;
const model = new Transformers({
  units: 64,
  seqLen,
  vocabSize: tokenizer.getVocabSize(),
  heads: 8,
  alpha: 0.0005,
  padTokenId: tokenizer.getPadId(),
});

model.load("./transformer-prev.json");
model.resizeVocab(tokenizer.getVocabSize());
model.compile({ alpha: 0.0005, optimizer: "adam", error: "softmaxCrossEntropy" });
```

## Step 3: training loop fine-tune
```ts
for (let epoch = 0; epoch < 5; epoch++) {
  let avg = 0;
  for (let i = 0; i < X.length; i++) {
    model.forward(X[i]);
    model.backward(Y[i]);
    avg += model.loss;
  }
  console.log("epoch", epoch, "loss", avg / X.length);
}
```

## Step 4: aktifkan profiling internal
```ts
model.enableProfiling(true);
model.forward(X[0]);
console.log(model.getProfilingReport(true));
```

## Step 5: generation sederhana
```ts
function argmaxCol0(m: any): number {
  let best = 0, max = -Infinity;
  for (let i = 0; i < m._shape[0]; i++) {
    const v = m._data[i * m._shape[1]];
    if (v > max) { max = v; best = i; }
  }
  return best;
}

let context = tokenizer.padSequence(tokenizer.encodeWithSpecial("instruksi: hitung 12 + 18"), seqLen);
for (let step = 0; step < 16; step++) {
  const logits = model.predict(mj.matrix(context.map((id)=>[id])));
  const next = argmaxCol0(logits);
  context = [...context.slice(1), next];
}
```

## Step 6: simpan checkpoint terbaru
```ts
model.save("./transformer-finetuned.json");
tokenizer.save("./bpe-finetuned.json");
```
