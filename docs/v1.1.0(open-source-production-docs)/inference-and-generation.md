# Inference and Generation

## Load model
```ts
const model = new Transformers({ units: 64, seqLen: 16, vocabSize: 1000, heads: 8, padTokenId: 0 });
model.load("./checkpoints/transformer.json");
model.eval();
```

## Forward inference
```ts
const logits = model.predict(inputMatrix); // [vocabSize, batch]
```

## Ambil token prediksi (argmax sederhana)
```ts
function argmaxCol0(m: any): number {
  let best = 0;
  let max = -Infinity;
  for (let i = 0; i < m._shape[0]; i++) {
    const v = m._data[i * m._shape[1]];
    if (v > max) { max = v; best = i; }
  }
  return best;
}
```

## Decode hasil
```ts
const nextId = argmaxCol0(logits);
const text = tokenizer.decode([nextId]);
```

## Generation loop sederhana
```ts
let context = tokenizer.padSequence(tokenizer.encodeWithSpecial(prompt), seqLen);
for (let step = 0; step < 20; step++) {
  const x = mj.matrix(context.map((id) => [id]));
  const out = model.predict(x);
  const next = argmaxCol0(out);
  context = [...context.slice(1), next];
}
```

## Catatan
Model transformer implementasi ini menggunakan representasi last-token state untuk output projector.
