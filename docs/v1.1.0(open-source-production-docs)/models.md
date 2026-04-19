# Models

## Sequential

### Constructor
```ts
new Sequential({ layers?: Layers[] })
```

### Method penting
- `add(layer)`
- `compile({ alpha?, optimizer?, error? })`
- `forward(x)`
- `backward(y)`
- `fit(X, y, epochs, cb?)`
- `predict(x)`
- `save(path)` / `load(path)`

### Kapan dipakai
- Klasifikasi/regresi sederhana.
- Stack layer linear (dense/conv/flatten/attention ringan).

### Contoh
```ts
const model = new Sequential();
model.add(new Dense({ units: 8, outputUnits: 16, activation: "relu", status: "input" }));
model.add(new Dense({ units: 16, outputUnits: 3, activation: "linear", status: "output", loss: "softmaxCrossEntropy" }));
model.compile({ alpha: 0.001, optimizer: "adam", error: "softmaxCrossEntropy" });
```

---

## Transformers

### Constructor
```ts
new Transformers({
  units,
  seqLen,
  vocabSize,
  heads = 8,
  dropoutRate = 0.1,
  alpha = 0.01,
  padTokenId,
})
```

### Method penting
- `compile({ alpha, optimizer, error })`
- `forward(x)` / `backward(y)`
- `fit(X, y, epochs, cb)`
- `resizeVocab(newVocabSize)`
- `save(path)` / `load(path)`
- profiling: `enableProfiling()`, `getProfilingReport()`

### Kapan dipakai
- Next-token prediction atau sequence modeling berbasis token.

### Contoh
```ts
const model = new Transformers({ units: 64, seqLen: 32, vocabSize: 1000, heads: 8, alpha: 0.001, padTokenId: 0 });
model.compile({ alpha: 0.001, optimizer: "adam", error: "softmaxCrossEntropy" });
```

---

## DimentionalityReduction

Turunan `Sequential` dengan split layer encoder/decoder berdasarkan status `outputReduction`.

### Method
- `encode(x)`
- `decode(enc)`
- `load(path)`

### Catatan
Jika tidak ada layer status `outputReduction`, seluruh layer akan dianggap encoder dan decoder kosong.
