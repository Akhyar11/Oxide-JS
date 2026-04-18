# API Reference (Ringkas)

## Entry module yang dipakai saat ini
- `../src/math`
- `../src/models`
- `../src/layers`
- `../src/tokenizer`

## Matrix & Math
- `mj.matrix(matrix2d)`
- `mj.dotProduct(a,b,out?,transA?,transB?)`
- `mj.add/sub/mul/div`
- `mj.reshape`, `mj.flatten`, `mj.transpose`
- `mj.sumAxis`, `mj.addBias`, `mj.clipGradients`

## Models
- `Sequential`
  - `add(layer)`
  - `compile({ alpha, optimizer, error })`
  - `forward(x)`, `backward(y)`
  - `fit(X, y, epochs, cb)`
  - `predict(x)`, `train()`, `eval()`
  - `save(path)`, `load(path)`, `summary()`
- `Transformers`
  - constructor `{ units, seqLen, vocabSize, heads?, dropoutRate?, alpha?, padTokenId? }`
  - `resizeVocab(newVocabSize)`
  - profiling API: `enableProfiling`, `disableProfiling`, `getProfilingReport`
- `DimentionalityReduction`
  - `encode(x)`, `decode(enc)`

## Layers
- `Dense`
- `Embedding`
- `LayerNormalization`
- `Dropout`
- `PositionalEncoding`
- `SelfAttention`
- `MultiHeadAttention`
- `Convolution`
- `Flatten`
- `Activation`

## Tokenizer
- `BPETokenizer`
  - `train`, `update`
  - `encode`, `encodeWithSpecial`, `decode`
  - `padSequence`, `getPadId`, `getVocabSize`
  - `save`, `static load`

## Peta dokumen detail
- Model detail: `models.md`
- Layer detail: `layers.md`
- Math detail: `math-and-matrix.md`
- Optimizer/loss/activation: `optimizer-loss-activation.md`
- Tokenizer detail: `tokenizer.md`
