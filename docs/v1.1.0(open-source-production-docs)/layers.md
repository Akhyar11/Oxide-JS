# Layers

## Dense
- Constructor: `{ units, outputUnits, activation?, optimizer?, status?, alpha?, loss? }`
- Forward: `W*x + b`, lalu activation.
- Backward: hitung grad, clipping, optimizer update.

## Embedding
- Constructor: `{ vocabSize, embeddingDim, alpha?, status?, optimizer?, padTokenId? }`
- Input: matrix index token.
- Output: `[embeddingDim, totalTokens]`.
- Mendukung `resize(newVocabSize)`.

## LayerNormalization
- Constructor: `{ units, status?, alpha?, optimizer? }`
- Normalisasi per kolom/token.
- Trainable params: `gamma`, `beta`.

## Dropout
- Constructor: `{ rate?, status? }`
- Aktif saat training mode.
- `Sequential.train()/eval()` akan mengubah mode dropout.

## PositionalEncoding
- Constructor: `{ dModel, maxSeqLen?, status? }`
- Sinusoidal fixed encoding (tanpa trainable params).

## MultiHeadAttention
- Constructor: `{ units, heads, seqLen, alpha?, status? }`
- Causal mask + pad mask.
- Proyeksi `q/k/v` + output projector dense.

## SelfAttention
- Constructor: `{ units, outputUnits?, seqLen?, alpha?, loss?, status? }`
- Attention tunggal (non multi-head).

## Flatten
- Constructor: `new Flatten(status?)`
- Ubah `[r,c] -> [r*c,1]`, backward melakukan reshape balik.

## Convolution
- Constructor: `{ kernelSize, inputShape, alpha?, status?, activation?, optimizer?, loss? }`
- Mendukung status `convOutput` untuk flatten output internal.

## Activation layer
- Wrapper layer untuk aktivasi (`relu/sigmoid/tanh/lRelu/linear/softmax`).

## Contoh komposisi layer
```ts
const model = new Sequential({
  layers: [
    new Embedding({ vocabSize: 500, embeddingDim: 32, padTokenId: 0 }),
    new PositionalEncoding({ dModel: 32, maxSeqLen: 16 }),
    new LayerNormalization({ units: 32 }),
    new Dropout({ rate: 0.1 }),
    new Dense({ units: 32, outputUnits: 500, activation: "linear", status: "output", loss: "softmaxCrossEntropy" }),
  ],
});
```
