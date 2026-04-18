# Optimizer, Loss, Activation

## Optimizer tersedia
- `sgd`
- `adaGrad`
- `momentum`
- `nag`
- `adam`

Semua optimizer diset via `compile({ optimizer })` atau constructor layer.

## Loss tersedia
- `mse`
- `crossEntropy`
- `binaryCrossEntropy`
- `softmaxCrossEntropy`

## Activation tersedia
- `linear`
- `sigmoid`
- `tanh`
- `relu`
- `lRelu`
- `softmax`

## Best practice pemilihan
- Klasifikasi multi-kelas sparse token: `activation: linear` + `loss: softmaxCrossEntropy`.
- Klasifikasi biner: `sigmoid` + `binaryCrossEntropy`.
- Regressi: `linear` + `mse`.
- Model transformer default lebih aman pakai `adam` + learning rate kecil.

## Contoh
```ts
const out = new Dense({
  units: 64,
  outputUnits: vocabSize,
  activation: "linear",
  status: "output",
  loss: "softmaxCrossEntropy",
});

model.compile({ alpha: 0.001, optimizer: "adam", error: "softmaxCrossEntropy" });
```
