# Practical Tutorial: Getting Started with Oxide-JS

This guide will walk you through the basics of using **Oxide-JS**, from simple matrix operations to training a small transformer model.

## 1. Basic Matrix Operations

All data in Oxide-JS is represented as `Matrix` objects. Use the `mj` module from `@oxide-js/core` to perform operations.

```ts
import { mj } from "@oxide-js/core";

// Create a 2x2 matrix
const a = mj.matrix([[1, 2], [3, 4]]);
const b = mj.matrix([[5, 6], [7, 8]]);

// Dot Product Multiplication
const c = mj.dotProduct(a, b);

// Element-wise Addition
const d = mj.add(c, 10);

c.print(); // Print matrix contents to console
console.log("Shape:", d._shape);
```

---

## 2. Building a Simple Model

You can use the `Sequential` class from `@oxide-js/models` to stack layers from `@oxide-js/layers`.

```ts
import { mj } from "@oxide-js/core";
import { Dense } from "@oxide-js/layers";
import { Sequential } from "@oxide-js/models";

const model = new Sequential({
  layers: [
    new Dense({ units: 2, outputUnits: 4, activation: "relu", status: "input" }),
    new Dense({ units: 4, outputUnits: 1, activation: "sigmoid", status: "output", loss: "mse" }),
  ],
});

// Compile model with optimizer and learning rate
model.compile({ alpha: 0.01, optimizer: "adam", error: "mse" });
```

---

## 3. Training the Model (Fit)

Use the `.fit()` method to train the model on a dataset.

```ts
// Simple XOR Data
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

// Train for 500 epochs
model.fit(X, Y, 500, {
  batchSize: 4,
  onEpochEnd: (epoch, loss) => {
    console.log(`Epoch ${epoch} | Loss: ${loss.toFixed(6)}`);
  }
});

// Prediction
const pred = model.predict(mj.matrix([[1], [0]]));
console.log("Prediction Result for [1, 0]:");
pred.print();
```

---

## 4. Using the BPE Tokenizer

For NLP tasks, you need to convert text into a sequence of numbers (token IDs). The tokenizer is available in `@oxide-js/core`.

```ts
import { BPETokenizer } from "@oxide-js/core";

const tokenizer = new BPETokenizer({ vocabSize: 100, minFrequency: 1 });

// Train tokenizer with text data
const corpus = ["i am learning ai", "ai is cool", "learning coding"];
tokenizer.train(corpus);

// Encode text to token IDs
const ids = tokenizer.encodeWithSpecial("i am learning coding");
console.log("Token IDs:", ids);

// Decode back to text
const text = tokenizer.decode(ids);
console.log("Decoded Text:", text);

// Save tokenizer for later use
tokenizer.save("./my-tokenizer.json");
```

For multilingual text, choose a Unicode-aware pre-tokenizer:

```ts
import { BPETokenizer } from "@oxide-js/core";

const tokenizer = new BPETokenizer({
  vocabSize: 1000,
  preTokenizer: "script-aware"
});

tokenizer.train([
  "hello world",
  "مرحبا بالعالم",
  "こんにちは世界",
  "你好世界",
  "ภาษาไทย",
  "한국어테스트",
  "ꦱꦺꦴꦥꦺꦴ",
  "x² + y² = z²",
  "hello ꦱꦺꦴꦥꦺꦴ 😊 你好"
]);

const ids = tokenizer.encode("hello ꦱꦺꦴꦥꦺꦴ 😊 你好");
console.log(tokenizer.decode(ids));
```

---

## 5. Sequence Modeling with GRU

Use `RecurrentModel` for high-level sequence training.

```ts
import { mj } from "@oxide-js/core";
import { RecurrentModel } from "@oxide-js/models";

const model = new RecurrentModel({
  kind: "gru",
  inputSize: 8,
  hiddenSizes: [16, 16],
  outputSize: 4,
  seqLen: 3,
  mode: "many-to-one",
  loss: "softmaxCrossEntropy",
});

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
model.fit([x], [y], 100, { batchSize: 1, shuffle: false });
```

---

## 6. Full-Sequence Causal LM with Transformers

For `Transformers`, you can use `predictMode` to switch between training-style output and inference-style next-token prediction.

```ts
import { mj } from "@oxide-js/core";
import { Transformers } from "@oxide-js/models";

const padTokenId = 0;
const model = new Transformers({
  units: 32,
  seqLen: 6,
  vocabSize: 100,
  heads: 4,
  alpha: 0.001,
  padTokenId,
  predictMode: "next-token",
});

const x = mj.matrix([[0], [11], [12], [13], [14], [15]]);
const y = mj.matrix([[0], [12], [13], [14], [15], [0]]);

model.train();
const trainLogits = model.forward(x);
model.backward(y);

model.eval();
const nextTokenLogits = model.predict(x); // Next token only

model.setPredictMode("full-sequence");
const allTokenLogits = model.predict(x); // All sequence tokens
```

---

## Development Tips

- **Modular Packages**: Import from `@oxide-js/core` for math/matrix, `@oxide-js/layers` for neural network components, and `@oxide-js/models` for model architectures.
- **Training vs Eval Mode**: Always call `model.train()` before backprop and `model.eval()` before prediction (especially for `Dropout` or `LayerNormalization`).
- **Native Check**: Verify acceleration with `isNativeAvailable()` from `@oxide-js/core`.

**Next Steps:**
Explore the [Full API Reference](../api/README.md) for a complete list of parameters and methods.
