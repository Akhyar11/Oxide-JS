# Quickstart

## 1) Operasi matrix minimal
```ts
import mj from "../src/math";

const a = mj.matrix([[1, 2], [3, 4]]);
const b = mj.matrix([[5, 6], [7, 8]]);
const c = mj.dotProduct(a, b);
c.print();
```

## 2) Training paling sederhana (Sequential + Dense)
```ts
import mj from "../src/math";
import { Sequential } from "../src/models";
import { Dense } from "../src/layers";

const model = new Sequential({
  layers: [
    new Dense({ units: 2, outputUnits: 4, activation: "relu", status: "input" }),
    new Dense({ units: 4, outputUnits: 1, activation: "sigmoid", status: "output", loss: "mse" }),
  ],
});

model.compile({ alpha: 0.01, optimizer: "adam", error: "mse" });

const X = [mj.matrix([[0], [0]]), mj.matrix([[0], [1]]), mj.matrix([[1], [0]]), mj.matrix([[1], [1]])];
const Y = [mj.matrix([[0]]), mj.matrix([[1]]), mj.matrix([[1]]), mj.matrix([[0]])];

model.fit(X, Y, 200, (loss) => console.log(loss));
```

## 3) Inference sederhana
```ts
const pred = model.predict(mj.matrix([[1], [0]]));
console.log(pred._value);
```

## 4) Quickstart tokenizer + transformer
```ts
import mj from "../src/math";
import { BPETokenizer } from "../src/tokenizer";
import { Transformers } from "../src/models";

const tokenizer = new BPETokenizer({ vocabSize: 120, minFrequency: 2 });
tokenizer.train(["aku suka matematika", "aku suka fisika"]);

const seqLen = 8;
const ids = tokenizer.padSequence(tokenizer.encodeWithSpecial("aku suka matematika"), seqLen);

const model = new Transformers({
  units: 32,
  seqLen,
  vocabSize: tokenizer.getVocabSize(),
  heads: 8,
  alpha: 0.001,
  padTokenId: tokenizer.getPadId(),
});

model.compile({ alpha: 0.001, optimizer: "adam", error: "softmaxCrossEntropy" });
model.forward(mj.matrix(ids.map((id) => [id])));
model.backward(mj.matrix([[tokenizer.getPadId()]]));
```
