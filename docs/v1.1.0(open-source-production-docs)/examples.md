# Examples

## 1) Matrix usage
```ts
const a = mj.matrix([[1,2],[3,4]]);
const b = mj.ones([2,2]);
const c = mj.add(a,b);
c.print();
```

## 2) Basic math ops
```ts
const x = mj.dotProduct(mj.matrix([[1,0],[0,1]]), mj.matrix([[2],[3]]));
const y = mj.mul(x, 2);
```

## 3) Dense layer
```ts
const dense = new Dense({ units: 2, outputUnits: 3, activation: "relu", status: "input" });
const out = dense.forward(mj.matrix([[0.5],[0.2]]));
```

## 4) Embedding layer
```ts
const emb = new Embedding({ vocabSize: 100, embeddingDim: 16, padTokenId: 0 });
const embOut = emb.forward(mj.matrix([[1],[2],[0],[5]]));
```

## 5) LayerNormalization
```ts
const ln = new LayerNormalization({ units: 16, optimizer: "adam" });
const lnOut = ln.forward(embOut);
```

## 6) Dropout
```ts
const drop = new Dropout({ rate: 0.1, status: "train" });
const dropOut = drop.forward(lnOut);
```

## 7) PositionalEncoding
```ts
const pe = new PositionalEncoding({ dModel: 16, maxSeqLen: 64 });
const peOut = pe.forward(embOut);
```

## 8) MultiHeadAttention
```ts
const mha = new MultiHeadAttention({ units: 16, heads: 4, seqLen: 4, alpha: 0.001, status: "train" });
const attnOut = mha.forward(mj.matrix([[1,2,3,4],[0,1,0,1],[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4],[5,5,5,5],[6,6,6,6],[7,7,7,7],[8,8,8,8],[9,9,9,9],[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4],[5,5,5,5]]));
```

## 9) Sequential model
```ts
const model = new Sequential();
model.add(new Dense({ units: 2, outputUnits: 4, activation: "relu", status: "input" }));
model.add(new Dense({ units: 4, outputUnits: 1, activation: "sigmoid", status: "output", loss: "mse" }));
model.compile({ alpha: 0.01, optimizer: "adam", error: "mse" });
```

## 10) Transformers model
```ts
const tr = new Transformers({ units: 64, seqLen: 8, vocabSize: 300, heads: 8, alpha: 0.001, padTokenId: 0 });
tr.compile({ alpha: 0.001, optimizer: "adam", error: "softmaxCrossEntropy" });
```

## 11) Tokenizer train/update/encode/decode
```ts
const t = new BPETokenizer({ vocabSize: 100, minFrequency: 2 });
t.train(["aku suka ai", "aku suka ml"]);
t.update(["aku suka matematika"], 120);
const ids = t.encodeWithSpecial("aku suka ai");
const text = t.decode(ids);
```

## 12) compile + fit + forward + backward
```ts
model.fit(X, Y, 50, (loss) => console.log(loss));
model.forward(X[0]);
model.backward(Y[0]);
```

## 13) Save/load model
```ts
model.save("./model.json");
model.load("./model.json");
```

## 14) Resize vocab + fine-tune
```ts
t.update(["token baru"], 180);
tr.resizeVocab(t.getVocabSize());
```

## 15) Inference
```ts
const logits = tr.predict(mj.matrix(paddedIds.map((id)=>[id])));
```
