# Tutorial Beginner

## Tujuan
Membuat model sederhana dari nol: matrix -> sequential -> train -> infer.

## Step 1: operasi matrix dasar
```ts
import mj from "../src/math";
const a = mj.matrix([[1,2],[3,4]]);
const b = mj.matrix([[5,6],[7,8]]);
console.log(mj.dotProduct(a,b)._value);
```

## Step 2: bangun model Sequential
```ts
import { Sequential } from "../src/models";
import { Dense } from "../src/layers";

const model = new Sequential();
model.add(new Dense({ units: 2, outputUnits: 4, activation: "relu", status: "input" }));
model.add(new Dense({ units: 4, outputUnits: 1, activation: "sigmoid", status: "output", loss: "mse" }));
model.compile({ alpha: 0.01, optimizer: "adam", error: "mse" });
```

## Step 3: train data XOR
```ts
const X = [mj.matrix([[0],[0]]), mj.matrix([[0],[1]]), mj.matrix([[1],[0]]), mj.matrix([[1],[1]])];
const Y = [mj.matrix([[0]]), mj.matrix([[1]]), mj.matrix([[1]]), mj.matrix([[0]])];
model.fit(X, Y, 200, (loss)=>console.log(loss));
```

## Step 4: inference
```ts
const pred = model.predict(mj.matrix([[1],[0]]));
console.log(pred._value);
```

## Step 5: save/load
```ts
model.save("./xor-model.json");
model.load("./xor-model.json");
```
