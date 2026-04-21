import mj from "../src/math";
import { Dense } from "../src/layers";
import { Sequential } from "../src/models";

const model = new Sequential({
  layers: [
    new Dense({ units: 2, outputUnits: 4, activation: "relu", status: "input" }),
    new Dense({ units: 4, outputUnits: 1, activation: "sigmoid", status: "output", loss: "mse" }),
  ],
});

model.compile({ alpha: 0.01, optimizer: "adam", error: "mse" });

const X = [mj.matrix([[0], [0]]), mj.matrix([[0], [1]]), mj.matrix([[1], [0]]), mj.matrix([[1], [1]])];
const y = [mj.matrix([[0]]), mj.matrix([[1]]), mj.matrix([[1]]), mj.matrix([[0]])];

const result = model.fit(X, y, 100, {
  batchSize: 2,
  validationSplit: 0.25,
  earlyStoppingPatience: 10,
  verbose: true,
  onEpochEnd: (epoch, loss, valLoss) => {
    console.log(`epoch=${epoch} loss=${loss} valLoss=${valLoss}`);
  },
});

console.log("bestEpoch:", result.bestEpoch);
console.log("bestLoss:", result.bestLoss);
console.log("history:", result.history);
