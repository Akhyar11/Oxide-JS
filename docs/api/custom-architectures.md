# Custom Architectures

Custom architectures are the recommended path when you want full control over topology, branching, residual paths, or mixed manual math.

## Import

```ts
import { EpisodeTrainer, Module, ModuleList, SequentialBlock, Trainer } from "@oxide-js/models";
import { Dense } from "@oxide-js/layers";
import { mj } from "@oxide-js/core";
```

## Overview

Instead of relying on a built-in model class such as `Sequential` or `Transformers`, you can define your own architecture by subclassing `Module`:

- store layers as class fields
- implement `forward(...)`
- use arbitrary JavaScript control flow and residual math
- train with `Trainer`, which uses the Gradient Tape

This is the primary architecture API for new work. High-level model classes remain available as legacy convenience wrappers.

## `Module`

`Module` is a recursive container for:

- layers with `getParams()` / `update()`
- nested `Module` instances
- arrays of layers or submodules

Built-in helpers:

- `parameters(): Matrix[]`
- `zeroGrad(): this`
- `compile(config): this`
- `step(alpha?): this`
- `train(): this`
- `eval(): this`
- `predict(...args)`

Ergonomic containers:

- `ModuleList<T>` for repeated trainable blocks that still participate in recursive parameter discovery
- `SequentialBlock` for lightweight sequential composition inside a larger custom `Module`

## `Trainer`

`Trainer` runs a generic supervised training loop over any `Module`.

```ts
const trainer = new Trainer(model, "mse");
const result = trainer.fit(X, y, 50, { batchSize: 8 });
```

It:

- batches `[rows, 1]` samples into `[rows, batchSize]`
- records the forward pass on the active tape
- injects the configured loss as a scalar autodiff node
- backpropagates through the custom graph
- calls `module.step()`

It also supports structured sample values:

- single `Matrix`
- arrays of `Matrix`
- objects whose leaves are `Matrix`

That means a custom `Module` can accept multi-input data and return multi-output predictions, as long as you provide a matching custom loss that returns structured gradients.

## `EpisodeTrainer`

`EpisodeTrainer` is the dedicated training loop for dynamic execution schedules such as:

- encoder runs once
- decoder runs multiple times
- loss is accumulated across steps
- backward happens once at the end of the episode

Use this when your architecture behaves more like a program or rollout than a single `forward(input) -> prediction` call.

```ts
const trainer = new EpisodeTrainer(model);

trainer.trainEpisode(inputEpisode, targetEpisode, ({ module, input, target }) => {
  const context = module.encode(input.source);
  let state = mj.zeros([hiddenUnits, 1]);
  let prev = input.start;
  let totalLoss = null;

  for (let step = 0; step < target._shape[0]; step++) {
    const decoded = module.decodeStep(context, prev, state);
    const stepTarget = mj.matrix([[target._data[step]]]);
    const stepLoss = mj.mean(mj.pow(mj.sub(decoded.prediction, stepTarget), 2));
    totalLoss = totalLoss ? mj.add(totalLoss, stepLoss) : stepLoss;
    prev = decoded.prediction;
    state = decoded.state;
  }

  return mj.mul(totalLoss, 1 / target._shape[0]);
});
```

For dataset-level training:

```ts
trainer.fit(episodesX, episodesY, 20, ({ module, input, target }) => {
  return module.runEpisode(input, target);
});
```

## Structured Input / Output

```ts
import { Module, SequentialBlock, Trainer } from "@oxide-js/models";
import { Dense } from "@oxide-js/layers";
import { mj, setLoss } from "@oxide-js/core";

class SiameseModel extends Module {
  left = new Dense({ units: 2, outputUnits: 4, activation: "relu", status: "input" });
  right = new Dense({ units: 2, outputUnits: 4, activation: "relu", status: "input" });
  shared = new SequentialBlock([
    new Dense({ units: 4, outputUnits: 4, activation: "relu", status: "train" }),
  ]);
  score = new Dense({ units: 4, outputUnits: 1, activation: "linear", status: "output", loss: "mse" });
  aux = new Dense({ units: 4, outputUnits: 1, activation: "linear", status: "output", loss: "mse" });

  forward(input) {
    const left = this.left.forward(input.left);
    const right = this.right.forward(input.right);
    const mixed = this.shared.forward(mj.add(left, right));
    return {
      score: this.score.forward(mixed),
      aux: this.aux.forward(mj.sub(left, right)),
    };
  }
}

const trainer = new Trainer(new SiameseModel(), (yTrue, yPred) => {
  const [scoreLoss, scoreGrad] = setLoss("mse")(yTrue.score, yPred.score);
  const [auxLoss, auxGrad] = setLoss("mse")(yTrue.aux, yPred.aux);
  return {
    loss: scoreLoss + auxLoss * 0.25,
    grads: {
      score: scoreGrad,
      aux: mj.mul(auxGrad, 0.25),
    },
  };
});
```

## Example

```ts
import { Module, ModuleList, SequentialBlock, Trainer } from "@oxide-js/models";
import { Dense } from "@oxide-js/layers";
import { mj } from "@oxide-js/core";

class ResidualMLP extends Module {
  input = new Dense({ units: 2, outputUnits: 8, activation: "relu", status: "input" });
  stem = new SequentialBlock([
    new Dense({ units: 8, outputUnits: 8, activation: "relu", status: "train" }),
  ]);
  blocks = new ModuleList([
    new Dense({ units: 8, outputUnits: 8, activation: "relu", status: "train" }),
    new Dense({ units: 8, outputUnits: 8, activation: "relu", status: "train" }),
  ]);
  output = new Dense({ units: 8, outputUnits: 1, activation: "linear", status: "output", loss: "mse" });

  forward(x) {
    let hidden = this.stem.forward(this.input.forward(x));
    for (const block of this.blocks) {
      hidden = mj.add(hidden, block.forward(hidden));
    }
    return this.output.forward(hidden);
  }
}

const model = new ResidualMLP();
model.compile({ alpha: 0.01, optimizer: "adam", error: "mse" });

const trainer = new Trainer(model, "mse");
trainer.fit(X, Y, 100, { batchSize: 4 });
```

## Positioning

- Use `Module` + `Trainer` for standard supervised custom architectures.
- Use `EpisodeTrainer` when execution spans multiple decoder steps or rollout-style state transitions before one backward pass.
- Use `ModuleList` when you want dynamic stacks, repeated blocks, or loops over trainable children.
- Use `SequentialBlock` when you want a small sequential subgraph inside a larger custom topology.
- Use `Sequential`, `RecurrentModel`, or `Transformers` only when you specifically want the existing legacy wrappers.
