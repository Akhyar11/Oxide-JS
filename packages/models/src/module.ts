import { Matrix } from "@oxide-js/core";

export type ModuleCompileConfig = Record<string, unknown>;

export interface TrainableLeaf {
  getParams?: () => Matrix[];
  update?: (alpha?: number) => void;
  compile?: (config: ModuleCompileConfig) => void;
  setTrainingMode?: (training: boolean) => void;
}

export interface ForwardLeaf extends TrainableLeaf {
  forward: (...args: any[]) => any;
}

function isObjectLike(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isModuleInstance(value: unknown): value is Module {
  return value instanceof Module;
}

function isTrainableLeaf(value: unknown): value is TrainableLeaf {
  return isObjectLike(value) && (
    typeof value.getParams === "function" ||
    typeof value.update === "function" ||
    typeof value.compile === "function" ||
    typeof value.setTrainingMode === "function"
  );
}

export default abstract class Module {
  private readonly registeredParameters = new Map<string, Matrix>();
  private isTrainingMode = true;

  abstract forward(...args: any[]): any;

  protected registerParameter(name: string, value: Matrix): Matrix {
    this.registeredParameters.set(name, value);
    return value;
  }

  parameters(): Matrix[] {
    const params: Matrix[] = [];
    const seenParams = new Set<Matrix>();
    const visited = new WeakSet<object>();

    const pushParam = (param: Matrix) => {
      if (!seenParams.has(param)) {
        seenParams.add(param);
        params.push(param);
      }
    };

    const visit = (value: unknown) => {
      if (!isObjectLike(value)) return;
      if (visited.has(value)) return;
      visited.add(value);

      if (isModuleInstance(value)) {
        value.registeredParameters.forEach(pushParam);
        for (const key of Object.keys(value)) {
          visit((value as Record<string, unknown>)[key]);
        }
        return;
      }

      if (isTrainableLeaf(value) && typeof value.getParams === "function") {
        for (const param of value.getParams()) {
          pushParam(param);
        }
      }

      if (Array.isArray(value)) {
        for (const item of value) visit(item);
        return;
      }

      for (const key of Object.keys(value)) {
        visit(value[key]);
      }
    };

    visit(this);
    return params;
  }

  zeroGrad(): this {
    for (const param of this.parameters()) {
      param.clearGrad();
    }
    return this;
  }

  compile(config: ModuleCompileConfig): this {
    const visited = new WeakSet<object>();

    const visit = (value: unknown) => {
      if (!isObjectLike(value)) return;
      if (visited.has(value)) return;
      visited.add(value);

      if (value !== this && isModuleInstance(value)) {
        value.compile(config);
        return;
      }

      if (isTrainableLeaf(value) && typeof value.compile === "function") {
        value.compile(config);
      }

      if (Array.isArray(value)) {
        for (const item of value) visit(item);
        return;
      }

      for (const key of Object.keys(value)) {
        visit(value[key]);
      }
    };

    for (const key of Object.keys(this)) {
      visit((this as Record<string, unknown>)[key]);
    }
    return this;
  }

  step(alpha?: number): this {
    const visited = new WeakSet<object>();

    const visit = (value: unknown) => {
      if (!isObjectLike(value)) return;
      if (visited.has(value)) return;
      visited.add(value);

      if (value !== this && isModuleInstance(value)) {
        value.step(alpha);
        return;
      }

      if (isTrainableLeaf(value) && typeof value.update === "function") {
        value.update(alpha);
      }

      if (Array.isArray(value)) {
        for (const item of value) visit(item);
        return;
      }

      for (const key of Object.keys(value)) {
        visit(value[key]);
      }
    };

    for (const key of Object.keys(this)) {
      visit((this as Record<string, unknown>)[key]);
    }
    return this;
  }

  train(): this {
    this.isTrainingMode = true;
    const visited = new WeakSet<object>();

    const visit = (value: unknown) => {
      if (!isObjectLike(value)) return;
      if (visited.has(value)) return;
      visited.add(value);

      if (value !== this && isModuleInstance(value)) {
        value.train();
        return;
      }

      if (isTrainableLeaf(value) && typeof value.setTrainingMode === "function") {
        value.setTrainingMode(true);
      }

      if (Array.isArray(value)) {
        for (const item of value) visit(item);
        return;
      }

      for (const key of Object.keys(value)) {
        visit(value[key]);
      }
    };

    for (const key of Object.keys(this)) {
      visit((this as Record<string, unknown>)[key]);
    }
    return this;
  }

  eval(): this {
    this.isTrainingMode = false;
    const visited = new WeakSet<object>();

    const visit = (value: unknown) => {
      if (!isObjectLike(value)) return;
      if (visited.has(value)) return;
      visited.add(value);

      if (value !== this && isModuleInstance(value)) {
        value.eval();
        return;
      }

      if (isTrainableLeaf(value) && typeof value.setTrainingMode === "function") {
        value.setTrainingMode(false);
      }

      if (Array.isArray(value)) {
        for (const item of value) visit(item);
        return;
      }

      for (const key of Object.keys(value)) {
        visit(value[key]);
      }
    };

    for (const key of Object.keys(this)) {
      visit((this as Record<string, unknown>)[key]);
    }
    return this;
  }

  predict<T>(...args: any[]): T {
    const wasTraining = this.isTrainingMode;
    this.eval();
    try {
      return this.forward(...args) as T;
    } finally {
      if (wasTraining) this.train();
    }
  }
}

export class ModuleList<TItem = unknown> extends Module implements Iterable<TItem> {
  readonly items: TItem[];

  constructor(items: TItem[] = []) {
    super();
    this.items = items;
  }

  get length(): number {
    return this.items.length;
  }

  at(index: number): TItem | undefined {
    return this.items[index];
  }

  push(...items: TItem[]): number {
    return this.items.push(...items);
  }

  forward<T>(input: T): T {
    return input;
  }

  [Symbol.iterator](): Iterator<TItem> {
    return this.items[Symbol.iterator]();
  }
}

export class SequentialBlock extends Module {
  readonly layers: Array<Module | ForwardLeaf>;

  constructor(layers: Array<Module | ForwardLeaf> = []) {
    super();
    this.layers = layers;
  }

  add(layer: Module | ForwardLeaf): this {
    this.layers.push(layer);
    return this;
  }

  forward(input: any): any {
    let current = input;
    for (const layer of this.layers) {
      current = layer.forward(current);
    }
    return current;
  }
}
