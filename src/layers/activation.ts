import { ActivationType, Cost, StatusLayer } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";
import setActivation from "../utils/setActivation";
import setLoss from "../utils/setLoss";
import { MemoryManager, WorkspaceConfig } from "../utils/memory";

export default class Activation {
  name: string = "activation layer";
  inputShape = [null, null];
  outputShape = [null, null];
  params = 0;
  loss = 0;
  activation: Function;
  lossFunc: Function;
  status: StatusLayer;
  activationName: ActivationType;
  lossName: Cost;
  private resultData: any = new Float32Array(0);
  private dResultData: any = new Float32Array(0);
  private evalBuffers: Record<string, any> = {};
  private result: Matrix;
  private dResult: Matrix;
  private sumLoss: number = 0;
  private index: number = 0;
  memoryConfig: WorkspaceConfig = {};
  constructor({
    activation,
    status = "input",
    loss = "mse",
  }: {
    activation: ActivationType;
    status?: StatusLayer;
    loss?: Cost;
  }) {
    this.activation = setActivation(activation);
    this.activationName = activation;
    this.status = status;
    this.lossFunc = setLoss(loss);
    this.lossName = loss;
    this.result = mj.matrix([]);
    this.dResult = mj.matrix([]);
  }

  save() {
    const data = {
      name: this.name,
      activation: this.activationName,
      status: this.status,
      loss: this.lossName,
    };
    return data;
  }

  load({
    activation,
    loss,
    status,
  }: {
    activation: ActivationType;
    loss: Cost;
    status: StatusLayer;
  }) {
    this.activation = setActivation(activation);
    this.lossFunc = setLoss(loss);
    this.activationName = activation;
    this.lossName = loss;
    this.status = status;
  }

  forward(x: Matrix, options?: { workspace?: "train" | "eval" }) {
    const required = x._data.length;
    this.ensureForwardBuffers(required, x._shape, options?.workspace);

    this.activation(x, { result: this.result, dResult: this.dResult });
    return this.result;
  }

  private ensureForwardBuffers(size: number, shape: [number, number], workspace: "train" | "eval" = "train") {
    if (workspace === "eval") {
      this.evalBuffers.resultData = MemoryManager.ensureCapacity(this.evalBuffers.resultData || new Float32Array(0), size, this.memoryConfig) as any;
      this.evalBuffers.dResultData = MemoryManager.ensureCapacity(this.evalBuffers.dResultData || new Float32Array(0), size, this.memoryConfig) as any;
      this.resultData = this.evalBuffers.resultData;
      this.dResultData = this.evalBuffers.dResultData;
    } else {
      this.resultData = MemoryManager.ensureCapacity(this.resultData, size, this.memoryConfig) as any;
      this.dResultData = MemoryManager.ensureCapacity(this.dResultData, size, this.memoryConfig) as any;
    }

    // Wrap matrices around the buffers
    this.result = Matrix.fromFlat(this.resultData.subarray(0, size) as any, shape);
    this.dResult = Matrix.fromFlat(this.dResultData.subarray(0, size) as any, shape);
  }

  backward(y: Matrix, err: Matrix) {
    let e = err;
    let loss;
    if (this.status === "output") {
      [loss, e] = this.lossFunc(y, this.result);
      this.index++;
      this.sumLoss += loss;
      this.loss = this.sumLoss / this.index;
    }
    const errActivation = mj.mul(e, this.dResult);
    return errActivation;
  }
}
