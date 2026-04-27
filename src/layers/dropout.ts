import { StatusLayer } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";
import { MemoryManager, WorkspaceConfig } from "../utils/memory";

export default class Dropout {
  name: string = "dropout layer";
  rate: number;
  status: StatusLayer;
  private training: boolean = false;

  inputShape: [number, number] = [0, 0];
  outputShape: [number, number] = [0, 0];
  params: number = 0;
  memoryConfig: WorkspaceConfig = {};
  private outputBufferData: any = new Float32Array(0);
  private maskData: any = new Float32Array(0);
  private outputBuffer: Matrix;
  private mask: Matrix;
  private evalBuffers: Record<string, any> = {};

  constructor({ rate = 0.5, status = "input", memoryConfig = {} }: { rate?: number; status?: StatusLayer; memoryConfig?: WorkspaceConfig }) {
    this.rate = rate;
    this.status = status;
    this.memoryConfig = memoryConfig;
    this.applyStatusTraining(status);
    this.outputBuffer = mj.matrix([]);
    this.mask = mj.matrix([]);
  }

  save() {
    return {
      name: this.name,
      status: this.status,
      rate: this.rate,
    };
  }

  load({ rate, status }: { rate: number; status: StatusLayer }) {
    this.rate = rate;
    this.status = status;
    this.applyStatusTraining(status);
  }

  forward(x: Matrix, options?: { workspace?: "train" | "eval" }): Matrix {
    this.inputShape = [x._shape[0], x._shape[1]];
    this.outputShape = [x._shape[0], x._shape[1]];

    // Hanya lakukan dropout JIKA statusnya adalah 'train'
    // Jika 'test' atau status lain, kembalikan input tanpa modifikasi
    if (!this.training || this.rate === 0) {
      return x;
    }

    const required = x._data.length;
    this.ensureForwardBuffers(required, x._shape, options?.workspace);

    const data = this.outputBufferData;
    const maskData = this.maskData;
    const xData = x._data;

    const scale = 1 / (1 - this.rate);
    for (let i = 0; i < required; i++) {
      if (Math.random() >= this.rate) {
        maskData[i] = scale;
        data[i] = xData[i] * scale;
      } else {
        maskData[i] = 0;
        data[i] = 0;
      }
    }

    return this.outputBuffer;
  }

  private ensureForwardBuffers(size: number, shape: [number, number], workspace: "train" | "eval" = "train") {
    if (workspace === "eval") {
      this.evalBuffers.outputData = MemoryManager.ensureCapacity(this.evalBuffers.outputData || new Float32Array(0), size, this.memoryConfig) as any;
      this.evalBuffers.maskData = MemoryManager.ensureCapacity(this.evalBuffers.maskData || new Float32Array(0), size, this.memoryConfig) as any;
      this.outputBufferData = this.evalBuffers.outputData;
      this.maskData = this.evalBuffers.maskData;
    } else {
      this.outputBufferData = MemoryManager.ensureCapacity(this.outputBufferData, size, this.memoryConfig) as any;
      this.maskData = MemoryManager.ensureCapacity(this.maskData, size, this.memoryConfig) as any;
    }

    this.outputBuffer = Matrix.fromFlat(this.outputBufferData.subarray(0, size) as any, shape);
    this.mask = Matrix.fromFlat(this.maskData.subarray(0, size) as any, shape);
  }

  releaseWorkspace(): void {
    this.evalBuffers = {};
    this.outputBufferData = new Float32Array(0);
    this.maskData = new Float32Array(0);
    this.outputBuffer = mj.matrix([]);
    this.mask = mj.matrix([]);
  }

  dispose(): void {
    this.releaseWorkspace();
  }

  backward(y: Matrix, err: Matrix): Matrix {
    if (!this.training || this.rate === 0) {
      return err;
    }
    return mj.mul(err, this.mask);
  }

  setTrainingMode(training: boolean): void {
    this.training = training;
    this.status = training ? "train" : "test";
  }

  isTraining(): boolean {
    return this.training;
  }

  private applyStatusTraining(status: StatusLayer): void {
    if (status === "train") this.training = true;
    else this.training = false;
  }
}
