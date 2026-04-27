import mj from "../math";
import Matrix from "../matrix";
import { StatusLayer } from "../@types/type";
import { MemoryManager, WorkspaceConfig } from "../utils/memory";
/**
 * Flatten Layer: Melebur matriks multi-dimensi/2D menjadi vector 1 dimensi.
 * Biasanya digunakan sebelum layer Dense di ujung ekor CNN atau Self-Attention.
 */
export default class Flatten {
  name = "flatten layer";
  status: StatusLayer;
  loss = 0;
  params = 0; // Flatten tidak punya bobot
  inputShape: [number, number] = [0, 0];
  outputShape: [number, number] = [0, 0];
  memoryConfig: WorkspaceConfig;

  private outputData: any = new Float32Array(0);
  private evalBuffers: Record<string, any> = {};
  private backwardData: any = new Float32Array(0);
  private output: Matrix;
  private backwardBuffer: Matrix;

  constructor({ status = "norm", memoryConfig = {} }: { status?: StatusLayer; memoryConfig?: WorkspaceConfig } = {}) {
    this.status = status;
    this.memoryConfig = memoryConfig;
    this.output = mj.matrix([]);
    this.backwardBuffer = mj.matrix([]);
  }
  
  forward(x: Matrix, options?: { workspace?: "train" | "eval" }): Matrix {
    this.inputShape = [x._shape[0], x._shape[1]];
    const n = x._shape[0] * x._shape[1];
    this.outputShape = [n, 1];
    
    this.ensureForwardBuffers(n, options?.workspace);
    this.output._data.set(x._data);
    return this.output;
  }
  
  backward(y: Matrix, err: Matrix): Matrix {
    this.ensureBackwardBuffers(this.inputShape[0] * this.inputShape[1]);
    this.backwardBuffer._data.set(err._data);
    return this.backwardBuffer;
  }
  
  resetLoss(): void {
    this.loss = 0;
  }
  
  save() { 
    return { name: this.name, status: this.status }; 
  }
  
  load(): void { }
  
  compile(): void { } // Kosong karena tidak ada bobot (weights)

  private ensureForwardBuffers(n: number, workspace: "train" | "eval" = "train"): void {
    if (workspace === "eval") {
      this.evalBuffers.outputData = MemoryManager.ensureCapacity(this.evalBuffers.outputData || new Float32Array(0), n, this.memoryConfig) as any;
      this.outputData = this.evalBuffers.outputData;
    } else {
      this.outputData = MemoryManager.ensureCapacity(this.outputData, n, this.memoryConfig) as any;
    }
    this.output = Matrix.fromFlat(this.outputData.subarray(0, n) as any, [n, 1]);
  }

  private ensureBackwardBuffers(n: number): void {
    this.backwardData = MemoryManager.ensureCapacity(this.backwardData, n, this.memoryConfig) as any;
    this.backwardBuffer = Matrix.fromFlat(this.backwardData.subarray(0, n) as any, this.inputShape);
  }

  releaseWorkspace(): void {
    this.evalBuffers = {};
    this.outputData = new Float32Array(0);
    this.backwardData = new Float32Array(0);
    this.output = mj.matrix([]);
    this.backwardBuffer = mj.matrix([]);
  }

  dispose(): void {
    this.releaseWorkspace();
  }
}
