import mj from "../math";
import { softmaxBackward } from "../activation";
import {
  ActivationType,
  Cost,
  Optimzier,
  OptimzierType,
  StatusLayer,
  matrix2d,
} from "../@types/type";
import setActivation from "../utils/setActivation";
import Matrix from "../matrix";
import setOptimizer from "../utils/setOptimizer";
import setLoss from "../utils/setLoss";

interface DenseLayers {
  units: number;
  outputUnits: number;
  alpha?: number;
  loss?: Cost;
  activation?: ActivationType;
  optimizer?: Optimzier;
  status?: StatusLayer;
}

export interface CompileDenseLayers {
  alpha?: number;
  optimizer?: Optimzier;
  error?: Cost;
}

export default class Dense {
  name = "dense layer";
  units: number;
  outputUnits: number;
  alpha: number;
  loss: number = 0;
  params: number;
  inputShape: [number, number];
  outputShape: [number, number];
  status: StatusLayer;
  bias: Matrix;
  weight: Matrix;
  private sumLoss: number = 0;
  private index: number = 0;
  private optimizerWeight: OptimzierType;
  private optimizerBias: OptimzierType;
  private activationName: ActivationType;
  private optimizerName: Optimzier;
  private lossName: Cost;
  private input: Matrix = mj.matrix([]);
  private dInput: Matrix = mj.matrix([]);
  private result: Matrix = mj.matrix([]);
  private lossFunc: Function;
  private activation: (a: Matrix) => [Matrix, Matrix];
  
  // Pre-allocated buffers for speed (REUSE)
  private z: Matrix;
  private errWeightBuffer: Matrix;

  constructor({
    units,
    outputUnits,
    activation = "linear",
    optimizer = "sgd",
    status = "input",
    alpha = 0.1,
    loss = "mse",
  }: DenseLayers) {
    this.units = units;
    this.outputUnits = outputUnits;
    this.inputShape = [units, 1];
    this.outputShape = [outputUnits, 1];
    
    // Gunakan Xavier initialization untuk stabilitas lebih baik
    this.weight = mj.xavier([outputUnits, units]);
    this.bias = mj.zeros([outputUnits, 1]);
    
    this.z = mj.zeros([outputUnits, 1]); // Buffer for dotProduct + bias
    this.errWeightBuffer = mj.zeros([outputUnits, units]); // Buffer for errWeight
    this.activation = setActivation(activation);
    this.activationName = activation;
    this.optimizerName = optimizer;
    this.lossName = loss;
    this.status = status;
    this.optimizerWeight = setOptimizer(optimizer, this.weight._shape, 1e-5);
    this.optimizerBias = setOptimizer(optimizer, this.bias._shape, 1e-5);
    this.lossFunc = setLoss(loss);
    this.alpha = alpha;
    this.params = outputUnits * units + outputUnits;
  }

  save() {
    return {
      name: this.name,
      status: this.status,
      units: this.units,
      outputUnits: this.outputUnits,
      activation: this.activationName,
      optimizer: this.optimizerName,
      loss: this.lossName,
      weight: this.weight._value,
      bias: this.bias._value,
    };
  }

  load(weight: matrix2d, bias: matrix2d): void {
    this.weight._value = weight;
    this.weight._shape = [weight.length, weight[0]?.length ?? 0];
    this.bias._value = bias;
    this.bias._shape = [bias.length, bias[0]?.length ?? 0];
    this.units = this.weight._shape[1];
    this.outputUnits = this.weight._shape[0];
    this.params = this.outputUnits * this.units + this.outputUnits;
    this.z = mj.zeros([this.outputUnits, 1]);
    this.errWeightBuffer = mj.zeros([this.outputUnits, this.units]);
  }

  compile({
    alpha = 0.1,
    optimizer = "sgd",
    error = "mse",
  }: CompileDenseLayers): void {
    this.alpha = alpha;
    this.optimizerWeight = setOptimizer(optimizer, this.weight._shape, 1e-5);
    this.optimizerBias = setOptimizer(optimizer, this.bias._shape, 1e-5);
    this.lossFunc = setLoss(error);
    this.optimizerName = optimizer;
    this.lossName = error;
  }

  forward(x: Matrix): Matrix {
    const [units, seqLen] = x._shape;
    this.input = x;
    
    // Periksa apakah buffer z perlu di-resize (misal untuk sequence length > 1)
    if (this.z._shape[0] !== this.outputUnits || this.z._shape[1] !== seqLen) {
        this.z = mj.zeros([this.outputUnits, seqLen]);
    }
    
    // 1. MatMul weight * input -> simpan di this.z 
    // [outputUnits, units] * [units, seqLen] -> [outputUnits, seqLen]
    mj.dotProduct(this.weight, this.input, this.z);
    
    // 2. Tambahkan bias secara broadcast (per kolom)
    const [rows, cols] = this.z._shape;
    const zData = this.z._data;
    const bData = this.bias._data;
    for (let j = 0; j < cols; j++) {
        for (let i = 0; i < rows; i++) {
            zData[i * cols + j] += bData[i];
        }
    }

    // 3. Activation
    const [result, dResult] = this.activation(this.z);
    this.dInput = dResult;
    this.result = result;
    return result;
  }

  backward(y: Matrix, err: Matrix): Matrix {
    const [rows, seqLen] = this.result._shape;
    let e: Matrix = mj.matrix([]);
    let lossValue = 0;
    if (this.status === "output") {
      // Safety check: Jika target adalah sparse index (1x1) tapi output bukan 1x1, 
      // maka MSE pasti akan error. Paksa gunakan SoftmaxCrossEntropy.
      if (y._shape[0] === 1 && y._shape[1] === 1 && this.result._shape[0] > 1) {
          const SoftmaxCrossEntropy = require("../cost/softmaxCrossEntropy").default;
          [lossValue, e] = SoftmaxCrossEntropy(y, this.result);
      } else {
          [lossValue, e] = this.lossFunc(y, this.result);
      }
      this.index++;
      this.sumLoss += lossValue;
      this.loss = this.sumLoss / this.index;
    } else {
      e = err;
    }

    let errActivation: Matrix;
    if (this.activationName === "softmax") {
      errActivation = softmaxBackward(this.result, e, false);
    } else {
      errActivation = mj.mul(e, this.dInput);
    }

    // 1. Hitung gradien weight
    // [outputUnits, seqLen] * [seqLen, units] -> [outputUnits, units]
    const gradWeight = mj.dotProduct(errActivation, this.input, this.errWeightBuffer, false, true);
    
    // 2. Hitung gradien bias (Sum sepanjang sequence/kolom)
    const gbData = new Float64Array(this.outputUnits);
    const eaData = errActivation._data;
    for (let j = 0; j < seqLen; j++) {
        for (let i = 0; i < this.outputUnits; i++) {
            gbData[i] += eaData[i * seqLen + j];
        }
    }
    const gradBias = Matrix.fromFlat(gbData, [this.outputUnits, 1]);

    // [New] Gradient Clipping: Batasi nilai gradien agar tidak meledak
    this.clipGradients(gradWeight, 1.0);
    this.clipGradients(gradBias, 1.0);

    // 3. Dapatkan update dari optimizer
    const updateWeight = this.optimizerWeight.calculate(gradWeight, this.alpha);
    const updateBias = this.optimizerBias.calculate(gradBias, this.alpha);

    // 4. Update In-Place!
    this.weight.subInPlace(updateWeight);
    this.bias.subInPlace(updateBias);

    // 5. Hitung gradien ke layer sebelumnya
    // [units, outputUnits] * [outputUnits, seqLen] -> [units, seqLen]
    return mj.dotProduct(this.weight, errActivation, undefined, true, false);
  }

  private clipGradients(m: Matrix, limit: number) {
    const data = m._data;
    for (let i = 0; i < data.length; i++) {
        if (data[i] > limit) data[i] = limit;
        else if (data[i] < -limit) data[i] = -limit;
    }
  }

  /**
   * Resize output units (e.g., when vocab size increases)
   * @param newOutputUnits - New number of output units
   */
  resize(newOutputUnits: number): void {
    if (newOutputUnits <= this.outputUnits) return;

    console.log(`[Dense] Resizing output units: ${this.outputUnits} -> ${newOutputUnits}`);

    // 1. Resize weights [newOutputUnits, units]
    const newWeight = mj.random([newOutputUnits, this.units]);
    const oldWeightData = this.weight._data;
    const newWeightData = newWeight._data;

    for (let i = 0; i < this.outputUnits; i++) {
        for (let j = 0; j < this.units; j++) {
            newWeightData[i * this.units + j] = oldWeightData[i * this.units + j];
        }
    }

    // 2. Resize bias [newOutputUnits, 1]
    const newBias = mj.zeros([newOutputUnits, 1]);
    const oldBiasData = this.bias._data;
    const newBiasData = newBias._data;
    for (let i = 0; i < this.outputUnits; i++) {
        newBiasData[i] = oldBiasData[i];
    }

    // 3. Update state
    this.weight = newWeight;
    this.bias = newBias;
    this.outputUnits = newOutputUnits;
    this.outputShape = [newOutputUnits, 1];
    this.params = newOutputUnits * this.units + newOutputUnits;

    // 4. Re-allocate buffers
    this.z = mj.zeros([newOutputUnits, 1]);
    this.errWeightBuffer = mj.zeros([newOutputUnits, this.units]);

    // 5. Reset optimizer for new shape
    this.optimizerWeight = setOptimizer(this.optimizerName, this.weight._shape, 1e-5);
    this.optimizerBias = setOptimizer(this.optimizerName, this.bias._shape, 1e-5);
  }

  /** Reset akumulasi loss — panggil di awal setiap epoch */
  resetLoss(): void {
    this.sumLoss = 0;
    this.index = 0;
    this.loss = 0;
  }
}
