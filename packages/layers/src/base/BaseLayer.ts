import { Matrix, mj, setActivation, setLoss, setOptimizer } from "@oxide-js/core";

export interface LayerConfig {
  name?: string;
  trainable?: boolean;
}

export abstract class BaseLayer {
  public name: string;
  public trainable: boolean;
  public dtype: "float32" | "float64" = "float32";
  public training: boolean = true;

  protected parameters: Map<string, { value: Matrix; trainable: boolean; logicalShape?: number[] }> = new Map();

  public isBuilt: boolean = false;

  public inputShape: number[] = [];
  public outputShape: number[] = [];

  constructor(config?: LayerConfig) {
    this.name = config?.name || this.constructor.name;
    this.trainable = config?.trainable ?? true;
  }

  /**
   * Mengubah status mode layer menjadi training mode.
   */
  public train(): void {
    this.training = true;
  }

  /**
   * Mengubah status mode layer menjadi evaluation/inference mode.
   */
  public eval(): void {
    this.training = false;
  }

  /**
   * Keras-style getter untuk mendapatkan seluruh matriks bobot layer.
   */
  public get weights(): Matrix[] {
    return Array.from(this.parameters.values()).map(p => p.value);
  }

  /**
   * Keras-style getter untuk bobot yang trainable.
   */
  public get trainableWeights(): Matrix[] {
    if (!this.trainable) return [];
    return Array.from(this.parameters.values())
      .filter(p => p.trainable)
      .map(p => p.value);
  }

  /**
   * Keras-style getter untuk bobot yang non-trainable (jika layer di-freeze atau parameter non-trainable).
   */
  public get nonTrainableWeights(): Matrix[] {
    if (!this.trainable) {
      return Array.from(this.parameters.values()).map(p => p.value);
    }
    return Array.from(this.parameters.values())
      .filter(p => !p.trainable)
      .map(p => p.value);
  }

  /**
   * Menghitung output shape berdasarkan input shape secara deterministik.
   * Sangat berguna sebelum alokasi memori penuh dilakukan.
   * Harus di-override oleh sub-class.
   */
  public abstract computeOutputShape(inputShape: number[]): number[];

  /**
   * Inisialisasi parameter (Weight/Bias) layer berdasarkan shape input.
   * Harus di-override oleh sub-class untuk menentukan outputShape & membuat weight.
   * 
   * @param inputShape Dimensi data input
   */
  public build(inputShape: number[]): void {
    this.inputShape = inputShape;
    this.outputShape = this.computeOutputShape(inputShape);
    this.isBuilt = true;
  }

  /**
   * Helper utility untuk inisialisasi bobot (Weight Initialization) standar Keras.
   * Mendukung 'he_normal'/'he_uniform', 'glorot_normal'/'glorot_uniform' (xavier), 'zeros', 'ones', 'random'.
   */
  protected createInitializer(initializerName: string, shape: [number, number]): Matrix {
    const init = initializerName.toLowerCase().replace(/_/g, "");
    switch (init) {
      case "zeros":
      case "zero":
        return mj.zeros(shape);
      case "ones":
      case "one":
        return mj.ones(shape);
      case "henormal":
      case "he":
        return mj.he(shape);
      case "heuniform":
        // TODO: Implement uniform variant in core, falling back to standard He Normal for now
        return mj.he(shape);
      case "glorotnormal":
      case "xavier":
        return mj.xavier(shape);
      case "glorotuniform":
        // TODO: Implement uniform variant in core, falling back to standard Xavier for now
        return mj.xavier(shape);
      case "random":
      case "randomuniform":
      case "randomnormal":
        return mj.random(shape);
      default:
        console.warn(`[BaseLayer] Initializer '${initializerName}' tidak dikenal. Menggunakan default 'glorot_normal' (Xavier).`);
        return mj.xavier(shape);
    }
  }

  /**
   * Helper utility untuk mencari fungsi aktivasi berdasarkan string nama (misal "relu", "sigmoid").
   */
  protected resolveActivation(activation: string, options?: { alpha?: number; row?: boolean }): any {
    return setActivation({
      activation: activation as any,
      alpha: options?.alpha,
      row: options?.row
    });
  }

  /**
   * Helper utility untuk mencari fungsi loss berdasarkan string nama (misal "mse", "crossEntropy").
   */
  protected resolveLoss(loss: string): any {
    return setLoss(loss as any);
  }

  /**
   * Helper utility untuk menginisialisasi optimizer berdasarkan string nama dan parameter.
   */
  protected resolveOptimizer(optimizer: string, shape: [number, number], alpha: number): any {
    return setOptimizer(optimizer as any, shape, alpha);
  }

  /**
   * Validasi kecocokan shape input dengan inputShape yang terdaftar pada build()
   * Mengabaikan dimensi batch (dimensi pertama index 0).
   */
  protected validateInputShape(inputs: Matrix): void {
    if (!this.isBuilt) return;

    const expected = this.inputShape.slice(1);
    const actual = inputs._shape.slice(1);

    if (expected.join(",") !== actual.join(",")) {
      throw new Error(
        `[${this.name}] Input shape mismatch. Expected [*, ${expected.join(", ")}], got [${inputs._shape.join(", ")}]`
      );
    }
  }

  /**
   * Core perhitungan matematika layer (Forward Pass).
   * 
   * @param inputs Matrix input
   * @param isTraining Flag apakah sedang fase training (misal untuk Dropout/BatchNorm)
   * @returns Matrix output
   */
  protected abstract compute(inputs: Matrix, isTraining?: boolean): Matrix;



  /**
   * Wrapper utama untuk Forward Pass.
   * Akan memanggil build() secara otomatis jika belum di-build.
   * 
   * @param inputs Matrix input
   * @param isTraining Flag training (default: default training state dari layer)
   * @returns Matrix output
   */
  public forward(inputs: Matrix, isTraining: boolean = this.training): Matrix {
    if (!this.isBuilt) {
      this.build(inputs._shape);
    } else {
      this.validateInputShape(inputs);
    }

    return this.compute(inputs, isTraining);
  }

  /**
   * Registrasi parameter (seperti 'weight' atau 'bias') ke dalam layer.
   */
  protected addParameter(name: string, param: Matrix, trainable = true, logicalShape?: number[]): Matrix {
    param.requiresGrad = trainable;
    this.parameters.set(name, { value: param, trainable, logicalShape });
    return param;
  }

  /**
   * Mendapatkan parameter spesifik berdasarkan nama.
   */
  public getParameter(name: string): Matrix | undefined {
    return this.parameters.get(name)?.value;
  }

  /**
   * Mengambil semua parameter yang dapat di-train (untuk Optimizer).
   * Jika layer di-freeze (trainable = false), return array kosong.
   */
  public getTrainableParameters(): Matrix[] {
    if (!this.trainable) return [];
    return Array.from(this.parameters.values())
      .filter(p => p.trainable)
      .map(p => p.value);
  }

  /**
   * Mengambil semua parameter layer (untuk Save Model / Export).
   */
  public getAllParameters(): Record<string, Matrix> {
    const record: Record<string, Matrix> = {};
    for (const [key, param] of this.parameters.entries()) {
      record[key] = param.value;
    }
    return record;
  }

  /**
   * Load bobot layer dari luar (untuk Load Model).
   */
  public setParameters(weights: Record<string, Matrix>): void {
    for (const [key, matrix] of Object.entries(weights)) {
      if (this.parameters.has(key)) {
        const entry = this.parameters.get(key)!;
        // Validasi shape tidak berubah
        if (entry.value._shape.join(',') !== matrix._shape.join(',')) {
          throw new Error(`[${this.name}] Shape parameter '${key}' tidak cocok. Ekspektasi: ${entry.value._shape}, Didapat: ${matrix._shape}`);
        }
        entry.value = matrix;
      } else {
        console.warn(`[${this.name}] Parameter '${key}' tidak dikenali di layer ini.`);
      }
    }
  }

  /**
   * Menghitung total parameter di dalam layer ini.
   */
  public countParams(): number {
    let total = 0;
    for (const param of this.parameters.values()) {
      total += param.value._shape.reduce((a, b) => a * b, 1);
    }
    return total;
  }

  /**
   * Clear semua gradient dari parameter layer ini (digunakan sebelum backward pass).
   */
  public clearGradients(): void {
    for (const param of this.parameters.values()) {
      param.value.clearGrad();
    }
  }

  /**
   * Konfigurasi spesifik layer (seperti units, activation).
   * Keras format meletakkan ini di dalam properti "config".
   */
  public getConfig(): Record<string, any> {
    return {
      name: this.name,
      trainable: this.trainable,
      dtype: this.dtype
    };
  }

  /**
   * Mengembalikan arsitektur layer dalam format topology Keras / TensorFlow.js.
   */
  public getKerasConfig() {
    return {
      class_name: this.constructor.name,
      config: this.getConfig()
    };
  }

  /**
   * Mengambil bobot untuk weightsManifest di format Keras/TF.js.
   * Berisi metadata shape (logicalShape asli jika ada) dan buffer Float32Array mentah.
   */
  public getWeights() {
    const weightsData = [];
    for (const [key, param] of this.parameters.entries()) {
      weightsData.push({
        name: `${this.name}/${key}`,
        shape: param.logicalShape ?? [...param.value._shape],
        physicalShape: [...param.value._shape],
        dtype: this.dtype,
        data: param.value._data // Float32Array mentah
      });
    }
    return weightsData;
  }

  /**
   * Helper utilitas untuk mengonversi bentuk dimensi apa pun (1D, 2D, 3D, atau 4D) 
   * menjadi bentuk 2D [rows, cols] yang kompatibel dengan kelas Matrix Oxide-JS.
   * Menyediakan berbagai mode flattening untuk disesuaikan dengan tipe layer (misal Dense vs Conv2D).
   */
  protected to2DShape(
    shape: number[],
    mode: "keepFirst" | "keepLast" | "flatRow" = "keepFirst"
  ): [number, number] {
    if (shape.length === 0) return [1, 1];
    if (shape.length === 1) return [1, shape[0] ?? 1];
    if (shape.length === 2) return [shape[0] ?? 1, shape[1] ?? 1];

    if (mode === "keepLast") {
      const rows = shape.slice(0, -1).reduce((a, b) => a * b, 1);
      const cols = shape[shape.length - 1] ?? 1;
      return [rows, cols];
    }

    if (mode === "flatRow") {
      const total = shape.reduce((a, b) => a * b, 1);
      return [1, total];
    }

    // Default: keepFirst
    const rows = shape[0] ?? 1;
    const cols = shape.slice(1).reduce((a, b) => a * b, 1);
    return [rows, cols];
  }

  /**
   * Memuat bobot dari buffer eksternal (proses Load Keras weights).
   */
  public setWeights(weightsData: { name: string, shape: number[], data: Float32Array }[]): void {
    for (const w of weightsData) {
      const key = w.name.split('/').pop() || w.name;
      
      if (this.parameters.has(key)) {
        const entry = this.parameters.get(key)!;
        
        // Validasi kesamaan ukuran total elemen
        const targetSize = entry.value._shape[0] * entry.value._shape[1];
        const incomingSize = w.shape.reduce((a, b) => a * b, 1);
        if (targetSize !== incomingSize) {
          throw new Error(
            `[${this.name}] Size mismatch untuk '${w.name}'. Ekspektasi: ${entry.value._shape.join("x")} (${targetSize} elemen), Diterima: ${w.shape.join("x")} (${incomingSize} elemen)`
          );
        }
        
        entry.value._data.set(w.data);
        entry.logicalShape = w.shape;
      } else {
        // Force inject jika belum terdefinisi
        const targetShape = this.to2DShape(w.shape);
        this.parameters.set(key, { 
          value: Matrix.fromFlat(w.data, targetShape), 
          trainable: true,
          logicalShape: w.shape
        });
      }
    }
    this.isBuilt = true;
  }

  /**
   * Mengembalikan objek informasi ringkas mengenai layer ini.
   * Digunakan oleh model container untuk mencetak ringkasan tabular secara terpadu.
   */
  public getSummaryInfo() {
    return {
      name: this.name,
      type: this.constructor.name,
      outputShape: this.isBuilt ? `[${this.outputShape.join(", ")}]` : "multiple/unbuilt",
      paramCount: this.countParams(),
      trainable: this.trainable
    };
  }

  /**
   * Mencetak ringkasan (summary) visual untuk layer ini ke console secara langsung.
   */
  public summary(): void {
    const info = this.getSummaryInfo();
    const divider = "=".repeat(45);
    console.log(divider);
    console.log(` Layer Name  : ${info.name}`);
    console.log(` Layer Type  : ${info.type}`);
    console.log(` Output Shape: ${info.outputShape}`);
    console.log(` Trainable   : ${info.trainable ? "Yes" : "No"}`);
    console.log(` Total Params: ${info.paramCount.toLocaleString()}`);
    console.log(divider);
  }
}

