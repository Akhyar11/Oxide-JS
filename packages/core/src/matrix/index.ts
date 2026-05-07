import { MatrixCollection, MatrixFlatData, MatrixShape, matrix2d } from "../@types/type.js";
import { isNativeAvailable, addInPlaceNative, subInPlaceNative, mulInPlaceNative } from "../math/rust_backend.js";

/**
 * Matrix class yang dioptimasi dengan Float32Array
 * 
 * Internal storage menggunakan Float32Array (flat, contiguous memory)
 * untuk performa yang jauh lebih baik dibanding number[][]
 * 
 * Akses: m._data[i * cols + j] (flat index)
 * Backward compatible: m._value[i][j] masih bisa digunakan (via getter/setter)
 */
export default class Matrix {
  /** Internal flat storage — GUNAKAN INI untuk operasi cepat */
  _data: Float32Array;
  _shape: MatrixShape;
  _version: number = 0; // Pelacakan untuk modifikasi in-place
  grad: Matrix | null = null;
  name?: string;

  constructor({ array }: { array: matrix2d }) {
    const rows = array.length;
    const cols = rows > 0 && array[0] !== undefined ? array[0].length : 0;
    this._shape = [rows, cols];
    this._data = new Float32Array(rows * cols);

    // Copy from 2D array ke flat Float32Array
    for (let i = 0; i < rows; i++) {
      const offset = i * cols;
      for (let j = 0; j < cols; j++) {
        this._data[offset + j] = array[i][j];
      }
    }
  }

  /**
   * Buat Matrix langsung dari Float32Arraydatar.
   * Data akan dinormalisasi ke Float32Array agar konsisten dengan backend native.
   */
  static fromFlat(data: MatrixFlatData | ArrayLike<number>, shape: MatrixShape): Matrix {
    const m = Object.create(Matrix.prototype) as Matrix;
    m._data = data instanceof Float32Array ? data : new Float32Array(data);
    m._shape = shape;
    m._version = 0;
    m.grad = null;
    return m;
  }

  /**
   * Backward compatible getter — konversi _data ke number[][]
   * PERINGATAN: Ini membuat array baru setiap kali dipanggil!
   * Untuk performa, gunakan _data langsung: _data[i * cols + j]
   */
  get _value(): matrix2d {
    const [rows, cols] = this._shape;
    const arr: matrix2d = new Array(rows);
    for (let i = 0; i < rows; i++) {
      const row = new Array(cols);
      const offset = i * cols;
      for (let j = 0; j < cols; j++) {
        row[j] = this._data[offset + j];
      }
      arr[i] = row;
    }
    return arr;
  }

  /**
   * Backward compatible setter — copy dari number[][] ke _data
   */
  set _value(arr: matrix2d) {
    const rows = arr.length;
    const cols = rows > 0 && arr[0] !== undefined ? arr[0].length : 0;
    this._shape = [rows, cols];
    this._data = new Float32Array(rows * cols);
    for (let i = 0; i < rows; i++) {
      const offset = i * cols;
      for (let j = 0; j < cols; j++) {
        this._data[offset + j] = arr[i][j];
      }
    }
  }

  /** Akses elemen cepat */
  get(i: number, j: number): number {
    return this._data[i * this._shape[1] + j];
  }

  /** Set elemen cepat */
  set(i: number, j: number, val: number): void {
    this._data[i * this._shape[1] + j] = val;
  }

  print(): void {
    const [rows, cols] = this._shape;
    const arr: number[][] = new Array(rows);
    for (let i = 0; i < rows; i++) {
      arr[i] = new Array(cols);
      const off = i * cols;
      for (let j = 0; j < cols; j++) {
        arr[i][j] = parseFloat(this._data[off + j].toFixed(2));
      }
    }
    console.table(arr);
  }

  /**
   * Ekstrak kolom sebagai Float32Array baru
   */
  getCol(colIndex: number): Float32Array {
    const [rows, cols] = this._shape;
    const col = new Float32Array(rows);
    for (let i = 0; i < rows; i++) {
      col[i] = this._data[i * cols + colIndex];
    }
    return col;
  }

  /**
   * Set data kolom dari typed array
   */
  setCol(colIndex: number, data: MatrixFlatData): void {
    const [rows, cols] = this._shape;
    for (let i = 0; i < rows; i++) {
      this._data[i * cols + colIndex] = data[i];
    }
  }

  map(func: (value: number) => number) {
    for (let i = 0; i < this._data.length; i++) {
      this._data[i] = func(this._data[i]);
    }
  }

  add(a: MatrixCollection) {
    if (typeof a === "number") {
      for (let i = 0; i < this._data.length; i++) this._data[i] += a;
    } else if (a instanceof Matrix) {
      if (this._shape[0] !== a._shape[0] || this._shape[1] !== a._shape[1]) {
        throw new Error(`bentuk dari a harus sama dengan matrix ${this._shape} != ${a._shape}`);
      }
      for (let i = 0; i < this._data.length; i++) this._data[i] += a._data[i];
    }
  }

  sub(a: MatrixCollection) {
    if (typeof a === "number") {
      for (let i = 0; i < this._data.length; i++) this._data[i] -= a;
    } else if (a instanceof Matrix) {
      if (this._shape[0] !== a._shape[0] || this._shape[1] !== a._shape[1]) {
        throw new Error(`bentuk dari a harus sama dengan matrix ${this._shape} != ${a._shape}`);
      }
      for (let i = 0; i < this._data.length; i++) this._data[i] -= a._data[i];
    }
  }

  mul(a: MatrixCollection) {
    if (typeof a === "number") {
      for (let i = 0; i < this._data.length; i++) this._data[i] *= a;
    } else if (a instanceof Matrix) {
      if (this._shape[0] !== a._shape[0] || this._shape[1] !== a._shape[1]) {
        throw new Error(`bentuk dari a harus sama dengan matrix ${this._shape} != ${a._shape}`);
      }
      for (let i = 0; i < this._data.length; i++) this._data[i] *= a._data[i];
    }
  }

  div(a: MatrixCollection) {
    if (typeof a === "number") {
      if (a === 0) throw new Error("Pembagian dengan nol (scalar = 0) tidak diizinkan");
      for (let i = 0; i < this._data.length; i++) this._data[i] /= a;
    } else if (a instanceof Matrix) {
      if (this._shape[0] !== a._shape[0] || this._shape[1] !== a._shape[1]) {
        throw new Error(`bentuk dari a harus sama dengan matrix ${this._shape} != ${a._shape}`);
      }
      for (let i = 0; i < this._data.length; i++) {
        if (a._data[i] === 0) throw new Error(`Pembagian dengan nol pada flat index [${i}]`);
        this._data[i] /= a._data[i];
      }
    }
  }

  flatten() {
    const n = this._data.length;
    this._shape = [1, n];
    // _data sudah flat, tidak perlu copy
  }

  reshape(shape: MatrixShape) {
    if (shape[0] * shape[1] !== this._shape[0] * this._shape[1]) {
      throw new Error(
        `Panjang dari shape baru tidak sama dengan yang lama ${this._shape[0] * this._shape[1]}!=${shape[0] * shape[1]}`
      );
    }
    this._shape = shape;
    // _data sudah flat dan urut, reshape hanya ubah interpretasi shape
  }

  /**
   * Copy data dari matrix lain ke matrix ini (tanpa alokasi baru)
   */
  copyFrom(other: Matrix): void {
    if (this._data.length !== other._data.length) {
      throw new Error("Ukuran matrix tidak sama untuk copy");
    }
    this._data.set(other._data);
  }

  /**
   * Clone matrix ini ke objek baru
   */
  clone(): Matrix {
    const newData = new Float32Array(this._data);
    return Matrix.fromFlat(newData, [...this._shape]);
  }

  clearGrad(): void {
    this.grad = null;
  }

  /**
   * Penjumlahan In-Place: Menjumlahkan matrix lain ke matrix ini
   */
  addInPlace(other: Matrix | number): void {
    if (typeof other === "number") {
      for (let i = 0; i < this._data.length; i++) this._data[i] += other;
    } else {
      if (this._data.length !== other._data.length) {
        throw new Error(`addInPlace: length mismatch ${this._data.length} !== ${other._data.length}`);
      }
      if (isNativeAvailable()) {
        addInPlaceNative(this._data as any, other._data as any);
      } else {
        for (let i = 0; i < this._data.length; i++) this._data[i] += other._data[i];
      }
    }
  }

  /**
   * Pengurangan In-Place: Mengurangi matrix ini dengan matrix lain
   */
  subInPlace(other: Matrix | number): void {
    if (typeof other === "number") {
      for (let i = 0; i < this._data.length; i++) this._data[i] -= other;
    } else {
      if (this._data.length !== other._data.length) {
        throw new Error(`subInPlace: length mismatch ${this._data.length} !== ${other._data.length}`);
      }
      if (isNativeAvailable()) {
        subInPlaceNative(this._data as any, other._data as any);
      } else {
        for (let i = 0; i < this._data.length; i++) this._data[i] -= other._data[i];
      }
    }
  }

  /**
   * Perkalian Elemen-per-Elemen In-Place
   */
  mulInPlace(other: Matrix | number): void {
    if (typeof other === "number") {
      for (let i = 0; i < this._data.length; i++) this._data[i] *= other;
    } else {
      if (this._data.length !== other._data.length) {
        throw new Error(`mulInPlace: length mismatch ${this._data.length} !== ${other._data.length}`);
      }
      if (isNativeAvailable()) {
        mulInPlaceNative(this._data as any, other._data as any);
      } else {
        for (let i = 0; i < this._data.length; i++) this._data[i] *= other._data[i];
      }
    }
  }
}
