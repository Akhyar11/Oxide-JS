import Matrix from "../matrix/index.js";
import Tape, { GradientFunc, TapeRecordOptions } from "./index.js";

export type GradTape<T> = Tape & { result: T };

class Engine {
  private tapeStack: Tape[] = [];

  startTape(): Tape {
    const tape = new Tape();
    tape.watch();
    this.tapeStack.push(tape);
    return tape;
  }

  endTape() {
    const tape = this.tapeStack.pop();
    if (tape) {
      tape.stop();
    }
  }

  get tape(): Tape | null {
    return this.tapeStack[this.tapeStack.length - 1] ?? null;
  }

  /**
   * Catat operasi ke tape aktif jika ada.
   */
  record(
    inputs: Matrix[],
    outputs: Matrix[],
    backward: GradientFunc,
    options?: TapeRecordOptions
  ): void {
    this.tape?.record(inputs, outputs, backward, options);
  }

  /**
   * Helper untuk menjalankan fungsi dalam konteks tape
   */
  grad<T>(fn: () => T): GradTape<T> {
    const tape = this.startTape();
    let result!: T;
    try {
      result = fn();
    } finally {
      this.endTape();
    }
    return Object.assign(tape, { result });
  }

  /**
   * Jalankan blok kode tanpa perekaman gradien global
   */
  noGrad<T>(fn: () => T): T {
    const activeTape = this.tape;
    if (activeTape) {
      return activeTape.noGrad(fn);
    }
    return fn();
  }
}

export const engine = new Engine();
