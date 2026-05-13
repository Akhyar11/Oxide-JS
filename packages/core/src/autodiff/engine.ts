import Tape from "./index.js";

export type GradTape<T> = Tape & { result: T };

class Engine {
  private activeTape: Tape | null = null;

  startTape(): Tape {
    const tape = new Tape();
    tape.watch();
    this.activeTape = tape;
    return tape;
  }

  endTape() {
    if (this.activeTape) {
      this.activeTape.stop();
    }
    this.activeTape = null;
  }

  get tape(): Tape | null {
    return this.activeTape;
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
}

export const engine = new Engine();
