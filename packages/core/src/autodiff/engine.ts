import Tape from "./index.js";

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
  grad(fn: () => any): Tape {
    const tape = this.startTape();
    fn();
    tape.stop();
    return tape;
  }
}

export const engine = new Engine();
