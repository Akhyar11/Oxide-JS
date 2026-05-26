/**
 * Poisson Encoder
 * 
 * Mengonversi nilai input kontinu (umumnya di-normalize antara 0.0 - 1.0)
 * menjadi spike biner berdasarkan probabilitas (Poisson process).
 * Semakin tinggi nilai input, semakin besar probabilitas menghasilkan spike.
 */
export class PoissonEncoder {
  public maxFiringRate: number; // Skala probabilitas maksimum (0.0 - 1.0)

  constructor(maxFiringRate: number = 1.0) {
    if (maxFiringRate <= 0.0 || maxFiringRate > 1.0) {
      throw new Error("maxFiringRate harus berada di antara 0.0 (eksklusif) dan 1.0 (inklusif)");
    }
    this.maxFiringRate = maxFiringRate;
  }

  /**
   * Encode satu nilai desimal menjadi satu probabilitas spike (1 atau 0).
   * @param value Nilai intensitas input (0.0 sampai 1.0 disarankan)
   * @returns 1 jika spike, 0 jika tidak
   */
  public encodeValue(value: number): number {
    const probability = Math.max(0.0, Math.min(1.0, value)) * this.maxFiringRate;
    return Math.random() < probability ? 1 : 0;
  }

  /**
   * Encode sebuah array input menjadi array spike untuk SATU time-step.
   * Cocok digunakan sebelum memasukkan (inject) spike ke `SpikingNetwork`.
   * @param values Array input (misalnya array piksel)
   * @returns Float32Array atau Uint8Array yang berisi spike
   */
  public encodeArray(values: number[] | Float32Array): Uint8Array {
    const spikes = new Uint8Array(values.length);
    for (let i = 0; i < values.length; i++) {
      spikes[i] = this.encodeValue(values[i]);
    }
    return spikes;
  }

  /**
   * Mensimulasikan encoding sebuah nilai konstan ke dalam beberapa time-step.
   * Mengembalikan sekuens spike (kereta spike / spike train).
   * @param value Nilai intensitas
   * @param timeSteps Jumlah step waktu
   */
  public generateSpikeTrain(value: number, timeSteps: number): Uint8Array {
    const train = new Uint8Array(timeSteps);
    for (let t = 0; t < timeSteps; t++) {
      train[t] = this.encodeValue(value);
    }
    return train;
  }
}
