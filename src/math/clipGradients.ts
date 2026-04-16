import Matrix from "../matrix";
import { isNativeAvailable, clipGradientsNative } from "./rust_backend";

/**
 * Membatasi nilai matrix (gradient clipping) secara in-place
 * @param a Matrix
 * @param limit Batas maksimum/minimum (abs)
 */
export default function clipGradients(a: Matrix, limit: number): void {
  if (isNativeAvailable()) {
    clipGradientsNative(a._data, limit);
  } else {
    const data = a._data;
    for (let i = 0; i < data.length; i++) {
        if (data[i] > limit) data[i] = limit;
        else if (data[i] < -limit) data[i] = -limit;
    }
  }
}
