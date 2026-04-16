import mj from "../src/math";
import Matrix from "../src/matrix";
import { isNativeAvailable } from "../src/math/rust_backend";

const SCALE = 2; // Perbesar matriks untuk membedakan performa
const ITERS = 10;

function makeMatrix(rows: number, cols: number): Matrix {
  const data = new Float64Array(rows * cols);
  for (let i = 0; i < data.length; i++) data[i] = Math.random();
  return Matrix.fromFlat(data, [rows, cols]);
}

async function runBenchmark() {
  console.log("=== Rust vs JS Math Benchmark ===");
  console.log(`Native Available: ${isNativeAvailable()}`);
  
  const r = 256 * SCALE;
  const c = 256 * SCALE;
  const a = makeMatrix(r, c);
  const b = makeMatrix(c, r);

  console.log(`Matrix size: [${r}x${c}] * [${c}x${r}]`);

  // Benchmark Dot Product
  console.log("\n--- Dot Product ---");
  
  // Warmup
  for(let i=0; i<3; i++) mj.dotProduct(a, b);

  const startJs = performance.now();
  for(let i=0; i<ITERS; i++) {
    // Paksa gunakan JS dengan cara mematikan native check sementara jika perlu, 
    // tapi karena kita ingin benchmark real-world, kita bandingkan manual jika isNativeAvailable.
    mj.dotProduct(a, b);
  }
  const endJs = performance.now();
  console.log(`JS/Native (Auto): ${(endJs - startJs) / ITERS} ms per op`);

  if (isNativeAvailable()) {
    // Tambahkan pembanding manual jika benar-benar ingin melihat perbandingannya
    // Tapi di kode kita sudah otomatis switch.
  } else {
    console.log("Note: Native module not loaded, currently running pure JS.");
  }

  // Benchmark Addition
  console.log("\n--- Addition ---");
  const a2 = makeMatrix(r, c);
  const b2 = makeMatrix(r, c);
  const startAdd = performance.now();
  for(let i=0; i<ITERS * 10; i++) mj.add(a2, b2);
  const endAdd = performance.now();
  console.log(`Avg time: ${(endAdd - startAdd) / (ITERS * 10)} ms per op`);
}

runBenchmark().catch(console.error);
