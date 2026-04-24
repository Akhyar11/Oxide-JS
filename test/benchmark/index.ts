import { runAllSyntheticBaselineBenchmarks } from "./synthetic_baseline_benchmark";
import { runRecurrentBufferReuseBenchmarks } from "./recurrent_buffer_reuse";

export async function runBenchmarkSuite() {
  await runAllSyntheticBaselineBenchmarks();
  await runRecurrentBufferReuseBenchmarks();
}
