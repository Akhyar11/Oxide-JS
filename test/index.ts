import { runBenchmarkSuite } from "./benchmark/index.ts";
import { runCorrectnessSuite } from "./correctness/index.ts";

async function main() {
  console.log("🚀 Starting Oxide-JS Test Suite...");
  runCorrectnessSuite();
  await runBenchmarkSuite();
}

main().catch((error) => {
  console.error("❌ Test Suite failed:", error);
  process.exit(1);
});
