import { runPerfBreakdown } from "./transformer_perf_breakdown";

type ScalingEntry = {
  numBlocks: number;
  perfMedian: {
    inferenceOnlyMsPerIter: number;
    forwardOnlyMsPerIter: number;
    backwardOnlyMsPerIter: number;
    trainingStepMsPerIter: number;
  };
  endToEndMedian: {
    msPerBatch: number;
    msPerSample: number;
    samplesPerSec: number;
  };
};

async function runTransformerNumBlocksScaling() {
  const numBlocksList = [2, 4, 6];
  const results: ScalingEntry[] = [];

  for (const numBlocks of numBlocksList) {
    const summary = await runPerfBreakdown({ numBlocks });
    results.push({
      numBlocks,
      perfMedian: summary.perfMedian,
      endToEndMedian: summary.endToEndMedian,
    });
  }

  console.log(JSON.stringify({ benchmark: "transformer_numblocks_scaling", results }, null, 2));
}

if (require.main === module) {
  runTransformerNumBlocksScaling().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}

export { runTransformerNumBlocksScaling };
