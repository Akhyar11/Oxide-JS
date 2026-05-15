import { runAdaptiveMemoryRNNCorrectnessSuite } from "./adaptiveMemoryRNN.test.ts";
import { runAttentionPoolingCorrectnessSuite } from "./attentionPooling.test.ts";
import { runCustomModuleCorrectnessSuite } from "./customModule.test.ts";
import { runEpisodeTrainerCorrectnessSuite } from "./episodeTrainer.test.ts";
import { runModuleLayerCompatibilitySuite } from "./module.layerCompatibility.test.ts";
import { runEmbeddingTrainableCorrectnessSuite } from "./embedding.trainable.test.ts";
import { runLSTMForgetBiasCorrectnessSuite } from "./lstm.forgetBias.test.ts";
import { runLossScalingCorrectnessSuite } from "./loss.scaling.test.ts";
import { runModelArchitectureCorrectnessSuite } from "./model.architecture.test.ts";
import { runRecurrentPoolingCorrectnessSuite } from "./recurrent.pooling.test.ts";
import { runRecurrentLearningCorrectnessSuite } from "./rnn.learning.test.ts";
import { runTokenizerMultilingualCorrectnessSuite } from "./tokenizer.multilingual.test.ts";
import { runTransformerApiCorrectnessSuite } from "./transformers.api.test.ts";
import { runTransformerLearningCorrectnessSuite } from "./transformers.learning.test.ts";
import { runMemoryBankCorrectnessSuite } from "./memoryBank.test.ts";
import { runMemoryBankRetrievalSuite } from "./memoryBank.retrieval.test.ts";
import { runAutoDiffGradientSuite } from "./autodiff.gradient.test.ts";
import { fileURLToPath } from "url";

export function runCorrectnessSuite(): void {
  console.log("\n✅ Running Correctness Suite...");
  runAutoDiffGradientSuite();
  runAdaptiveMemoryRNNCorrectnessSuite();
  runAttentionPoolingCorrectnessSuite();
  runCustomModuleCorrectnessSuite();
  runEpisodeTrainerCorrectnessSuite();
  runModuleLayerCompatibilitySuite();
  runEmbeddingTrainableCorrectnessSuite();
  runLSTMForgetBiasCorrectnessSuite();
  runLossScalingCorrectnessSuite();
  runModelArchitectureCorrectnessSuite();
  runRecurrentPoolingCorrectnessSuite();
  runTokenizerMultilingualCorrectnessSuite();
  runRecurrentLearningCorrectnessSuite();
  runTransformerApiCorrectnessSuite();
  runTransformerLearningCorrectnessSuite();
  runMemoryBankCorrectnessSuite();
  runMemoryBankRetrievalSuite();
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === (process.argv[1]);

if (isMain) {
  runCorrectnessSuite();
}
