import { runAdaptiveMemoryRNNCorrectnessSuite } from "./adaptiveMemoryRNN.test";
import { runEmbeddingTrainableCorrectnessSuite } from "./embedding.trainable.test";
import { runLossScalingCorrectnessSuite } from "./loss.scaling.test";
import { runModelArchitectureCorrectnessSuite } from "./model.architecture.test";
import { runRecurrentLearningCorrectnessSuite } from "./rnn.learning.test";
import { runTokenizerMultilingualCorrectnessSuite } from "./tokenizer.multilingual.test";
import { runTransformerApiCorrectnessSuite } from "./transformers.api.test";
import { runTransformerLearningCorrectnessSuite } from "./transformers.learning.test";

export function runCorrectnessSuite(): void {
  runAdaptiveMemoryRNNCorrectnessSuite();
  runEmbeddingTrainableCorrectnessSuite();
  runLossScalingCorrectnessSuite();
  runModelArchitectureCorrectnessSuite();
  runTokenizerMultilingualCorrectnessSuite();
  runRecurrentLearningCorrectnessSuite();
  runTransformerApiCorrectnessSuite();
  runTransformerLearningCorrectnessSuite();
}

if (require.main === module) {
  runCorrectnessSuite();
}
