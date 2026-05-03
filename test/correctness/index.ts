import { runAdaptiveMemoryRNNCorrectnessSuite } from "./adaptiveMemoryRNN.test";
import { runEmbeddingTrainableCorrectnessSuite } from "./embedding.trainable.test";
import { runLSTMForgetBiasCorrectnessSuite } from "./lstm.forgetBias.test";
import { runLossScalingCorrectnessSuite } from "./loss.scaling.test";
import { runModelArchitectureCorrectnessSuite } from "./model.architecture.test";
import { runRecurrentPoolingCorrectnessSuite } from "./recurrent.pooling.test";
import { runRecurrentLearningCorrectnessSuite } from "./rnn.learning.test";
import { runTokenizerMultilingualCorrectnessSuite } from "./tokenizer.multilingual.test";
import { runTransformerApiCorrectnessSuite } from "./transformers.api.test";
import { runTransformerLearningCorrectnessSuite } from "./transformers.learning.test";
import { runMemoryBankCorrectnessSuite } from "./memoryBank.test";
import { runMemoryBankRetrievalSuite } from "./memoryBank.retrieval.test";

export function runCorrectnessSuite(): void {
  runAdaptiveMemoryRNNCorrectnessSuite();
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

if (require.main === module) {
  runCorrectnessSuite();
}
