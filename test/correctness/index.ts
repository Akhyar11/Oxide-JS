import { runAdaptiveMemoryRNNCorrectnessSuite } from "./adaptiveMemoryRNN.test";
import { runRecurrentLearningCorrectnessSuite } from "./rnn.learning.test";
import { runTokenizerMultilingualCorrectnessSuite } from "./tokenizer.multilingual.test";
import { runTransformerApiCorrectnessSuite } from "./transformers.api.test";
import { runTransformerLearningCorrectnessSuite } from "./transformers.learning.test";

export function runCorrectnessSuite(): void {
  runAdaptiveMemoryRNNCorrectnessSuite();
  runTokenizerMultilingualCorrectnessSuite();
  runRecurrentLearningCorrectnessSuite();
  runTransformerApiCorrectnessSuite();
  runTransformerLearningCorrectnessSuite();
}

if (require.main === module) {
  runCorrectnessSuite();
}
