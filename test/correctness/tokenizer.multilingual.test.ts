import { BPETokenizer, scriptAwarePreTokenizer, unicodeGraphemePreTokenizer } from "@oxide-js/core";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}

function assertEqualArray(actual: string[], expected: string[], message: string): void {
  const sameLength = actual.length === expected.length;
  const sameValues = sameLength && actual.every((value, index) => value === expected[index]);
  if (!sameValues) {
    throw new Error(`${message}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

export function runTokenizerMultilingualCorrectnessSuite(): void {
  assertEqualArray(scriptAwarePreTokenizer("hello world"), ["hello", "world"], "script-aware Latin");
  assert(scriptAwarePreTokenizer("مرحبا بالعالم").length > 0, "script-aware Arabic should produce tokens");
  assert(scriptAwarePreTokenizer("こんにちは世界").length > 0, "script-aware Japanese should produce tokens");
  assert(scriptAwarePreTokenizer("你好世界").length > 0, "script-aware Mandarin should produce tokens");
  assert(scriptAwarePreTokenizer("ภาษาไทย").length > 0, "script-aware Thai should produce tokens");
  assert(scriptAwarePreTokenizer("한국어테스트").length > 0, "script-aware Korean should produce tokens");

  if (typeof (Intl as typeof Intl & { Segmenter?: unknown }).Segmenter !== "undefined") {
    assert(unicodeGraphemePreTokenizer("👨‍👩‍👧‍👦").length === 1, "unicode-grapheme should keep emoji ZWJ family together");
  } else {
    assert(unicodeGraphemePreTokenizer("👨‍👩‍👧‍👦").length > 0, "unicode-grapheme fallback should produce tokens");
  }

  const mathTokens = scriptAwarePreTokenizer("x² + y² = z²");
  assert(mathTokens.includes("+"), "script-aware math should contain +");
  assert(mathTokens.includes("="), "script-aware math should contain =");

  assertEqualArray(scriptAwarePreTokenizer("ꦱꦺꦴꦥꦺꦴ"), ["ꦱꦺꦴ", "ꦥꦺꦴ"], "script-aware Javanese");
  assert(scriptAwarePreTokenizer("hello ꦱꦺꦴꦥꦺꦴ 😊 你好").length > 0, "script-aware mixed text should produce tokens");

  const tokenizer = new BPETokenizer({
    vocabSize: 300,
    preTokenizer: "script-aware",
  });

  tokenizer.train([
    "hello world",
    "مرحبا بالعالم",
    "こんにちは世界",
    "你好世界",
    "ภาษาไทย",
    "한국어테스트",
    "ꦱꦺꦴꦥꦺꦴ",
    "x² + y² = z²",
    "hello ꦱꦺꦴꦥꦺꦴ 😊 你好",
  ]);

  const ids = tokenizer.encode("hello ꦱꦺꦴꦥꦺꦴ 😊 你好");
  const decoded = tokenizer.decode(ids);
  assert(ids.length > 0, "script-aware BPE encode should produce ids");
  assert(decoded.length > 0, "script-aware BPE decode should produce text");

  const savePath = path.join(os.tmpdir(), `ml-v1-script-aware-tokenizer-${process.pid}.json`);
  tokenizer.save(savePath);
  const saved = JSON.parse(fs.readFileSync(savePath, "utf-8")) as { config?: { preTokenizer?: string } };
  assert(saved.config?.preTokenizer === "script-aware", "save should persist built-in preTokenizer name");
  const loaded = BPETokenizer.load(savePath);
  assert(loaded.encode("hello ꦱꦺꦴ").length > 0, "load should restore built-in preTokenizer");
  fs.unlinkSync(savePath);

  const trainingTokenizer = new BPETokenizer({
    vocabSize: 128,
    minFrequency: 1,
    preTokenizer: "unicode-grapheme",
  });
  trainingTokenizer.train([
    "belajar model",
    "belajar tokenizer",
    "belajar subword",
    "belajar belajar",
  ]);

  const wholeWordId = trainingTokenizer.getTokenId("▁belajar");
  assert(wholeWordId !== undefined, "training tokenizer should learn whole-word token");

  const normalIds = trainingTokenizer.encode("belajar");
  assert(normalIds.length === 1, "normal encode should use whole-word token when available");

  const forcedLookupIds = trainingTokenizer.encodeForTraining("belajar", {
    fullWordLookupProbability: 1,
    random: () => 0,
  });
  assert(
    forcedLookupIds.length === 1 && forcedLookupIds[0] === normalIds[0],
    "encodeForTraining should preserve whole-word lookup when random branch enables it"
  );

  const forcedSubwordIds = trainingTokenizer.encodeForTraining("belajar", {
    fullWordLookupProbability: 0,
    random: () => 0.999,
  });
  assert(forcedSubwordIds.length > 1, "encodeForTraining should split into subwords when whole-word lookup is skipped");
  assert(trainingTokenizer.decode(forcedSubwordIds) === "belajar", "training-mode split should remain decodable");

  console.log("=== Tokenizer Multilingual Correctness ===");
  console.table([
    { check: "script-aware multilingual pre-tokenizers", status: "pass" },
    { check: "unicode grapheme emoji segmentation", status: "pass" },
    { check: "script-aware BPE integration", status: "pass" },
    { check: "preTokenizer save/load metadata", status: "pass" },
    { check: "training encode whole-word vs subword branching", status: "pass" },
  ]);
}

