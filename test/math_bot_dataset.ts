import { buildChatPrompt, normalizeMathRecord, recordsToCorpus } from "../project/math-bot/data";

let passed = 0;
let failed = 0;

function assert(condition: boolean, name: string) {
  if (condition) {
    console.log(`  PASS: ${name}`);
    passed++;
    return;
  }

  console.error(`  FAIL: ${name}`);
  failed++;
}

console.log("\n=== Math Bot Dataset ===");

const normalizedWithInput = normalizeMathRecord({
  instruction: "Hitung hasil penjumlahan.",
  input: "45 + 55",
  output: "100",
});

assert(
  normalizedWithInput === "instruksi: hitung hasil penjumlahan.\ninput: 45 + 55\njawaban: 100",
  "normalizes record with input",
);

const normalizedWithoutInput = normalizeMathRecord({
  instruction: "Berapakah 150 + 350?",
  input: "",
  output: "500",
});

assert(
  normalizedWithoutInput === "instruksi: berapakah 150 + 350?\njawaban: 500",
  "omits empty input line",
);

const corpus = recordsToCorpus([
  { instruction: "Hitung hasil penjumlahan.", input: "45 + 55", output: "100" },
  { instruction: "Berapakah 150 + 350?", input: "", output: "500" },
  { instruction: "invalid", input: "x", output: 42 as unknown as string },
]);

assert(corpus.length === 2, "skips invalid records");
assert(
  corpus[1] === "instruksi: berapakah 150 + 350?\njawaban: 500",
  "keeps valid normalized order",
);

const prompt = buildChatPrompt("berapa 12 x 12?");
assert(
  prompt === "instruksi: jawab pertanyaan matematika berikut.\ninput: berapa 12 x 12?\njawaban:",
  "builds chat prompt with answer prefix",
);

console.log(`\nPASSED: ${passed}  FAILED: ${failed}`);

if (failed > 0) {
  process.exit(1);
}
