import * as fs from "fs";

type RawMathRecord = {
  instruction?: unknown;
  input?: unknown;
  output?: unknown;
  prompt?: unknown;
  response?: unknown;
};

function toNonEmptyString(value: unknown): string {
  if (typeof value !== "string") return "";
  return value.trim();
}

export function normalizeMathRecord(record: RawMathRecord): string | null {
  const instruction = toNonEmptyString(record.instruction);
  const input = toNonEmptyString(record.input);
  const output = toNonEmptyString(record.output);
  const prompt = toNonEmptyString(record.prompt);
  const response = toNonEmptyString(record.response);

  if (instruction.length > 0 && output.length > 0) {
    const lines = [`instruksi: ${instruction.toLowerCase()}`];
    if (input.length > 0) {
      lines.push(`input: ${input.toLowerCase()}`);
    }
    lines.push(`jawaban: ${output.toLowerCase()}`);
    return lines.join("\n");
  }

  if (prompt.length > 0 && response.length > 0) {
    return [
      "instruksi: jawab pertanyaan matematika berikut.",
      `input: ${prompt.toLowerCase()}`,
      `jawaban: ${response.toLowerCase()}`,
    ].join("\n");
  }

  return null;
}

export function recordsToCorpus(records: RawMathRecord[]): string[] {
  const corpus: string[] = [];
  for (const record of records) {
    const normalized = normalizeMathRecord(record);
    if (normalized) {
      corpus.push(normalized);
    }
  }
  return corpus;
}

export function buildChatPrompt(question: string): string {
  return [
    "instruksi: jawab pertanyaan matematika berikut.",
    `input: ${question.trim().toLowerCase()}`,
    "jawaban:",
  ].join("\n");
}

export function loadMathTrainingCorpus(datasetPath: string): string[] {
  const raw = fs.readFileSync(datasetPath, "utf-8");
  const parsed = JSON.parse(raw) as RawMathRecord[];
  return recordsToCorpus(parsed);
}
