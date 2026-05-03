/**
 * Raw-text MemoryBank dataset loader for BPETokenizer.
 *
 * This file intentionally does NOT define token IDs for inputs.
 * It uses the repository BPETokenizer to encode each turn text.
 */

import { readFileSync } from "fs";
import Matrix from "../../src/matrix";
import mj from "../../src/math";
import BPETokenizer from "../../src/tokenizer/bpe";

export interface BpeMemoryTurn {
  op: "STORE" | "UPDATE" | "QUERY" | "NOOP";
  text: string;
  key_text?: string;
  value_text?: string;
  old_value_text?: string;
}

export interface BpeMemoryQuery {
  turn_index: number;
  query_text: string;
  key_text: string;
  target_value_text: string;
  target_class: number;
}

export interface BpeMemoryEpisode {
  id: string;
  turns: BpeMemoryTurn[];
  episode_text: string;
  queries: BpeMemoryQuery[];
  num_turns: number;
  has_update: boolean;
  trace: Array<Record<string, string | number>>;
}

export function loadBpeMemoryEpisodes(path: string): BpeMemoryEpisode[] {
  const raw = readFileSync(path, "utf-8").trim();
  if (!raw) return [];
  if (raw.startsWith("[")) return JSON.parse(raw) as BpeMemoryEpisode[];
  return raw.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line) as BpeMemoryEpisode);
}

export function loadBpeTrainingCorpus(path: string): string[] {
  return readFileSync(path, "utf-8").split(/\r?\n/).map((x) => x.trim()).filter(Boolean);
}

export function trainMemoryBpeTokenizer(corpusPath: string, vocabSize = 256): BPETokenizer {
  const tokenizer = new BPETokenizer({ vocabSize, minFrequency: 2, preTokenizer: "char" });
  tokenizer.train(loadBpeTrainingCorpus(corpusPath));
  return tokenizer;
}

export function encodeTurnToMatrix(
  tokenizer: BPETokenizer,
  text: string,
  maxTurnTokens: number,
): Matrix {
  const ids = tokenizer.padSequence(tokenizer.encode(text), maxTurnTokens);
  return Matrix.fromFlat(Float32Array.from(ids), [maxTurnTokens, 1]);
}

export function encodeEpisodeTurns(
  tokenizer: BPETokenizer,
  episode: BpeMemoryEpisode,
  maxTurnTokens: number,
): Matrix[] {
  return episode.turns.map((turn) => encodeTurnToMatrix(tokenizer, turn.text, maxTurnTokens));
}

export function makeTargetMatrix(targetClass: number): Matrix {
  return mj.matrix([[targetClass]]);
}

export function isQueryTurn(turn: BpeMemoryTurn): boolean {
  return turn.op === "QUERY";
}

export function getQueryForTurn(episode: BpeMemoryEpisode, turnIndex: number): BpeMemoryQuery | undefined {
  return episode.queries.find((q) => q.turn_index === turnIndex);
}
