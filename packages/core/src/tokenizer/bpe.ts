import * as fs from "node:fs";
import {
  BuiltInPreTokenizer,
  PreTokenizer,
  charPreTokenizer,
  isEmojiLike,
  isMathSymbol,
  isPunctuation,
  resolvePreTokenizer,
  unicodeGraphemePreTokenizer,
} from "./pretokenizers.js";

/**
 * BPE (Byte Pair Encoding) Tokenizer
 * 
 * Algoritma:
 * 1. Memulai dari level karakter individual
 * 2. Menghitung pasangan karakter yang paling sering muncul
 * 3. Menggabungkan pasangan tersebut menjadi token baru
 * 4. Mengulangi proses sampai vocabulary mencapai ukuran target
 * 
 * Keunggulan:
 * - Bisa menangani kata-kata baru (OOV / Out-Of-Vocabulary)
 * - Kata yang sering muncul jadi 1 token, kata langka dipecah jadi subword
 * - Efisien untuk vocabulary berukuran terbatas
 */

// Token khusus
const PAD_TOKEN = "<PAD>";
const UNK_TOKEN = "<UNK>";
const BOS_TOKEN = "<BOS>";  // Beginning of Sequence
const EOS_TOKEN = "<EOS>";  // End of Sequence
const WORD_BOUNDARY = "▁";  // Penanda awal kata (seperti SentencePiece)
const PAIR_SEPARATOR = "\0";

export type BPETokenizerOptions = {
  vocabSize?: number;
  minFrequency?: number;
  preTokenizer?: BuiltInPreTokenizer | PreTokenizer;
};

export type BPETrainingEncodeOptions = {
  fullWordLookupProbability?: number;
  random?: () => number;
};

export interface BPEConfig extends BPETokenizerOptions {
  vocabSize: number;        // Ukuran vocabulary target
  specialTokens?: string[]; // Token khusus tambahan
}

export type SerializedBPEConfig = Omit<BPEConfig, "preTokenizer"> & {
  preTokenizer?: BuiltInPreTokenizer | "custom";
};

export interface BPEVocabData {
  vocab: Record<string, number>;       // token → id
  merges: [string, string][];          // Daftar merge rules, urut dari pertama dipelajari
  config: SerializedBPEConfig;
}

type WordSymbols = { symbols: string[]; freq: number };

export default class BPETokenizer {
  private vocab: Map<string, number> = new Map();
  private reverseVocab: Map<number, string> = new Map();
  private merges: [string, string][] = [];
  private vocabSize: number;
  private minFrequency: number;
  private specialTokens: string[];
  private preTokenizer: PreTokenizer;
  private preTokenizerName: BuiltInPreTokenizer | "custom";
  private encodeCache: Map<string, number[]> = new Map();
  private readonly maxEncodeCacheSize = 8192;

  constructor(config: (BPETokenizerOptions & { specialTokens?: string[] }) = {}) {
    this.vocabSize = config.vocabSize ?? 1000;
    this.minFrequency = config.minFrequency ?? 2;
    this.preTokenizerName = typeof config.preTokenizer === "string" ? config.preTokenizer : config.preTokenizer ? "custom" : "char";
    this.preTokenizer = resolvePreTokenizer(config.preTokenizer ?? "char");
    this.specialTokens = Array.from(new Set([
      PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN,
      ...(config.specialTokens ?? [])
    ]));
  }

  /**
   * Melatih tokenizer dari corpus teks
   * @param texts - Array of strings sebagai training data
   */
  train(texts: string[]): void {
    // === STEP 1: Inisialisasi vocabulary dengan special tokens ===
    this.vocab.clear();
    this.merges = [];
    this.clearEncodeCache();
    let nextId = 0;

    for (const token of this.specialTokens) {
      this.vocab.set(token, nextId++);
    }

    this.sanitize(); // Clean existing state before training

    // === STEP 2: Tokenisasi awal dengan pre-tokenizer terpilih ===
    const wordFreq: Map<string, number> = new Map();
    for (const text of texts) {
      const words = this.preTokenize(text);
      for (const word of words) {
        if (word.length === 0) continue;
        const key = WORD_BOUNDARY + word; // Tambah boundary marker
        wordFreq.set(key, (wordFreq.get(key) ?? 0) + 1);
      }
    }

    const corpus: WordSymbols[] = [];

    for (const [word, freq] of wordFreq) {
      const chars = this.createInitialSymbols(word);
      corpus.push({ symbols: chars, freq });

      // Tambahkan setiap karakter ke vocab jika belum ada
      for (const char of chars) {
        if (!this.vocab.has(char)) {
          this.vocab.set(char, nextId++);
        }
      }
    }

    this.runBPE(corpus, nextId);
  }

  /**
   * Update tokenizer dengan data baru (Incremental Training)
   * Menambahkan karakter baru atau merge baru tanpa merubah ID lama.
   */
  update(texts: string[], newVocabSize?: number): void {
    if (newVocabSize && newVocabSize > this.vocabSize) {
      this.vocabSize = newVocabSize;
    }

    this.clearEncodeCache();
    this.sanitize(); // Clean existing state before update

    let nextId = 0;
    for (const id of this.vocab.values()) {
      if (id >= nextId) nextId = id + 1;
    }

    const wordFreq: Map<string, number> = new Map();
    for (const text of texts) {
      const words = this.preTokenize(text);
      for (const word of words) {
        if (word.length === 0) continue;
        const key = WORD_BOUNDARY + word;
        wordFreq.set(key, (wordFreq.get(key) ?? 0) + 1);
      }
    }

    const corpus: WordSymbols[] = [];

    for (const [word, freq] of wordFreq) {
      const symbols = this.createInitialSymbols(word);

      // 1. Pastikan karakter dasar ada di vocab
      for (const char of symbols) {
        if (!this.vocab.has(char)) {
          const allocatedId = this.allocateTokenId(char, nextId);
          if (allocatedId >= nextId) {
            nextId = allocatedId + 1;
          }
        }
      }

      // 2. Terapkan merge rules yang sudah ada untuk melihat seberapa "pecah" kata ini
      this.applyMergeRulesInPlace(symbols, this.merges);

      // Hal pertama yang perlu di cek dari korpus baru: apakah ada kombinasi > 3 token?
      // Jika ada, masukkan ke token alokasi (placeholder) atau buat ID baru.
      // UPDATE: Hanya lakukan ini untuk kata (alfabet), simbol kombinasi (1-2-3) diabaikan.
      if (symbols.length > 3) {
        const fullWord = symbols.join("");
        const isAlphabeticWord = this.isMergeableToken(fullWord);

        if (isAlphabeticWord) {
          if (!this.vocab.has(fullWord)) {
            console.log(`[BPE] Kata kompleks terdeteksi: "${fullWord}" (${symbols.length} token). Mengalokasikan token baru.`);
            const allocatedId = this.allocateTokenId(fullWord, nextId);
            if (allocatedId >= nextId) {
              nextId = allocatedId + 1;
            }
          }
          // Gunakan token utuh yang baru (atau lama) agar panjangnya jadi 1 token
          symbols.length = 1;
          symbols[0] = fullWord;
        } else {
          // console.log(`[BPE] Kombinasi simbol terdeteksi: "${fullWord}" (${symbols.length} token). Biarkan tetap di korpus BPE.`);
        }
      }

      // Hanya masukkan ke korpus training jika kata tersebut (entah bagaimana) masih "kompleks"
      // Seharusnya sekarang sudah jadi 1 token jika melewati blok di atas.
      if (symbols.length > 3) {
        corpus.push({ symbols, freq });
      }
    }

    if (corpus.length === 0) {
      console.log("[BPE] Semua kata sudah sangat ringkas (<= 3 token). Tidak ada merge baru yang perlu dipelajari.");
      this.buildReverseVocab();
      return;
    }

    console.log(`[BPE] Melanjutkan training dengan ${corpus.length} kata kompleks. Vocab saat ini: ${this.vocab.size}, Target: ${this.vocabSize}`);
    this.runBPE(corpus, nextId);
    this.clearEncodeCache();
  }

  private preTokenize(text: string): string[] {
    const rawTokens = this.preTokenizer(text);

    if (this.preTokenizerName === "char" || this.preTokenizerName === "unicode-grapheme") {
      return this.groupWhitespaceDelimitedTokens(rawTokens);
    }

    const tokens: string[] = [];
    for (const token of rawTokens) {
      if (token.length > 0 && token.trim().length > 0) tokens.push(token);
    }
    return tokens;
  }

  private groupWhitespaceDelimitedTokens(parts: string[]): string[] {
    const tokens: string[] = [];
    let current = "";

    for (const part of parts) {
      if (part.length === 0) continue;
      if (/^\s+$/u.test(part)) {
        if (current.length > 0) {
          tokens.push(current);
          current = "";
        }
        continue;
      }
      current += part;
    }

    if (current.length > 0) tokens.push(current);
    return tokens;
  }

  private createInitialSymbols(token: string): string[] {
    if (this.preTokenizerName === "char") {
      return charPreTokenizer(token);
    }

    if (token.startsWith(WORD_BOUNDARY)) {
      const body = token.slice(WORD_BOUNDARY.length);
      return [WORD_BOUNDARY, ...unicodeGraphemePreTokenizer(body)];
    }

    return unicodeGraphemePreTokenizer(token);
  }

  private runBPE(corpus: WordSymbols[], nextId: number): void {
    // === STEP 3: Iterasi BPE — gabungkan pasangan paling sering ===
    // Terus berjalan selama ada target vocab yang belum tercapai 
    // ATAU masih ada kata yang terlalu panjang (> 3 token).
    while (this.vocab.size < this.vocabSize || this.hasLongCorpusEntry(corpus)) {
      // 3a. Hitung frekuensi semua pasangan yang bersebelahan
      const pairFreq: Map<string, number> = new Map();

      for (const { symbols, freq } of corpus) {
        for (let i = 0; i < symbols.length - 1; i++) {
          const left = symbols[i];
          const right = symbols[i + 1];
          const merged = left + right;

          // Hanya izinkan merge untuk unit kata/script. Simbol matematika,
          // emoji, dan tanda baca tetap menjadi token terpisah.
          if (!this.isMergeableToken(merged)) continue;

          // Gunakan separator NULL yang hampir mustahil ada di text dataset
          const pair = left + PAIR_SEPARATOR + right;
          pairFreq.set(pair, (pairFreq.get(pair) ?? 0) + freq);
        }
      }

      if (pairFreq.size === 0) break; // Tidak ada lagi yang bisa di-merge

      // 3b. Cari pasangan dengan frekuensi tertinggi
      let bestPair = "";
      let bestFreq = 0;
      for (const [pair, freq] of pairFreq) {
        if (freq > bestFreq) {
          bestFreq = freq;
          bestPair = pair;
        }
      }

      // Hentikan jika frekuensi pasangan terbaik di bawah minimum
      if (bestFreq < this.minFrequency) {
        console.log(`[BPE] Berhenti: frekuensi tertinggi (${bestFreq}) < minimum (${this.minFrequency})`);
        break;
      }

      // 3c. Gabungkan pasangan terbaik
      const separatorIndex = bestPair.indexOf(PAIR_SEPARATOR);
      const left = bestPair.substring(0, separatorIndex);
      const right = bestPair.substring(separatorIndex + 1);
      const merged = left + right;

      // HANYA tambah ke merges jika belum ada
      const alreadyMerged = this.merges.some(([l, r]) => l === left && r === right);

      if (!alreadyMerged) {
        if (!this.vocab.has(merged)) {
          this.merges.push([left, right]);
          const allocatedId = this.allocateTokenId(merged, nextId);
          if (allocatedId >= nextId) {
            nextId = allocatedId + 1;
          }
        }

        // 3d. Terapkan merge ke seluruh corpus
        for (const entry of corpus) {
          this.applyMergeInPlace(entry.symbols, left, right, merged);
        }
      } else {
        // Jika sudah ada tapi kita sampai sini, berarti pasangan ini tidak bisa di-merge lagi
        // (biasanya karena bug split atau data corrupt). Kita hentikan untuk mencegah infinite loop.
        console.warn(`[BPE] Warning: Pasangan "${left}" + "${right}" sudah di-merge tapi terpilih kembali. Menghentikan.`);
        break;
      }

      if (this.merges.length % 100 === 0) {
        console.log(`[BPE] Merge #${this.merges.length}: "${left}" + "${right}" → "${merged}" (freq: ${bestFreq}), vocab: ${this.vocab.size}`);
      }
    }

    // === STEP 4: Isi sisa kapasitas dengan token placeholder jika belum mencapai target ===
    let unusedIdx = 0;
    while (this.vocab.size < this.vocabSize) {
      const placeholder = `<UNUSED_${unusedIdx++}>`;
      if (!this.vocab.has(placeholder)) {
        this.vocab.set(placeholder, nextId++);
      }
    }

    // Build reverse vocab
    this.buildReverseVocab();

    console.log(`[BPE] Selesai! Vocabulary size: ${this.vocab.size}, Merges: ${this.merges.length}`);
  }

  private applyMergeRulesInPlace(symbols: string[], merges: [string, string][]): void {
    for (const [left, right] of merges) {
      this.applyMergeInPlace(symbols, left, right, left + right);
    }
  }

  /**
   * Terapkan satu merge rule dengan compact in-place agar training/update tidak
   * mengalokasikan array baru untuk setiap kata dan setiap merge.
   */
  private applyMergeInPlace(symbols: string[], left: string, right: string, merged: string): boolean {
    let readIdx = 0;
    let writeIdx = 0;
    let changed = false;
    const lastMergeableIndex = symbols.length - 1;

    while (readIdx < symbols.length) {
      if (readIdx < lastMergeableIndex && symbols[readIdx] === left && symbols[readIdx + 1] === right) {
        symbols[writeIdx++] = merged;
        readIdx += 2;
        changed = true;
      } else {
        symbols[writeIdx++] = symbols[readIdx++];
      }
    }

    if (changed) {
      symbols.length = writeIdx;
    }
    return changed;
  }

  private hasLongCorpusEntry(corpus: WordSymbols[]): boolean {
    for (const entry of corpus) {
      if (entry.symbols.length > 3) return true;
    }
    return false;
  }

  private isMergeableToken(token: string): boolean {
    if (token.length === 0) return false;
    if (this.specialTokens.includes(token) || token.startsWith("<UNUSED_") || token.startsWith("<RESERVED_")) {
      return true;
    }

    const withoutBoundary = token.replace(new RegExp(WORD_BOUNDARY, "g"), "");
    if (withoutBoundary.length === 0) return true;

    for (const cluster of unicodeGraphemePreTokenizer(withoutBoundary)) {
      if (isEmojiLike(cluster) || isMathSymbol(cluster) || isPunctuation(cluster) || /^\s+$/u.test(cluster)) {
        return false;
      }
    }

    return true;
  }

  private allocateTokenId(token: string, nextId: number): number {
    const existingId = this.vocab.get(token);
    if (existingId !== undefined) {
      return existingId;
    }

    const reusablePlaceholder = this.findReusablePlaceholder();
    if (reusablePlaceholder) {
      console.log(`[BPE] Menempati token alokasi: "${reusablePlaceholder.token}" (${reusablePlaceholder.id}) -> "${token}"`);
      this.vocab.delete(reusablePlaceholder.token);
      this.vocab.set(token, reusablePlaceholder.id);
      return reusablePlaceholder.id;
    }

    // Jika tidak ada token alokasi/placeholder, buat ID baru.
    // Ini mendukung ekspansi vocabulary secara dinamis.
    this.vocab.set(token, nextId);
    if (this.vocab.size > this.vocabSize) {
      this.vocabSize = this.vocab.size;
    }
    return nextId;
  }

  /**
   * Proactively remove polluted tokens (compound symbols/digits)
   * that might have been learned in previous sessions.
   */
  private sanitize(): void {
    const beforeCount = this.vocab.size;

    // 1. Filter merges
    const initialMergeCount = this.merges.length;
    this.merges = this.merges.filter(([l, r]) => this.isMergeableToken(l + r));
    if (this.merges.length < initialMergeCount) {
      console.log(`[BPE] Sanitize: Removed ${initialMergeCount - this.merges.length} polluted merge rules.`);
    }

    // 2. Filter vocab (remove compound symbols)
    for (const [token, id] of this.vocab) {
      const isSpecial = this.specialTokens.includes(token) ||
        token.startsWith("<UNUSED_") ||
        token.startsWith("<RESERVED_");
      if (isSpecial) continue;

      // Single characters are always kept
      const isSingleChar = charPreTokenizer(token).length <= 1;
      if (isSingleChar) continue;

      if (!this.isMergeableToken(token)) {
        console.log(`[BPE] Removing polluted token: "${token}" (ID: ${id})`);
        this.vocab.delete(token);
      }
    }

    if (this.vocab.size < beforeCount) {
      this.buildReverseVocab();
      console.log(`[BPE] Sanitize: Removed ${beforeCount - this.vocab.size} polluted tokens.`);
    }
  }

  private findReusablePlaceholder(): { token: string; id: number } | null {
    // Cari token <UNUSED_*> dulu (prioritas pertama)
    for (const [token, id] of this.vocab) {
      if (token.startsWith("<UNUSED_") && token.endsWith(">")) {
        return { token, id };
      }
    }

    // Jika tidak ada <UNUSED_*>, cari <RESERVED_*> (wadah cadangan)
    for (const [token, id] of this.vocab) {
      if (token.startsWith("<RESERVED_") && token.endsWith(">")) {
        return { token, id };
      }
    }

    return null;
  }

  /**
   * Encode teks menjadi array token ID
   */
  encode(text: string): number[] {
    return this.encodeInternal(text, {
      allowWholeWordLookup: true,
      allowEncodeCache: true,
      preserveWholeWordAsSingleToken: true,
    });
  }

  /**
   * Encode khusus untuk training.
   * Secara acak dapat melewati pengecekan kata utuh pada vocab agar kata yang
   * sebenarnya dikenal tetap kadang dipecah menjadi kombinasi subword.
   */
  encodeForTraining(text: string, options: BPETrainingEncodeOptions = {}): number[] {
    const probability = options.fullWordLookupProbability ?? 0.5;
    const random = options.random ?? Math.random;
    const shouldLookupWholeWord = random() < probability;

    return this.encodeInternal(text, {
      allowWholeWordLookup: shouldLookupWholeWord,
      allowEncodeCache: shouldLookupWholeWord,
      preserveWholeWordAsSingleToken: shouldLookupWholeWord,
    });
  }

  /**
   * Encode teks dan bungkus dengan BOS/EOS
   */
  encodeWithSpecial(text: string): number[] {
    const bos = this.vocab.get(BOS_TOKEN)!;
    const eos = this.vocab.get(EOS_TOKEN)!;
    return [bos, ...this.encode(text), eos];
  }

  /**
   * Decode array token ID kembali menjadi teks
   */
  decode(ids: number[]): string {
    const tokens: string[] = [];
    for (const id of ids) {
      const token = this.reverseVocab.get(id);
      if (token && token !== BOS_TOKEN && token !== EOS_TOKEN && token !== PAD_TOKEN) {
        tokens.push(token);
      }
    }
    // Gabungkan lalu ganti word boundary dengan spasi
    return tokens.join("").replace(new RegExp(WORD_BOUNDARY, "g"), " ").trim();
  }

  /**
   * Dapatkan ukuran vocabulary saat ini
   */
  getVocabSize(): number {
    return this.vocab.size;
  }

  /**
   * Dapatkan kapasitas vocabulary berdasarkan ID tertinggi + 1.
   * Ini penting karena ID token bisa tidak selalu rapat setelah placeholder direuse.
   */
  getVocabularyCapacity(): number {
    let maxId = -1;
    for (const id of this.vocab.values()) {
      if (id > maxId) maxId = id;
    }
    return maxId + 1;
  }

  /**
   * Dapatkan ID dari sebuah token
   */
  getTokenId(token: string): number | undefined {
    return this.vocab.get(token);
  }

  /**
   * Dapatkan token dari sebuah ID
   */
  getToken(id: number): string | undefined {
    return this.reverseVocab.get(id);
  }

  /**
   * Dapatkan ID untuk PAD token (berguna untuk padding sequences)
   */
  getPadId(): number {
    return this.vocab.get(PAD_TOKEN)!;
  }

  /**
   * Pad sequences agar panjangnya sama
   */
  padSequence(ids: number[], maxLength: number): number[] {
    const padId = this.getPadId();
    if (ids.length >= maxLength) {
      return ids.slice(0, maxLength);
    }
    return [...ids, ...Array(maxLength - ids.length).fill(padId)];
  }

  // === SAVE / LOAD ===

  /**
   * Simpan vocabulary dan merge rules ke file JSON
   */
  save(filepath: string): void {
    const data: BPEVocabData = {
      vocab: Object.fromEntries(this.vocab),
      merges: this.merges,
      config: {
        vocabSize: this.vocabSize,
        minFrequency: this.minFrequency,
        specialTokens: this.specialTokens,
        preTokenizer: this.preTokenizerName,
      }
    };
    fs.writeFileSync(filepath, JSON.stringify(data, null, 2), "utf-8");
    console.log(`[BPE] Vocabulary disimpan ke: ${filepath}`);
  }

  /**
   * Muat vocabulary dan merge rules dari file JSON
   */
  static load(filepath: string, options?: { preTokenizer?: BuiltInPreTokenizer | PreTokenizer }): BPETokenizer {
    const raw = fs.readFileSync(filepath, "utf-8");
    const data: BPEVocabData = JSON.parse(raw);

    const serializedPreTokenizer = data.config.preTokenizer;
    const preTokenizer = serializedPreTokenizer === "custom"
      ? options?.preTokenizer
      : options?.preTokenizer ?? serializedPreTokenizer;

    if (serializedPreTokenizer === "custom" && typeof preTokenizer !== "function") {
      throw new Error("[BPE] Tokenizer ini disimpan dengan custom preTokenizer. Berikan function yang sama saat load().");
    }

    const tokenizer = new BPETokenizer({
      ...data.config,
      preTokenizer: preTokenizer ?? "char",
    });
    tokenizer.vocab = new Map(Object.entries(data.vocab).map(([k, v]) => [k, v as number]));
    tokenizer.merges = data.merges;
    tokenizer.clearEncodeCache();
    tokenizer.sanitize(); // Clean up loaded data
    tokenizer.buildReverseVocab();

    console.log(`[BPE] Vocabulary dimuat dari: ${filepath} (${tokenizer.vocab.size} tokens, ${tokenizer.merges.length} merges)`);
    return tokenizer;
  }

  /**
   * Build reverse vocab map (id → token)
   */
  private buildReverseVocab(): void {
    this.reverseVocab.clear();
    for (const [token, id] of this.vocab) {
      this.reverseVocab.set(id, token);
    }
  }

  private setEncodeCache(word: string, ids: number[]): void {
    if (this.encodeCache.size >= this.maxEncodeCacheSize) {
      const oldestKey = this.encodeCache.keys().next().value;
      if (oldestKey !== undefined) this.encodeCache.delete(oldestKey);
    }
    this.encodeCache.set(word, ids);
  }

  private clearEncodeCache(): void {
    this.encodeCache.clear();
  }

  private encodeInternal(
    text: string,
    options: {
      allowWholeWordLookup: boolean;
      allowEncodeCache: boolean;
      preserveWholeWordAsSingleToken: boolean;
    }
  ): number[] {
    const words = this.preTokenize(text);
    const tokenIds: number[] = [];

    for (const word of words) {
      if (word.length === 0) continue;
      const wordTokenIds = this.encodeWord(word, options);
      for (const id of wordTokenIds) tokenIds.push(id);
    }

    return tokenIds;
  }

  private encodeWord(
    word: string,
    options: {
      allowWholeWordLookup: boolean;
      allowEncodeCache: boolean;
      preserveWholeWordAsSingleToken: boolean;
    }
  ): number[] {
    const fullWord = WORD_BOUNDARY + word;

    if (options.allowWholeWordLookup) {
      const existingId = this.vocab.get(fullWord);
      if (existingId !== undefined) {
        return [existingId];
      }
    }

    if (options.allowEncodeCache) {
      const cachedIds = this.encodeCache.get(fullWord);
      if (cachedIds !== undefined) {
        return [...cachedIds];
      }
    }

    const symbols = this.createInitialSymbols(fullWord);
    this.applyMergeRulesInPlaceWithOptions(symbols, this.merges, {
      blockedMergedToken: options.preserveWholeWordAsSingleToken ? null : fullWord,
    });

    const wordTokenIds = this.symbolsToIds(symbols);
    if (options.allowEncodeCache) {
      this.setEncodeCache(fullWord, wordTokenIds);
    }
    return wordTokenIds;
  }

  private applyMergeRulesInPlaceWithOptions(
    symbols: string[],
    merges: [string, string][],
    options: { blockedMergedToken: string | null }
  ): void {
    for (const [left, right] of merges) {
      const merged = left + right;
      if (options.blockedMergedToken !== null && merged === options.blockedMergedToken) {
        continue;
      }
      this.applyMergeInPlace(symbols, left, right, merged);
    }
  }

  private symbolsToIds(symbols: string[]): number[] {
    const wordTokenIds: number[] = [];
    for (const sym of symbols) {
      const id = this.vocab.get(sym);
      if (id !== undefined) {
        wordTokenIds.push(id);
      } else {
        wordTokenIds.push(this.vocab.get(UNK_TOKEN)!);
      }
    }
    return wordTokenIds;
  }

  /**
   * Print vocabulary summary
   */
  summary(): void {
    console.log("=== BPE Tokenizer Summary ===");
    console.log(`Vocabulary size : ${this.vocab.size}`);
    console.log(`Merge rules     : ${this.merges.length}`);
    console.log(`Special tokens  : ${this.specialTokens.join(", ")}`);
    console.log(`\nSample vocabulary (first 20):`);
    let count = 0;
    for (const [token, id] of this.vocab) {
      if (count >= 20) break;
      const display = token.replace(WORD_BOUNDARY, "▁");
      console.log(`  [${id}] "${display}"`);
      count++;
    }
    console.log("=============================");
  }
}
