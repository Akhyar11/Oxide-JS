import * as fs from "fs";

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

export interface BPEConfig {
  vocabSize: number;        // Ukuran vocabulary target
  minFrequency?: number;    // Frekuensi minimum untuk merge (default: 2)
  specialTokens?: string[]; // Token khusus tambahan
}

export interface BPEVocabData {
  vocab: Record<string, number>;       // token → id
  merges: [string, string][];          // Daftar merge rules, urut dari pertama dipelajari
  config: BPEConfig;
}

export default class BPETokenizer {
  private vocab: Map<string, number> = new Map();
  private reverseVocab: Map<number, string> = new Map();
  private merges: [string, string][] = [];
  private vocabSize: number;
  private minFrequency: number;
  private specialTokens: string[];

  constructor(config: BPEConfig) {
    this.vocabSize = config.vocabSize;
    this.minFrequency = config.minFrequency ?? 2;
    this.specialTokens = [
      PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN,
      ...(config.specialTokens ?? [])
    ];
  }

  /**
   * Melatih tokenizer dari corpus teks
   * @param texts - Array of strings sebagai training data
   */
  train(texts: string[]): void {
    // === STEP 1: Inisialisasi vocabulary dengan special tokens ===
    this.vocab.clear();
    this.merges = [];
    let nextId = 0;

    for (const token of this.specialTokens) {
      this.vocab.set(token, nextId++);
    }

    // === STEP 2: Tokenisasi awal — pecah setiap kata jadi karakter ===
    // Pre-tokenize: split berdasarkan spasi, tambahkan word boundary marker
    // "saya makan" → ["▁s", "a", "y", "a", " ", "▁m", "a", "k", "a", "n"]
    
    // Hitung frekuensi setiap kata
    const wordFreq: Map<string, number> = new Map();
    for (const text of texts) {
      const words = text.trim().split(/\s+/);
      for (const word of words) {
        if (word.length === 0) continue;
        const key = WORD_BOUNDARY + word; // Tambah boundary marker
        wordFreq.set(key, (wordFreq.get(key) ?? 0) + 1);
      }
    }

    // Pecah setiap kata menjadi karakter individual
    // "▁makan" → ["▁m", "a", "k", "a", "n"] → Tapi kita simpan sebagai ["▁", "m", "a", "k", "a", "n"]
    // Kita gunakan representasi: list of symbols per word
    type WordSymbols = { symbols: string[]; freq: number };
    const corpus: WordSymbols[] = [];

    for (const [word, freq] of wordFreq) {
      const chars = [...word]; // Split Unicode-safe
      corpus.push({ symbols: chars, freq });

      // Tambahkan setiap karakter ke vocab jika belum ada
      for (const char of chars) {
        if (!this.vocab.has(char)) {
          this.vocab.set(char, nextId++);
        }
      }
    }

    // === STEP 3: Iterasi BPE — gabungkan pasangan paling sering ===
    console.log(`[BPE] Vocabulary awal: ${this.vocab.size} tokens (target: ${this.vocabSize})`);

    while (this.vocab.size < this.vocabSize) {
      // 3a. Hitung frekuensi semua pasangan yang bersebelahan
      const pairFreq: Map<string, number> = new Map();

      for (const { symbols, freq } of corpus) {
        for (let i = 0; i < symbols.length - 1; i++) {
          const pair = symbols[i] + "|" + symbols[i + 1]; // Delimiter untuk key
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
      const [left, right] = bestPair.split("|");
      const merged = left + right;

      // Simpan merge rule
      this.merges.push([left, right]);

      // Tambahkan token baru ke vocab
      if (!this.vocab.has(merged)) {
        this.vocab.set(merged, nextId++);
      }

      // 3d. Terapkan merge ke seluruh corpus
      for (const entry of corpus) {
        entry.symbols = this.applyMerge(entry.symbols, left, right, merged);
      }

      if (this.merges.length % 100 === 0) {
        console.log(`[BPE] Merge #${this.merges.length}: "${left}" + "${right}" → "${merged}" (freq: ${bestFreq}), vocab: ${this.vocab.size}`);
      }
    }

    // Build reverse vocab
    this.buildReverseVocab();

    console.log(`[BPE] Training selesai! Vocabulary size: ${this.vocab.size}, Merges: ${this.merges.length}`);
  }

  /**
   * Terapkan satu merge rule ke array symbols
   */
  private applyMerge(symbols: string[], left: string, right: string, merged: string): string[] {
    const result: string[] = [];
    let i = 0;
    while (i < symbols.length) {
      if (i < symbols.length - 1 && symbols[i] === left && symbols[i + 1] === right) {
        result.push(merged);
        i += 2; // Skip kedua simbol yang di-merge
      } else {
        result.push(symbols[i]);
        i++;
      }
    }
    return result;
  }

  /**
   * Encode teks menjadi array token ID
   */
  encode(text: string): number[] {
    const words = text.trim().split(/\s+/);
    const tokenIds: number[] = [];

    for (const word of words) {
      if (word.length === 0) continue;

      // Pecah kata jadi karakter dengan word boundary
      let symbols = [...(WORD_BOUNDARY + word)];

      // Terapkan semua merge rules secara berurutan
      for (const [left, right] of this.merges) {
        const merged = left + right;
        symbols = this.applyMerge(symbols, left, right, merged);
      }

      // Convert symbols ke ID
      for (const sym of symbols) {
        const id = this.vocab.get(sym);
        if (id !== undefined) {
          tokenIds.push(id);
        } else {
          // Token tidak dikenal → UNK
          tokenIds.push(this.vocab.get(UNK_TOKEN)!);
        }
      }
    }

    return tokenIds;
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
      }
    };
    fs.writeFileSync(filepath, JSON.stringify(data, null, 2), "utf-8");
    console.log(`[BPE] Vocabulary disimpan ke: ${filepath}`);
  }

  /**
   * Muat vocabulary dan merge rules dari file JSON
   */
  static load(filepath: string): BPETokenizer {
    const raw = fs.readFileSync(filepath, "utf-8");
    const data: BPEVocabData = JSON.parse(raw);

    const tokenizer = new BPETokenizer(data.config);
    tokenizer.vocab = new Map(Object.entries(data.vocab).map(([k, v]) => [k, v as number]));
    tokenizer.merges = data.merges;
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
