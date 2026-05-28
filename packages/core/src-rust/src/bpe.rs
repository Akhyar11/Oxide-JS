use napi_derive::napi;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use rayon::prelude::*;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BPEConfigSerde {
    pub vocab_size: u32,
    pub min_frequency: u32,
    pub pre_tokenizer: String,
    pub special_tokens: Vec<String>,
}

#[derive(Serialize)]
pub struct BPESaveFormat {
    pub vocab: HashMap<String, u32>,
    pub merges: Vec<Vec<String>>,
    pub config: BPEConfigSerde,
}

// Struktur untuk mereturn Vocab hasil
#[napi(object)]
pub struct NativeVocabResult {
    pub vocab: HashMap<String, u32>,
    pub merges: Vec<Vec<String>>, // Array of [left, right]
}

#[napi]
pub struct NativeBPETrainer {
    vocab_size: u32,
    special_tokens: Vec<String>,
}

#[napi]
impl NativeBPETrainer {
    #[napi(constructor)]
    pub fn new(vocab_size: u32, special_tokens: Vec<String>) -> Self {
        Self {
            vocab_size,
            special_tokens,
        }
    }

    #[napi]
    pub fn train_from_file(&self, filepath: String) -> NativeVocabResult {
        println!("[Rust BPE] Memulai pembacaan file: {}", filepath);
        
        // 1. Inisialisasi peta ID
        let mut string_to_id: HashMap<String, u32> = HashMap::new();
        let mut id_to_string: HashMap<u32, String> = HashMap::new();
        let mut next_id: u32 = 0;

        // Daftarkan special tokens
        for token in &self.special_tokens {
            string_to_id.insert(token.clone(), next_id);
            id_to_string.insert(next_id, token.clone());
            next_id += 1;
        }

        // 2. Baca file dan bangun word frequencies
        let file = File::open(&filepath).expect("Gagal membuka file dataset");
        let reader = BufReader::new(file);
        
        let mut word_freqs: HashMap<String, u32> = HashMap::new();
        for line in reader.lines() {
            if let Ok(l) = line {
                for word in l.split_whitespace() {
                    if word.is_empty() { continue; }
                    let word_boundary = format!("▁{}", word);
                    *word_freqs.entry(word_boundary).or_insert(0) += 1;
                }
            }
        }
        
        println!("[Rust BPE] Berhasil membaca {} kata unik.", word_freqs.len());

        // 3. Konversi karakter menjadi Token IDs awal
        // Corpus: list kata, di mana tiap kata adalah rentetan ID dan frekuensinya
        let mut corpus: Vec<(Vec<u32>, u32)> = Vec::with_capacity(word_freqs.len());
        
        for (word, freq) in word_freqs {
            let mut ids = Vec::new();
            // Implementasi split sederhana per karakter (UTF-8 char)
            let mut is_first = true;
            let mut chars = word.chars().peekable();
            
            while let Some(c) = chars.next() {
                let s = if is_first && c == '▁' {
                    is_first = false;
                    "▁".to_string()
                } else if is_first {
                    is_first = false;
                    c.to_string()
                } else {
                    c.to_string()
                };
                
                let id = *string_to_id.entry(s.clone()).or_insert_with(|| {
                    let new_id = next_id;
                    id_to_string.insert(new_id, s);
                    next_id += 1;
                    new_id
                });
                ids.push(id);
            }
            corpus.push((ids, freq));
        }

        println!("[Rust BPE] Karakter dasar (Initial Vocab): {}", next_id);

        let mut merges_history: Vec<Vec<String>> = Vec::new();

        // 4. Proses iterasi BPE Merging (Super cepat karena menggunakan integer/u32)
        while next_id < self.vocab_size {
            // Hitung frekuensi pasangan adjacent (kiri, kanan) menggunakan Rayon (Map-Reduce)
            let pair_freqs: HashMap<(u32, u32), u32> = corpus.par_iter()
                .fold(
                    || HashMap::new(),
                    |mut local_map, (ids, freq)| {
                        if ids.len() >= 2 {
                            for i in 0..ids.len() - 1 {
                                let pair = (ids[i], ids[i+1]);
                                *local_map.entry(pair).or_insert(0) += freq;
                            }
                        }
                        local_map
                    }
                )
                .reduce(
                    || HashMap::new(),
                    |mut map1, map2| {
                        for (k, v) in map2 {
                            *map1.entry(k).or_insert(0) += v;
                        }
                        map1
                    }
                );

            // Cari pasangan dengan frekuensi tertinggi
            let best_pair = pair_freqs.into_iter().max_by_key(|&(_, count)| count);
            
            if let Some(((left_id, right_id), best_freq)) = best_pair {
                if best_freq < 2 {
                    println!("[Rust BPE] Merges dihentikan. Frekuensi maksimal < 2.");
                    break; 
                }

                // Buat String Gabungan
                let left_str = id_to_string.get(&left_id).unwrap().clone();
                let right_str = id_to_string.get(&right_id).unwrap().clone();
                let merged_str = format!("{}{}", left_str, right_str);

                // Buat ID baru
                let new_id = next_id;
                next_id += 1;
                string_to_id.insert(merged_str.clone(), new_id);
                id_to_string.insert(new_id, merged_str.clone());
                
                merges_history.push(vec![left_str, right_str]);

                if merges_history.len() % 1000 == 0 {
                    println!("[Rust BPE] Progress Merges: {}/{} (Vocab Size: {})", merges_history.len(), self.vocab_size, next_id);
                }

                // Terapkan merge ke seluruh corpus (Parallel in-place modification)
                corpus.par_iter_mut().for_each(|(ids, _)| {
                    let mut i = 0;
                    let mut new_ids = Vec::with_capacity(ids.len());
                    while i < ids.len() {
                        if i < ids.len() - 1 && ids[i] == left_id && ids[i+1] == right_id {
                            new_ids.push(new_id);
                            i += 2; // Lewati elemen kanan yang sudah di-merge
                        } else {
                            new_ids.push(ids[i]);
                            i += 1;
                        }
                    }
                    *ids = new_ids; // Update array dengan yang sudah dipadatkan
                });
            } else {
                break; // Tidak ada pasangan lagi
            }
        }

        println!("[Rust BPE] Selesai! Final Vocab Size: {}", string_to_id.len());

        NativeVocabResult {
            vocab: string_to_id,
            merges: merges_history,
        }
    }

    #[napi]
    pub fn train_and_save(&self, filepath: String, save_path: String) -> bool {
        let result = self.train_from_file(filepath);
        
        let save_data = BPESaveFormat {
            vocab: result.vocab,
            merges: result.merges,
            config: BPEConfigSerde {
                vocab_size: self.vocab_size,
                min_frequency: 2,
                pre_tokenizer: "char".to_string(),
                special_tokens: self.special_tokens.clone(),
            },
        };

        let json_str = serde_json::to_string(&save_data).expect("Gagal serialize BPE data");
        let mut file = File::create(&save_path).expect("Gagal membuat file save");
        file.write_all(json_str.as_bytes()).expect("Gagal menulis ke file");

        println!("[Rust BPE] ✅ Model ({} tokens) berhasil di-serialize dan ditulis langsung ke: {}", self.vocab_size, save_path);
        true
    }
}

