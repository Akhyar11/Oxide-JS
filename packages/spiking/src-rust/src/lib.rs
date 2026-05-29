#![deny(clippy::all)]

use napi_derive::napi;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

#[derive(Serialize, Deserialize)]
pub struct NetworkState {
    pub num_neurons: u32,
    pub beta: f32,
    pub thresholds: Vec<f32>,
    pub post_synaptic_indices: Vec<Vec<u32>>,
    pub weights: Vec<Vec<f32>>,
}

#[napi]
pub struct NativeSpikingNetwork {
  num_neurons: u32,
  potentials: Vec<f32>,
  thresholds: Vec<f32>,
  spikes: Vec<u8>,
  
  // Adjacency lists
  post_synaptic_indices: Vec<Vec<u32>>,
  weights: Vec<Vec<f32>>,
  
  // STDP Traces
  pre_traces: Vec<f32>,
  post_traces: Vec<f32>,
  
  beta: f32,
}

#[napi]
impl NativeSpikingNetwork {
  #[napi(constructor)]
  pub fn new(num_neurons: u32, beta: f64, default_threshold: f64) -> Self {
    Self {
      num_neurons,
      potentials: vec![0.0; num_neurons as usize],
      thresholds: vec![default_threshold as f32; num_neurons as usize],
      spikes: vec![0; num_neurons as usize],
      post_synaptic_indices: vec![Vec::new(); num_neurons as usize],
      weights: vec![Vec::new(); num_neurons as usize],
      pre_traces: vec![0.0; num_neurons as usize],
      post_traces: vec![0.0; num_neurons as usize],
      beta: beta as f32,
    }
  }

  #[napi]
  pub fn connect(&mut self, pre: u32, post: u32, weight: f64) {
    let pre_idx = pre as usize;
    if pre_idx >= self.num_neurons as usize || post as usize >= self.num_neurons as usize {
      return;
    }

    if let Some(pos) = self.post_synaptic_indices[pre_idx].iter().position(|&p| p == post) {
      self.weights[pre_idx][pos] = weight as f32;
    } else {
      self.post_synaptic_indices[pre_idx].push(post);
      self.weights[pre_idx].push(weight as f32);
    }
  }

  #[napi]
  pub fn inject_current(&mut self, neuron_idx: u32, current: f64) {
    let idx = neuron_idx as usize;
    if idx < self.potentials.len() {
      self.potentials[idx] += current as f32;
    }
  }

  #[napi]
  pub fn inhibit_range(&mut self, start_idx: u32, end_idx: u32, except_idx: u32, current: f64) {
    let start = start_idx as usize;
    let end = end_idx as usize;
    let ex = except_idx as usize;
    let c = current as f32;

    for i in start..end {
      if i != ex && i < self.potentials.len() {
        self.potentials[i] += c;
      }
    }
  }

  #[napi]
  pub fn step(&mut self) {
    // 1. Accumulate pre-synaptic spikes (Sequential for now to avoid data races)
    // We only process neurons that spiked in the previous step
    for i in 0..self.num_neurons as usize {
      if self.spikes[i] == 1 {
        let targets = &self.post_synaptic_indices[i];
        let w = &self.weights[i];
        for k in 0..targets.len() {
          let j = targets[k] as usize;
          self.potentials[j] += w[k];
        }
      }
    }

    // Reset spikes safely (Paralel)
    self.spikes.par_iter_mut().for_each(|s| *s = 0);

    // 2. Membrane decay, threshold check, and fire (Paralel via Rayon)
    let beta = self.beta;
    self.potentials
      .par_iter_mut()
      .zip(self.thresholds.par_iter())
      .zip(self.spikes.par_iter_mut())
      .for_each(|((pot, &thresh), spike)| {
        *pot *= beta; // Leaky integration
        if *pot < 1e-5 { *pot = 0.0; } // Prevent denormals
        if *pot >= thresh {
          *spike = 1;
          *pot -= thresh; // Soft reset
        }
      });
  }

  #[napi]
  pub fn reset_state(&mut self) {
    self.potentials.par_iter_mut().for_each(|p| *p = 0.0);
    self.spikes.par_iter_mut().for_each(|s| *s = 0);
  }

  // Helpers to get data to JS if needed
  #[napi]
  pub fn get_spikes(&self) -> Vec<u8> {
    self.spikes.clone()
  }

  #[napi]
  pub fn get_potentials(&self) -> Vec<f32> {
    self.potentials.clone()
  }

  #[napi]
  pub fn update_stdp(
    &mut self,
    learning_rate: f64,
    tau_plus: f64,
    tau_minus: f64,
    a_plus: f64,
    a_minus: f64,
    w_max: f64,
    w_min: f64,
  ) {
    let lr = learning_rate as f32;
    let t_plus = tau_plus as f32;
    let t_minus = tau_minus as f32;
    let ap = a_plus as f32;
    let am = a_minus as f32;
    let wmax = w_max as f32;
    let wmin = w_min as f32;

    // 1. Update trace decay and trace spikes
    self.pre_traces
      .par_iter_mut()
      .zip(self.post_traces.par_iter_mut())
      .zip(self.spikes.par_iter())
      .for_each(|((pre, post), &spike)| {
        *pre *= t_plus;
        *post *= t_minus;
        
        if *pre < 1e-5 { *pre = 0.0; }
        if *post < 1e-5 { *post = 0.0; }

        if spike == 1 {
          *pre = 1.0;
          *post = 1.0;
        }
      });

    // 2. Event-driven weight updates
    // Since we mutate self.weights, we can parallelize over the outer slice
    // because each sub-vector weights[i] belongs to exclusively one thread.
    // However, we need immutable access to self.post_traces and self.spikes.
    // To do this cleanly in Rust with Rayon without violating borrow rules:
    
    let spikes = &self.spikes;
    let pre_traces = &self.pre_traces;
    let post_traces = &self.post_traces;
    let post_synaptic_indices = &self.post_synaptic_indices;

    self.weights
      .par_iter_mut()
      .enumerate()
      .for_each(|(i, w_list)| {
        let pre_spiked = spikes[i] == 1;
        let pre_trace = pre_traces[i];

        if pre_trace == 0.0 && !pre_spiked {
          return;
        }

        let targets = &post_synaptic_indices[i];

        for k in 0..targets.len() {
          let j = targets[k] as usize;
          let post_spiked = spikes[j] == 1;
          
          let mut dw = 0.0;
          
          // LTP
          if post_spiked {
            dw += lr * ap * pre_trace;
          }
          // LTD
          if pre_spiked {
            dw -= lr * am * post_traces[j];
          }

          if dw != 0.0 {
            w_list[k] += dw;
            if w_list[k] > wmax {
              w_list[k] = wmax;
            } else if w_list[k] < wmin {
              w_list[k] = wmin;
            }
          }
        }
      });
  }

  #[napi]
  pub fn save_to_file(&self, filepath: String) {
      let state = NetworkState {
          num_neurons: self.num_neurons,
          beta: self.beta,
          thresholds: self.thresholds.clone(),
          post_synaptic_indices: self.post_synaptic_indices.clone(),
          weights: self.weights.clone(),
      };
      let file = File::create(filepath).expect("Gagal membuat file penyimpanan");
      let writer = BufWriter::new(file);
      serde_json::to_writer(writer, &state).expect("Gagal menyimpan NetworkState");
  }

  #[napi]
  pub fn load_from_file(&mut self, filepath: String) {
      let file = File::open(filepath).expect("Gagal membuka file penyimpanan");
      let reader = BufReader::new(file);
      let state: NetworkState = serde_json::from_reader(reader).expect("Gagal memuat NetworkState");
      
      self.num_neurons = state.num_neurons;
      self.beta = state.beta;
      self.thresholds = state.thresholds;
      self.post_synaptic_indices = state.post_synaptic_indices;
      self.weights = state.weights;
      
      // Sesuaikan memori untuk traces & potentials
      let len = self.num_neurons as usize;
      self.potentials.resize(len, 0.0);
      self.spikes.resize(len, 0);
      self.pre_traces.resize(len, 0.0);
      self.post_traces.resize(len, 0.0);
      
      // Reset state internal
      self.potentials.fill(0.0);
      self.spikes.fill(0);
      self.pre_traces.fill(0.0);
      self.post_traces.fill(0.0);
  }
}

#[napi]
pub fn dot_product_add_only_native(
    a_data: napi::bindgen_prelude::Float32Array,
    a_rows_orig: u32,
    a_cols_orig: u32,
    b_data: napi::bindgen_prelude::Float32Array,
    b_rows_orig: u32,
    b_cols_orig: u32,
    trans_a: bool,
    trans_b: bool,
    mut out_data: napi::bindgen_prelude::Float32Array,
) {
    let a_rows = if trans_a { a_cols_orig } else { a_rows_orig } as usize;
    let a_cols = if trans_a { a_rows_orig } else { a_cols_orig } as usize;
    let b_rows = if trans_b { b_cols_orig } else { b_rows_orig } as usize;
    let b_cols = if trans_b { b_rows_orig } else { b_cols_orig } as usize;

    let a_slice = &*a_data;
    let b_slice = &*b_data;
    let out_slice = &mut *out_data;

    let mut a_is_binary = true;
    for &val in a_slice {
        if val != 0.0 && val != 1.0 {
            a_is_binary = false;
            break;
        }
    }
    
    let mut b_is_binary = true;
    if !a_is_binary {
        for &val in b_slice {
            if val != 0.0 && val != 1.0 {
                b_is_binary = false;
                break;
            }
        }
    }

    if !a_is_binary && !b_is_binary {
        panic!("SNN Error: Kedua matriks adalah floating-point. Setidaknya salah satu matriks harus hanya berisi 0 dan 1.");
    }

    out_slice.par_iter_mut().for_each(|x| *x = 0.0);
    
    let a_rows_orig = a_rows_orig as usize;
    let a_cols_orig = a_cols_orig as usize;
    let b_cols_orig = b_cols_orig as usize;

    out_slice.par_chunks_mut(b_cols).enumerate().for_each(|(i, out_row)| {
        if !trans_b {
            for k in 0..a_cols {
                let aik = if trans_a { a_slice[k * a_rows_orig + i] } else { a_slice[i * a_cols_orig + k] };
                if aik == 0.0 { continue; }
                let k_offset = k * b_cols;
                
                if a_is_binary {
                    // aik must be 1.0
                    for j in 0..b_cols {
                        out_row[j] += b_slice[k_offset + j];
                    }
                } else {
                    // b_is_binary
                    for j in 0..b_cols {
                        if b_slice[k_offset + j] == 1.0 {
                            out_row[j] += aik;
                        }
                    }
                }
            }
        } else {
            // trans_b == true
            for j in 0..b_cols {
                let mut sum = 0.0;
                for k in 0..a_cols {
                    let aik = if trans_a { a_slice[k * a_rows_orig + i] } else { a_slice[i * a_cols_orig + k] };
                    let bjk = b_slice[j * b_cols_orig + k];
                    if a_is_binary {
                        if aik == 1.0 { sum += bjk; }
                    } else {
                        if bjk == 1.0 { sum += aik; }
                    }
                }
                out_row[j] = sum;
            }
        }
    });
}

#[napi]
pub fn lif_step_native(
    mut potentials: napi::bindgen_prelude::Float32Array,
    dot: napi::bindgen_prelude::Float32Array,
    mut spikes: napi::bindgen_prelude::Float32Array,
    mut last_potentials: napi::bindgen_prelude::Float32Array,
    beta: f64,
    threshold: f64,
) {
    let pot_slice = &mut *potentials;
    let dot_slice = &*dot;
    let spike_slice = &mut *spikes;
    let lp_slice = &mut *last_potentials;
    let b = beta as f32;
    let th = threshold as f32;
    
    pot_slice.par_iter_mut()
        .zip(dot_slice.par_iter())
        .zip(spike_slice.par_iter_mut())
        .zip(lp_slice.par_iter_mut())
        .for_each(|(((p, d), s), lp)| {
            *p = (*p * b) + d;
            *lp = *p;
            if *p >= th {
                *s = 1.0;
                *p -= th;
            } else {
                *s = 0.0;
            }
        });
}

#[napi]
pub fn mask_surrogate_native(
    mut error_signal: napi::bindgen_prelude::Float32Array,
    potentials: napi::bindgen_prelude::Float32Array,
    threshold: f64,
    window_size: f64,
) {
    let err_slice = &mut *error_signal;
    let pot_slice = &*potentials;
    let th = threshold as f32;
    let win = window_size as f32;
    
    err_slice.par_iter_mut()
        .zip(pot_slice.par_iter())
        .for_each(|(e, p)| {
            if (*p - th).abs() > win {
                *e = 0.0;
            }
        });
}

#[napi]
pub fn apply_add_only_delta_native(
    mut kernel: napi::bindgen_prelude::Float32Array,
    mut bias: napi::bindgen_prelude::Float32Array,
    inputs: napi::bindgen_prelude::Float32Array,
    error_signal: napi::bindgen_prelude::Float32Array,
    learning_rate: f64,
    batch: u32,
    in_features: u32,
    units: u32,
    use_bias: bool,
) {
    let k_slice = &mut *kernel;
    let b_slice = &mut *bias;
    let in_slice = &*inputs;
    let err_slice = &*error_signal;
    let lr = learning_rate as f32;
    
    let batch = batch as usize;
    let in_f = in_features as usize;
    let u = units as usize;

    k_slice.par_chunks_mut(u).enumerate().for_each(|(k, k_row)| {
        let mut row_update = vec![0.0; u];
        for b in 0..batch {
            let in_offset = b * in_f;
            if in_slice[in_offset + k] > 0.5 {
                let err_offset = b * u;
                for j in 0..u {
                    row_update[j] += err_slice[err_offset + j];
                }
            }
        }
        for j in 0..u {
            k_row[j] += lr * row_update[j];
        }
    });

    if use_bias && b_slice.len() >= u {
        b_slice.par_iter_mut().enumerate().for_each(|(j, b_val)| {
            let mut b_update = 0.0;
            for b in 0..batch {
                let err_offset = b * u;
                b_update += err_slice[err_offset + j];
            }
            *b_val += (lr * b_update) / (batch as f32);
        });
    }
}

#[napi]
pub fn learn_hebbian_native(
    mut kernel: napi::bindgen_prelude::Float32Array,
    tokens: napi::bindgen_prelude::Float32Array,
    positive_context: napi::bindgen_prelude::Float32Array,
    negative_contexts: napi::bindgen_prelude::Float32Array,
    num_negatives: u32,
    input_dim: u32,
    output_dim: u32,
    learning_rate: f64,
    margin_positive: f64,
    margin_negative: f64,
) {
    let k_slice = &mut *kernel;
    let t_slice = &*tokens;
    let pos_slice = &*positive_context;
    let neg_slice = &*negative_contexts;
    let lr = learning_rate as f32;
    let mp = margin_positive as f32;
    let mn = margin_negative as f32;
    let dim = output_dim as usize;
    let in_dim = input_dim as usize;
    let num_neg = num_negatives as usize;

    let mut updated_mask = vec![false; in_dim];

    for &t in t_slice {
        let token_id = t.round() as usize;
        if token_id < in_dim {
            updated_mask[token_id] = true;
            let k_offset = token_id * dim;

            // 1. Update Positive Context (selalu dieksekusi)
            for j in 0..dim {
                let pos_grad = pos_slice[j] - k_slice[k_offset + j];
                k_slice[k_offset + j] += lr * pos_grad * mp;
            }

            // 2. Update Negative Contexts (jika ada)
            for n in 0..num_neg {
                let neg_offset = n * dim;
                let neg_mean = &neg_slice[neg_offset..neg_offset + dim];
                for j in 0..dim {
                    let neg_grad = k_slice[k_offset + j] - neg_mean[j];
                    k_slice[k_offset + j] += lr * neg_grad * mn;
                }
            }
        }
    }

    // L2 Normalization in parallel
    k_slice.par_chunks_mut(dim).enumerate().for_each(|(i, k_row)| {
        if updated_mask[i] {
            let mut norm = 0.0;
            for j in 0..dim {
                norm += k_row[j] * k_row[j];
            }
            norm = (norm + 1e-8).sqrt();
            for j in 0..dim {
                k_row[j] /= norm;
            }
        }
    });
}
