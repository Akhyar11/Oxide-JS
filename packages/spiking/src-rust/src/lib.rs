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
        let targets = &post_synaptic_indices[i];
        let pre_spiked = spikes[i] == 1;
        let pre_trace = pre_traces[i];

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
