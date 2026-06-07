use napi_derive::napi;
use napi::bindgen_prelude::Float32Array;
use rayon::prelude::*;

#[napi]
pub fn contrastive_hebbian_native(
    spikes: Float32Array,
    mut err_data: Float32Array,
    num_pairs: u32,
    sequence_length: u32,
    d_model: u32,
) -> f64 {
    let spikes_slice: &[f32] = &spikes;
    let err_slice: &mut [f32] = &mut err_data;
    
    let num_pairs = num_pairs as usize;
    let seq_len = sequence_length as usize;
    let d_model = d_model as usize;
    let chunk_size = seq_len * d_model;

    let total_loss: f32 = err_slice.par_chunks_mut(chunk_size).enumerate().map(|(b, chunk)| {
        let mut local_loss = 0.0f32;

        if b < num_pairs {
            // Ini adalah vektor Q
            let i = b;
            let p_offset = (num_pairs + i) * chunk_size;
            let n_offset = (num_pairs + ((i + 1) % num_pairs)) * chunk_size;
            let q_offset = i * chunk_size;
            
            for rem in 0..chunk_size {
                let q_spike = spikes_slice[q_offset + rem];
                let p_spike = spikes_slice[p_offset + rem];
                let n_spike = spikes_slice[n_offset + rem];
                
                let mut pull = p_spike - q_spike;
                if q_spike == 0.0 && p_spike == 0.0 && n_spike == 0.0 {
                    pull = 0.05; // Suntik energi
                }
                let push = (q_spike * n_spike) * 0.2;
                
                chunk[rem] = pull - push;
                
                if pull != 0.0 || push != 0.0 {
                    local_loss += pull.abs() + push;
                }
            }
        } else {
            // Ini adalah vektor P atau N
            let p_index = b - num_pairs;
            
            // Peran sebagai P untuk i = p_index
            let q_offset_p = p_index * chunk_size;
            let n_offset_p = (num_pairs + ((p_index + 1) % num_pairs)) * chunk_size;
            
            // Peran sebagai N untuk i = p_index - 1 (dengan wrap around)
            let i_n = if p_index == 0 { num_pairs - 1 } else { p_index - 1 };
            let q_offset_n = i_n * chunk_size;
            
            for rem in 0..chunk_size {
                let q_spike_p = spikes_slice[q_offset_p + rem];
                let p_spike_p = spikes_slice[b * chunk_size + rem];
                let n_spike_p = spikes_slice[n_offset_p + rem];
                
                let mut pull_p = p_spike_p - q_spike_p;
                if q_spike_p == 0.0 && p_spike_p == 0.0 && n_spike_p == 0.0 {
                    pull_p = 0.05;
                }
                let contrib_p = -pull_p;
                
                let q_spike_n = spikes_slice[q_offset_n + rem];
                let n_spike_n = spikes_slice[b * chunk_size + rem];
                
                let push_n = (q_spike_n * n_spike_n) * 0.2;
                let contrib_n = -push_n;
                
                chunk[rem] = contrib_p + contrib_n;
            }
        }
        
        local_loss
    }).sum();

    total_loss as f64
}
