use napi_derive::napi;
use napi::bindgen_prelude::Float32Array;
use rayon::prelude::*;

#[napi]
pub fn lif_step_native(
    mut potentials: Float32Array,
    dot: Float32Array,
    mut spikes: Float32Array,
    mut last_potentials: Float32Array,
    beta: Float32Array,
    threshold: Float32Array
) {
    let units = beta.len();
    if units == 0 { return; }
    let batch = potentials.len() / units;
    
    let pot_slice: &mut [f32] = &mut potentials;
    let dot_slice: &[f32] = &dot;
    let spikes_slice: &mut [f32] = &mut spikes;
    let last_pot_slice: &mut [f32] = &mut last_potentials;
    let beta_slice: &[f32] = &beta;
    let thresh_slice: &[f32] = &threshold;

    pot_slice.par_chunks_mut(units)
        .zip(dot_slice.par_chunks(units))
        .zip(spikes_slice.par_chunks_mut(units))
        .zip(last_pot_slice.par_chunks_mut(units))
        .for_each(|(((pot_chunk, dot_chunk), spike_chunk), last_pot_chunk)| {
            for i in 0..units {
                let mut pot = (pot_chunk[i] * beta_slice[i]) + dot_chunk[i];
                pot = pot.min(1.0); // Clamp potential max 1.0
                last_pot_chunk[i] = pot;
                
                if pot >= thresh_slice[i] {
                    spike_chunk[i] = 1.0;
                    pot -= thresh_slice[i];
                } else {
                    spike_chunk[i] = 0.0;
                }
                pot_chunk[i] = pot;
            }
        });
}
