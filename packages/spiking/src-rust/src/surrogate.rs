use napi_derive::napi;
use napi::bindgen_prelude::Float32Array;
use rayon::prelude::*;

#[napi]
pub fn mask_surrogate_native(
    mut error_signal: Float32Array,
    potentials: Float32Array,
    threshold: Float32Array,
    window_size: f64
) {
    let units = threshold.len();
    if units == 0 { return; }
    
    let err_slice: &mut [f32] = &mut error_signal;
    let pot_slice: &[f32] = &potentials;
    let thresh_slice: &[f32] = &threshold;

    err_slice.par_chunks_mut(units)
        .zip(pot_slice.par_chunks(units))
        .for_each(|(err_chunk, pot_chunk)| {
            for i in 0..units {
                if (pot_chunk[i] - thresh_slice[i]).abs() > window_size as f32 {
                    err_chunk[i] = 0.0;
                }
            }
        });
}
