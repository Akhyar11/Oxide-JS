use napi_derive::napi;
use napi::bindgen_prelude::Float32Array;
use rayon::prelude::*;

#[napi]
pub fn apply_add_only_delta_native(
    mut kernel: Float32Array,
    mut bias: Float32Array,
    inputs: Float32Array,
    error_signal: Float32Array,
    learning_rate: f64,
    batch: u32,
    in_features: u32,
    units: u32,
    use_bias: bool
) {
    let in_feat = in_features as usize;
    let u = units as usize;
    let b_size = batch as usize;
    let lr = learning_rate as f32;

    let kernel_slice: &mut [f32] = &mut kernel;
    let bias_slice: &mut [f32] = &mut bias;
    let in_slice: &[f32] = &inputs;
    let err_slice: &[f32] = &error_signal;

    kernel_slice.par_chunks_mut(u).enumerate().for_each(|(k, kernel_row)| {
        for b in 0..b_size {
            if in_slice[b * in_feat + k] > 0.5 {
                let err_offset = b * u;
                for j in 0..u {
                    kernel_row[j] += lr * err_slice[err_offset + j];
                }
            }
        }
        for j in 0..u {
            kernel_row[j] = kernel_row[j].clamp(-1.0, 1.0);
        }
    });

    if use_bias {
        bias_slice.par_iter_mut().enumerate().for_each(|(j, b_val)| {
            let mut sum = 0.0;
            for b in 0..b_size {
                sum += err_slice[b * u + j];
            }
            *b_val += (lr * sum) / (b_size as f32);
            *b_val = b_val.clamp(-1.0, 1.0);
        });
    }
}
