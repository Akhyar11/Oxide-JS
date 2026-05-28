use napi_derive::napi;
use rayon::prelude::*;

pub mod activation;
pub mod layers;
pub mod loss;
pub mod math;
pub mod optimizer;
pub mod bpe;

const CLIP_PARALLEL_THRESHOLD: usize = 16 * 1024;

#[inline(always)]
fn clip_chunk(data: &mut [f32], neg_limit: f32, limit: f32) {
    for val in data.iter_mut() {
        *val = val.clamp(neg_limit, limit);
    }
}

#[napi]
pub fn clip_gradients_native(mut data: napi::bindgen_prelude::Float32Array, limit: f64) {
    let limit = limit as f32;
    let neg_limit = -limit;
    let data_slice = &mut *data;
    if data_slice.len() < CLIP_PARALLEL_THRESHOLD {
        clip_chunk(data_slice, neg_limit, limit);
    } else {
        const CHUNK: usize = 1024;
        data_slice.par_chunks_mut(CHUNK).for_each(|chunk| {
            clip_chunk(chunk, neg_limit, limit);
        });
    }
}
