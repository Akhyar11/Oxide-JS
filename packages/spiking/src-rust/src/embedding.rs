use napi_derive::napi;
use napi::bindgen_prelude::Float32Array;

#[napi]
pub fn apply_embedding_delta_native(
    mut embeddings: Float32Array,
    inputs: Float32Array,
    error_signal: Float32Array,
    learning_rate: f64,
    input_dim: u32,
    output_dim: u32
) {
    let batch = inputs.len();
    let out_dim = output_dim as usize;
    
    for b in 0..batch {
        let token_idx = inputs[b] as i32;
        if token_idx >= 0 && token_idx < input_dim as i32 {
            let token_idx = token_idx as usize;
            let emb_offset = token_idx * out_dim;
            let err_offset = b * out_dim;
            for j in 0..out_dim {
                embeddings[emb_offset + j] += learning_rate as f32 * error_signal[err_offset + j];
                embeddings[emb_offset + j] = embeddings[emb_offset + j].clamp(-1.0, 1.0);
            }
        }
    }
}
