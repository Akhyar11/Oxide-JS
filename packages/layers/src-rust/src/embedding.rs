use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
pub fn embedding_forward_native(
    inputs: Float32Array,
    embeddings: Float32Array,
    vocab_size: u32,
    embedding_dim: u32,
    mut out: Float32Array,
) -> napi::Result<()> {
    let total_tokens = inputs.len();
    let embed_dim = embedding_dim as usize;
    let v_size = vocab_size as usize;

    let inputs_slice = &*inputs;
    let embed_slice = &*embeddings;
    let out_slice = &mut *out;

    for i in 0..total_tokens {
        let idx = inputs_slice[i].floor() as isize;
        if idx < 0 || idx >= v_size as isize {
            return Err(napi::Error::from_reason(format!(
                "[Embedding] Token index {} is out of vocabulary bounds [0, {}].",
                idx,
                v_size - 1
            )));
        }
        let dest_offset = i * embed_dim;
        let src_offset = idx as usize * embed_dim;
        for j in 0..embed_dim {
            out_slice[dest_offset + j] = embed_slice[src_offset + j];
        }
    }
    Ok(())
}

#[napi]
pub fn embedding_backward_native(
    grad_out: Float32Array,
    inputs: Float32Array,
    vocab_size: u32,
    embedding_dim: u32,
    mut grad_embed: Float32Array,
) -> napi::Result<()> {
    let total_tokens = inputs.len();
    let embed_dim = embedding_dim as usize;
    let v_size = vocab_size as usize;

    let grad_out_slice = &*grad_out;
    let inputs_slice = &*inputs;
    let grad_embed_slice = &mut *grad_embed;

    for i in 0..total_tokens {
        let idx = inputs_slice[i].floor() as isize;
        if idx < 0 || idx >= v_size as isize {
            return Err(napi::Error::from_reason(format!(
                "[Embedding] Token index {} is out of vocabulary bounds [0, {}].",
                idx,
                v_size - 1
            )));
        }
        let src_offset = i * embed_dim;
        let dest_offset = idx as usize * embed_dim;
        for j in 0..embed_dim {
            grad_embed_slice[dest_offset + j] += grad_out_slice[src_offset + j];
        }
    }
    Ok(())
}
