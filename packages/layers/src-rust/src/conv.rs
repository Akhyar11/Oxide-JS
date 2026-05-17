use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
pub fn seq2col_native(
    inputs: Float32Array,
    batch_size: u32,
    sequence_length: u32,
    input_dim: u32,
    kernel_size: u32,
    strides: u32,
    pad_left: u32,
    mut out: Float32Array,
) {
    let b_size = batch_size as usize;
    let seq_len = sequence_length as usize;
    let in_dim = input_dim as usize;
    let k_size = kernel_size as usize;
    let strd = strides as usize;
    let p_left = pad_left as usize;

    let l_out = out.len() / (b_size * k_size * in_dim);
    let patch_cols = k_size * in_dim;

    let inputs_slice = &*inputs;
    let out_slice = &mut *out;

    for b in 0..b_size {
        for i in 0..l_out {
            let out_row_idx = b * l_out + i;
            let dest_offset = out_row_idx * patch_cols;
            let t_start = (i * strd) as isize - p_left as isize;

            for k in 0..k_size {
                let t = t_start + k as isize;
                let kernel_offset = k * in_dim;

                if t >= 0 && t < seq_len as isize {
                    let src_offset = (b * seq_len + t as usize) * in_dim;
                    for c in 0..in_dim {
                        out_slice[dest_offset + kernel_offset + c] = inputs_slice[src_offset + c];
                    }
                }
            }
        }
    }
}

#[napi]
pub fn col2seq_native(
    grad_out: Float32Array,
    batch_size: u32,
    sequence_length: u32,
    input_dim: u32,
    kernel_size: u32,
    strides: u32,
    pad_left: u32,
    mut grad_in: Float32Array,
) {
    let b_size = batch_size as usize;
    let seq_len = sequence_length as usize;
    let in_dim = input_dim as usize;
    let k_size = kernel_size as usize;
    let strd = strides as usize;
    let p_left = pad_left as usize;

    let l_out = grad_out.len() / (b_size * k_size * in_dim);
    let patch_cols = k_size * in_dim;

    let grad_out_slice = &*grad_out;
    let grad_in_slice = &mut *grad_in;

    for b in 0..b_size {
        for i in 0..l_out {
            let out_row_idx = b * l_out + i;
            let src_offset = out_row_idx * patch_cols;
            let t_start = (i * strd) as isize - p_left as isize;

            for k in 0..k_size {
                let t = t_start + k as isize;
                let kernel_offset = k * in_dim;

                if t >= 0 && t < seq_len as isize {
                    let dest_offset = (b * seq_len + t as usize) * in_dim;
                    for c in 0..in_dim {
                        grad_in_slice[dest_offset + c] += grad_out_slice[src_offset + kernel_offset + c];
                    }
                }
            }
        }
    }
}

#[napi]
pub fn grid2col_native(
    inputs: Float32Array,
    batch_size: u32,
    height: u32,
    width: u32,
    channels: u32,
    kernel_rows: u32,
    kernel_cols: u32,
    stride_rows: u32,
    stride_cols: u32,
    pad_top: u32,
    pad_left: u32,
    h_out: u32,
    w_out: u32,
    mut out: Float32Array,
) {
    let b_size = batch_size as usize;
    let h = height as usize;
    let w = width as usize;
    let c = channels as usize;
    let kr = kernel_rows as usize;
    let kc = kernel_cols as usize;
    let sr = stride_rows as usize;
    let sc = stride_cols as usize;
    let p_top = pad_top as usize;
    let p_left = pad_left as usize;
    let ho = h_out as usize;
    let wo = w_out as usize;

    let patch_cols = kr * kc * c;
    let inputs_slice = &*inputs;
    let out_slice = &mut *out;

    for b in 0..b_size {
        for i in 0..ho {
            for j in 0..wo {
                let out_row_idx = b * ho * wo + i * wo + j;
                let dest_offset = out_row_idx * patch_cols;
                let h_start = (i * sr) as isize - p_top as isize;
                let w_start = (j * sc) as isize - p_left as isize;

                for kr_idx in 0..kr {
                    let h_idx = h_start + kr_idx as isize;
                    let kernel_row_offset = kr_idx * kc * c;

                    if h_idx >= 0 && h_idx < h as isize {
                        for kc_idx in 0..kc {
                            let w_idx = w_start + kc_idx as isize;
                            let kernel_col_offset = kc_idx * c;

                            if w_idx >= 0 && w_idx < w as isize {
                                let src_offset = (b * h * w + h_idx as usize * w + w_idx as usize) * c;
                                let kernel_offset = kernel_row_offset + kernel_col_offset;
                                for c_idx in 0..c {
                                    out_slice[dest_offset + kernel_offset + c_idx] = inputs_slice[src_offset + c_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[napi]
pub fn col2grid_native(
    grad_out: Float32Array,
    batch_size: u32,
    height: u32,
    width: u32,
    channels: u32,
    kernel_rows: u32,
    kernel_cols: u32,
    stride_rows: u32,
    stride_cols: u32,
    pad_top: u32,
    pad_left: u32,
    h_out: u32,
    w_out: u32,
    mut grad_in: Float32Array,
) {
    let b_size = batch_size as usize;
    let h = height as usize;
    let w = width as usize;
    let c = channels as usize;
    let kr = kernel_rows as usize;
    let kc = kernel_cols as usize;
    let sr = stride_rows as usize;
    let sc = stride_cols as usize;
    let p_top = pad_top as usize;
    let p_left = pad_left as usize;
    let ho = h_out as usize;
    let wo = w_out as usize;

    let patch_cols = kr * kc * c;
    let grad_out_slice = &*grad_out;
    let grad_in_slice = &mut *grad_in;

    for b in 0..b_size {
        for i in 0..ho {
            for j in 0..wo {
                let out_row_idx = b * ho * wo + i * wo + j;
                let src_offset = out_row_idx * patch_cols;
                let h_start = (i * sr) as isize - p_top as isize;
                let w_start = (j * sc) as isize - p_left as isize;

                for kr_idx in 0..kr {
                    let h_idx = h_start + kr_idx as isize;
                    let kernel_row_offset = kr_idx * kc * c;

                    if h_idx >= 0 && h_idx < h as isize {
                        for kc_idx in 0..kc {
                            let w_idx = w_start + kc_idx as isize;
                            let kernel_col_offset = kc_idx * c;

                            if w_idx >= 0 && w_idx < w as isize {
                                let dest_offset = (b * h * w + h_idx as usize * w + w_idx as usize) * c;
                                let kernel_offset = kernel_row_offset + kernel_col_offset;
                                for c_idx in 0..c {
                                    grad_in_slice[dest_offset + c_idx] += grad_out_slice[src_offset + kernel_offset + c_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
