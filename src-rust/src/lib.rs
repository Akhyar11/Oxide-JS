use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;
use std::collections::HashMap;

pub mod rnn;
pub use rnn::*;

const ELEMENTWISE_PARALLEL_THRESHOLD: usize = 16 * 1024;
const ADAM_PARALLEL_THRESHOLD: usize = 8 * 1024;
const MASKED_SPARSE_SOFTMAX_PARALLEL_THRESHOLD: usize = 2048;
const MASKED_SPARSE_SOFTMAX_TOKEN_BLOCK: usize = 16;
const DENSE_LINEAR_BACKWARD_PARALLEL_THRESHOLD: usize = 64 * 64;

#[napi(object)]
pub struct MaskedSparseSoftmaxCrossEntropyResult {
    pub loss: f64,
    pub valid_tokens: u32,
}

#[napi(object)]
pub struct EmbeddingSparseBackwardResult {
    pub unique_indices: Int32Array,
    pub grad: Float32Array,
}

#[napi]
pub fn dot_product(
    a_data: Float32Array,
    a_shape: Vec<u32>,
    b_data: Float32Array,
    b_shape: Vec<u32>,
    trans_a: bool,
    trans_b: bool,
) -> Float32Array {
    let a_rows = if trans_a { a_shape[1] } else { a_shape[0] } as usize;
    let b_cols = if trans_b { b_shape[0] } else { b_shape[1] } as usize;
    let result = vec![0.0; a_rows * b_cols];
    
    // We'll keep the existing one for compatibility but optimize it later
    // or just call the into version
    let out_array = Float32Array::from(result);
    dot_product_into(a_data, a_shape, b_data, b_shape, out_array.clone(), trans_a, trans_b);
    out_array
}

#[napi]
pub fn dot_product_into(
    a_data: Float32Array,
    a_shape: Vec<u32>,
    b_data: Float32Array,
    b_shape: Vec<u32>,
    mut out_data: Float32Array,
    trans_a: bool,
    trans_b: bool,
) {
    let a_rows_orig = a_shape[0] as usize;
    let a_cols_orig = a_shape[1] as usize;
    let b_rows_orig = b_shape[0] as usize;
    let b_cols_orig = b_shape[1] as usize;
    dot_product_into_impl(
        a_data,
        a_rows_orig,
        a_cols_orig,
        b_data,
        b_rows_orig,
        b_cols_orig,
        out_data,
        trans_a,
        trans_b,
    );
}

#[napi]
pub fn dot_product_into_dims(
    a_data: Float32Array,
    a_rows: u32,
    a_cols: u32,
    b_data: Float32Array,
    b_rows: u32,
    b_cols: u32,
    out_data: Float32Array,
    trans_a: bool,
    trans_b: bool,
) {
    dot_product_into_impl(
        a_data,
        a_rows as usize,
        a_cols as usize,
        b_data,
        b_rows as usize,
        b_cols as usize,
        out_data,
        trans_a,
        trans_b,
    );
}

fn dot_product_into_impl(
    a_data: Float32Array,
    a_rows_orig: usize,
    a_cols_orig: usize,
    b_data: Float32Array,
    b_rows_orig: usize,
    b_cols_orig: usize,
    mut out_data: Float32Array,
    trans_a: bool,
    trans_b: bool,
) {
    let m = if trans_a { a_cols_orig } else { a_rows_orig };
    let k = if trans_a { a_rows_orig } else { a_cols_orig };
    let b_rows = if trans_b { b_cols_orig } else { b_rows_orig };
    let n = if trans_b { b_rows_orig } else { b_cols_orig };

    if k != b_rows {
        panic!("Dimension mismatch: {}x{} * {}x{}", m, k, b_rows, n);
    }

    let (rsa, csa) = if trans_a {
        (1, a_cols_orig as isize)
    } else {
        (a_cols_orig as isize, 1)
    };

    let (rsb, csb) = if trans_b {
        (1, b_cols_orig as isize)
    } else {
        (b_cols_orig as isize, 1)
    };

    let rsc = n as isize;
    let csc = 1;

    unsafe {
        matrixmultiply::sgemm(
            m, k, n,
            1.0,
            a_data.as_ptr(), rsa, csa,
            b_data.as_ptr(), rsb, csb,
            0.0,
            out_data.as_mut_ptr(), rsc, csc,
        );
    }
}

#[napi]
pub fn add_matrices_into(a: Float32Array, b: Float32Array, mut out: Float32Array) {
    let a_slice = &*a;
    let b_slice = &*b;
    let out_slice = &mut *out;
    if out_slice.len() < ELEMENTWISE_PARALLEL_THRESHOLD {
        for i in 0..out_slice.len() {
            out_slice[i] = a_slice[i] + b_slice[i];
        }
    } else {
        out_slice.par_iter_mut().enumerate().for_each(|(i, val)| {
            *val = a_slice[i] + b_slice[i];
        });
    }
}

#[napi]
pub fn sub_matrices_into(a: Float32Array, b: Float32Array, mut out: Float32Array) {
    let a_slice = &*a;
    let b_slice = &*b;
    let out_slice = &mut *out;
    if out_slice.len() < ELEMENTWISE_PARALLEL_THRESHOLD {
        for i in 0..out_slice.len() {
            out_slice[i] = a_slice[i] - b_slice[i];
        }
    } else {
        out_slice.par_iter_mut().enumerate().for_each(|(i, val)| {
            *val = a_slice[i] - b_slice[i];
        });
    }
}

#[napi]
pub fn mul_matrices_into(a: Float32Array, b: Float32Array, mut out: Float32Array) {
    let a_slice = &*a;
    let b_slice = &*b;
    let out_slice = &mut *out;
    if out_slice.len() < ELEMENTWISE_PARALLEL_THRESHOLD {
        for i in 0..out_slice.len() {
            out_slice[i] = a_slice[i] * b_slice[i];
        }
    } else {
        out_slice.par_iter_mut().enumerate().for_each(|(i, val)| {
            *val = a_slice[i] * b_slice[i];
        });
    }
}

#[napi]
pub fn div_matrices_into(a: Float32Array, b: Float32Array, mut out: Float32Array) {
    let a_slice = &*a;
    let b_slice = &*b;
    let out_slice = &mut *out;
    if out_slice.len() < ELEMENTWISE_PARALLEL_THRESHOLD {
        for i in 0..out_slice.len() {
            out_slice[i] = a_slice[i] / b_slice[i];
        }
    } else {
        out_slice.par_iter_mut().enumerate().for_each(|(i, val)| {
            *val = a_slice[i] / b_slice[i];
        });
    }
}

#[napi]
pub fn softmax_native_into(data: Float32Array, rows: u32, cols: u32, is_row: bool, mut out: Float32Array) {
    let r = rows as usize;
    let c = cols as usize;
    out.copy_from_slice(&data);

    if is_row {
        for i in 0..r {
            let offset = i * c;
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..c { if out[offset + j] > max_val { max_val = out[offset + j]; } }
            if !max_val.is_finite() {
                let uniform = 1.0 / c as f32;
                for j in 0..c { out[offset + j] = uniform; }
                continue;
            }
            let mut sum_exp = 0.0f32;
            for j in 0..c {
                let exp_val = (out[offset + j] - max_val).exp();
                out[offset + j] = exp_val;
                sum_exp += exp_val;
            }
            if !sum_exp.is_finite() || sum_exp <= 0.0 {
                let uniform = 1.0 / c as f32;
                for j in 0..c { out[offset + j] = uniform; }
                continue;
            }
            for j in 0..c { out[offset + j] /= sum_exp; }
        }
    } else {
        for j in 0..c {
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..r { if out[i * c + j] > max_val { max_val = out[i * c + j]; } }
            if !max_val.is_finite() {
                let uniform = 1.0 / r as f32;
                for i in 0..r { out[i * c + j] = uniform; }
                continue;
            }
            let mut sum_exp = 0.0f32;
            for i in 0..r {
                let idx = i * c + j;
                let exp_val = (out[idx] - max_val).exp();
                out[idx] = exp_val;
                sum_exp += exp_val;
            }
            if !sum_exp.is_finite() || sum_exp <= 0.0 {
                let uniform = 1.0 / r as f32;
                for i in 0..r { out[i * c + j] = uniform; }
                continue;
            }
            for i in 0..r { out[i * c + j] /= sum_exp; }
        }
    }
}

#[napi]
pub fn softmax_backward_native_into(s_data: Float32Array, g_data: Float32Array, rows: u32, cols: u32, is_row: bool, mut out: Float32Array) {
    let r = rows as usize;
    let c = cols as usize;
    if is_row {
        for i in 0..r {
            let offset = i * c;
            let mut sum_grad_s = 0.0;
            for j in 0..c { sum_grad_s += s_data[offset + j] * g_data[offset + j]; }
            for j in 0..c {
                let idx = offset + j;
                out[idx] = s_data[idx] * (g_data[idx] - sum_grad_s);
            }
        }
    } else {
        for j in 0..c {
            let mut sum_grad_s = 0.0;
            for i in 0..r { sum_grad_s += s_data[i * c + j] * g_data[i * c + j]; }
            for i in 0..r {
                let idx = i * c + j;
                out[idx] = s_data[idx] * (g_data[idx] - sum_grad_s);
            }
        }
    }
}

#[napi]
pub fn layer_norm_native_into(
    x_data: Float32Array,
    gamma: Float32Array,
    beta: Float32Array,
    rows: u32,
    cols: u32,
    eps: f64,
    mut out_res: Float32Array,
    mut out_norm: Float32Array,
    mut out_means: Float32Array,
    mut out_stds: Float32Array,
) {
    let r = rows as usize;
    let c = cols as usize;
    let eps_f32 = eps as f32;
    let x_slice = &*x_data;
    let gamma_slice = &*gamma;
    let beta_slice = &*beta;
    let means_slice = &mut *out_means;
    let stds_slice = &mut *out_stds;
    let norm_slice = &mut *out_norm;
    let res_slice = &mut *out_res;

    means_slice
        .par_iter_mut()
        .zip(stds_slice.par_iter_mut())
        .enumerate()
        .for_each(|(j, (mean_out, std_out))| {
            let mut sum = 0.0f32;
            for i in 0..r {
                sum += x_slice[i * c + j];
            }
            let m = sum / (r as f32);
            *mean_out = m;

            let mut sum_sq = 0.0f32;
            for i in 0..r {
                let diff = x_slice[i * c + j] - m;
                sum_sq += diff * diff;
            }
            *std_out = (sum_sq / (r as f32) + eps_f32).sqrt();
        });

    norm_slice
        .par_chunks_mut(c)
        .zip(res_slice.par_chunks_mut(c))
        .enumerate()
        .for_each(|(i, (norm_row, res_row))| {
            let g = gamma_slice[i];
            let b = beta_slice[i];
            let row_offset = i * c;
            for j in 0..c {
                let idx = row_offset + j;
                let norm = (x_slice[idx] - means_slice[j]) / stds_slice[j];
                norm_row[j] = norm;
                res_row[j] = norm * g + b;
            }
        });
}

#[napi]
pub fn layer_norm_backward_native_into(
    err_data: Float32Array,
    norm_data: Float32Array,
    gamma_data: Float32Array,
    rows: u32,
    cols: u32,
    std_data: Float32Array,
    mut d_gamma_out: Float32Array,
    mut d_beta_out: Float32Array,
    mut dx_out: Float32Array,
) {
    let r = rows as usize;
    let c = cols as usize;
    let err_slice = &*err_data;
    let norm_slice = &*norm_data;
    let gamma_slice = &*gamma_data;
    let std_slice = &*std_data;

    let d_gamma_slice = &mut *d_gamma_out;
    let d_beta_slice = &mut *d_beta_out;
    let dx_slice = &mut *dx_out;

    d_gamma_slice
        .par_iter_mut()
        .zip(d_beta_slice.par_iter_mut())
        .enumerate()
        .for_each(|(i, (dg, db))| {
            let mut sum_g = 0.0f32;
            let mut sum_b = 0.0f32;
            let row_offset = i * c;
            for j in 0..c {
                let idx = row_offset + j;
                sum_g += err_slice[idx] * norm_slice[idx];
                sum_b += err_slice[idx];
            }
            *dg = sum_g;
            *db = sum_b;
        });

    let mut sum1_cols = vec![0.0f32; c];
    let mut sum2_cols = vec![0.0f32; c];
    sum1_cols
        .par_iter_mut()
        .zip(sum2_cols.par_iter_mut())
        .enumerate()
        .for_each(|(j, (sum1_out, sum2_out))| {
            let mut sum1 = 0.0f32;
            let mut sum2 = 0.0f32;
            for i in 0..r {
                let idx = i * c + j;
                let e = err_slice[idx] * gamma_slice[i];
                sum1 += e;
                sum2 += e * norm_slice[idx];
            }
            *sum1_out = sum1;
            *sum2_out = sum2;
        });

    let inv_r = 1.0f32 / (r as f32);
    dx_slice
        .par_chunks_mut(c)
        .enumerate()
        .for_each(|(i, dx_row)| {
            let g = gamma_slice[i];
            let row_offset = i * c;
            for j in 0..c {
                let idx = row_offset + j;
                dx_row[j] =
                    (g * err_slice[idx] - (sum1_cols[j] * inv_r) - (norm_slice[idx] * sum2_cols[j] * inv_r))
                        / std_slice[j];
            }
        });
}

#[napi]
pub fn relu_native_into(input: Float32Array, mut out_res: Float32Array, mut out_grad: Float32Array) {
    for i in 0..input.len() {
        let val = input[i];
        if val > 0.0 {
            out_res[i] = val;
            out_grad[i] = 1.0;
        } else {
            out_res[i] = 0.0;
            out_grad[i] = 0.0;
        }
    }
}

#[napi]
pub fn sigmoid_native_into(input: Float32Array, mut out_res: Float32Array, mut out_grad: Float32Array) {
    for i in 0..input.len() {
        let val = 1.0 / (1.0 + (-input[i]).exp());
        out_res[i] = val;
        out_grad[i] = val * (1.0 - val);
    }
}

#[napi]
pub fn tanh_native_into(input: Float32Array, mut out_res: Float32Array, mut out_grad: Float32Array) {
    for i in 0..input.len() {
        let val = input[i].tanh();
        out_res[i] = val;
        out_grad[i] = 1.0 - val * val;
    }
}

#[napi]
pub fn embedding_forward_native_into(
    indices: Vec<f64>,
    weight_data: Float32Array,
    vocab_size: u32,
    embedding_dim: u32,
    pad_token_id: Option<i32>,
    mut out: Float32Array
) {
    let seq_len = indices.len();
    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;
    for i in 0..out.len() { out[i] = 0.0; }
    for j in 0..seq_len {
        let raw = indices[j];
        if !raw.is_finite() || raw < 0.0 || raw >= v_size as f64 { continue; }
        let token_idx = raw as usize;
        if let Some(pad_id) = pad_token_id { if pad_id >= 0 && token_idx == pad_id as usize { continue; } }
        for i in 0..dim { out[i * seq_len + j] = weight_data[i * v_size + token_idx]; }
    }
}

#[napi]
pub fn embedding_forward_native_int32_into(
    indices: Int32Array,
    weight_data: Float32Array,
    vocab_size: u32,
    embedding_dim: u32,
    pad_token_id: Option<i32>,
    mut out: Float32Array
) {
    let seq_len = indices.len();
    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;
    for i in 0..out.len() { out[i] = 0.0; }
    for j in 0..seq_len {
        let raw = indices[j];
        if raw < 0 || raw >= vocab_size as i32 { continue; }
        let token_idx = raw as usize;
        if let Some(pad_id) = pad_token_id { if pad_id >= 0 && token_idx == pad_id as usize { continue; } }
        for i in 0..dim { out[i * seq_len + j] = weight_data[i * v_size + token_idx]; }
    }
}

#[napi]
pub fn embedding_backward_native(
    indices: Vec<f64>,
    err_data: Float32Array,
    mut grad_data: Float32Array,
    vocab_size: u32,
    embedding_dim: u32,
    pad_token_id: Option<i32>
) {
    let seq_len = indices.len();
    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;

    for i in 0..dim {
        for j in 0..seq_len {
            let raw = indices[j];
            if !raw.is_finite() || raw < 0.0 || raw >= v_size as f64 { continue; }
            let token_idx = raw as usize;
            if let Some(pad_id) = pad_token_id {
                if pad_id >= 0 && token_idx == pad_id as usize { continue; }
            }
            grad_data[i * v_size + token_idx] += err_data[i * seq_len + j];
        }
    }
}

#[napi]
pub fn embedding_backward_native_int32(
    indices: Int32Array,
    err_data: Float32Array,
    mut grad_data: Float32Array,
    vocab_size: u32,
    embedding_dim: u32,
    pad_token_id: Option<i32>
) {
    let seq_len = indices.len();
    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;

    for i in 0..dim {
        for j in 0..seq_len {
            let raw = indices[j];
            if raw < 0 || raw >= vocab_size as i32 { continue; }
            let token_idx = raw as usize;
            if let Some(pad_id) = pad_token_id {
                if pad_id >= 0 && token_idx == pad_id as usize { continue; }
            }
            grad_data[i * v_size + token_idx] += err_data[i * seq_len + j];
        }
    }
}

#[napi]
pub fn embedding_backward_sparse_native(
    indices: Int32Array,
    err_data: Float32Array,
    embedding_dim: u32,
    pad_token_id: Option<i32>,
) -> EmbeddingSparseBackwardResult {
    let mut unique_map = HashMap::new();
    let mut unique_vec = Vec::new();
    let seq_len = indices.len();
    let mut pos_in_unique = Vec::with_capacity(seq_len);

    for &idx in indices.iter() {
        if let Some(pad_id) = pad_token_id {
            if idx == pad_id {
                pos_in_unique.push(-1);
                continue;
            }
        }
        let pos = *unique_map.entry(idx).or_insert_with(|| {
            let p = unique_vec.len();
            unique_vec.push(idx);
            p
        });
        pos_in_unique.push(pos as i32);
    }

    let num_unique = unique_vec.len();
    let dim = embedding_dim as usize;
    let mut grad = vec![0.0f32; num_unique * dim];

    for j in 0..seq_len {
        let u_idx = pos_in_unique[j];
        if u_idx < 0 { continue; }
        let u_ptr = u_idx as usize;
        for i in 0..dim {
            grad[i * num_unique + u_ptr] += err_data[i * seq_len + j];
        }
    }

    EmbeddingSparseBackwardResult {
        unique_indices: Int32Array::from(unique_vec),
        grad: Float32Array::from(grad),
    }
}

#[napi]
pub fn convolution_native_into(
    a_data: Float32Array,
    a_rows: u32,
    a_cols: u32,
    k_data: Float32Array,
    k_rows: u32,
    k_cols: u32,
    mut out: Float32Array
) {
    let ac = a_cols as usize;
    let kr = k_rows as usize;
    let kc = k_cols as usize;
    let out_rows = (a_rows - k_rows + 1) as usize;
    let out_cols = (a_cols - k_cols + 1) as usize;

    for i in 0..out_rows {
        let r_offset = i * out_cols;
        for j in 0..out_cols {
            let mut sum = 0.0;
            for k in 0..kr {
                let a_offset = (i + k) * ac + j;
                let k_offset = k * kc;
                for l in 0..kc {
                    sum += a_data[a_offset + l] * k_data[k_offset + l];
                }
            }
            out[r_offset + j] = sum;
        }
    }
}

#[napi]
pub fn conv_backward_input_native_into(
    err_data: Float32Array,
    err_rows: u32,
    err_cols: u32,
    input_data: Float32Array,
    input_rows: u32,
    input_cols: u32,
    out_rows: u32, 
    out_cols: u32,
    mut out: Float32Array
) {
    let er = err_rows as usize;
    let ec = err_cols as usize;
    let ic = input_cols as usize;
    let oc = out_cols as usize;

    for k in 0..er {
        for l in 0..ec {
            let err_val = err_data[k * ec + l];
            if err_val == 0.0 { continue; }
            for m in 0..input_rows as usize {
                for n in 0..ic {
                    out[(m + k) * oc + (n + l)] += err_val * input_data[m * ic + n];
                }
            }
        }
    }
}

#[napi]
pub fn apply_attention_mask_native(
    mut data: Float32Array,
    pad_mask: Vec<bool>,
    rows: u32,
    cols: u32,
    scale: f64
) {
    let r = rows as usize;
    let c = cols as usize;
    let masked_value = -1e9 as f32;
    let scale_f32 = scale as f32;

    for query in 0..c {
        if pad_mask[query] {
            for key in 0..r {
                data[key * c + query] = masked_value;
            }
            data[query * c + query] = 0.0;
            continue;
        }

        for key in 0..r {
            if pad_mask[key] || key > query {
                data[key * c + query] = masked_value;
            } else {
                data[key * c + query] *= scale_f32;
            }
        }
    }
}

#[napi]
pub fn adam_update_native(
    grad: Float32Array,
    mut m: Float32Array,
    mut v: Float32Array,
    mut buffer: Float32Array,
    t: u32,
    alpha: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64
) {
    let alpha = alpha as f32;
    let beta1 = beta1 as f32;
    let beta2 = beta2 as f32;
    let epsilon = epsilon as f32;

    let one_minus_beta1 = 1.0 - beta1;
    let one_minus_beta2 = 1.0 - beta2;
    let bias_correction1 = 1.0 / (1.0 - beta1.powi(t as i32));
    let bias_correction2 = 1.0 / (1.0 - beta2.powi(t as i32));

    let grad_slice = &*grad;
    let m_slice = &mut *m;
    let v_slice = &mut *v;
    let buffer_slice = &mut *buffer;

    if buffer_slice.len() < ADAM_PARALLEL_THRESHOLD {
        for i in 0..buffer_slice.len() {
            let g = grad_slice[i];
            let m_new = beta1 * m_slice[i] + one_minus_beta1 * g;
            let v_new = beta2 * v_slice[i] + one_minus_beta2 * g * g;
            m_slice[i] = m_new;
            v_slice[i] = v_new;

            let m_hat = m_new * bias_correction1;
            let v_hat = v_new * bias_correction2;
            buffer_slice[i] = alpha * m_hat / (v_hat.sqrt() + epsilon);
        }
    } else {
        buffer_slice.par_iter_mut()
            .zip(grad_slice.par_iter())
            .zip(m_slice.par_iter_mut())
            .zip(v_slice.par_iter_mut())
            .for_each(|(((b_val, &g), m_val), v_val)| {
                let m_new = beta1 * (*m_val) + one_minus_beta1 * g;
                let v_new = beta2 * (*v_val) + one_minus_beta2 * g * g;
                *m_val = m_new;
                *v_val = v_new;

                let m_hat = m_new * bias_correction1;
                let v_hat = v_new * bias_correction2;
                *b_val = alpha * m_hat / (v_hat.sqrt() + epsilon);
            });
    }
}

#[napi]
pub fn adam_sparse_update_native(
    indices: Int32Array,
    grad: Float32Array,
    mut weight: Float32Array,
    mut m: Float32Array,
    mut v: Float32Array,
    t: u32,
    alpha: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    vocab_size: u32,
    embedding_dim: u32,
) {
    let alpha = alpha as f32;
    let beta1 = beta1 as f32;
    let beta2 = beta2 as f32;
    let epsilon = epsilon as f32;

    let one_minus_beta1 = 1.0 - beta1;
    let one_minus_beta2 = 1.0 - beta2;
    let bias_correction1 = 1.0 / (1.0 - beta1.powi(t as i32));
    let bias_correction2 = 1.0 / (1.0 - beta2.powi(t as i32));

    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;
    let num_unique = indices.len();

    let indices_slice = &*indices;
    let grad_slice = &*grad;
    let weight_slice = &mut *weight;
    let m_slice = &mut *m;
    let v_slice = &mut *v;

    for j in 0..num_unique {
        let token_idx = indices_slice[j] as usize;
        for i in 0..dim {
            let full_idx = i * v_size + token_idx;
            let grad_idx = i * num_unique + j;

            let g = grad_slice[grad_idx];
            let m_new = beta1 * m_slice[full_idx] + one_minus_beta1 * g;
            let v_new = beta2 * v_slice[full_idx] + one_minus_beta2 * g * g;
            m_slice[full_idx] = m_new;
            v_slice[full_idx] = v_new;

            let m_hat = m_new * bias_correction1;
            let v_hat = v_new * bias_correction2;
            weight_slice[full_idx] -= alpha * m_hat / (v_hat.sqrt() + epsilon);
        }
    }
}

#[napi]
pub fn sgd_update_native(grad: Float32Array, mut out: Float32Array, alpha: f64) {
    let alpha = alpha as f32;
    let grad_slice = &*grad;
    let out_slice = &mut *out;
    if out_slice.len() < ELEMENTWISE_PARALLEL_THRESHOLD {
        for i in 0..out_slice.len() { out_slice[i] = grad_slice[i] * alpha; }
    } else {
        out_slice.par_iter_mut().zip(grad_slice.par_iter()).for_each(|(o, &g)| {
            *o = g * alpha;
        });
    }
}

#[napi]
pub fn sgd_sparse_update_native(
    indices: Int32Array,
    grad: Float32Array,
    mut weight: Float32Array,
    alpha: f64,
    vocab_size: u32,
    embedding_dim: u32,
) {
    let alpha = alpha as f32;
    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;
    let num_unique = indices.len();
    let indices_slice = &*indices;
    let grad_slice = &*grad;
    let weight_slice = &mut *weight;

    for j in 0..num_unique {
        let token_idx = indices_slice[j] as usize;
        for i in 0..dim {
            let full_idx = i * v_size + token_idx;
            let grad_idx = i * num_unique + j;
            weight_slice[full_idx] -= alpha * grad_slice[grad_idx];
        }
    }
}

#[napi]
pub fn adagrad_update_native(
    grad: Float32Array,
    mut sum: Float32Array,
    mut out: Float32Array,
    alpha: f64,
    epsilon: f64
) {
    let alpha = alpha as f32;
    let eps = epsilon as f32;
    let grad_slice = &*grad;
    let sum_slice = &mut *sum;
    let out_slice = &mut *out;

    if out_slice.len() < ELEMENTWISE_PARALLEL_THRESHOLD {
        for i in 0..out_slice.len() {
            let g = grad_slice[i];
            sum_slice[i] += g * g;
            out_slice[i] = alpha * g / (sum_slice[i].sqrt() + eps);
        }
    } else {
        out_slice.par_iter_mut()
            .zip(grad_slice.par_iter())
            .zip(sum_slice.par_iter_mut())
            .for_each(|((o, &g), s)| {
                *s += g * g;
                *o = alpha * g / (s.sqrt() + eps);
            });
    }
}

#[napi]
pub fn adagrad_sparse_update_native(
    indices: Int32Array,
    grad: Float32Array,
    mut weight: Float32Array,
    mut sum: Float32Array,
    alpha: f64,
    epsilon: f64,
    vocab_size: u32,
    embedding_dim: u32,
) {
    let alpha = alpha as f32;
    let eps = epsilon as f32;
    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;
    let num_unique = indices.len();
    let indices_slice = &*indices;
    let grad_slice = &*grad;
    let weight_slice = &mut *weight;
    let sum_slice = &mut *sum;

    for j in 0..num_unique {
        let token_idx = indices_slice[j] as usize;
        for i in 0..dim {
            let full_idx = i * v_size + token_idx;
            let grad_idx = i * num_unique + j;
            let g = grad_slice[grad_idx];
            sum_slice[full_idx] += g * g;
            weight_slice[full_idx] -= alpha * g / (sum_slice[full_idx].sqrt() + eps);
        }
    }
}

#[napi]
pub fn momentum_update_native(
    grad: Float32Array,
    mut v: Float32Array,
    mut out: Float32Array,
    alpha: f64,
    beta: f64
) {
    let alpha = alpha as f32;
    let beta = beta as f32;
    let grad_slice = &*grad;
    let v_slice = &mut *v;
    let out_slice = &mut *out;

    if out_slice.len() < ELEMENTWISE_PARALLEL_THRESHOLD {
        for i in 0..out_slice.len() {
            v_slice[i] = beta * v_slice[i] + alpha * grad_slice[i];
            out_slice[i] = v_slice[i];
        }
    } else {
        out_slice.par_iter_mut()
            .zip(grad_slice.par_iter())
            .zip(v_slice.par_iter_mut())
            .for_each(|((o, &g), v_val)| {
                *v_val = beta * (*v_val) + alpha * g;
                *o = *v_val;
            });
    }
}

#[napi]
pub fn momentum_sparse_update_native(
    indices: Int32Array,
    grad: Float32Array,
    mut weight: Float32Array,
    mut v: Float32Array,
    alpha: f64,
    beta: f64,
    vocab_size: u32,
    embedding_dim: u32,
) {
    let alpha = alpha as f32;
    let beta = beta as f32;
    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;
    let num_unique = indices.len();
    let indices_slice = &*indices;
    let grad_slice = &*grad;
    let weight_slice = &mut *weight;
    let v_slice = &mut *v;

    for j in 0..num_unique {
        let token_idx = indices_slice[j] as usize;
        for i in 0..dim {
            let full_idx = i * v_size + token_idx;
            let grad_idx = i * num_unique + j;
            v_slice[full_idx] = beta * v_slice[full_idx] + alpha * grad_slice[grad_idx];
            weight_slice[full_idx] -= v_slice[full_idx];
        }
    }
}

#[napi]
pub fn nag_update_native(
    grad: Float32Array,
    mut v: Float32Array,
    mut out: Float32Array,
    alpha: f64,
    beta: f64
) {
    let alpha = alpha as f32;
    let beta = beta as f32;
    let grad_slice = &*grad;
    let v_slice = &mut *v;
    let out_slice = &mut *out;

    if out_slice.len() < ELEMENTWISE_PARALLEL_THRESHOLD {
        for i in 0..out_slice.len() {
            let v_old = v_slice[i];
            let v_new = beta * v_old + alpha * (grad_slice[i] - beta * v_old);
            v_slice[i] = v_new;
            out_slice[i] = v_new;
        }
    } else {
        out_slice.par_iter_mut()
            .zip(grad_slice.par_iter())
            .zip(v_slice.par_iter_mut())
            .for_each(|((o, &g), v_val)| {
                let v_old = *v_val;
                let v_new = beta * v_old + alpha * (g - beta * v_old);
                *v_val = v_new;
                *o = v_new;
            });
    }
}

#[napi]
pub fn nag_sparse_update_native(
    indices: Int32Array,
    grad: Float32Array,
    mut weight: Float32Array,
    mut v: Float32Array,
    alpha: f64,
    beta: f64,
    vocab_size: u32,
    embedding_dim: u32,
) {
    let alpha = alpha as f32;
    let beta = beta as f32;
    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;
    let num_unique = indices.len();
    let indices_slice = &*indices;
    let grad_slice = &*grad;
    let weight_slice = &mut *weight;
    let v_slice = &mut *v;

    for j in 0..num_unique {
        let token_idx = indices_slice[j] as usize;
        for i in 0..dim {
            let full_idx = i * v_size + token_idx;
            let grad_idx = i * num_unique + j;
            let v_old = v_slice[full_idx];
            let v_new = beta * v_old + alpha * (grad_slice[grad_idx] - beta * v_old);
            v_slice[full_idx] = v_new;
            weight_slice[full_idx] -= v_new;
        }
    }
}

#[napi]
pub fn add_in_place(mut a: Float32Array, b: Float32Array) {
    assert_eq!(a.len(), b.len(), "add_in_place: length mismatch {} != {}", a.len(), b.len());
    for i in 0..a.len() { a[i] += b[i]; }
}

#[napi]
pub fn sub_in_place(mut a: Float32Array, b: Float32Array) {
    assert_eq!(a.len(), b.len(), "sub_in_place: length mismatch {} != {}", a.len(), b.len());
    for i in 0..a.len() { a[i] -= b[i]; }
}

#[napi]
pub fn mul_in_place(mut a: Float32Array, b: Float32Array) {
    assert_eq!(a.len(), b.len(), "mul_in_place: length mismatch {} != {}", a.len(), b.len());
    for i in 0..a.len() { a[i] *= b[i]; }
}

#[napi]
pub fn mse_native(y_true: Float32Array, y_pred: Float32Array) -> Vec<f64> {
    let mut sum_sq = 0.0;
    let n = y_true.len() as f32;
    for i in 0..y_true.len() {
        let diff = y_true[i] - y_pred[i];
        sum_sq += diff * diff;
    }
    vec![(sum_sq / n) as f64]
}
#[napi]
pub fn add_bias_native(mut data: Float32Array, bias: Float32Array, rows: u32, cols: u32) {
    let r = rows as usize;
    let c = cols as usize;
    let bias_slice = &*bias;
    let data_slice = &mut *data;
    if data_slice.len() < ELEMENTWISE_PARALLEL_THRESHOLD {
        for i in 0..r {
            let bias_value = bias_slice[i];
            let row = &mut data_slice[i * c..(i + 1) * c];
            for value in row.iter_mut() {
                *value += bias_value;
            }
        }
    } else {
        data_slice
            .par_chunks_mut(c)
            .enumerate()
            .for_each(|(i, row)| {
                let bias_value = bias_slice[i];
                for value in row.iter_mut() {
                    *value += bias_value;
                }
            });
    }
}

#[napi]
pub fn sum_axis_native(data: Float32Array, rows: u32, cols: u32, axis: u32, mut out: Float32Array) {
    let r = rows as usize;
    let c = cols as usize;
    if axis == 1 {
        // Sum across columns (result is [rows x 1])
        for i in 0..r {
            let mut sum = 0.0;
            for j in 0..c {
                sum += data[i * c + j];
            }
            out[i] = sum;
        }
    } else {
        // Sum across rows (result is [1 x cols])
        for j in 0..c {
            let mut sum = 0.0;
            for i in 0..r {
                sum += data[i * c + j];
            }
            out[j] = sum;
        }
    }
}

#[napi]
pub fn clip_gradients_native(mut data: Float32Array, limit: f64) {
    let limit = limit as f32;
    for i in 0..data.len() {
        if data[i] > limit {
            data[i] = limit;
        } else if data[i] < -limit {
            data[i] = -limit;
        }
    }
}

#[napi]
pub fn dense_linear_backward_native_into(
    err_activation: Float32Array,
    input: Float32Array,
    weight: Float32Array,
    output_units: u32,
    units: u32,
    seq_len: u32,
    clip_limit: f64,
    mut grad_weight_out: Float32Array,
    mut grad_bias_out: Float32Array,
    mut prev_err_out: Float32Array,
) {
    let output_units = output_units as usize;
    let units = units as usize;
    let seq_len = seq_len as usize;
    let clip_limit = clip_limit as f32;

    if err_activation.len() != output_units * seq_len {
        panic!(
            "dense_linear_backward_native_into: err_activation length mismatch {} != {}",
            err_activation.len(),
            output_units * seq_len
        );
    }
    if input.len() != units * seq_len {
        panic!(
            "dense_linear_backward_native_into: input length mismatch {} != {}",
            input.len(),
            units * seq_len
        );
    }
    if weight.len() != output_units * units {
        panic!(
            "dense_linear_backward_native_into: weight length mismatch {} != {}",
            weight.len(),
            output_units * units
        );
    }
    if grad_weight_out.len() != output_units * units {
        panic!(
            "dense_linear_backward_native_into: grad_weight_out length mismatch {} != {}",
            grad_weight_out.len(),
            output_units * units
        );
    }
    if grad_bias_out.len() != output_units {
        panic!(
            "dense_linear_backward_native_into: grad_bias_out length mismatch {} != {}",
            grad_bias_out.len(),
            output_units
        );
    }
    if prev_err_out.len() != units * seq_len {
        panic!(
            "dense_linear_backward_native_into: prev_err_out length mismatch {} != {}",
            prev_err_out.len(),
            units * seq_len
        );
    }

    let err_slice = &*err_activation;
    let input_slice = &*input;
    let weight_slice = &*weight;
    let grad_weight_slice = &mut *grad_weight_out;
    let grad_bias_slice = &mut *grad_bias_out;
    let prev_err_slice = &mut *prev_err_out;

    let should_clip = clip_limit >= 0.0;

    if output_units * units >= DENSE_LINEAR_BACKWARD_PARALLEL_THRESHOLD {
        grad_weight_slice
            .par_chunks_mut(units)
            .zip(grad_bias_slice.par_iter_mut())
            .enumerate()
            .for_each(|(out_idx, (grad_weight_row, grad_bias_ref))| {
                let err_row = &err_slice[out_idx * seq_len..(out_idx + 1) * seq_len];
                let mut bias_sum = 0.0f32;
                for unit_idx in 0..units {
                    let input_row = &input_slice[unit_idx * seq_len..(unit_idx + 1) * seq_len];
                    let mut sum = 0.0f32;
                    for token_idx in 0..seq_len {
                        let e = err_row[token_idx];
                        sum += e * input_row[token_idx];
                    }
                    grad_weight_row[unit_idx] = if should_clip {
                        sum.clamp(-clip_limit, clip_limit)
                    } else {
                        sum
                    };
                }
                for &e in err_row {
                    bias_sum += e;
                }
                *grad_bias_ref = if should_clip {
                    bias_sum.clamp(-clip_limit, clip_limit)
                } else {
                    bias_sum
                };
            });
    } else {
        for out_idx in 0..output_units {
            let err_row = &err_slice[out_idx * seq_len..(out_idx + 1) * seq_len];
            let grad_weight_row = &mut grad_weight_slice[out_idx * units..(out_idx + 1) * units];
            let mut bias_sum = 0.0f32;
            for unit_idx in 0..units {
                let input_row = &input_slice[unit_idx * seq_len..(unit_idx + 1) * seq_len];
                let mut sum = 0.0f32;
                for token_idx in 0..seq_len {
                    let e = err_row[token_idx];
                    sum += e * input_row[token_idx];
                }
                grad_weight_row[unit_idx] = if should_clip {
                    sum.clamp(-clip_limit, clip_limit)
                } else {
                    sum
                };
            }
            for &e in err_row {
                bias_sum += e;
            }
            grad_bias_slice[out_idx] = if should_clip {
                bias_sum.clamp(-clip_limit, clip_limit)
            } else {
                bias_sum
            };
        }
    }

    if units * seq_len >= DENSE_LINEAR_BACKWARD_PARALLEL_THRESHOLD {
        prev_err_slice
            .par_chunks_mut(seq_len)
            .enumerate()
            .for_each(|(unit_idx, prev_err_row)| {
                for token_idx in 0..seq_len {
                    let mut sum = 0.0f32;
                    for out_idx in 0..output_units {
                        sum += weight_slice[out_idx * units + unit_idx] * err_slice[out_idx * seq_len + token_idx];
                    }
                    prev_err_row[token_idx] = sum;
                }
            });
    } else {
        for unit_idx in 0..units {
            let prev_err_row = &mut prev_err_slice[unit_idx * seq_len..(unit_idx + 1) * seq_len];
            for token_idx in 0..seq_len {
                let mut sum = 0.0f32;
                for out_idx in 0..output_units {
                    sum += weight_slice[out_idx * units + unit_idx] * err_slice[out_idx * seq_len + token_idx];
                }
                prev_err_row[token_idx] = sum;
            }
        }
    }
}

#[napi]
pub fn masked_sparse_softmax_cross_entropy_into(
    logits: Float32Array,
    input_tokens: Float32Array,
    targets: Float32Array,
    seq_len: u32,
    batch_size: u32,
    vocab_size: u32,
    pad_token_id: Option<i32>,
    mut out_grad: Float32Array,
) -> MaskedSparseSoftmaxCrossEntropyResult {
    let seq_len = seq_len as usize;
    let batch_size = batch_size as usize;
    let vocab_size = vocab_size as usize;
    let total_tokens = seq_len * batch_size;
    let epsilon = 1e-15f32;
    let pad_token_id = pad_token_id.unwrap_or(-1);

    if logits.len() != vocab_size * total_tokens {
        panic!(
            "masked_sparse_softmax_cross_entropy_into: logits length mismatch {} != {}",
            logits.len(),
            vocab_size * total_tokens
        );
    }
    if input_tokens.len() != total_tokens || targets.len() != total_tokens {
        panic!(
            "masked_sparse_softmax_cross_entropy_into: token length mismatch input={} targets={} total_tokens={}",
            input_tokens.len(),
            targets.len(),
            total_tokens
        );
    }
    if out_grad.len() != logits.len() {
        panic!(
            "masked_sparse_softmax_cross_entropy_into: grad length mismatch {} != {}",
            out_grad.len(),
            logits.len()
        );
    }

    let logits_slice = &*logits;
    let input_slice = &*input_tokens;
    let target_slice = &*targets;
    let grad_slice = &mut *out_grad;

    let process_block = |block_index: usize, grad_ptr_addr: usize, logits_ptr_addr: usize| -> (f64, usize) {
        let start_token = block_index * MASKED_SPARSE_SOFTMAX_TOKEN_BLOCK;
        let end_token = (start_token + MASKED_SPARSE_SOFTMAX_TOKEN_BLOCK).min(total_tokens);
        let block_len = end_token - start_token;
        let grad_ptr = grad_ptr_addr as *mut f32;
        let logits_ptr = logits_ptr_addr as *const f32;

        let mut max_logits = [f32::NEG_INFINITY; MASKED_SPARSE_SOFTMAX_TOKEN_BLOCK];
        let mut sum_exps = [0.0f32; MASKED_SPARSE_SOFTMAX_TOKEN_BLOCK];
        let mut inv_sum_exps = [0.0f32; MASKED_SPARSE_SOFTMAX_TOKEN_BLOCK];
        let mut target_ids = [0usize; MASKED_SPARSE_SOFTMAX_TOKEN_BLOCK];
        let mut valid_mask = [false; MASKED_SPARSE_SOFTMAX_TOKEN_BLOCK];

        let mut valid_tokens = 0usize;
        for local_idx in 0..block_len {
            let token_index = start_token + local_idx;
            let batch_idx = token_index / seq_len;
            let pos = token_index % seq_len;
            let source_index = pos * batch_size + batch_idx;
            let source_token = input_slice[source_index] as i32;
            let target_token = target_slice[source_index] as i32;
            let is_valid_position =
                pos < seq_len - 1
                    && (pad_token_id < 0 || (source_token != pad_token_id && target_token != pad_token_id));

            if !is_valid_position {
                continue;
            }
            if target_token < 0 || target_token as usize >= vocab_size {
                panic!(
                    "masked_sparse_softmax_cross_entropy_into: target token {} out of range 0..{} at batch {} pos {}",
                    target_token,
                    vocab_size.saturating_sub(1),
                    batch_idx,
                    pos
                );
            }

            valid_mask[local_idx] = true;
            target_ids[local_idx] = target_token as usize;
            valid_tokens += 1;
        }

        if valid_tokens == 0 {
            for vocab_idx in 0..vocab_size {
                let row_offset = vocab_idx * total_tokens + start_token;
                unsafe {
                    std::ptr::write_bytes(grad_ptr.add(row_offset), 0, block_len);
                }
            }
            return (0.0, 0);
        }

        for vocab_idx in 0..vocab_size {
            let row_offset = vocab_idx * total_tokens + start_token;
            for local_idx in 0..block_len {
                if !valid_mask[local_idx] {
                    continue;
                }
                let value = unsafe { *logits_ptr.add(row_offset + local_idx) };
                if value > max_logits[local_idx] {
                    max_logits[local_idx] = value;
                }
            }
        }

        for vocab_idx in 0..vocab_size {
            let row_offset = vocab_idx * total_tokens + start_token;
            for local_idx in 0..block_len {
                let grad_ref = unsafe { &mut *grad_ptr.add(row_offset + local_idx) };
                if !valid_mask[local_idx] {
                    *grad_ref = 0.0;
                    continue;
                }
                let exp_val = (unsafe { *logits_ptr.add(row_offset + local_idx) } - max_logits[local_idx]).exp();
                *grad_ref = exp_val;
                sum_exps[local_idx] += exp_val;
            }
        }

        let mut total_loss = 0.0f64;
        for local_idx in 0..block_len {
            if !valid_mask[local_idx] {
                continue;
            }
            if !sum_exps[local_idx].is_finite() || sum_exps[local_idx] <= 0.0 {
                inv_sum_exps[local_idx] = 0.0;
                total_loss -= (1.0f32 / vocab_size as f32).max(epsilon).ln() as f64;
            } else {
                inv_sum_exps[local_idx] = 1.0 / sum_exps[local_idx];
            }
        }

        for vocab_idx in 0..vocab_size {
            let row_offset = vocab_idx * total_tokens + start_token;
            for local_idx in 0..block_len {
                let grad_ref = unsafe { &mut *grad_ptr.add(row_offset + local_idx) };
                if !valid_mask[local_idx] {
                    continue;
                }
                if inv_sum_exps[local_idx] == 0.0 {
                    *grad_ref = 1.0 / vocab_size as f32;
                } else {
                    *grad_ref *= inv_sum_exps[local_idx];
                }
            }
        }

        for local_idx in 0..block_len {
            if !valid_mask[local_idx] {
                continue;
            }
            let token_index = start_token + local_idx;
            let target_offset = target_ids[local_idx] * total_tokens + token_index;
            let grad_ref = unsafe { &mut *grad_ptr.add(target_offset) };
            if inv_sum_exps[local_idx] != 0.0 {
                total_loss -= (*grad_ref).max(epsilon).ln() as f64;
            }
            *grad_ref -= 1.0;
        }

        (total_loss, valid_tokens)
    };

    let total_blocks = total_tokens.div_ceil(MASKED_SPARSE_SOFTMAX_TOKEN_BLOCK);
    let (total_loss, valid_tokens) = if total_tokens >= MASKED_SPARSE_SOFTMAX_PARALLEL_THRESHOLD {
        let grad_ptr_addr = grad_slice.as_mut_ptr() as usize;
        let logits_ptr_addr = logits_slice.as_ptr() as usize;
        (0..total_blocks)
            .into_par_iter()
            .map(|block_index| process_block(block_index, grad_ptr_addr, logits_ptr_addr))
            .reduce(|| (0.0f64, 0usize), |(loss_a, valid_a), (loss_b, valid_b)| {
                (loss_a + loss_b, valid_a + valid_b)
            })
    } else {
        let grad_ptr_addr = grad_slice.as_mut_ptr() as usize;
        let logits_ptr_addr = logits_slice.as_ptr() as usize;
        let mut total_loss = 0.0f64;
        let mut valid_tokens = 0usize;
        for block_index in 0..total_blocks {
            let (loss, valid) = process_block(block_index, grad_ptr_addr, logits_ptr_addr);
            total_loss += loss;
            valid_tokens += valid;
        }
        (total_loss, valid_tokens)
    };

    if valid_tokens == 0 {
        panic!("masked_sparse_softmax_cross_entropy_into: no valid tokens");
    }

    MaskedSparseSoftmaxCrossEntropyResult {
        loss: total_loss / valid_tokens as f64,
        valid_tokens: valid_tokens as u32,
    }
}

#[napi]
pub fn project_last_token_logits_native_into(
    hidden: Float32Array,
    weight: Float32Array,
    bias: Float32Array,
    units: u32,
    seq_len: u32,
    batch_size: u32,
    vocab_size: u32,
    mut out: Float32Array,
) {
    let units = units as usize;
    let seq_len = seq_len as usize;
    let batch_size = batch_size as usize;
    let vocab_size = vocab_size as usize;
    let total_cols = seq_len * batch_size;

    if hidden.len() != units * total_cols {
        panic!(
            "project_last_token_logits_native_into: hidden length mismatch {} != {}",
            hidden.len(),
            units * total_cols
        );
    }
    if weight.len() != vocab_size * units {
        panic!(
            "project_last_token_logits_native_into: weight length mismatch {} != {}",
            weight.len(),
            vocab_size * units
        );
    }
    if bias.len() != vocab_size {
        panic!(
            "project_last_token_logits_native_into: bias length mismatch {} != {}",
            bias.len(),
            vocab_size
        );
    }
    if out.len() != vocab_size * batch_size {
        panic!(
            "project_last_token_logits_native_into: out length mismatch {} != {}",
            out.len(),
            vocab_size * batch_size
        );
    }

    let hidden_slice = &*hidden;
    let weight_slice = &*weight;
    let bias_slice = &*bias;
    let out_slice = &mut *out;

    out_slice
        .par_chunks_mut(batch_size)
        .enumerate()
        .for_each(|(vocab_idx, out_row)| {
            let weight_offset = vocab_idx * units;
            let bias_value = bias_slice[vocab_idx];
            for batch_idx in 0..batch_size {
                let token_col = (batch_idx + 1) * seq_len - 1;
                let mut sum = bias_value;
                for unit_idx in 0..units {
                    sum += weight_slice[weight_offset + unit_idx] * hidden_slice[unit_idx * total_cols + token_col];
                }
                out_row[batch_idx] = sum;
            }
        });
}

fn mha_forward_head_into(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    pad_mask: &[bool],
    out_head: &mut [f32],
    attention_head: &mut [f32],
    total_cols: usize,
    head_idx: usize,
    head_units: usize,
    seq_len: usize,
    batch_size: usize,
    scale: f32,
) {
    let head_row_start = head_idx * head_units;

    for sample_idx in 0..batch_size {
        let sample_offset = sample_idx * seq_len;
        let attn_offset = sample_idx * seq_len * seq_len;
        let attn_block = &mut attention_head[attn_offset..attn_offset + seq_len * seq_len];

        for q_pos in 0..seq_len {
            let q_col = sample_offset + q_pos;
            if pad_mask[q_col] {
                for k_pos in 0..seq_len {
                    attn_block[k_pos * seq_len + q_pos] = 0.0;
                }
                for i in 0..head_units {
                    out_head[i * total_cols + q_col] = 0.0;
                }
                continue;
            }

            let mut max_score = f32::NEG_INFINITY;
            for k_pos in 0..seq_len {
                let k_col = sample_offset + k_pos;
                let idx = k_pos * seq_len + q_pos;
                if pad_mask[k_col] || k_pos > q_pos {
                    attn_block[idx] = f32::NEG_INFINITY;
                    continue;
                }

                let mut score = 0.0f32;
                for i in 0..head_units {
                    let row = head_row_start + i;
                    score += k_data[row * total_cols + k_col] * q_data[row * total_cols + q_col];
                }
                score *= scale;
                attn_block[idx] = score;
                if score > max_score {
                    max_score = score;
                }
            }

            if !max_score.is_finite() {
                continue;
            }

            let mut sum_exp = 0.0f32;
            for k_pos in 0..seq_len {
                let idx = k_pos * seq_len + q_pos;
                let score = attn_block[idx];
                if !score.is_finite() {
                    attn_block[idx] = 0.0;
                    continue;
                }
                let exp_val = (score - max_score).exp();
                attn_block[idx] = exp_val;
                sum_exp += exp_val;
            }

            if sum_exp <= 0.0 || !sum_exp.is_finite() {
                for k_pos in 0..seq_len {
                    attn_block[k_pos * seq_len + q_pos] = 0.0;
                }
                continue;
            }

            let inv_sum = 1.0f32 / sum_exp;
            for k_pos in 0..seq_len {
                let idx = k_pos * seq_len + q_pos;
                attn_block[idx] *= inv_sum;
            }

            for i in 0..head_units {
                let row = head_row_start + i;
                let out_idx = i * total_cols + q_col;
                let mut sum = 0.0f32;
                for k_pos in 0..seq_len {
                    let k_col = sample_offset + k_pos;
                    sum += v_data[row * total_cols + k_col] * attn_block[k_pos * seq_len + q_pos];
                }
                out_head[out_idx] = sum;
            }
        }
    }
}

fn mha_backward_head_into(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    attention_head: &[f32],
    d_out_data: &[f32],
    pad_mask: &[bool],
    d_q_head: &mut [f32],
    d_k_head: &mut [f32],
    d_v_head: &mut [f32],
    total_cols: usize,
    head_idx: usize,
    head_units: usize,
    seq_len: usize,
    batch_size: usize,
    scale: f32,
) {
    let head_row_start = head_idx * head_units;
    let mut err_attention = vec![0.0f32; seq_len];

    for sample_idx in 0..batch_size {
        let sample_offset = sample_idx * seq_len;
        let attn_offset = sample_idx * seq_len * seq_len;
        let attn_block = &attention_head[attn_offset..attn_offset + seq_len * seq_len];

        for q_pos in 0..seq_len {
            let q_col = sample_offset + q_pos;
            if pad_mask[q_col] {
                for i in 0..head_units {
                    d_q_head[i * total_cols + q_col] = 0.0;
                }
                continue;
            }

            err_attention.fill(0.0);

            for i in 0..head_units {
                let row = head_row_start + i;
                let d_out_val = d_out_data[row * total_cols + q_col];
                for k_pos in 0..seq_len {
                    let attn_idx = k_pos * seq_len + q_pos;
                    let k_col = sample_offset + k_pos;
                    d_v_head[i * total_cols + k_col] += d_out_val * attn_block[attn_idx];
                    err_attention[k_pos] += v_data[row * total_cols + k_col] * d_out_val;
                }
            }

            let mut dot = 0.0f32;
            for k_pos in 0..seq_len {
                let attn_idx = k_pos * seq_len + q_pos;
                dot += attn_block[attn_idx] * err_attention[k_pos];
            }

            for i in 0..head_units {
                let row = head_row_start + i;
                let q_val = q_data[row * total_cols + q_col];
                let mut dq_sum = 0.0f32;
                for k_pos in 0..seq_len {
                    let k_col = sample_offset + k_pos;
                    let attn_idx = k_pos * seq_len + q_pos;
                    let score_grad = attn_block[attn_idx] * (err_attention[k_pos] - dot) * scale;
                    dq_sum += k_data[row * total_cols + k_col] * score_grad;
                    d_k_head[i * total_cols + k_col] += q_val * score_grad;
                }
                d_q_head[i * total_cols + q_col] = dq_sum;
            }
        }
    }
}

#[napi]
pub fn multi_head_attention_forward_native_into(
    q_data: Float32Array,
    k_data: Float32Array,
    v_data: Float32Array,
    pad_mask: Vec<bool>,
    heads: u32,
    head_units: u32,
    seq_len: u32,
    batch_size: u32,
    scale: f64,
    mut out_data: Float32Array,
    mut attention_data: Float32Array,
) {
    let h = heads as usize;
    let hu = head_units as usize;
    let sl = seq_len as usize;
    let bs = batch_size as usize;
    let total_cols = sl * bs;
    let scale_f32 = scale as f32;

    let q_slice = &*q_data;
    let k_slice = &*k_data;
    let v_slice = &*v_data;

    let out_slice = &mut *out_data;
    let attention_slice = &mut *attention_data;
    debug_assert_eq!(out_slice.len(), h * hu * total_cols);
    debug_assert_eq!(attention_slice.len(), h * bs * sl * sl);

    out_slice
        .par_chunks_mut(hu * total_cols)
        .zip(attention_slice.par_chunks_mut(bs * sl * sl))
        .enumerate()
        .for_each(|(head_idx, (out_head, attention_head))| {
            mha_forward_head_into(
                q_slice,
                k_slice,
                v_slice,
                &pad_mask,
                out_head,
                attention_head,
                total_cols,
                head_idx,
                hu,
                sl,
                bs,
                scale_f32,
            );
        });
}

#[napi]
pub fn multi_head_attention_backward_native_into(
    q_data: Float32Array,
    k_data: Float32Array,
    v_data: Float32Array,
    attention_data: Float32Array,
    d_out_data: Float32Array,
    pad_mask: Vec<bool>,
    heads: u32,
    head_units: u32,
    seq_len: u32,
    batch_size: u32,
    scale: f64,
    mut d_q_out: Float32Array,
    mut d_k_out: Float32Array,
    mut d_v_out: Float32Array,
) {
    let h = heads as usize;
    let hu = head_units as usize;
    let sl = seq_len as usize;
    let bs = batch_size as usize;
    let total_cols = sl * bs;
    let scale_f32 = scale as f32;

    let q_slice = &*q_data;
    let k_slice = &*k_data;
    let v_slice = &*v_data;
    let d_out_slice = &*d_out_data;
    let attn_slice = &*attention_data;

    let d_q_slice = &mut *d_q_out;
    let d_k_slice = &mut *d_k_out;
    let d_v_slice = &mut *d_v_out;
    debug_assert_eq!(d_q_slice.len(), h * hu * total_cols);
    debug_assert_eq!(d_k_slice.len(), h * hu * total_cols);
    debug_assert_eq!(d_v_slice.len(), h * hu * total_cols);
    debug_assert_eq!(attn_slice.len(), h * bs * sl * sl);

    d_q_slice
        .par_chunks_mut(hu * total_cols)
        .zip(d_k_slice.par_chunks_mut(hu * total_cols))
        .zip(d_v_slice.par_chunks_mut(hu * total_cols))
        .zip(attn_slice.par_chunks(bs * sl * sl))
        .enumerate()
        .for_each(|(head_idx, (((d_q_head, d_k_head), d_v_head), attention_head))| {
            d_q_head.fill(0.0);
            d_k_head.fill(0.0);
            d_v_head.fill(0.0);
            mha_backward_head_into(
                q_slice,
                k_slice,
                v_slice,
                attention_head,
                d_out_slice,
                &pad_mask,
                d_q_head,
                d_k_head,
                d_v_head,
                total_cols,
                head_idx,
                hu,
                sl,
                bs,
                scale_f32,
            );
        });
}
