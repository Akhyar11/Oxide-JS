use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;

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
    out_slice.par_iter_mut().enumerate().for_each(|(i, val)| {
        *val = a_slice[i] + b_slice[i];
    });
}

#[napi]
pub fn sub_matrices_into(a: Float32Array, b: Float32Array, mut out: Float32Array) {
    let a_slice = &*a;
    let b_slice = &*b;
    let out_slice = &mut *out;
    out_slice.par_iter_mut().enumerate().for_each(|(i, val)| {
        *val = a_slice[i] - b_slice[i];
    });
}

#[napi]
pub fn mul_matrices_into(a: Float32Array, b: Float32Array, mut out: Float32Array) {
    let a_slice = &*a;
    let b_slice = &*b;
    let out_slice = &mut *out;
    out_slice.par_iter_mut().enumerate().for_each(|(i, val)| {
        *val = a_slice[i] * b_slice[i];
    });
}

#[napi]
pub fn div_matrices_into(a: Float32Array, b: Float32Array, mut out: Float32Array) {
    let a_slice = &*a;
    let b_slice = &*b;
    let out_slice = &mut *out;
    out_slice.par_iter_mut().enumerate().for_each(|(i, val)| {
        *val = a_slice[i] / b_slice[i];
    });
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
            let mut sum_exp = 0.0;
            for j in 0..c {
                let exp_val = (out[offset + j] - max_val).exp();
                out[offset + j] = exp_val;
                sum_exp += exp_val;
            }
            for j in 0..c { out[offset + j] /= sum_exp; }
        }
    } else {
        for j in 0..c {
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..r { if out[i * c + j] > max_val { max_val = out[i * c + j]; } }
            let mut sum_exp = 0.0;
            for i in 0..r {
                let idx = i * c + j;
                let exp_val = (out[idx] - max_val).exp();
                out[idx] = exp_val;
                sum_exp += exp_val;
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
    for j in 0..c {
        let mut sum = 0.0;
        for i in 0..r { sum += x_data[i * c + j]; }
        let m = sum / (r as f32);
        out_means[j] = m;
        let mut sum_sq = 0.0;
        for i in 0..r {
            let diff = x_data[i * c + j] - m;
            sum_sq += diff * diff;
        }
        let s = (sum_sq / (r as f32) + eps_f32).sqrt();
        out_stds[j] = s;
        for i in 0..r {
            let idx = i * c + j;
            let norm = (x_data[idx] - m) / s;
            out_norm[idx] = norm;
            out_res[idx] = norm * gamma[i] + beta[i];
        }
    }
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

    for i in 0..r {
        let mut sum_g = 0.0f32;
        let mut sum_b = 0.0f32;
        for j in 0..c {
            let idx = i * c + j;
            sum_g += err_data[idx] * norm_data[idx];
            sum_b += err_data[idx];
        }
        d_gamma_out[i] = sum_g;
        d_beta_out[i] = sum_b;
    }

    for j in 0..c {
        let s = std_data[j];
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        for i in 0..r {
            let idx = i * c + j;
            let e = err_data[idx] * gamma_data[i];
            sum1 += e;
            sum2 += e * norm_data[idx];
        }

        let inv_r = 1.0f32 / (r as f32);
        for i in 0..r {
            let idx = i * c + j;
            dx_out[idx] =
                (gamma_data[i] * err_data[idx] - (sum1 * inv_r) - (norm_data[idx] * sum2 * inv_r)) / s;
        }
    }
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
        let token_idx = indices[j] as usize;
        if let Some(pad_id) = pad_token_id { if token_idx == pad_id as usize { continue; } }
        if token_idx >= v_size { continue; }
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
            let token_idx = indices[j] as usize;
            if let Some(pad_id) = pad_token_id {
                if token_idx == pad_id as usize { continue; }
            }
            if token_idx >= v_size { continue; }
            
            grad_data[i * v_size + token_idx] += err_data[i * seq_len + j];
        }
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

#[napi]
pub fn add_in_place(mut a: Float32Array, b: Float32Array) {
    for i in 0..a.len() { a[i] += b[i]; }
}

#[napi]
pub fn sub_in_place(mut a: Float32Array, b: Float32Array) {
    for i in 0..a.len() { a[i] -= b[i]; }
}

#[napi]
pub fn mul_in_place(mut a: Float32Array, b: Float32Array) {
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
    for j in 0..c {
        let offset = j; // assuming column major or specific broadcasting? 
        // In dense.ts: zData[i * cols + j] += bData[i]
        // This is [rows x cols] where bias is [rows x 1].
        for i in 0..r {
            data[i * c + j] += bias[i];
        }
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

fn mha_forward_block(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    pad_mask: &[bool],
    total_cols: usize,
    head_idx: usize,
    sample_idx: usize,
    head_units: usize,
    seq_len: usize,
    scale: f32,
) -> (usize, Vec<f32>, Vec<f32>) {
    let mut out_block = vec![0.0f32; head_units * seq_len];
    let mut attn_block = vec![0.0f32; seq_len * seq_len];
    let sample_offset = sample_idx * seq_len;
    let head_row_start = head_idx * head_units;

    for q_pos in 0..seq_len {
        let q_col = sample_offset + q_pos;
        if pad_mask[q_col] {
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
            let out_idx = i * seq_len + q_pos;
            let mut sum = 0.0f32;
            for k_pos in 0..seq_len {
                let k_col = sample_offset + k_pos;
                sum += v_data[row * total_cols + k_col] * attn_block[k_pos * seq_len + q_pos];
            }
            out_block[out_idx] = sum;
        }
    }

    (head_idx * 10_000_000 + sample_idx, out_block, attn_block)
}

fn mha_backward_block(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    attn_block: &[f32],
    d_out_data: &[f32],
    pad_mask: &[bool],
    total_cols: usize,
    head_idx: usize,
    sample_idx: usize,
    head_units: usize,
    seq_len: usize,
    scale: f32,
) -> (usize, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut d_q_block = vec![0.0f32; head_units * seq_len];
    let mut d_k_block = vec![0.0f32; head_units * seq_len];
    let mut d_v_block = vec![0.0f32; head_units * seq_len];
    let mut err_attention = vec![0.0f32; seq_len * seq_len];
    let mut err_score = vec![0.0f32; seq_len * seq_len];

    let sample_offset = sample_idx * seq_len;
    let head_row_start = head_idx * head_units;

    for q_pos in 0..seq_len {
        let q_col = sample_offset + q_pos;
        if pad_mask[q_col] {
            continue;
        }

        for i in 0..head_units {
            let row = head_row_start + i;
            let d_out_val = d_out_data[row * total_cols + q_col];
            for k_pos in 0..seq_len {
                let attn_idx = k_pos * seq_len + q_pos;
                d_v_block[i * seq_len + k_pos] += d_out_val * attn_block[attn_idx];
                let k_col = sample_offset + k_pos;
                err_attention[attn_idx] += v_data[row * total_cols + k_col] * d_out_val;
            }
        }

        let mut dot = 0.0f32;
        for k_pos in 0..seq_len {
            let attn_idx = k_pos * seq_len + q_pos;
            dot += attn_block[attn_idx] * err_attention[attn_idx];
        }

        for k_pos in 0..seq_len {
            let attn_idx = k_pos * seq_len + q_pos;
            err_score[attn_idx] = attn_block[attn_idx] * (err_attention[attn_idx] - dot) * scale;
        }

        for i in 0..head_units {
            let row = head_row_start + i;
            let mut dq_sum = 0.0f32;
            for k_pos in 0..seq_len {
                let k_col = sample_offset + k_pos;
                let score_grad = err_score[k_pos * seq_len + q_pos];
                dq_sum += k_data[row * total_cols + k_col] * score_grad;
                d_k_block[i * seq_len + k_pos] += q_data[row * total_cols + q_col] * score_grad;
            }
            d_q_block[i * seq_len + q_pos] = dq_sum;
        }
    }

    (head_idx * 10_000_000 + sample_idx, d_q_block, d_k_block, d_v_block)
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

    for i in 0..out_data.len() {
        out_data[i] = 0.0;
    }
    for i in 0..attention_data.len() {
        attention_data[i] = 0.0;
    }

    let q_slice = &*q_data;
    let k_slice = &*k_data;
    let v_slice = &*v_data;

    let blocks: Vec<(usize, usize)> = (0..h)
        .flat_map(|head_idx| (0..bs).map(move |sample_idx| (head_idx, sample_idx)))
        .collect();

    let results: Vec<(usize, Vec<f32>, Vec<f32>)> = blocks
        .into_par_iter()
        .map(|(head_idx, sample_idx)| {
            mha_forward_block(
                q_slice,
                k_slice,
                v_slice,
                &pad_mask,
                total_cols,
                head_idx,
                sample_idx,
                hu,
                sl,
                scale_f32,
            )
        })
        .collect();

    for (block_id, out_block, attn_block) in results {
        let head_idx = block_id / 10_000_000;
        let sample_idx = block_id % 10_000_000;
        let sample_offset = sample_idx * sl;
        let head_row_start = head_idx * hu;

        for i in 0..hu {
            let row = head_row_start + i;
            let src_offset = i * sl;
            for q_pos in 0..sl {
                out_data[row * total_cols + sample_offset + q_pos] = out_block[src_offset + q_pos];
            }
        }

        let attn_offset = (head_idx * bs + sample_idx) * sl * sl;
        for i in 0..attn_block.len() {
            attention_data[attn_offset + i] = attn_block[i];
        }
    }
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

    for i in 0..d_q_out.len() {
        d_q_out[i] = 0.0;
        d_k_out[i] = 0.0;
        d_v_out[i] = 0.0;
    }

    let q_slice = &*q_data;
    let k_slice = &*k_data;
    let v_slice = &*v_data;
    let d_out_slice = &*d_out_data;
    let attn_slice = &*attention_data;

    let blocks: Vec<(usize, usize)> = (0..h)
        .flat_map(|head_idx| (0..bs).map(move |sample_idx| (head_idx, sample_idx)))
        .collect();

    let results: Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)> = blocks
        .into_par_iter()
        .map(|(head_idx, sample_idx)| {
            let attn_offset = (head_idx * bs + sample_idx) * sl * sl;
            let attn_block = &attn_slice[attn_offset..attn_offset + sl * sl];
            mha_backward_block(
                q_slice,
                k_slice,
                v_slice,
                attn_block,
                d_out_slice,
                &pad_mask,
                total_cols,
                head_idx,
                sample_idx,
                hu,
                sl,
                scale_f32,
            )
        })
        .collect();

    for (block_id, d_q_block, d_k_block, d_v_block) in results {
        let head_idx = block_id / 10_000_000;
        let sample_idx = block_id % 10_000_000;
        let sample_offset = sample_idx * sl;
        let head_row_start = head_idx * hu;

        for i in 0..hu {
            let row = head_row_start + i;
            let src_offset = i * sl;
            for pos in 0..sl {
                let dst_idx = row * total_cols + sample_offset + pos;
                d_q_out[dst_idx] = d_q_block[src_offset + pos];
                d_k_out[dst_idx] = d_k_block[src_offset + pos];
                d_v_out[dst_idx] = d_v_block[src_offset + pos];
            }
        }
    }
}
