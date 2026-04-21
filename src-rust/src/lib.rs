use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;

const ELEMENTWISE_PARALLEL_THRESHOLD: usize = 16 * 1024;
const ADAM_PARALLEL_THRESHOLD: usize = 8 * 1024;

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
