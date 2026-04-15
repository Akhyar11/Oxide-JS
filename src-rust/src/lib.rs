use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
pub fn dot_product(
    a_data: Float64Array,
    a_shape: Vec<u32>,
    b_data: Float64Array,
    b_shape: Vec<u32>,
    trans_a: bool,
    trans_b: bool,
) -> Float64Array {
    let a_rows_orig = a_shape[0] as usize;
    let a_cols_orig = a_shape[1] as usize;
    let b_rows_orig = b_shape[0] as usize;
    let b_cols_orig = b_shape[1] as usize;

    let a_rows = if trans_a { a_cols_orig } else { a_rows_orig };
    let a_cols = if trans_a { a_rows_orig } else { a_cols_orig };
    let b_rows = if trans_b { b_cols_orig } else { b_rows_orig };
    let b_cols = if trans_b { b_rows_orig } else { b_cols_orig };

    if a_cols != b_rows {
        panic!("Dimension mismatch: {}x{} * {}x{}", a_rows, a_cols, b_rows, b_cols);
    }

    let mut result = vec![0.0; a_rows * b_cols];

    // Pointer access for speed
    let a_ptr = a_data.as_ptr();
    let b_ptr = b_data.as_ptr();

    if !trans_b {
        for i in 0..a_rows {
            let r_offset = i * b_cols;
            for k in 0..a_cols {
                let aik = if trans_a {
                    unsafe { *a_ptr.add(k * a_rows + i) }
                } else {
                    unsafe { *a_ptr.add(i * a_cols + k) }
                };
                
                if aik == 0.0 { continue; }
                
                let k_offset = k * b_cols;
                for j in 0..b_cols {
                    unsafe {
                        result[r_offset + j] += aik * (*b_ptr.add(k_offset + j));
                    }
                }
            }
        }
    } else {
        // Case: A * B^T
        for i in 0..a_rows {
            let r_offset = i * b_cols;
            for j in 0..b_cols {
                let mut sum = 0.0;
                let b_offset = j * a_cols; 
                for k in 0..a_cols {
                    let aik = if trans_a {
                        unsafe { *a_ptr.add(k * a_rows + i) }
                    } else {
                        unsafe { *a_ptr.add(i * a_cols + k) }
                    };
                    let bjk = unsafe { *b_ptr.add(b_offset + k) };
                    sum += aik * bjk;
                }
                result[r_offset + j] = sum;
            }
        }
    }

    Float64Array::from(result)
}

#[napi]
pub fn add_matrices(a: Float64Array, b: Float64Array) -> Float64Array {
    let mut res = a.to_vec();
    for i in 0..res.len() {
        res[i] += b[i];
    }
    Float64Array::from(res)
}

#[napi]
pub fn sub_matrices(a: Float64Array, b: Float64Array) -> Float64Array {
    let mut res = a.to_vec();
    for i in 0..res.len() {
        res[i] -= b[i];
    }
    Float64Array::from(res)
}

#[napi]
pub fn mul_matrices(a: Float64Array, b: Float64Array) -> Float64Array {
    let mut res = a.to_vec();
    for i in 0..res.len() {
        res[i] *= b[i];
    }
    Float64Array::from(res)
}

#[napi]
pub fn div_matrices(a: Float64Array, b: Float64Array) -> Float64Array {
    let mut res = a.to_vec();
    for i in 0..res.len() {
        res[i] /= b[i];
    }
    Float64Array::from(res)
}

#[napi]
pub fn add_in_place(mut a: Float64Array, b: Float64Array) {
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

#[napi]
pub fn sub_in_place(mut a: Float64Array, b: Float64Array) {
    for i in 0..a.len() {
        a[i] -= b[i];
    }
}

#[napi]
pub fn mul_in_place(mut a: Float64Array, b: Float64Array) {
    for i in 0..a.len() {
        a[i] *= b[i];
    }
}

#[napi]
pub fn softmax_native(data: Float64Array, rows: u32, cols: u32, is_row: bool) -> Float64Array {
    let mut res = data.to_vec();
    let r = rows as usize;
    let c = cols as usize;

    if is_row {
        for i in 0..r {
            let offset = i * c;
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..c {
                if res[offset + j] > max_val { max_val = res[offset + j]; }
            }

            let mut sum_exp = 0.0;
            for j in 0..c {
                let exp_val = (res[offset + j] - max_val).exp();
                res[offset + j] = exp_val;
                sum_exp += exp_val;
            }

            for j in 0..c {
                res[offset + j] /= sum_exp;
            }
        }
    } else {
        for j in 0..c {
            let mut max_val = f64::NEG_INFINITY;
            for i in 0..r {
                if res[i * c + j] > max_val { max_val = res[i * c + j]; }
            }

            let mut sum_exp = 0.0;
            for i in 0..r {
                let idx = i * c + j;
                let exp_val = (res[idx] - max_val).exp();
                res[idx] = exp_val;
                sum_exp += exp_val;
            }

            for i in 0..r {
                res[i * c + j] /= sum_exp;
            }
        }
    }
    Float64Array::from(res)
}

#[napi]
pub fn softmax_backward_native(s_data: Float64Array, g_data: Float64Array, rows: u32, cols: u32, is_row: bool) -> Float64Array {
    let mut res = vec![0.0; s_data.len()];
    let r = rows as usize;
    let c = cols as usize;

    if is_row {
        for i in 0..r {
            let offset = i * c;
            let mut sum_grad_s = 0.0;
            for j in 0..c {
                let idx = offset + j;
                sum_grad_s += s_data[idx] * g_data[idx];
            }
            for j in 0..c {
                let idx = offset + j;
                res[idx] = s_data[idx] * (g_data[idx] - sum_grad_s);
            }
        }
    } else {
        for j in 0..c {
            let mut sum_grad_s = 0.0;
            for i in 0..r {
                let idx = i * c + j;
                sum_grad_s += s_data[idx] * g_data[idx];
            }
            for i in 0..r {
                let idx = i * c + j;
                res[idx] = s_data[idx] * (g_data[idx] - sum_grad_s);
            }
        }
    }
    Float64Array::from(res)
}

#[napi]
pub fn layer_norm_native(
    x_data: Float64Array,
    gamma: Float64Array,
    beta: Float64Array,
    rows: u32,
    cols: u32,
    eps: f64
) -> Vec<Float64Array> {
    let r = rows as usize;
    let c = cols as usize;
    let mut res = vec![0.0; r * c];
    let mut norm_data = vec![0.0; r * c];
    let mut means = vec![0.0; c];
    let mut stds = vec![0.0; c];

    for j in 0..c {
        let mut sum = 0.0;
        for i in 0..r {
            sum += x_data[i * c + j];
        }
        let m = sum / (r as f64);
        means[j] = m;

        let mut sum_sq = 0.0;
        for i in 0..r {
            let diff = x_data[i * c + j] - m;
            sum_sq += diff * diff;
        }
        let s = (sum_sq / (r as f64) + eps).sqrt();
        stds[j] = s;

        for i in 0..r {
            let idx = i * c + j;
            let norm = (x_data[idx] - m) / s;
            norm_data[idx] = norm;
            res[idx] = norm * gamma[i] + beta[i];
        }
    }

    vec![
        Float64Array::from(res),
        Float64Array::from(norm_data),
        Float64Array::from(means),
        Float64Array::from(stds),
    ]
}

#[napi]
pub fn apply_attention_mask_native(
    mut data: Float64Array,
    pad_mask: Vec<bool>,
    rows: u32,
    cols: u32,
    scale: f64
) {
    let r = rows as usize;
    let c = cols as usize;
    let masked_value = -1e9;

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
                data[key * c + query] *= scale;
            }
        }
    }
}
#[napi]
pub fn adam_update_native(
    grad: Float64Array,
    mut m: Float64Array,
    mut v: Float64Array,
    mut buffer: Float64Array,
    t: u32,
    alpha: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64
) {
    let one_minus_beta1 = 1.0 - beta1;
    let one_minus_beta2 = 1.0 - beta2;
    let bias_correction1 = 1.0 / (1.0 - beta1.powi(t as i32));
    let bias_correction2 = 1.0 / (1.0 - beta2.powi(t as i32));

    for i in 0..grad.len() {
        let g = grad[i];
        let m_new = beta1 * m[i] + one_minus_beta1 * g;
        let v_new = beta2 * v[i] + one_minus_beta2 * g * g;
        m[i] = m_new;
        v[i] = v_new;

        let m_hat = m_new * bias_correction1;
        let v_hat = v_new * bias_correction2;
        buffer[i] = alpha * m_hat / (v_hat.sqrt() + epsilon);
    }
}

#[napi]
pub fn relu_native(input: Float64Array) -> Vec<Float64Array> {
    let mut res = vec![0.0; input.len()];
    let mut grad = vec![0.0; input.len()];
    for i in 0..input.len() {
        let val = input[i];
        if val > 0.0 {
            res[i] = val;
            grad[i] = 1.0;
        } else {
            res[i] = 0.0;
            grad[i] = 0.0;
        }
    }
    vec![Float64Array::from(res), Float64Array::from(grad)]
}

#[napi]
pub fn sigmoid_native(input: Float64Array) -> Vec<Float64Array> {
    let mut res = vec![0.0; input.len()];
    let mut grad = vec![0.0; input.len()];
    for i in 0..input.len() {
        let val = 1.0 / (1.0 + (-input[i]).exp());
        res[i] = val;
        grad[i] = val * (1.0 - val);
    }
    vec![Float64Array::from(res), Float64Array::from(grad)]
}

#[napi]
pub fn tanh_native(input: Float64Array) -> Vec<Float64Array> {
    let mut res = vec![0.0; input.len()];
    let mut grad = vec![0.0; input.len()];
    for i in 0..input.len() {
        let val = input[i].tanh();
        res[i] = val;
        grad[i] = 1.0 - val * val;
    }
    vec![Float64Array::from(res), Float64Array::from(grad)]
}

#[napi]
pub fn mse_native(y_true: Float64Array, y_pred: Float64Array) -> Vec<f64> {
    let mut sum_sq = 0.0;
    let n = y_true.len() as f64;
    for i in 0..y_true.len() {
        let diff = y_true[i] - y_pred[i];
        sum_sq += diff * diff;
    }
    vec![sum_sq / n]
}

#[napi]
pub fn embedding_forward_native(
    indices: Vec<f64>,
    weight_data: Float64Array,
    vocab_size: u32,
    embedding_dim: u32,
    pad_token_id: Option<i32>
) -> Float64Array {
    let seq_len = indices.len();
    let dim = embedding_dim as usize;
    let v_size = vocab_size as usize;
    let mut out = vec![0.0; dim * seq_len];

    for j in 0..seq_len {
        let token_idx = indices[j] as usize;
        if let Some(pad_id) = pad_token_id {
            if token_idx == pad_id as usize { continue; }
        }
        if token_idx >= v_size { continue; }

        for i in 0..dim {
            out[i * seq_len + j] = weight_data[i * v_size + token_idx];
        }
    }
    Float64Array::from(out)
}

#[napi]
pub fn embedding_backward_native(
    indices: Vec<f64>,
    err_data: Float64Array,
    mut grad_data: Float64Array,
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
pub fn convolution_native(
    a_data: Float64Array,
    a_rows: u32,
    a_cols: u32,
    k_data: Float64Array,
    k_rows: u32,
    k_cols: u32
) -> Float64Array {
    let out_rows = (a_rows - k_rows + 1) as usize;
    let out_cols = (a_cols - k_cols + 1) as usize;
    let ar = a_rows as usize;
    let ac = a_cols as usize;
    let kr = k_rows as usize;
    let kc = k_cols as usize;
    
    let mut out = vec![0.0; out_rows * out_cols];

    for i in 0..out_rows {
        for j in 0..out_cols {
            let mut sum = 0.0;
            for k in 0..kr {
                let a_offset = (i + k) * ac + j;
                let k_offset = k * kc;
                for l in 0..kc {
                    sum += a_data[a_offset + l] * k_data[k_offset + l];
                }
            }
            out[i * out_cols + j] = sum;
        }
    }
    Float64Array::from(out)
}

#[napi]
pub fn conv_backward_input_native(
    err_data: Float64Array,
    err_rows: u32,
    err_cols: u32,
    input_data: Float64Array,
    input_rows: u32,
    input_cols: u32,
    out_rows: u32,
    out_cols: u32
) -> Float64Array {
    let er = err_rows as usize;
    let ec = err_cols as usize;
    let ir = input_rows as usize;
    let ic = input_cols as usize;
    let or = out_rows as usize;
    let oc = out_cols as usize;
    
    let mut out = vec![0.0; or * oc];

    for k in 0..er {
        for l in 0..ec {
            let err_val = err_data[k * ec + l];
            if err_val == 0.0 { continue; }
            
            for m in 0..ir {
                for n in 0..ic {
                    out[(m + k) * oc + (n + l)] += err_val * input_data[m * ic + n];
                }
            }
        }
    }
    Float64Array::from(out)
}
