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
    let a_rows = if trans_a { a_shape[1] } else { a_shape[0] } as usize;
    let b_cols = if trans_b { b_shape[0] } else { b_shape[1] } as usize;
    let result = vec![0.0; a_rows * b_cols];
    
    // We'll keep the existing one for compatibility but optimize it later
    // or just call the into version
    let out_array = Float64Array::from(result);
    dot_product_into(a_data, a_shape, b_data, b_shape, out_array.clone(), trans_a, trans_b);
    out_array
}

#[napi]
pub fn dot_product_into(
    a_data: Float64Array,
    a_shape: Vec<u32>,
    b_data: Float64Array,
    b_shape: Vec<u32>,
    mut out_data: Float64Array,
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
        matrixmultiply::dgemm(
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
pub fn add_matrices_into(a: Float64Array, b: Float64Array, mut out: Float64Array) {
    for i in 0..a.len() { out[i] = a[i] + b[i]; }
}

#[napi]
pub fn sub_matrices_into(a: Float64Array, b: Float64Array, mut out: Float64Array) {
    for i in 0..a.len() { out[i] = a[i] - b[i]; }
}

#[napi]
pub fn mul_matrices_into(a: Float64Array, b: Float64Array, mut out: Float64Array) {
    for i in 0..a.len() { out[i] = a[i] * b[i]; }
}

#[napi]
pub fn div_matrices_into(a: Float64Array, b: Float64Array, mut out: Float64Array) {
    for i in 0..a.len() { out[i] = a[i] / b[i]; }
}

#[napi]
pub fn softmax_native_into(data: Float64Array, rows: u32, cols: u32, is_row: bool, mut out: Float64Array) {
    let r = rows as usize;
    let c = cols as usize;
    out.copy_from_slice(&data);

    if is_row {
        for i in 0..r {
            let offset = i * c;
            let mut max_val = f64::NEG_INFINITY;
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
            let mut max_val = f64::NEG_INFINITY;
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
pub fn softmax_backward_native_into(s_data: Float64Array, g_data: Float64Array, rows: u32, cols: u32, is_row: bool, mut out: Float64Array) {
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
    x_data: Float64Array,
    gamma: Float64Array,
    beta: Float64Array,
    rows: u32,
    cols: u32,
    eps: f64,
    mut out_res: Float64Array,
    mut out_norm: Float64Array,
    mut out_means: Float64Array,
    mut out_stds: Float64Array,
) {
    let r = rows as usize;
    let c = cols as usize;
    for j in 0..c {
        let mut sum = 0.0;
        for i in 0..r { sum += x_data[i * c + j]; }
        let m = sum / (r as f64);
        out_means[j] = m;
        let mut sum_sq = 0.0;
        for i in 0..r {
            let diff = x_data[i * c + j] - m;
            sum_sq += diff * diff;
        }
        let s = (sum_sq / (r as f64) + eps).sqrt();
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
pub fn relu_native_into(input: Float64Array, mut out_res: Float64Array, mut out_grad: Float64Array) {
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
pub fn sigmoid_native_into(input: Float64Array, mut out_res: Float64Array, mut out_grad: Float64Array) {
    for i in 0..input.len() {
        let val = 1.0 / (1.0 + (-input[i]).exp());
        out_res[i] = val;
        out_grad[i] = val * (1.0 - val);
    }
}

#[napi]
pub fn tanh_native_into(input: Float64Array, mut out_res: Float64Array, mut out_grad: Float64Array) {
    for i in 0..input.len() {
        let val = input[i].tanh();
        out_res[i] = val;
        out_grad[i] = 1.0 - val * val;
    }
}

#[napi]
pub fn embedding_forward_native_into(
    indices: Vec<f64>,
    weight_data: Float64Array,
    vocab_size: u32,
    embedding_dim: u32,
    pad_token_id: Option<i32>,
    mut out: Float64Array
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
pub fn convolution_native_into(
    a_data: Float64Array,
    a_rows: u32,
    a_cols: u32,
    k_data: Float64Array,
    k_rows: u32,
    k_cols: u32,
    mut out: Float64Array
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
    err_data: Float64Array,
    err_rows: u32,
    err_cols: u32,
    input_data: Float64Array,
    input_rows: u32,
    input_cols: u32,
    out_rows: u32, 
    out_cols: u32,
    mut out: Float64Array
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
pub fn add_in_place(mut a: Float64Array, b: Float64Array) {
    for i in 0..a.len() { a[i] += b[i]; }
}

#[napi]
pub fn sub_in_place(mut a: Float64Array, b: Float64Array) {
    for i in 0..a.len() { a[i] -= b[i]; }
}

#[napi]
pub fn mul_in_place(mut a: Float64Array, b: Float64Array) {
    for i in 0..a.len() { a[i] *= b[i]; }
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
pub fn add_bias_native(mut data: Float64Array, bias: Float64Array, rows: u32, cols: u32) {
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
pub fn sum_axis_native(data: Float64Array, rows: u32, cols: u32, axis: u32, mut out: Float64Array) {
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
pub fn clip_gradients_native(mut data: Float64Array, limit: f64) {
    for i in 0..data.len() {
        if data[i] > limit {
            data[i] = limit;
        } else if data[i] < -limit {
            data[i] = -limit;
        }
    }
}
