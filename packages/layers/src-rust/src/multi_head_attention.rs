use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;

#[derive(Copy, Clone)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    #[inline(always)]
    pub fn get(self) -> *mut f32 {
        self.0
    }
}

#[derive(Copy, Clone)]
struct SendPtrConst(*const f32);
unsafe impl Send for SendPtrConst {}
unsafe impl Sync for SendPtrConst {}

impl SendPtrConst {
    #[inline(always)]
    pub fn get(self) -> *const f32 {
        self.0
    }
}

#[napi]
pub fn multi_head_attention_forward_native(
    inputs_q: Float32Array,
    inputs_k: Float32Array,
    inputs_v: Float32Array,
    w_q: Float32Array,
    w_k: Float32Array,
    w_v: Float32Array,
    w_o: Float32Array,
    b_q: Float32Array,
    b_k: Float32Array,
    b_v: Float32Array,
    b_o: Float32Array,
    batch_size: u32,
    seq_len_q: u32,
    seq_len_k: u32,
    input_dim_q: u32,
    input_dim_k: u32,
    input_dim_v: u32,
    num_heads: u32,
    key_dim: u32,
    value_dim: u32,
    output_dim: u32,
    use_bias: bool,
    mut out: Float32Array,
    mut q: Float32Array,
    mut k: Float32Array,
    mut v: Float32Array,
    mut scores: Float32Array,
    mut probs: Float32Array,
    mut out_concat: Float32Array,
) {
    let b_size = batch_size as usize;
    let sl_q = seq_len_q as usize;
    let sl_k = seq_len_k as usize;
    let in_dim_q = input_dim_q as usize;
    let in_dim_k = input_dim_k as usize;
    let in_dim_v = input_dim_v as usize;
    let n_heads = num_heads as usize;
    let k_dim = key_dim as usize;
    let v_dim = value_dim as usize;
    let o_dim = output_dim as usize;

    let inputs_q_ptr = SendPtrConst(inputs_q.as_ptr());
    let inputs_k_ptr = SendPtrConst(inputs_k.as_ptr());
    let inputs_v_ptr = SendPtrConst(inputs_v.as_ptr());
    let w_q_ptr = SendPtrConst(w_q.as_ptr());
    let w_k_ptr = SendPtrConst(w_k.as_ptr());
    let w_v_ptr = SendPtrConst(w_v.as_ptr());
    let w_o_ptr = SendPtrConst(w_o.as_ptr());
    let b_q_ptr = SendPtrConst(b_q.as_ptr());
    let b_k_ptr = SendPtrConst(b_k.as_ptr());
    let b_v_ptr = SendPtrConst(b_v.as_ptr());
    let b_o_ptr = SendPtrConst(b_o.as_ptr());

    let out_ptr = SendPtr(out.as_mut_ptr());
    let q_ptr = SendPtr(q.as_mut_ptr());
    let k_ptr = SendPtr(k.as_mut_ptr());
    let v_ptr = SendPtr(v.as_mut_ptr());
    let scores_ptr = SendPtr(scores.as_mut_ptr());
    let probs_ptr = SendPtr(probs.as_mut_ptr());
    let out_concat_ptr = SendPtr(out_concat.as_mut_ptr());

    let qk_scale = 1.0 / (k_dim as f32).sqrt();

    // 1. Linear Projection for Q, K, V in parallel per batch element
    (0..b_size).into_par_iter().for_each(move |b| {
        unsafe {
            // Project Q: [sl_q, in_dim_q] * [in_dim_q, n_heads * k_dim] -> [sl_q, n_heads, k_dim]
            for t in 0..sl_q {
                let row_offset = (b * sl_q + t) * in_dim_q;
                let out_offset = (b * sl_q + t) * n_heads * k_dim;

                for h in 0..n_heads {
                    for d in 0..k_dim {
                        let j = h * k_dim + d;
                        let mut sum = if use_bias { *b_q_ptr.get().add(j) } else { 0.0 };
                        for c in 0..in_dim_q {
                            sum += *inputs_q_ptr.get().add(row_offset + c) * *w_q_ptr.get().add(c * (n_heads * k_dim) + j);
                        }
                        *q_ptr.get().add(out_offset + j) = sum;
                    }
                }
            }

            // Project K: [sl_k, in_dim_k] * [in_dim_k, n_heads * k_dim] -> [sl_k, n_heads, k_dim]
            for t in 0..sl_k {
                let row_offset = (b * sl_k + t) * in_dim_k;
                let out_offset = (b * sl_k + t) * n_heads * k_dim;

                for h in 0..n_heads {
                    for d in 0..k_dim {
                        let j = h * k_dim + d;
                        let mut sum = if use_bias { *b_k_ptr.get().add(j) } else { 0.0 };
                        for c in 0..in_dim_k {
                            sum += *inputs_k_ptr.get().add(row_offset + c) * *w_k_ptr.get().add(c * (n_heads * k_dim) + j);
                        }
                        *k_ptr.get().add(out_offset + j) = sum;
                    }
                }
            }

            // Project V: [sl_k, in_dim_v] * [in_dim_v, n_heads * v_dim] -> [sl_k, n_heads, v_dim]
            for t in 0..sl_k {
                let row_offset = (b * sl_k + t) * in_dim_v;
                let out_offset = (b * sl_k + t) * n_heads * v_dim;

                for h in 0..n_heads {
                    for d in 0..v_dim {
                        let j = h * v_dim + d;
                        let mut sum = if use_bias { *b_v_ptr.get().add(j) } else { 0.0 };
                        for c in 0..in_dim_v {
                            sum += *inputs_v_ptr.get().add(row_offset + c) * *w_v_ptr.get().add(c * (n_heads * v_dim) + j);
                        }
                        *v_ptr.get().add(out_offset + j) = sum;
                    }
                }
            }
        }
    });

    // 2. Compute Self-Attention per task (batch, head)
    let total_attention_tasks = b_size * n_heads;
    (0..total_attention_tasks).into_par_iter().for_each(move |task_id| {
        let b = task_id / n_heads;
        let h = task_id % n_heads;

        unsafe {
            // Compute attention scores: [sl_q, k_dim] * [k_dim, sl_k] -> [sl_q, sl_k]
            for i in 0..sl_q {
                let q_row_offset = (b * sl_q + i) * n_heads * k_dim + h * k_dim;
                let score_row_offset = (b * sl_q + i) * n_heads * sl_k + h * sl_k;

                let mut max_val = -f32::INFINITY;

                for j in 0..sl_k {
                    let k_row_offset = (b * sl_k + j) * n_heads * k_dim + h * k_dim;
                    let mut sum = 0.0;
                    for d in 0..k_dim {
                        sum += *q_ptr.get().add(q_row_offset + d) * *k_ptr.get().add(k_row_offset + d);
                    }
                    let score_val = sum * qk_scale;
                    *scores_ptr.get().add(score_row_offset + j) = score_val;
                    if score_val > max_val {
                        max_val = score_val;
                    }
                }

                // Softmax
                let mut sum_exp = 0.0;
                for j in 0..sl_k {
                    let s_val = *scores_ptr.get().add(score_row_offset + j);
                    let exp_val = (s_val - max_val).exp();
                    *probs_ptr.get().add(score_row_offset + j) = exp_val;
                    sum_exp += exp_val;
                }

                for j in 0..sl_k {
                    *probs_ptr.get().add(score_row_offset + j) /= sum_exp;
                }
            }

            // Compute context value representation: [sl_q, sl_k] * [sl_k, v_dim] -> [sl_q, v_dim]
            for i in 0..sl_q {
                let prob_row_offset = (b * sl_q + i) * n_heads * sl_k + h * sl_k;
                let concat_row_offset = (b * sl_q + i) * n_heads * v_dim + h * v_dim;

                for d in 0..v_dim {
                    let mut sum = 0.0;
                    for j in 0..sl_k {
                        let v_row_offset = (b * sl_k + j) * n_heads * v_dim + h * v_dim;
                        sum += *probs_ptr.get().add(prob_row_offset + j) * *v_ptr.get().add(v_row_offset + d);
                    }
                    *out_concat_ptr.get().add(concat_row_offset + d) = sum;
                }
            }
        }
    });

    // 3. Final Output Projection: [B * sl_q, n_heads * v_dim] * [n_heads * v_dim, o_dim] -> [B * sl_q, o_dim]
    let total_rows_q = b_size * sl_q;
    let concat_dim = n_heads * v_dim;

    (0..total_rows_q).into_par_iter().for_each(move |i| {
        unsafe {
            for j in 0..o_dim {
                let mut sum = if use_bias { *b_o_ptr.get().add(j) } else { 0.0 };
                for c in 0..concat_dim {
                    sum += *out_concat_ptr.get().add(i * concat_dim + c) * *w_o_ptr.get().add(c * o_dim + j);
                }
                *out_ptr.get().add(i * o_dim + j) = sum;
            }
        }
    });
}

#[napi]
pub fn multi_head_attention_backward_native(
    grad_out: Float32Array,
    inputs_q: Float32Array,
    inputs_k: Float32Array,
    inputs_v: Float32Array,
    q: Float32Array,
    k: Float32Array,
    v: Float32Array,
    probs: Float32Array,
    out_concat: Float32Array,
    w_q: Float32Array,
    w_k: Float32Array,
    w_v: Float32Array,
    w_o: Float32Array,
    batch_size: u32,
    seq_len_q: u32,
    seq_len_k: u32,
    input_dim_q: u32,
    input_dim_k: u32,
    input_dim_v: u32,
    num_heads: u32,
    key_dim: u32,
    value_dim: u32,
    output_dim: u32,
    use_bias: bool,
    mut grad_in_q: Float32Array,
    mut grad_in_k: Float32Array,
    mut grad_in_v: Float32Array,
    mut grad_w_q: Float32Array,
    mut grad_w_k: Float32Array,
    mut grad_w_v: Float32Array,
    mut grad_w_o: Float32Array,
    mut grad_b_q: Float32Array,
    mut grad_b_k: Float32Array,
    mut grad_b_v: Float32Array,
    mut grad_b_o: Float32Array,
) {
    let b_size = batch_size as usize;
    let sl_q = seq_len_q as usize;
    let sl_k = seq_len_k as usize;
    let in_dim_q = input_dim_q as usize;
    let in_dim_k = input_dim_k as usize;
    let in_dim_v = input_dim_v as usize;
    let n_heads = num_heads as usize;
    let k_dim = key_dim as usize;
    let v_dim = value_dim as usize;
    let o_dim = output_dim as usize;

    let grad_out_ptr = SendPtrConst(grad_out.as_ptr());
    let inputs_q_ptr = SendPtrConst(inputs_q.as_ptr());
    let inputs_k_ptr = SendPtrConst(inputs_k.as_ptr());
    let inputs_v_ptr = SendPtrConst(inputs_v.as_ptr());
    let q_ptr = SendPtrConst(q.as_ptr());
    let k_ptr = SendPtrConst(k.as_ptr());
    let v_ptr = SendPtrConst(v.as_ptr());
    let probs_ptr = SendPtrConst(probs.as_ptr());
    let out_concat_ptr = SendPtrConst(out_concat.as_ptr());
    let w_q_ptr = SendPtrConst(w_q.as_ptr());
    let w_k_ptr = SendPtrConst(w_k.as_ptr());
    let w_v_ptr = SendPtrConst(w_v.as_ptr());
    let w_o_ptr = SendPtrConst(w_o.as_ptr());

    let grad_in_q_ptr = SendPtr(grad_in_q.as_mut_ptr());
    let grad_in_k_ptr = SendPtr(grad_in_k.as_mut_ptr());
    let grad_in_v_ptr = SendPtr(grad_in_v.as_mut_ptr());

    let total_rows_q = b_size * sl_q;
    let concat_dim = n_heads * v_dim;

    let qk_scale = 1.0 / (k_dim as f32).sqrt();

    // 1. Output Projector Backpropagation
    let mut grad_out_concat = vec![0.0f32; total_rows_q * concat_dim];
    let grad_out_concat_ptr = SendPtr(grad_out_concat.as_mut_ptr());

    let (final_gwo, final_gbo) = (0..total_rows_q)
        .into_par_iter()
        .fold(
            move || (vec![0.0f32; concat_dim * o_dim], vec![0.0f32; o_dim]),
            move |(mut local_gwo, mut local_gbo), i| {
                unsafe {
                    for j in 0..o_dim {
                        let d_out = *grad_out_ptr.get().add(i * o_dim + j);
                        if use_bias {
                            local_gbo[j] += d_out;
                        }
                        for c in 0..concat_dim {
                            local_gwo[c * o_dim + j] += *out_concat_ptr.get().add(i * concat_dim + c) * d_out;
                            *grad_out_concat_ptr.get().add(i * concat_dim + c) += d_out * *w_o_ptr.get().add(c * o_dim + j);
                        }
                    }
                }
                (local_gwo, local_gbo)
            },
        )
        .reduce(
            move || (vec![0.0f32; concat_dim * o_dim], vec![0.0f32; o_dim]),
            move |(mut wo1, mut bo1), (wo2, bo2)| {
                for i in 0..wo1.len() { wo1[i] += wo2[i]; }
                for i in 0..bo1.len() { bo1[i] += bo2[i]; }
                (wo1, bo1)
            },
        );

    // Write w_o and b_o gradients
    let grad_w_o_slice = &mut *grad_w_o;
    let grad_b_o_slice = &mut *grad_b_o;
    for i in 0..grad_w_o_slice.len() { grad_w_o_slice[i] += final_gwo[i]; }
    for i in 0..grad_b_o_slice.len() { grad_b_o_slice[i] += final_gbo[i]; }

    // 2. Backward Self-Attention for Q_h, K_h, V_h per task (batch, head)
    let total_attention_tasks = b_size * n_heads;

    let mut grad_q = vec![0.0f32; b_size * sl_q * n_heads * k_dim];
    let mut grad_k = vec![0.0f32; b_size * sl_k * n_heads * k_dim];
    let mut grad_v = vec![0.0f32; b_size * sl_k * n_heads * v_dim];

    let grad_q_ptr = SendPtr(grad_q.as_mut_ptr());
    let grad_k_ptr = SendPtr(grad_k.as_mut_ptr());
    let grad_v_ptr = SendPtr(grad_v.as_mut_ptr());

    (0..total_attention_tasks).into_par_iter().for_each(move |task_id| {
        let b = task_id / n_heads;
        let h = task_id % n_heads;

        unsafe {
            let mut local_dprobs = vec![0.0f32; sl_q * sl_k];
            let mut local_dscores = vec![0.0f32; sl_q * sl_k];

            // A. Output backward to probs and local V
            for i in 0..sl_q {
                let prob_row_offset = (b * sl_q + i) * n_heads * sl_k + h * sl_k;
                let concat_row_offset = (b * sl_q + i) * n_heads * v_dim + h * v_dim;

                for d in 0..v_dim {
                    let d_out_concat = *grad_out_concat_ptr.get().add(concat_row_offset + d);

                    for j in 0..sl_k {
                        let v_row_offset = (b * sl_k + j) * n_heads * v_dim + h * v_dim;
                        local_dprobs[i * sl_k + j] += d_out_concat * *v_ptr.get().add(v_row_offset + d);
                        *grad_v_ptr.get().add(v_row_offset + d) += d_out_concat * *probs_ptr.get().add(prob_row_offset + j);
                    }
                }
            }

            // B. Softmax backward
            for i in 0..sl_q {
                let prob_row_offset = (b * sl_q + i) * n_heads * sl_k + h * sl_k;

                let mut dot_prod = 0.0;
                for k_idx in 0..sl_k {
                    dot_prod += local_dprobs[i * sl_k + k_idx] * *probs_ptr.get().add(prob_row_offset + k_idx);
                }

                for j in 0..sl_k {
                    let p_val = *probs_ptr.get().add(prob_row_offset + j);
                    local_dscores[i * sl_k + j] = p_val * (local_dprobs[i * sl_k + j] - dot_prod);
                }
            }

            // C. Scores backprop to local Q and local K
            for i in 0..sl_q {
                let q_row_offset = (b * sl_q + i) * n_heads * k_dim + h * k_dim;

                for j in 0..sl_k {
                    let k_row_offset = (b * sl_k + j) * n_heads * k_dim + h * k_dim;
                    let ds_val = local_dscores[i * sl_k + j] * qk_scale;

                    for d in 0..k_dim {
                        *grad_q_ptr.get().add(q_row_offset + d) += ds_val * *k_ptr.get().add(k_row_offset + d);
                        *grad_k_ptr.get().add(k_row_offset + d) += ds_val * *q_ptr.get().add(q_row_offset + d);
                    }
                }
            }
        }
    });

    // 3. Projections Backward Pass in parallel per batch element
    let grad_q_ptr_const = SendPtrConst(grad_q.as_ptr());
    let grad_k_ptr_const = SendPtrConst(grad_k.as_ptr());
    let grad_v_ptr_const = SendPtrConst(grad_v.as_ptr());

    let (final_gwq, final_gwk, final_gwv, final_gbq, final_gbk, final_gbv) = (0..b_size)
        .into_par_iter()
        .fold(
            move || (
                vec![0.0f32; in_dim_q * n_heads * k_dim],
                vec![0.0f32; in_dim_k * n_heads * k_dim],
                vec![0.0f32; in_dim_v * n_heads * v_dim],
                vec![0.0f32; n_heads * k_dim],
                vec![0.0f32; n_heads * k_dim],
                vec![0.0f32; n_heads * v_dim],
            ),
            move |(mut lgwq, mut lgwk, mut lgwv, mut lgbq, mut lgbk, mut lgbv), b| {
                unsafe {
                    // Q Backward
                    for t in 0..sl_q {
                        let row_offset = (b * sl_q + t) * in_dim_q;
                        let out_offset = (b * sl_q + t) * n_heads * k_dim;

                        for j in 0..(n_heads * k_dim) {
                            let dq_val = *grad_q_ptr_const.get().add(out_offset + j);
                            if use_bias {
                                lgbq[j] += dq_val;
                            }
                            for c in 0..in_dim_q {
                                lgwq[c * (n_heads * k_dim) + j] += *inputs_q_ptr.get().add(row_offset + c) * dq_val;
                            }
                        }

                        // Gradient for inputs_q
                        for c in 0..in_dim_q {
                            let mut sum = 0.0;
                            for j in 0..(n_heads * k_dim) {
                                sum += *grad_q_ptr_const.get().add(out_offset + j) * *w_q_ptr.get().add(c * (n_heads * k_dim) + j);
                            }
                            *grad_in_q_ptr.get().add(row_offset + c) = sum;
                        }
                    }

                    // K Backward
                    for t in 0..sl_k {
                        let row_offset = (b * sl_k + t) * in_dim_k;
                        let out_offset = (b * sl_k + t) * n_heads * k_dim;

                        for j in 0..(n_heads * k_dim) {
                            let dk_val = *grad_k_ptr_const.get().add(out_offset + j);
                            if use_bias {
                                lgbk[j] += dk_val;
                            }
                            for c in 0..in_dim_k {
                                lgwk[c * (n_heads * k_dim) + j] += *inputs_k_ptr.get().add(row_offset + c) * dk_val;
                            }
                        }

                        // Gradient for inputs_k
                        for c in 0..in_dim_k {
                            let mut sum = 0.0;
                            for j in 0..(n_heads * k_dim) {
                                sum += *grad_k_ptr_const.get().add(out_offset + j) * *w_k_ptr.get().add(c * (n_heads * k_dim) + j);
                            }
                            *grad_in_k_ptr.get().add(row_offset + c) = sum;
                        }
                    }

                    // V Backward
                    for t in 0..sl_k {
                        let row_offset = (b * sl_k + t) * in_dim_v;
                        let out_offset = (b * sl_k + t) * n_heads * v_dim;

                        for j in 0..(n_heads * v_dim) {
                            let dv_val = *grad_v_ptr_const.get().add(out_offset + j);
                            if use_bias {
                                lgbv[j] += dv_val;
                            }
                            for c in 0..in_dim_v {
                                lgwv[c * (n_heads * v_dim) + j] += *inputs_v_ptr.get().add(row_offset + c) * dv_val;
                            }
                        }

                        // Gradient for inputs_v
                        for c in 0..in_dim_v {
                            let mut sum = 0.0;
                            for j in 0..(n_heads * v_dim) {
                                sum += *grad_v_ptr_const.get().add(out_offset + j) * *w_v_ptr.get().add(c * (n_heads * v_dim) + j);
                            }
                            *grad_in_v_ptr.get().add(row_offset + c) = sum;
                        }
                    }
                }
                (lgwq, lgwk, lgwv, lgbq, lgbk, lgbv)
            },
        )
        .reduce(
            move || (
                vec![0.0f32; in_dim_q * n_heads * k_dim],
                vec![0.0f32; in_dim_k * n_heads * k_dim],
                vec![0.0f32; in_dim_v * n_heads * v_dim],
                vec![0.0f32; n_heads * k_dim],
                vec![0.0f32; n_heads * k_dim],
                vec![0.0f32; n_heads * v_dim],
            ),
            move |(mut wq1, mut wk1, mut wv1, mut bq1, mut bk1, mut bv1), (wq2, wk2, wv2, bq2, bk2, bv2)| {
                for i in 0..wq1.len() { wq1[i] += wq2[i]; }
                for i in 0..wk1.len() { wk1[i] += wk2[i]; }
                for i in 0..wv1.len() { wv1[i] += wv2[i]; }
                for i in 0..bq1.len() { bq1[i] += bq2[i]; }
                for i in 0..bk1.len() { bk1[i] += bk2[i]; }
                for i in 0..bv1.len() { bv1[i] += bv2[i]; }
                (wq1, wk1, wv1, bq1, bk1, bv1)
            },
        );

    // Add back into original NAPI output Float32Arrays
    let grad_w_q_slice = &mut *grad_w_q;
    let grad_w_k_slice = &mut *grad_w_k;
    let grad_w_v_slice = &mut *grad_w_v;
    let grad_b_q_slice = &mut *grad_b_q;
    let grad_b_k_slice = &mut *grad_b_k;
    let grad_b_v_slice = &mut *grad_b_v;

    for i in 0..grad_w_q_slice.len() { grad_w_q_slice[i] += final_gwq[i]; }
    for i in 0..grad_w_k_slice.len() { grad_w_k_slice[i] += final_gwk[i]; }
    for i in 0..grad_w_v_slice.len() { grad_w_v_slice[i] += final_gwv[i]; }
    for i in 0..grad_b_q_slice.len() { grad_b_q_slice[i] += final_gbq[i]; }
    for i in 0..grad_b_k_slice.len() { grad_b_k_slice[i] += final_gbk[i]; }
    for i in 0..grad_b_v_slice.len() { grad_b_v_slice[i] += final_gbv[i]; }
}
