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
pub fn attention_forward_native(
    inputs_q: Float32Array,
    inputs_k: Float32Array,
    w_q: Float32Array,
    w_k: Float32Array,
    w_v: Float32Array,
    b_q: Float32Array,
    b_k: Float32Array,
    b_v: Float32Array,
    batch_size: u32,
    seq_len_q: u32,
    seq_len_k: u32,
    input_dim: u32,
    units: u32,
    use_bias: bool,
    mut out: Float32Array,
    mut q: Float32Array,
    mut k: Float32Array,
    mut v: Float32Array,
    mut scores: Float32Array,
    mut probs: Float32Array,
) {
    let b_size = batch_size as usize;
    let sl_q = seq_len_q as usize;
    let sl_k = seq_len_k as usize;
    let in_dim = input_dim as usize;
    let h_dim = units as usize;

    let inputs_q_ptr = SendPtrConst(inputs_q.as_ptr());
    let inputs_k_ptr = SendPtrConst(inputs_k.as_ptr());
    let w_q_ptr = SendPtrConst(w_q.as_ptr());
    let w_k_ptr = SendPtrConst(w_k.as_ptr());
    let w_v_ptr = SendPtrConst(w_v.as_ptr());
    let b_q_ptr = SendPtrConst(b_q.as_ptr());
    let b_k_ptr = SendPtrConst(b_k.as_ptr());
    let b_v_ptr = SendPtrConst(b_v.as_ptr());

    let out_ptr = SendPtr(out.as_mut_ptr());
    let q_ptr = SendPtr(q.as_mut_ptr());
    let k_ptr = SendPtr(k.as_mut_ptr());
    let v_ptr = SendPtr(v.as_mut_ptr());
    let scores_ptr = SendPtr(scores.as_mut_ptr());
    let probs_ptr = SendPtr(probs.as_mut_ptr());

    let scale = 1.0 / (h_dim as f32).sqrt();

    (0..b_size).into_par_iter().for_each(move |b| {
        unsafe {
            // 1. Project inputs_q to Q
            for t in 0..sl_q {
                let row_offset = (b * sl_q + t) * in_dim;
                let out_offset = (b * sl_q + t) * h_dim;

                for j in 0..h_dim {
                    let mut sum_q = if use_bias { *b_q_ptr.get().add(j) } else { 0.0 };
                    for c in 0..in_dim {
                        sum_q += *inputs_q_ptr.get().add(row_offset + c) * *w_q_ptr.get().add(c * h_dim + j);
                    }
                    *q_ptr.get().add(out_offset + j) = sum_q;
                }
            }

            // 2. Project inputs_k to K and V
            for t in 0..sl_k {
                let row_offset = (b * sl_k + t) * in_dim;
                let out_offset = (b * sl_k + t) * h_dim;

                for j in 0..h_dim {
                    let mut sum_k = if use_bias { *b_k_ptr.get().add(j) } else { 0.0 };
                    let mut sum_v = if use_bias { *b_v_ptr.get().add(j) } else { 0.0 };

                    for c in 0..in_dim {
                        let val = *inputs_k_ptr.get().add(row_offset + c);
                        sum_k += val * *w_k_ptr.get().add(c * h_dim + j);
                        sum_v += val * *w_v_ptr.get().add(c * h_dim + j);
                    }

                    *k_ptr.get().add(out_offset + j) = sum_k;
                    *v_ptr.get().add(out_offset + j) = sum_v;
                }
            }

            // 3. Compute scores & Softmax
            for i in 0..sl_q {
                let score_offset = (b * sl_q + i) * sl_k;
                let q_offset = (b * sl_q + i) * h_dim;

                let mut max_val = -f32::INFINITY;

                for j in 0..sl_k {
                    let k_offset = (b * sl_k + j) * h_dim;
                    let mut sum = 0.0;
                    for h in 0..h_dim {
                        sum += *q_ptr.get().add(q_offset + h) * *k_ptr.get().add(k_offset + h);
                    }
                    let score_val = sum * scale;
                    *scores_ptr.get().add(score_offset + j) = score_val;
                    if score_val > max_val {
                        max_val = score_val;
                    }
                }

                // Softmax
                let mut sum_exp = 0.0;
                for j in 0..sl_k {
                    let s_val = *scores_ptr.get().add(score_offset + j);
                    let exp_val = (s_val - max_val).exp();
                    *probs_ptr.get().add(score_offset + j) = exp_val;
                    sum_exp += exp_val;
                }

                for j in 0..sl_k {
                    *probs_ptr.get().add(score_offset + j) /= sum_exp;
                }
            }

            // 4. Compute Output
            for i in 0..sl_q {
                let prob_offset = (b * sl_q + i) * sl_k;
                let out_offset = (b * sl_q + i) * h_dim;

                for h in 0..h_dim {
                    let mut sum = 0.0;
                    for j in 0..sl_k {
                        let v_offset = (b * sl_k + j) * h_dim;
                        sum += *probs_ptr.get().add(prob_offset + j) * *v_ptr.get().add(v_offset + h);
                    }
                    *out_ptr.get().add(out_offset + h) = sum;
                }
            }
        }
    });
}

#[napi]
pub fn attention_backward_native(
    grad_out: Float32Array,
    inputs_q: Float32Array,
    inputs_k: Float32Array,
    q: Float32Array,
    k: Float32Array,
    v: Float32Array,
    probs: Float32Array,
    w_q: Float32Array,
    w_k: Float32Array,
    w_v: Float32Array,
    batch_size: u32,
    seq_len_q: u32,
    seq_len_k: u32,
    input_dim: u32,
    units: u32,
    use_bias: bool,
    mut grad_in_q: Float32Array,
    mut grad_in_k: Float32Array,
    mut grad_w_q: Float32Array,
    mut grad_w_k: Float32Array,
    mut grad_w_v: Float32Array,
    mut grad_b_q: Float32Array,
    mut grad_b_k: Float32Array,
    mut grad_b_v: Float32Array,
) {
    let b_size = batch_size as usize;
    let sl_q = seq_len_q as usize;
    let sl_k = seq_len_k as usize;
    let in_dim = input_dim as usize;
    let h_dim = units as usize;

    let grad_out_ptr = SendPtrConst(grad_out.as_ptr());
    let inputs_q_ptr = SendPtrConst(inputs_q.as_ptr());
    let inputs_k_ptr = SendPtrConst(inputs_k.as_ptr());
    let q_ptr = SendPtrConst(q.as_ptr());
    let k_ptr = SendPtrConst(k.as_ptr());
    let v_ptr = SendPtrConst(v.as_ptr());
    let probs_ptr = SendPtrConst(probs.as_ptr());
    let w_q_ptr = SendPtrConst(w_q.as_ptr());
    let w_k_ptr = SendPtrConst(w_k.as_ptr());
    let w_v_ptr = SendPtrConst(w_v.as_ptr());

    let grad_in_q_ptr = SendPtr(grad_in_q.as_mut_ptr());
    let grad_in_k_ptr = SendPtr(grad_in_k.as_mut_ptr());

    let scale = 1.0 / (h_dim as f32).sqrt();

    let (final_gwq, final_gwk, final_gwv, final_gbq, final_gbk, final_gbv) = (0..b_size)
        .into_par_iter()
        .fold(
            move || (
                vec![0.0f32; in_dim * h_dim],
                vec![0.0f32; in_dim * h_dim],
                vec![0.0f32; in_dim * h_dim],
                vec![0.0f32; h_dim],
                vec![0.0f32; h_dim],
                vec![0.0f32; h_dim],
            ),
            move |(mut local_gwq, mut local_gwk, mut local_gwv, mut local_gbq, mut local_gbk, mut local_gbv), b| {
                unsafe {
                    let mut dq = vec![0.0f32; sl_q * h_dim];
                    let mut dk = vec![0.0f32; sl_k * h_dim];
                    let mut dv = vec![0.0f32; sl_k * h_dim];
                    let mut dprobs = vec![0.0f32; sl_q * sl_k];
                    let mut dscores = vec![0.0f32; sl_q * sl_k];

                    // 1. Output backprop
                    for i in 0..sl_q {
                        let out_offset = (b * sl_q + i) * h_dim;
                        let prob_offset = (b * sl_q + i) * sl_k;

                        for h in 0..h_dim {
                            let d_out = *grad_out_ptr.get().add(out_offset + h);

                            for j in 0..sl_k {
                                let v_offset = (b * sl_k + j) * h_dim;
                                dprobs[i * sl_k + j] += d_out * *v_ptr.get().add(v_offset + h);
                                dv[j * h_dim + h] += d_out * *probs_ptr.get().add(prob_offset + j);
                            }
                        }
                    }

                    // 2. Softmax backward
                    for i in 0..sl_q {
                        let prob_offset = (b * sl_q + i) * sl_k;

                        let mut dot_prod = 0.0;
                        for k in 0..sl_k {
                            dot_prod += dprobs[i * sl_k + k] * *probs_ptr.get().add(prob_offset + k);
                        }

                        for j in 0..sl_k {
                            let p_val = *probs_ptr.get().add(prob_offset + j);
                            dscores[i * sl_k + j] = p_val * (dprobs[i * sl_k + j] - dot_prod);
                        }
                    }

                    // 3. Score backprop
                    for i in 0..sl_q {
                        let q_offset = (b * sl_q + i) * h_dim;

                        for j in 0..sl_k {
                            let k_offset = (b * sl_k + j) * h_dim;
                            let ds_val = dscores[i * sl_k + j] * scale;

                            for h in 0..h_dim {
                                dq[i * h_dim + h] += ds_val * *k_ptr.get().add(k_offset + h);
                                dk[j * h_dim + h] += ds_val * *q_ptr.get().add(q_offset + h);
                            }
                        }
                    }

                    // 4. Linear projection backward & input gradient accumulation
                    for t in 0..sl_q {
                        let row_offset = (b * sl_q + t) * in_dim;

                        for j in 0..h_dim {
                            let dq_val = dq[t * h_dim + j];

                            if use_bias {
                                local_gbq[j] += dq_val;
                            }

                            for c in 0..in_dim {
                                local_gwq[c * h_dim + j] += *inputs_q_ptr.get().add(row_offset + c) * dq_val;
                            }
                        }

                        // Gradient of inputs_q
                        for c in 0..in_dim {
                            let mut sum_in = 0.0;
                            for h in 0..h_dim {
                                sum_in += dq[t * h_dim + h] * *w_q_ptr.get().add(c * h_dim + h);
                            }
                            *grad_in_q_ptr.get().add(row_offset + c) = sum_in;
                        }
                    }

                    for t in 0..sl_k {
                        let row_offset = (b * sl_k + t) * in_dim;

                        for j in 0..h_dim {
                            let dk_val = dk[t * h_dim + j];
                            let dv_val = dv[t * h_dim + j];

                            if use_bias {
                                local_gbk[j] += dk_val;
                                local_gbv[j] += dv_val;
                            }

                            for c in 0..in_dim {
                                let input_val = *inputs_k_ptr.get().add(row_offset + c);
                                local_gwk[c * h_dim + j] += input_val * dk_val;
                                local_gwv[c * h_dim + j] += input_val * dv_val;
                            }
                        }

                        // Gradient of inputs_k
                        for c in 0..in_dim {
                            let mut sum_in = 0.0;
                            for h in 0..h_dim {
                                sum_in += dk[t * h_dim + h] * *w_k_ptr.get().add(c * h_dim + h);
                                sum_in += dv[t * h_dim + h] * *w_v_ptr.get().add(c * h_dim + h);
                            }
                            *grad_in_k_ptr.get().add(row_offset + c) = sum_in;
                        }
                    }
                }
                (local_gwq, local_gwk, local_gwv, local_gbq, local_gbk, local_gbv)
            },
        )
        .reduce(
            move || (
                vec![0.0f32; in_dim * h_dim],
                vec![0.0f32; in_dim * h_dim],
                vec![0.0f32; in_dim * h_dim],
                vec![0.0f32; h_dim],
                vec![0.0f32; h_dim],
                vec![0.0f32; h_dim],
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
