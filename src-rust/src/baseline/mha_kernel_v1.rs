// Baseline snapshot of MHA native kernel before allocation/copy optimization.
// Source: src-rust/src/lib.rs (pre-rewrite)

pub fn mha_forward_block_v1(
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

pub fn mha_backward_block_v1(
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
