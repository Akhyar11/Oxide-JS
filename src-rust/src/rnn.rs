use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;

#[napi]
pub fn rnn_forward_native_into(
    x_data: Float32Array,
    wxh_data: Float32Array,
    whh_data: Float32Array,
    bh_data: Float32Array,
    seq_len: u32,
    batch_size: u32,
    units: u32,
    hidden_units: u32,
    is_relu: bool,
    mut h_seq_data: Float32Array,
    mut dact_data: Float32Array,
) {
    let seq_len = seq_len as usize;
    let batch_size = batch_size as usize;
    let units = units as usize;
    let hidden_units = hidden_units as usize;

    let total_cols = seq_len * batch_size;
    let mut projected = vec![0.0; hidden_units * batch_size];
    let mut recurrent = vec![0.0; hidden_units * batch_size];

    for step in 0..seq_len {
        let col_offset = step * batch_size;
        
        // projected = Wxh * X_t
        for i in 0..hidden_units {
            for b in 0..batch_size {
                let mut sum = 0.0;
                let col_offset = step * batch_size;
                for j in 0..units {
                    let w = wxh_data[i * units + j];
                    let x = x_data[j * total_cols + col_offset + b];
                    sum += w * x;
                }
                projected[i * batch_size + b] = sum;
            }
        }

        // recurrent = Whh * H_{t-1}
        for i in 0..hidden_units {
            for b in 0..batch_size {
                let mut sum = 0.0;
                for j in 0..hidden_units {
                    let w = whh_data[i * hidden_units + j];
                    let h_prev = h_seq_data[step * hidden_units * batch_size + j * batch_size + b];
                    sum += w * h_prev;
                }
                recurrent[i * batch_size + b] = sum;
            }
        }

        // h_t = activation(projected + recurrent + bh)
        for i in 0..hidden_units {
            let bias = bh_data[i];
            for b in 0..batch_size {
                let idx = i * batch_size + b;
                let sum = projected[idx] + recurrent[idx] + bias;
                
                let h_val;
                let dact_val;
                
                if is_relu {
                    if sum > 0.0 {
                        h_val = sum;
                        dact_val = 1.0;
                    } else {
                        h_val = 0.0;
                        dact_val = 0.0;
                    }
                } else {
                    let tv = sum.tanh();
                    h_val = tv;
                    dact_val = 1.0 - tv * tv;
                }
                
                h_seq_data[(step + 1) * hidden_units * batch_size + idx] = h_val;
                dact_data[step * hidden_units * batch_size + idx] = dact_val;
            }
        }
    }
}

#[napi]
pub fn rnn_backward_native_into(
    x_data: Float32Array,
    wxh_data: Float32Array,
    whh_data: Float32Array,
    h_seq_data: Float32Array,
    dact_data: Float32Array,
    ext_err_data: Float32Array,
    seq_len: u32,
    batch_size: u32,
    units: u32,
    hidden_units: u32,
    mut dwxh_data: Float32Array,
    mut dwhh_data: Float32Array,
    mut dbh_data: Float32Array,
    mut dx_data: Float32Array,
) {
    let seq_len = seq_len as usize;
    let batch_size = batch_size as usize;
    let units = units as usize;
    let hidden_units = hidden_units as usize;
    let total_cols = seq_len * batch_size;

    let mut dh_next = vec![0.0; hidden_units * batch_size];
    let mut dz = vec![0.0; hidden_units * batch_size];

    for step in (0..seq_len).rev() {
        let col_offset = step * batch_size;
        
        // compute dz = (ext_err + dh_next) * dact
        for i in 0..hidden_units {
            for b in 0..batch_size {
                let idx = i * batch_size + b;
                let step_idx = step * hidden_units * batch_size + idx;
                let mut dh = ext_err_data[step_idx] + dh_next[idx];
                dz[idx] = dh * dact_data[step_idx];
                
                // dbh += dz
                dbh_data[i] += dz[idx];
            }
        }

        // dWxh += dz * X_t^T
        for i in 0..hidden_units {
            for j in 0..units {
                let mut sum = 0.0;
                for b in 0..batch_size {
                    sum += dz[i * batch_size + b] * x_data[j * total_cols + col_offset + b];
                }
                dwxh_data[i * units + j] += sum;
            }
        }

        // dWhh += dz * H_{t-1}^T
        let prev_step = step * hidden_units * batch_size;
        for i in 0..hidden_units {
            for j in 0..hidden_units {
                let mut sum = 0.0;
                for b in 0..batch_size {
                    sum += dz[i * batch_size + b] * h_seq_data[prev_step + j * batch_size + b];
                }
                dwhh_data[i * hidden_units + j] += sum;
            }
        }

        // dx_t = Wxh^T * dz
        for j in 0..units {
            for b in 0..batch_size {
                let mut sum = 0.0;
                for i in 0..hidden_units {
                    sum += wxh_data[i * units + j] * dz[i * batch_size + b];
                }
                dx_data[j * total_cols + col_offset + b] = sum;
            }
        }

        // dh_next = Whh^T * dz
        for j in 0..hidden_units {
            for b in 0..batch_size {
                let mut sum = 0.0;
                for i in 0..hidden_units {
                    sum += whh_data[i * hidden_units + j] * dz[i * batch_size + b];
                }
                dh_next[j * batch_size + b] = sum;
            }
        }
    }
}

#[napi]
pub fn lstm_forward_native_into(
    x_data: Float32Array,
    wxi_data: Float32Array,
    whi_data: Float32Array,
    bi_data: Float32Array,
    wxf_data: Float32Array,
    whf_data: Float32Array,
    bf_data: Float32Array,
    wxo_data: Float32Array,
    who_data: Float32Array,
    bo_data: Float32Array,
    wxg_data: Float32Array,
    whg_data: Float32Array,
    bg_data: Float32Array,
    seq_len: u32,
    batch_size: u32,
    units: u32,
    hidden_units: u32,
    mut h_seq_data: Float32Array,
    mut c_seq_data: Float32Array,
    mut i_seq_data: Float32Array,
    mut f_seq_data: Float32Array,
    mut o_seq_data: Float32Array,
    mut g_seq_data: Float32Array,
) {
    let seq_len = seq_len as usize;
    let batch_size = batch_size as usize;
    let units = units as usize;
    let hidden_units = hidden_units as usize;
    let total_cols = seq_len * batch_size;

    for step in 0..seq_len {
        let col_offset = step * batch_size;
        let step_offset = step * hidden_units * batch_size;
        let h_prev_offset = step * hidden_units * batch_size;
        let h_next_offset = (step + 1) * hidden_units * batch_size;
        let c_prev_offset = step * hidden_units * batch_size;
        let c_next_offset = (step + 1) * hidden_units * batch_size;

        for i in 0..hidden_units {
            let bi = bi_data[i];
            let bf = bf_data[i];
            let bo = bo_data[i];
            let bg = bg_data[i];

            for b in 0..batch_size {
                let mut sum_i = bi;
                let mut sum_f = bf;
                let mut sum_o = bo;
                let mut sum_g = bg;

                for j in 0..units {
                    let x_val = x_data[j * total_cols + col_offset + b];
                    sum_i += wxi_data[i * units + j] * x_val;
                    sum_f += wxf_data[i * units + j] * x_val;
                    sum_o += wxo_data[i * units + j] * x_val;
                    sum_g += wxg_data[i * units + j] * x_val;
                }

                for j in 0..hidden_units {
                    let h_prev_val = h_seq_data[h_prev_offset + j * batch_size + b];
                    sum_i += whi_data[i * hidden_units + j] * h_prev_val;
                    sum_f += whf_data[i * hidden_units + j] * h_prev_val;
                    sum_o += who_data[i * hidden_units + j] * h_prev_val;
                    sum_g += whg_data[i * hidden_units + j] * h_prev_val;
                }

                let gate_idx = step_offset + i * batch_size + b;
                let i_val = 1.0 / (1.0 + (-sum_i).exp());
                let f_val = 1.0 / (1.0 + (-sum_f).exp());
                let o_val = 1.0 / (1.0 + (-sum_o).exp());
                let g_val = sum_g.tanh();

                i_seq_data[gate_idx] = i_val;
                f_seq_data[gate_idx] = f_val;
                o_seq_data[gate_idx] = o_val;
                g_seq_data[gate_idx] = g_val;

                let c_prev = c_seq_data[c_prev_offset + i * batch_size + b];
                let c_next = f_val * c_prev + i_val * g_val;
                c_seq_data[c_next_offset + i * batch_size + b] = c_next;
                h_seq_data[h_next_offset + i * batch_size + b] = o_val * c_next.tanh();
            }
        }
    }
}

#[napi]
pub fn lstm_backward_native_into(
    x_data: Float32Array,
    wxi_data: Float32Array,
    whi_data: Float32Array,
    wxf_data: Float32Array,
    whf_data: Float32Array,
    wxo_data: Float32Array,
    who_data: Float32Array,
    wxg_data: Float32Array,
    whg_data: Float32Array,
    h_seq_data: Float32Array,
    c_seq_data: Float32Array,
    i_seq_data: Float32Array,
    f_seq_data: Float32Array,
    o_seq_data: Float32Array,
    g_seq_data: Float32Array,
    ext_err_data: Float32Array,
    seq_len: u32,
    batch_size: u32,
    units: u32,
    hidden_units: u32,
    mut dwxi_data: Float32Array,
    mut dwhi_data: Float32Array,
    mut dbi_data: Float32Array,
    mut dwxf_data: Float32Array,
    mut dwhf_data: Float32Array,
    mut dbf_data: Float32Array,
    mut dwxo_data: Float32Array,
    mut dwho_data: Float32Array,
    mut dbo_data: Float32Array,
    mut dwxg_data: Float32Array,
    mut dwhg_data: Float32Array,
    mut dbg_data: Float32Array,
    mut dx_data: Float32Array,
) {
    let seq_len = seq_len as usize;
    let batch_size = batch_size as usize;
    let units = units as usize;
    let hidden_units = hidden_units as usize;
    let total_cols = seq_len * batch_size;

    let mut dh_next = vec![0.0; hidden_units * batch_size];
    let mut dc_next = vec![0.0; hidden_units * batch_size];
    let mut dzi = vec![0.0; hidden_units * batch_size];
    let mut dzf = vec![0.0; hidden_units * batch_size];
    let mut dzo = vec![0.0; hidden_units * batch_size];
    let mut dzg = vec![0.0; hidden_units * batch_size];

    for step in (0..seq_len).rev() {
        let col_offset = step * batch_size;
        let step_offset = step * hidden_units * batch_size;
        let h_prev_offset = step * hidden_units * batch_size;
        let c_prev_offset = step * hidden_units * batch_size;
        let c_next_offset = (step + 1) * hidden_units * batch_size;

        for i in 0..hidden_units {
            for b in 0..batch_size {
                let idx = i * batch_size + b;
                let gate_idx = step_offset + idx;
                
                let dh = ext_err_data[gate_idx] + dh_next[idx];
                let c_next = c_seq_data[c_next_offset + idx];
                let c_prev = c_seq_data[c_prev_offset + idx];
                let tanh_c = c_next.tanh();
                
                let i_val = i_seq_data[gate_idx];
                let f_val = f_seq_data[gate_idx];
                let o_val = o_seq_data[gate_idx];
                let g_val = g_seq_data[gate_idx];

                let do_gate = dh * tanh_c;
                let dc = dh * o_val * (1.0 - tanh_c * tanh_c) + dc_next[idx];
                let df = dc * c_prev;
                let di = dc * g_val;
                let dg = dc * i_val;

                dzi[idx] = di * i_val * (1.0 - i_val);
                dzf[idx] = df * f_val * (1.0 - f_val);
                dzo[idx] = do_gate * o_val * (1.0 - o_val);
                dzg[idx] = dg * (1.0 - g_val * g_val);

                dbi_data[i] += dzi[idx];
                dbf_data[i] += dzf[idx];
                dbo_data[i] += dzo[idx];
                dbg_data[i] += dzg[idx];

                // dW += dz * x^T
                for j in 0..units {
                    let x_val = x_data[j * total_cols + col_offset + b];
                    dwxi_data[i * units + j] += dzi[idx] * x_val;
                    dwxf_data[i * units + j] += dzf[idx] * x_val;
                    dwxo_data[i * units + j] += dzo[idx] * x_val;
                    dwxg_data[i * units + j] += dzg[idx] * x_val;
                }

                // dWh
                for j in 0..hidden_units {
                    let h_prev_val = h_seq_data[h_prev_offset + j * batch_size + b];
                    dwhi_data[i * hidden_units + j] += dzi[idx] * h_prev_val;
                    dwhf_data[i * hidden_units + j] += dzf[idx] * h_prev_val;
                    dwho_data[i * hidden_units + j] += dzo[idx] * h_prev_val;
                    dwhg_data[i * hidden_units + j] += dzg[idx] * h_prev_val;
                }

                dc_next[idx] = dc * f_val;
            }
        }

        // dx = W^T * dz
        for j in 0..units {
            for b in 0..batch_size {
                let mut sum = 0.0;
                for i in 0..hidden_units {
                    let idx = i * batch_size + b;
                    sum += wxi_data[i * units + j] * dzi[idx];
                    sum += wxf_data[i * units + j] * dzf[idx];
                    sum += wxo_data[i * units + j] * dzo[idx];
                    sum += wxg_data[i * units + j] * dzg[idx];
                }
                dx_data[j * total_cols + col_offset + b] = sum;
            }
        }

        // dh_next = Wh^T * dz
        for j in 0..hidden_units {
            for b in 0..batch_size {
                let mut sum = 0.0;
                for i in 0..hidden_units {
                    let idx = i * batch_size + b;
                    sum += whi_data[i * hidden_units + j] * dzi[idx];
                    sum += whf_data[i * hidden_units + j] * dzf[idx];
                    sum += who_data[i * hidden_units + j] * dzo[idx];
                    sum += whg_data[i * hidden_units + j] * dzg[idx];
                }
                dh_next[j * batch_size + b] = sum;
            }
        }
    }
}

#[napi]
pub fn gru_forward_native_into(
    x_data: Float32Array,
    wxr_data: Float32Array,
    whr_data: Float32Array,
    br_data: Float32Array,
    wxz_data: Float32Array,
    whz_data: Float32Array,
    bz_data: Float32Array,
    wxh_data: Float32Array,
    whh_data: Float32Array,
    bh_data: Float32Array,
    seq_len: u32,
    batch_size: u32,
    units: u32,
    hidden_units: u32,
    reverse: bool,
    mut h_seq_data: Float32Array,
    mut r_seq_data: Float32Array,
    mut z_seq_data: Float32Array,
    mut n_seq_data: Float32Array,
) {
    let seq_len = seq_len as usize;
    let batch_size = batch_size as usize;
    let units = units as usize;
    let hidden_units = hidden_units as usize;
    let total_cols = seq_len * batch_size;

    for step in 0..seq_len {
        let t = if reverse { seq_len - 1 - step } else { step };
        let col_offset = t * batch_size;
        let step_offset = step * hidden_units * batch_size;
        let h_prev_offset = step * hidden_units * batch_size;
        let h_next_offset = (step + 1) * hidden_units * batch_size;

        for i in 0..hidden_units {
            for b in 0..batch_size {
                let mut sum_r = br_data[i];
                let mut sum_z = bz_data[i];

                for j in 0..units {
                    let x_val = x_data[j * total_cols + col_offset + b];
                    sum_r += wxr_data[i * units + j] * x_val;
                    sum_z += wxz_data[i * units + j] * x_val;
                }

                for j in 0..hidden_units {
                    let h_prev_val = h_seq_data[h_prev_offset + j * batch_size + b];
                    sum_r += whr_data[i * hidden_units + j] * h_prev_val;
                    sum_z += whz_data[i * hidden_units + j] * h_prev_val;
                }

                let gate_idx = step_offset + i * batch_size + b;
                let r_val = 1.0 / (1.0 + (-sum_r).exp());
                let z_val = 1.0 / (1.0 + (-sum_z).exp());

                r_seq_data[gate_idx] = r_val;
                z_seq_data[gate_idx] = z_val;
            }
        }

        for i in 0..hidden_units {
            for b in 0..batch_size {
                let mut h_mix = 0.0;
                for j in 0..hidden_units {
                    let r_j = r_seq_data[step_offset + j * batch_size + b];
                    let h_prev_j = h_seq_data[h_prev_offset + j * batch_size + b];
                    h_mix += whh_data[i * hidden_units + j] * (r_j * h_prev_j);
                }

                let mut x_term = bh_data[i];
                for j in 0..units {
                    let x_val = x_data[j * total_cols + col_offset + b];
                    x_term += wxh_data[i * units + j] * x_val;
                }

                let gate_idx = step_offset + i * batch_size + b;
                let n_val = (x_term + h_mix).tanh();
                n_seq_data[gate_idx] = n_val;

                let z_val = z_seq_data[gate_idx];
                let h_prev = h_seq_data[h_prev_offset + i * batch_size + b];
                h_seq_data[h_next_offset + i * batch_size + b] = (1.0 - z_val) * n_val + z_val * h_prev;
            }
        }
    }
}

#[napi]
pub fn gru_backward_native_into(
    x_seq_data: Float32Array,
    wxr_data: Float32Array,
    whr_data: Float32Array,
    wxz_data: Float32Array,
    whz_data: Float32Array,
    wxh_data: Float32Array,
    whh_data: Float32Array,
    h_seq_data: Float32Array,
    r_seq_data: Float32Array,
    z_seq_data: Float32Array,
    n_seq_data: Float32Array,
    ext_err_data: Float32Array,
    seq_len: u32,
    batch_size: u32,
    units: u32,
    hidden_units: u32,
    reverse: bool,
    mut dwxr_data: Float32Array,
    mut dwhr_data: Float32Array,
    mut dbr_data: Float32Array,
    mut dwxz_data: Float32Array,
    mut dwhz_data: Float32Array,
    mut dbz_data: Float32Array,
    mut dwxh_data: Float32Array,
    mut dwhh_data: Float32Array,
    mut dbh_data: Float32Array,
    mut dx_data: Float32Array,
) {
    let seq_len = seq_len as usize;
    let batch_size = batch_size as usize;
    let units = units as usize;
    let hidden_units = hidden_units as usize;

    let mut dh_next = vec![0.0; hidden_units * batch_size];
    let mut dan = vec![0.0; hidden_units * batch_size];
    let mut daz = vec![0.0; hidden_units * batch_size];
    let mut dar = vec![0.0; hidden_units * batch_size];

    for step in (0..seq_len).rev() {
        let t = if reverse { seq_len - 1 - step } else { step };
        let step_offset = step * hidden_units * batch_size;
        let h_prev_offset = step * hidden_units * batch_size;
        let x_step_offset = step * units * batch_size;

        for i in 0..hidden_units {
            for b in 0..batch_size {
                let idx = i * batch_size + b;
                let gate_idx = step_offset + idx;
                
                let dh = ext_err_data[t * hidden_units * batch_size + idx] + dh_next[idx];
                let h_prev = h_seq_data[h_prev_offset + idx];
                let n_val = n_seq_data[gate_idx];
                let z_val = z_seq_data[gate_idx];

                let dn = dh * (1.0 - z_val);
                let dz = dh * (h_prev - n_val);
                dan[idx] = dn * (1.0 - n_val * n_val);
                daz[idx] = dz * z_val * (1.0 - z_val);

                dbh_data[i] += dan[idx];
                dbz_data[i] += daz[idx];

                for j in 0..units {
                    let x_val = x_seq_data[x_step_offset + j * batch_size + b];
                    dwxh_data[i * units + j] += dan[idx] * x_val;
                    dwxz_data[i * units + j] += daz[idx] * x_val;
                }

                for j in 0..hidden_units {
                    let h_prev_j = h_seq_data[h_prev_offset + j * batch_size + b];
                    dwhz_data[i * hidden_units + j] += daz[idx] * h_prev_j;
                    
                    let r_j = r_seq_data[step_offset + j * batch_size + b];
                    dwhh_data[i * hidden_units + j] += dan[idx] * (r_j * h_prev_j);
                }
            }
        }

        for j in 0..hidden_units {
            for b in 0..batch_size {
                let idx = j * batch_size + b;
                let mut dr_from_n = 0.0;
                for i in 0..hidden_units {
                    dr_from_n += whh_data[i * hidden_units + j] * dan[i * batch_size + b];
                }
                let h_prev_j = h_seq_data[h_prev_offset + idx];
                let r_val = r_seq_data[step_offset + idx];
                dar[idx] = dr_from_n * h_prev_j * r_val * (1.0 - r_val);
            }
        }

        for i in 0..hidden_units {
            for b in 0..batch_size {
                let idx = i * batch_size + b;
                dbr_data[i] += dar[idx];
                for j in 0..units {
                    let x_val = x_seq_data[x_step_offset + j * batch_size + b];
                    dwxr_data[i * units + j] += dar[idx] * x_val;
                }
                for j in 0..hidden_units {
                    let h_prev_j = h_seq_data[h_prev_offset + j * batch_size + b];
                    dwhr_data[i * hidden_units + j] += dar[idx] * h_prev_j;
                }
            }
        }

        for j in 0..units {
            for b in 0..batch_size {
                let mut sum = 0.0;
                for i in 0..hidden_units {
                    let idx = i * batch_size + b;
                    sum += wxh_data[i * units + j] * dan[idx];
                    sum += wxr_data[i * units + j] * dar[idx];
                    sum += wxz_data[i * units + j] * daz[idx];
                }
                dx_data[t * units * batch_size + j * batch_size + b] = sum;
            }
        }

        for j in 0..hidden_units {
            for b in 0..batch_size {
                let idx = j * batch_size + b;
                let mut sum = 0.0;
                for i in 0..hidden_units {
                    let i_idx = i * batch_size + b;
                    sum += whr_data[i * hidden_units + j] * dar[i_idx];
                    sum += whz_data[i * hidden_units + j] * daz[i_idx];
                    
                    let r_j = r_seq_data[step_offset + idx];
                    sum += whh_data[i * hidden_units + j] * dan[i_idx] * r_j;
                }
                let dh = ext_err_data[t * hidden_units * batch_size + idx] + dh_next[idx];
                let z_val = z_seq_data[step_offset + idx];
                dh_next[idx] = sum + dh * z_val;
            }
        }
    }
}
