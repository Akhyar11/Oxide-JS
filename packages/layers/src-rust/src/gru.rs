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

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[napi]
pub fn gru_forward_native(
    inputs: Float32Array,
    kernel: Float32Array,
    recurrent_kernel: Float32Array,
    bias: Float32Array,
    batch_size: u32,
    sequence_length: u32,
    input_dim: u32,
    units: u32,
    return_sequences: bool,
    mut out: Float32Array,
    mut hidden_states: Float32Array,
    mut gate_values: Float32Array,
) {
    let b_size = batch_size as usize;
    let seq_len = sequence_length as usize;
    let in_dim = input_dim as usize;
    let h_dim = units as usize;

    let inputs_ptr = SendPtrConst(inputs.as_ptr());
    let kernel_ptr = SendPtrConst(kernel.as_ptr());
    let recurrent_kernel_ptr = SendPtrConst(recurrent_kernel.as_ptr());
    let bias_ptr = SendPtrConst(bias.as_ptr());
    let out_ptr = SendPtr(out.as_mut_ptr());
    let hidden_states_ptr = SendPtr(hidden_states.as_mut_ptr());
    let gate_values_ptr = SendPtr(gate_values.as_mut_ptr());

    (0..b_size).into_par_iter().for_each(move |b| {
        let h = vec![0.0f32; h_dim];
        let mut a = vec![0.0f32; 3 * h_dim];

        for t in 0..seq_len {
            let step_offset = (b * seq_len + t) * in_dim;
            let hidden_offset = (b * seq_len + t) * h_dim;
            let gate_offset = (b * seq_len + t) * 3 * h_dim;

            unsafe {
                // 1. Calculate x_t * W + bias
                for j in 0..(3 * h_dim) {
                    let mut sum = *bias_ptr.get().add(j);
                    for c_idx in 0..in_dim {
                        sum += *inputs_ptr.get().add(step_offset + c_idx) * *kernel_ptr.get().add(c_idx * 3 * h_dim + j);
                    }
                    a[j] = sum;
                }

                // 2. Compute gates z and r
                let prev_h_ptr = if t > 0 {
                    hidden_states_ptr.get().add((b * seq_len + t - 1) * h_dim)
                } else {
                    h.as_ptr()
                };

                // Update gate (z) and Reset gate (r) pre-activations and activations
                for j in 0..(2 * h_dim) {
                    let mut sum = 0.0;
                    for k in 0..h_dim {
                        sum += *prev_h_ptr.add(k) * *recurrent_kernel_ptr.get().add(k * 3 * h_dim + j);
                    }
                    a[j] += sum;
                }

                for j in 0..h_dim {
                    let gate_z = sigmoid(a[j]);
                    let gate_r = sigmoid(a[h_dim + j]);

                    *gate_values_ptr.get().add(gate_offset + j) = gate_z;
                    *gate_values_ptr.get().add(gate_offset + h_dim + j) = gate_r;
                }

                // 3. Candidate hidden state (h_tilde)
                // a_h = x_t * W_h + b_h + (r_t * h_{t-1}) * U_h
                let mut sum_h = 0.0;
                for k in 0..h_dim {
                    let r_val = *gate_values_ptr.get().add(gate_offset + h_dim + k);
                    let h_val = *prev_h_ptr.add(k);
                    sum_h += (r_val * h_val) * *recurrent_kernel_ptr.get().add(k * 3 * h_dim + 2 * h_dim);
                }
                a[2 * h_dim] = sum_h; // Will add to each units slot below

                for j in 0..h_dim {
                    let mut sum = 0.0;
                    for k in 0..h_dim {
                        let r_val = *gate_values_ptr.get().add(gate_offset + h_dim + k);
                        let h_val = *prev_h_ptr.add(k);
                        sum += (r_val * h_val) * *recurrent_kernel_ptr.get().add(k * 3 * h_dim + 2 * h_dim + j);
                    }
                    let candidate_a = a[2 * h_dim + j] + sum;
                    let gate_h = candidate_a.tanh();

                    *gate_values_ptr.get().add(gate_offset + 2 * h_dim + j) = gate_h;

                    // Compute new hidden state: h_t = (1 - z) * h_{t-1} + z * h_tilde
                    let z_val = *gate_values_ptr.get().add(gate_offset + j);
                    let h_prev_val = *prev_h_ptr.add(j);
                    let new_h = (1.0 - z_val) * h_prev_val + z_val * gate_h;

                    *hidden_states_ptr.get().add(hidden_offset + j) = new_h;
                }

                // 4. Save to out if return_sequences is true
                if return_sequences {
                    let out_offset = (b * seq_len + t) * h_dim;
                    for j in 0..h_dim {
                        *out_ptr.get().add(out_offset + j) = *hidden_states_ptr.get().add(hidden_offset + j);
                    }
                }
            }
        }

        // 5. If return_sequences is false, save the last hidden state to out
        if !return_sequences {
            unsafe {
                let last_hidden_offset = (b * seq_len + seq_len - 1) * h_dim;
                let out_offset = b * h_dim;
                for j in 0..h_dim {
                    *out_ptr.get().add(out_offset + j) = *hidden_states_ptr.get().add(last_hidden_offset + j);
                }
            }
        }
    });
}

#[napi]
pub fn gru_backward_native(
    grad_out: Float32Array,
    inputs: Float32Array,
    hidden_states: Float32Array,
    gate_values: Float32Array,
    kernel: Float32Array,
    recurrent_kernel: Float32Array,
    batch_size: u32,
    sequence_length: u32,
    input_dim: u32,
    units: u32,
    return_sequences: bool,
    mut grad_in: Float32Array,
    mut grad_kernel: Float32Array,
    mut grad_recurrent_kernel: Float32Array,
    mut grad_bias: Float32Array,
) {
    let b_size = batch_size as usize;
    let seq_len = sequence_length as usize;
    let in_dim = input_dim as usize;
    let h_dim = units as usize;

    let grad_out_ptr = SendPtrConst(grad_out.as_ptr());
    let inputs_ptr = SendPtrConst(inputs.as_ptr());
    let hidden_states_ptr = SendPtrConst(hidden_states.as_ptr());
    let gate_values_ptr = SendPtrConst(gate_values.as_ptr());
    let kernel_ptr = SendPtrConst(kernel.as_ptr());
    let recurrent_kernel_ptr = SendPtrConst(recurrent_kernel.as_ptr());
    let grad_in_ptr = SendPtr(grad_in.as_mut_ptr());

    let (final_grad_kernel, final_grad_recurrent_kernel, final_grad_bias) = (0..b_size)
        .into_par_iter()
        .fold(
            move || (
                vec![0.0f32; in_dim * 3 * h_dim],
                vec![0.0f32; h_dim * 3 * h_dim],
                vec![0.0f32; 3 * h_dim],
            ),
            move |(mut local_gk, mut local_grk, mut local_gb), b| {
                let mut dh_next = vec![0.0f32; h_dim];
                let mut da = vec![0.0f32; 3 * h_dim];

                for t in (0..seq_len).rev() {
                    let step_offset = (b * seq_len + t) * in_dim;
                    let gate_offset = (b * seq_len + t) * 3 * h_dim;

                    unsafe {
                        // 1. Get gradient from output layer (d_out)
                        let mut dh = vec![0.0f32; h_dim];
                        for j in 0..h_dim {
                            let d_out = if return_sequences {
                                *grad_out_ptr.get().add((b * seq_len + t) * h_dim + j)
                            } else {
                                if t == seq_len - 1 {
                                    *grad_out_ptr.get().add(b * h_dim + j)
                                } else {
                                    0.0
                                }
                            };
                            dh[j] = d_out + dh_next[j];
                        }

                        // 2. Compute gate pre-activation gradients
                        let prev_h_ptr = if t > 0 {
                            hidden_states_ptr.get().add((b * seq_len + t - 1) * h_dim)
                        } else {
                            [0.0f32; 512].as_ptr() // safe fallback pointer for 0.0 values
                        };

                        let mut dh_reset = vec![0.0f32; h_dim];

                        for j in 0..h_dim {
                            let gate_z = *gate_values_ptr.get().add(gate_offset + j);
                            let gate_h = *gate_values_ptr.get().add(gate_offset + 2 * h_dim + j);

                            let h_prev_val = if t > 0 { *prev_h_ptr.add(j) } else { 0.0 };

                            let d_h_tilde = dh[j] * gate_z;
                            let da_h = d_h_tilde * (1.0 - gate_h * gate_h);

                            let d_z = dh[j] * (gate_h - h_prev_val);
                            let da_z = d_z * gate_z * (1.0 - gate_z);

                            da[j] = da_z;
                            da[2 * h_dim + j] = da_h;
                        }

                        // Reseted gradient contribution from Candidate preactivation da_h
                        for k in 0..h_dim {
                            let mut sum = 0.0;
                            for j in 0..h_dim {
                                sum += da[2 * h_dim + j] * *recurrent_kernel_ptr.get().add(k * 3 * h_dim + 2 * h_dim + j);
                            }
                            dh_reset[k] = sum;
                        }

                        for j in 0..h_dim {
                            let gate_r = *gate_values_ptr.get().add(gate_offset + h_dim + j);
                            let h_prev_val = if t > 0 { *prev_h_ptr.add(j) } else { 0.0 };

                            let d_r = dh_reset[j] * h_prev_val;
                            let da_r = d_r * gate_r * (1.0 - gate_r);

                            da[h_dim + j] = da_r;
                        }

                        // 3. Accumulate local kernel, recurrent kernel and bias gradients
                        for j in 0..(3 * h_dim) {
                            local_gb[j] += da[j];

                            for c_idx in 0..in_dim {
                                local_gk[c_idx * 3 * h_dim + j] += *inputs_ptr.get().add(step_offset + c_idx) * da[j];
                            }

                            if t > 0 {
                                // For U_z (0..H) and U_r (H..2*H), the input is hidden_states_{t-1}
                                for k in 0..h_dim {
                                    let h_prev_val = *prev_h_ptr.add(k);
                                    if j < 2 * h_dim {
                                        local_grk[k * 3 * h_dim + j] += h_prev_val * da[j];
                                    } else {
                                        // For U_h, input is r_t * h_{t-1}
                                        let r_val = *gate_values_ptr.get().add(gate_offset + h_dim + k);
                                        local_grk[k * 3 * h_dim + j] += (r_val * h_prev_val) * da[j];
                                    }
                                }
                            }
                        }

                        // 4. Input gradient (dx_t)
                        for c_idx in 0..in_dim {
                            let mut sum = 0.0;
                            for g in 0..(3 * h_dim) {
                                sum += da[g] * *kernel_ptr.get().add(c_idx * 3 * h_dim + g);
                            }
                            *grad_in_ptr.get().add(step_offset + c_idx) = sum;
                        }

                        // 5. Update dh_next for the next backward step (t-1)
                        for k in 0..h_dim {
                            let mut sum = 0.0;

                            // Direct recurrence z gate contribution
                            let gate_z = *gate_values_ptr.get().add(gate_offset + k);
                            sum += dh[k] * (1.0 - gate_z);

                            // Reseted contribution
                            let gate_r = *gate_values_ptr.get().add(gate_offset + h_dim + k);
                            sum += dh_reset[k] * gate_r;

                            // U_z and U_r recurrent contribution
                            for j in 0..(2 * h_dim) {
                                sum += da[j] * *recurrent_kernel_ptr.get().add(k * 3 * h_dim + j);
                            }

                            dh_next[k] = sum;
                        }
                    }
                }
                (local_gk, local_grk, local_gb)
            },
        )
        .reduce(
            move || (
                vec![0.0f32; in_dim * 3 * h_dim],
                vec![0.0f32; h_dim * 3 * h_dim],
                vec![0.0f32; 3 * h_dim],
            ),
            move |(mut gk1, mut grk1, mut gb1), (gk2, grk2, gb2)| {
                for i in 0..gk1.len() { gk1[i] += gk2[i]; }
                for i in 0..grk1.len() { grk1[i] += grk2[i]; }
                for i in 0..gb1.len() { gb1[i] += gb2[i]; }
                (gk1, grk1, gb1)
            },
        );

    let grad_kernel_slice = &mut *grad_kernel;
    let grad_recurrent_kernel_slice = &mut *grad_recurrent_kernel;
    let grad_bias_slice = &mut *grad_bias;

    for i in 0..grad_kernel_slice.len() { grad_kernel_slice[i] += final_grad_kernel[i]; }
    for i in 0..grad_recurrent_kernel_slice.len() { grad_recurrent_kernel_slice[i] += final_grad_recurrent_kernel[i]; }
    for i in 0..grad_bias_slice.len() { grad_bias_slice[i] += final_grad_bias[i]; }
}
