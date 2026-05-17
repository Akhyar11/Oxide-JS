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
pub fn lstm_forward_native(
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
    mut cell_states: Float32Array,
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
    let cell_states_ptr = SendPtr(cell_states.as_mut_ptr());
    let gate_values_ptr = SendPtr(gate_values.as_mut_ptr());

    (0..b_size).into_par_iter().for_each(move |b| {
        let h = vec![0.0f32; h_dim];
        let c = vec![0.0f32; h_dim];
        let mut a = vec![0.0f32; 4 * h_dim];

        for t in 0..seq_len {
            let step_offset = (b * seq_len + t) * in_dim;
            let hidden_offset = (b * seq_len + t) * h_dim;
            let gate_offset = (b * seq_len + t) * 4 * h_dim;

            unsafe {
                // 1. Calculate x_t * W + bias
                for j in 0..(4 * h_dim) {
                    let mut sum = *bias_ptr.get().add(j);
                    for c_idx in 0..in_dim {
                        sum += *inputs_ptr.get().add(step_offset + c_idx) * *kernel_ptr.get().add(c_idx * 4 * h_dim + j);
                    }
                    a[j] = sum;
                }

                // 2. Add h_{t-1} * U if t > 0
                if t > 0 {
                    let prev_hidden_offset = (b * seq_len + t - 1) * h_dim;
                    for j in 0..(4 * h_dim) {
                        let mut sum = 0.0;
                        for k in 0..h_dim {
                            sum += *hidden_states_ptr.get().add(prev_hidden_offset + k) * *recurrent_kernel_ptr.get().add(k * 4 * h_dim + j);
                        }
                        a[j] += sum;
                    }
                } else {
                    for j in 0..(4 * h_dim) {
                        let mut sum = 0.0;
                        for k in 0..h_dim {
                            sum += h[k] * *recurrent_kernel_ptr.get().add(k * 4 * h_dim + j);
                        }
                        a[j] += sum;
                    }
                }

                // 3. Compute gates
                // 0..h_dim -> input gate i
                // h_dim..2*h_dim -> forget gate f
                // 2*h_dim..3*h_dim -> candidate cell c_tilde
                // 3*h_dim..4*h_dim -> output gate o
                for j in 0..h_dim {
                    let gate_i = sigmoid(a[j]);
                    let gate_f = sigmoid(a[h_dim + j]);
                    let gate_c = a[2 * h_dim + j].tanh();
                    let gate_o = sigmoid(a[3 * h_dim + j]);

                    *gate_values_ptr.get().add(gate_offset + j) = gate_i;
                    *gate_values_ptr.get().add(gate_offset + h_dim + j) = gate_f;
                    *gate_values_ptr.get().add(gate_offset + 2 * h_dim + j) = gate_c;
                    *gate_values_ptr.get().add(gate_offset + 3 * h_dim + j) = gate_o;

                    // Update cell state
                    let prev_c = if t > 0 {
                        *cell_states_ptr.get().add((b * seq_len + t - 1) * h_dim + j)
                    } else {
                        c[j]
                    };
                    let new_c = gate_f * prev_c + gate_i * gate_c;
                    *cell_states_ptr.get().add(hidden_offset + j) = new_c;

                    // Update hidden state
                    let new_h = gate_o * new_c.tanh();
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
pub fn lstm_backward_native(
    grad_out: Float32Array,
    inputs: Float32Array,
    hidden_states: Float32Array,
    cell_states: Float32Array,
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
    let cell_states_ptr = SendPtrConst(cell_states.as_ptr());
    let gate_values_ptr = SendPtrConst(gate_values.as_ptr());
    let kernel_ptr = SendPtrConst(kernel.as_ptr());
    let recurrent_kernel_ptr = SendPtrConst(recurrent_kernel.as_ptr());
    let grad_in_ptr = SendPtr(grad_in.as_mut_ptr());

    let (final_grad_kernel, final_grad_recurrent_kernel, final_grad_bias) = (0..b_size)
        .into_par_iter()
        .fold(
            move || (
                vec![0.0f32; in_dim * 4 * h_dim],
                vec![0.0f32; h_dim * 4 * h_dim],
                vec![0.0f32; 4 * h_dim],
            ),
            move |(mut local_gk, mut local_grk, mut local_gb), b| {
                let mut dh_next = vec![0.0f32; h_dim];
                let mut dc_next = vec![0.0f32; h_dim];
                let mut da = vec![0.0f32; 4 * h_dim];

                for t in (0..seq_len).rev() {
                    let step_offset = (b * seq_len + t) * in_dim;
                    let hidden_offset = (b * seq_len + t) * h_dim;
                    let gate_offset = (b * seq_len + t) * 4 * h_dim;

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
                        for j in 0..h_dim {
                            let c_val = *cell_states_ptr.get().add(hidden_offset + j);
                            let c_tanh = c_val.tanh();

                            let gate_i = *gate_values_ptr.get().add(gate_offset + j);
                            let gate_f = *gate_values_ptr.get().add(gate_offset + h_dim + j);
                            let gate_c = *gate_values_ptr.get().add(gate_offset + 2 * h_dim + j);
                            let gate_o = *gate_values_ptr.get().add(gate_offset + 3 * h_dim + j);

                            let d_o = dh[j] * c_tanh;
                            let da_o = d_o * gate_o * (1.0 - gate_o);

                            let dc = dh[j] * gate_o * (1.0 - c_tanh * c_tanh) + dc_next[j];

                            let d_c_tilde = dc * gate_i;
                            let da_c = d_c_tilde * (1.0 - gate_c * gate_c);

                            let prev_c = if t > 0 {
                                *cell_states_ptr.get().add((b * seq_len + t - 1) * h_dim + j)
                            } else {
                                0.0
                            };
                            let d_f = dc * prev_c;
                            let da_f = d_f * gate_f * (1.0 - gate_f);

                            let d_i = dc * gate_c;
                            let da_i = d_i * gate_i * (1.0 - gate_i);

                            da[j] = da_i;
                            da[h_dim + j] = da_f;
                            da[2 * h_dim + j] = da_c;
                            da[3 * h_dim + j] = da_o;

                            dc_next[j] = dc * gate_f;
                        }

                        // 3. Accumulate local kernel, recurrent kernel and bias gradients
                        for j in 0..(4 * h_dim) {
                            local_gb[j] += da[j];

                            for c_idx in 0..in_dim {
                                local_gk[c_idx * 4 * h_dim + j] += *inputs_ptr.get().add(step_offset + c_idx) * da[j];
                            }

                            if t > 0 {
                                let prev_hidden_offset = (b * seq_len + t - 1) * h_dim;
                                for k in 0..h_dim {
                                    local_grk[k * 4 * h_dim + j] += *hidden_states_ptr.get().add(prev_hidden_offset + k) * da[j];
                                }
                            }
                        }

                        // 4. Input gradient (dx_t)
                        for c_idx in 0..in_dim {
                            let mut sum = 0.0;
                            for g in 0..(4 * h_dim) {
                                sum += da[g] * *kernel_ptr.get().add(c_idx * 4 * h_dim + g);
                            }
                            *grad_in_ptr.get().add(step_offset + c_idx) = sum;
                        }

                        // 5. Update dh_next for the next backward step (t-1)
                        for k in 0..h_dim {
                            let mut sum = 0.0;
                            for g in 0..(4 * h_dim) {
                                sum += da[g] * *recurrent_kernel_ptr.get().add(k * 4 * h_dim + g);
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
                vec![0.0f32; in_dim * 4 * h_dim],
                vec![0.0f32; h_dim * 4 * h_dim],
                vec![0.0f32; 4 * h_dim],
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
