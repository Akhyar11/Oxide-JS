use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;

const ACTIVATION_PARALLEL_THRESHOLD: usize = 16 * 1024;

/// Process a chunk of elements for relu with a pattern that encourages
/// auto-vectorization (no branches inside the inner loop body).
#[inline(always)]
fn relu_chunk(input: &[f32], out_res: &mut [f32], out_grad: &mut [f32]) {
    debug_assert_eq!(input.len(), out_res.len());
    debug_assert_eq!(input.len(), out_grad.len());
    for i in 0..input.len() {
        let val = input[i];
        // branchless: select via comparison mask
        let mask = if val > 0.0 { 1.0f32 } else { 0.0f32 };
        out_res[i] = val * mask;
        out_grad[i] = mask;
    }
}

#[napi]
pub fn relu_native_into(
    input: Float32Array,
    mut out_res: Float32Array,
    mut out_grad: Float32Array,
) {
    let input_slice = &*input;
    let out_res_slice = &mut *out_res;
    let out_grad_slice = &mut *out_grad;
    if input_slice.len() < ACTIVATION_PARALLEL_THRESHOLD {
        relu_chunk(input_slice, out_res_slice, out_grad_slice);
    } else {
        // Use chunks_exact for better auto-vectorization alignment
        const CHUNK: usize = 64;
        let n = input_slice.len();
        let full = n - (n % CHUNK);
        out_res_slice[..full]
            .par_chunks_mut(CHUNK)
            .zip(out_grad_slice[..full].par_chunks_mut(CHUNK))
            .enumerate()
            .for_each(|(chunk_idx, (res_chunk, grad_chunk))| {
                let start = chunk_idx * CHUNK;
                let inp = &input_slice[start..start + CHUNK];
                relu_chunk(inp, res_chunk, grad_chunk);
            });
        // remainder
        if full < n {
            relu_chunk(
                &input_slice[full..],
                &mut out_res_slice[full..],
                &mut out_grad_slice[full..],
            );
        }
    }
}

#[inline(always)]
fn sigmoid_chunk(input: &[f32], out_res: &mut [f32], out_grad: &mut [f32]) {
    for i in 0..input.len() {
        let val = 1.0 / (1.0 + (-input[i]).exp());
        out_res[i] = val;
        out_grad[i] = val * (1.0 - val);
    }
}

#[napi]
pub fn sigmoid_native_into(
    input: Float32Array,
    mut out_res: Float32Array,
    mut out_grad: Float32Array,
) {
    let input_slice = &*input;
    let out_res_slice = &mut *out_res;
    let out_grad_slice = &mut *out_grad;
    if input_slice.len() < ACTIVATION_PARALLEL_THRESHOLD {
        sigmoid_chunk(input_slice, out_res_slice, out_grad_slice);
    } else {
        const CHUNK: usize = 64;
        let n = input_slice.len();
        let full = n - (n % CHUNK);
        out_res_slice[..full]
            .par_chunks_mut(CHUNK)
            .zip(out_grad_slice[..full].par_chunks_mut(CHUNK))
            .enumerate()
            .for_each(|(chunk_idx, (res_chunk, grad_chunk))| {
                let start = chunk_idx * CHUNK;
                let inp = &input_slice[start..start + CHUNK];
                sigmoid_chunk(inp, res_chunk, grad_chunk);
            });
        if full < n {
            sigmoid_chunk(
                &input_slice[full..],
                &mut out_res_slice[full..],
                &mut out_grad_slice[full..],
            );
        }
    }
}

#[inline(always)]
fn tanh_chunk(input: &[f32], out_res: &mut [f32], out_grad: &mut [f32]) {
    for i in 0..input.len() {
        let x = input[i];
        if x == 0.0 {
            out_res[i] = 0.0;
            out_grad[i] = 1.0;
        } else {
            let val = x.tanh();
            out_res[i] = val;
            out_grad[i] = 1.0 - val * val;
        }
    }
}

#[napi]
pub fn tanh_native_into(
    input: Float32Array,
    mut out_res: Float32Array,
    mut out_grad: Float32Array,
) {
    let input_slice = &*input;
    let out_res_slice = &mut *out_res;
    let out_grad_slice = &mut *out_grad;
    if input_slice.len() < ACTIVATION_PARALLEL_THRESHOLD {
        tanh_chunk(input_slice, out_res_slice, out_grad_slice);
    } else {
        const CHUNK: usize = 64;
        let n = input_slice.len();
        let full = n - (n % CHUNK);
        out_res_slice[..full]
            .par_chunks_mut(CHUNK)
            .zip(out_grad_slice[..full].par_chunks_mut(CHUNK))
            .enumerate()
            .for_each(|(chunk_idx, (res_chunk, grad_chunk))| {
                let start = chunk_idx * CHUNK;
                let inp = &input_slice[start..start + CHUNK];
                tanh_chunk(inp, res_chunk, grad_chunk);
            });
        if full < n {
            tanh_chunk(
                &input_slice[full..],
                &mut out_res_slice[full..],
                &mut out_grad_slice[full..],
            );
        }
    }
}

#[inline(always)]
fn lrelu_chunk(input: &[f32], out_res: &mut [f32], out_grad: &mut [f32]) {
    for i in 0..input.len() {
        let val = input[i];
        if val == 0.0 {
            out_res[i] = 0.0;
            out_grad[i] = 1.0;
        } else if val < 0.0 {
            out_res[i] = val * 1e-5;
            out_grad[i] = 1e-5;
        } else {
            out_res[i] = val;
            out_grad[i] = 1.0;
        }
    }
}

#[napi]
pub fn l_relu_native_into(
    input: Float32Array,
    mut out_res: Float32Array,
    mut out_grad: Float32Array,
) {
    let input_slice = &*input;
    let out_res_slice = &mut *out_res;
    let out_grad_slice = &mut *out_grad;
    if input_slice.len() < ACTIVATION_PARALLEL_THRESHOLD {
        lrelu_chunk(input_slice, out_res_slice, out_grad_slice);
    } else {
        const CHUNK: usize = 64;
        let n = input_slice.len();
        let full = n - (n % CHUNK);
        out_res_slice[..full]
            .par_chunks_mut(CHUNK)
            .zip(out_grad_slice[..full].par_chunks_mut(CHUNK))
            .enumerate()
            .for_each(|(chunk_idx, (res_chunk, grad_chunk))| {
                let start = chunk_idx * CHUNK;
                let inp = &input_slice[start..start + CHUNK];
                lrelu_chunk(inp, res_chunk, grad_chunk);
            });
        if full < n {
            lrelu_chunk(
                &input_slice[full..],
                &mut out_res_slice[full..],
                &mut out_grad_slice[full..],
            );
        }
    }
}


#[inline(always)]
pub fn elu_chunk(input: &[f32], out_res: &mut [f32], out_grad: &mut [f32], alpha: f32) {
    for i in 0..input.len() {
        let x = input[i];
        if x == 0.0 {
            out_res[i] = 0.0;
            out_grad[i] = 1.0 + alpha;
        } else if x > 0.0 {
            out_res[i] = x;
            out_grad[i] = 1.0;
        } else {
            let val = alpha * (x.exp() - 1.0);
            out_res[i] = val;
            out_grad[i] = val + alpha;
        }
    }
}

#[inline(always)]
pub fn selu_chunk(input: &[f32], out_res: &mut [f32], out_grad: &mut [f32]) {
    const ALPHA: f32 = 1.6732632423543772;
    const SCALE: f32 = 1.0507009873554805;
    for i in 0..input.len() {
        let x = input[i];
        if x == 0.0 {
            out_res[i] = 0.0;
            out_grad[i] = SCALE * ALPHA;
        } else if x > 0.0 {
            out_res[i] = SCALE * x;
            out_grad[i] = SCALE;
        } else {
            let val = SCALE * ALPHA * (x.exp() - 1.0);
            out_res[i] = val;
            out_grad[i] = SCALE * ALPHA * x.exp();
        }
    }
}

#[inline(always)]
pub fn softplus_chunk(input: &[f32], out_res: &mut [f32], out_grad: &mut [f32]) {
    for i in 0..input.len() {
        let x = input[i];
        out_res[i] = (1.0 + x.exp()).ln();
        out_grad[i] = 1.0 / (1.0 + (-x).exp()); // sigmoid
    }
}

#[inline(always)]
pub fn softsign_chunk(input: &[f32], out_res: &mut [f32], out_grad: &mut [f32]) {
    for i in 0..input.len() {
        let x = input[i];
        if x == 0.0 {
            out_res[i] = 0.0;
            out_grad[i] = 1.0;
        } else {
            let denom = 1.0 + x.abs();
            out_res[i] = x / denom;
            out_grad[i] = 1.0 / (denom * denom);
        }
    }
}

#[inline(always)]
pub fn swish_chunk(input: &[f32], out_res: &mut [f32], out_grad: &mut [f32]) {
    for i in 0..input.len() {
        let x = input[i];
        if x == 0.0 {
            out_res[i] = 0.0;
            out_grad[i] = 0.5;
        } else {
            let sig = 1.0 / (1.0 + (-x).exp());
            let swish = x * sig;
            out_res[i] = swish;
            out_grad[i] = swish + sig * (1.0 - swish);
        }
    }
}

#[inline(always)]
pub fn gelu_chunk(input: &[f32], out_res: &mut [f32], out_grad: &mut [f32]) {
    // GELU Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    const COEF: f32 = 0.044715;
    for i in 0..input.len() {
        let x = input[i];
        if x == 0.0 {
            out_res[i] = 0.0;
            out_grad[i] = 0.5;
        } else {
            let x3 = x * x * x;
            let inner = SQRT_2_OVER_PI * (x + COEF * x3);
            let tanh_inner = inner.tanh();
            out_res[i] = 0.5 * x * (1.0 + tanh_inner);
            
            let sech2 = 1.0 - tanh_inner * tanh_inner;
            let d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * COEF * x * x);
            out_grad[i] = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner;
        }
    }
}

#[inline(always)]
pub fn mish_chunk(input: &[f32], out_res: &mut [f32], out_grad: &mut [f32]) {
    for i in 0..input.len() {
        let x = input[i];
        if x == 0.0 {
            out_res[i] = 0.0;
            out_grad[i] = 0.60000002;
        } else {
            let sp = (1.0 + x.exp()).ln();
            let tanh_sp = sp.tanh();
            out_res[i] = x * tanh_sp;
            
            let sech2 = 1.0 - tanh_sp * tanh_sp;
            let sp_grad = 1.0 / (1.0 + (-x).exp()); // sigmoid
            out_grad[i] = tanh_sp + x * sech2 * sp_grad;
        }
    }
}

#[inline(always)]
pub fn hard_sigmoid_chunk(input: &[f32], out_res: &mut [f32], out_grad: &mut [f32]) {
    for i in 0..input.len() {
        let x = input[i];
        if x <= -3.0 {
            out_res[i] = 0.0;
            out_grad[i] = 0.0;
        } else if x >= 3.0 {
            out_res[i] = 1.0;
            out_grad[i] = 0.0;
        } else {
            out_res[i] = (x + 3.0) / 6.0;
            out_grad[i] = 1.0 / 6.0;
        }
    }
}

#[inline(always)]
pub fn hard_swish_chunk(input: &[f32], out_res: &mut [f32], out_grad: &mut [f32]) {
    for i in 0..input.len() {
        let x = input[i];
        if x == 0.0 {
            out_res[i] = 0.0;
            out_grad[i] = 0.5;
        } else if x <= -3.0 {
            out_res[i] = 0.0;
            out_grad[i] = 0.0;
        } else if x >= 3.0 {
            out_res[i] = x;
            out_grad[i] = 1.0;
        } else {
            out_res[i] = x * (x + 3.0) / 6.0;
            out_grad[i] = (2.0 * x + 3.0) / 6.0;
        }
    }
}


macro_rules! define_activation {
    ($name:ident, $chunk_fn:ident) => {
        #[napi]
        pub fn $name(
            input: Float32Array,
            mut out_res: Float32Array,
            mut out_grad: Float32Array,
        ) {
            let input_slice = &*input;
            let out_res_slice = &mut *out_res;
            let out_grad_slice = &mut *out_grad;
            if input_slice.len() < ACTIVATION_PARALLEL_THRESHOLD {
                $chunk_fn(input_slice, out_res_slice, out_grad_slice);
            } else {
                const CHUNK: usize = 64;
                let n = input_slice.len();
                let full = n - (n % CHUNK);
                out_res_slice[..full]
                    .par_chunks_mut(CHUNK)
                    .zip(out_grad_slice[..full].par_chunks_mut(CHUNK))
                    .enumerate()
                    .for_each(|(chunk_idx, (res_chunk, grad_chunk))| {
                        let start = chunk_idx * CHUNK;
                        let inp = &input_slice[start..start + CHUNK];
                        $chunk_fn(inp, res_chunk, grad_chunk);
                    });
                if full < n {
                    $chunk_fn(
                        &input_slice[full..],
                        &mut out_res_slice[full..],
                        &mut out_grad_slice[full..],
                    );
                }
            }
        }
    };
}

define_activation!(selu_native_into, selu_chunk);
define_activation!(softplus_native_into, softplus_chunk);
define_activation!(softsign_native_into, softsign_chunk);
define_activation!(swish_native_into, swish_chunk);
define_activation!(gelu_native_into, gelu_chunk);
define_activation!(mish_native_into, mish_chunk);
define_activation!(hard_sigmoid_native_into, hard_sigmoid_chunk);
define_activation!(hard_swish_native_into, hard_swish_chunk);

#[napi]
pub fn elu_native_into(
    input: Float32Array,
    alpha: f64,
    mut out_res: Float32Array,
    mut out_grad: Float32Array,
) {
    let input_slice = &*input;
    let out_res_slice = &mut *out_res;
    let out_grad_slice = &mut *out_grad;
    let alpha_f32 = alpha as f32;
    if input_slice.len() < ACTIVATION_PARALLEL_THRESHOLD {
        elu_chunk(input_slice, out_res_slice, out_grad_slice, alpha_f32);
    } else {
        const CHUNK: usize = 64;
        let n = input_slice.len();
        let full = n - (n % CHUNK);
        out_res_slice[..full]
            .par_chunks_mut(CHUNK)
            .zip(out_grad_slice[..full].par_chunks_mut(CHUNK))
            .enumerate()
            .for_each(|(chunk_idx, (res_chunk, grad_chunk))| {
                let start = chunk_idx * CHUNK;
                let inp = &input_slice[start..start + CHUNK];
                elu_chunk(inp, res_chunk, grad_chunk, alpha_f32);
            });
        if full < n {
            elu_chunk(
                &input_slice[full..],
                &mut out_res_slice[full..],
                &mut out_grad_slice[full..],
                alpha_f32,
            );
        }
    }
}

#[inline(always)]
pub fn threshold_chunk(input: &[f32], out_res: &mut [f32], out_grad: &mut [f32], threshold: f32, leak: f32) {
    for i in 0..input.len() {
        let x = input[i];
        out_res[i] = if x > threshold { 1.0 } else { 0.0 };
        out_grad[i] = if x > threshold { leak } else { 1.0 };
    }
}

#[napi]
pub fn threshold_native_into(
    input: Float32Array,
    threshold: f64,
    leak: f64,
    mut out_res: Float32Array,
    mut out_grad: Float32Array,
) {
    let input_slice = &*input;
    let out_res_slice = &mut *out_res;
    let out_grad_slice = &mut *out_grad;
    let threshold_f32 = threshold as f32;
    let leak_f32 = leak as f32;
    if input_slice.len() < ACTIVATION_PARALLEL_THRESHOLD {
        threshold_chunk(input_slice, out_res_slice, out_grad_slice, threshold_f32, leak_f32);
    } else {
        const CHUNK: usize = 64;
        let n = input_slice.len();
        let full = n - (n % CHUNK);
        out_res_slice[..full]
            .par_chunks_mut(CHUNK)
            .zip(out_grad_slice[..full].par_chunks_mut(CHUNK))
            .enumerate()
            .for_each(|(chunk_idx, (res_chunk, grad_chunk))| {
                let start = chunk_idx * CHUNK;
                let inp = &input_slice[start..start + CHUNK];
                threshold_chunk(inp, res_chunk, grad_chunk, threshold_f32, leak_f32);
            });
        if full < n {
            threshold_chunk(
                &input_slice[full..],
                &mut out_res_slice[full..],
                &mut out_grad_slice[full..],
                threshold_f32,
                leak_f32,
            );
        }
    }
}


