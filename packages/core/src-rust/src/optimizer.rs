use crate::math::{SafeRawPtr, SafeRawPtrMut};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;

const ADAM_PARALLEL_THRESHOLD: usize = 8 * 1024;
const ELEMENTWISE_PARALLEL_THRESHOLD: usize = 16 * 1024;

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
    epsilon: f64,
) {
    let alpha = alpha as f32;
    let beta1 = beta1 as f32;
    let beta2 = beta2 as f32;
    let epsilon = epsilon as f32;

    let one_minus_beta1 = 1.0 - beta1;
    let one_minus_beta2 = 1.0 - beta2;
    let bias_correction1 = 1.0 / (1.0 - beta1.powi(t as i32));
    let bias_correction2 = 1.0 / (1.0 - beta2.powi(t as i32));

    let len = buffer.len();
    let grad_p = SafeRawPtr(grad.as_ptr() as usize);
    let m_p = SafeRawPtrMut(m.as_ptr() as usize);
    let v_p = SafeRawPtrMut(v.as_ptr() as usize);
    let buf_p = SafeRawPtrMut(buffer.as_ptr() as usize);

    if len < ADAM_PARALLEL_THRESHOLD {
        for i in 0..len {
            let g = grad[i];
            let m_new = beta1 * m[i] + one_minus_beta1 * g;
            let v_new = beta2 * v[i] + one_minus_beta2 * g * g;
            m[i] = m_new;
            v[i] = v_new;

            let m_hat = m_new * bias_correction1;
            let v_hat = v_new * bias_correction2;
            buffer[i] = alpha * m_hat / (v_hat.sqrt() + epsilon);
        }
    } else {
        (0..len).into_par_iter().for_each(|i| unsafe {
            let g_ptr = std::slice::from_raw_parts(grad_p.0 as *const f32, len);
            let m_ptr = std::slice::from_raw_parts_mut(m_p.0 as *mut f32, len);
            let v_ptr = std::slice::from_raw_parts_mut(v_p.0 as *mut f32, len);
            let b_ptr = std::slice::from_raw_parts_mut(buf_p.0 as *mut f32, len);

            let g = g_ptr[i];
            let m_new = beta1 * m_ptr[i] + one_minus_beta1 * g;
            let v_new = beta2 * v_ptr[i] + one_minus_beta2 * g * g;
            m_ptr[i] = m_new;
            v_ptr[i] = v_new;

            let m_hat = m_new * bias_correction1;
            let v_hat = v_new * bias_correction2;
            b_ptr[i] = alpha * m_hat / (v_hat.sqrt() + epsilon);
        });
    }
}

#[napi]
pub fn sgd_update_native(grad: Float32Array, mut out: Float32Array, alpha: f64) {
    let alpha = alpha as f32;
    let len = out.len();
    let grad_p = SafeRawPtr(grad.as_ptr() as usize);
    let out_p = SafeRawPtrMut(out.as_ptr() as usize);

    if len < ELEMENTWISE_PARALLEL_THRESHOLD {
        for i in 0..len {
            out[i] = grad[i] * alpha;
        }
    } else {
        (0..len).into_par_iter().for_each(|i| unsafe {
            let g_ptr = std::slice::from_raw_parts(grad_p.0 as *const f32, len);
            let o_ptr = std::slice::from_raw_parts_mut(out_p.0 as *mut f32, len);
            o_ptr[i] = g_ptr[i] * alpha;
        });
    }
}

#[napi]
pub fn adagrad_update_native(
    grad: Float32Array,
    mut sum: Float32Array,
    mut out: Float32Array,
    alpha: f64,
    epsilon: f64,
) {
    let alpha = alpha as f32;
    let eps = epsilon as f32;
    let len = out.len();
    let grad_p = SafeRawPtr(grad.as_ptr() as usize);
    let sum_p = SafeRawPtrMut(sum.as_ptr() as usize);
    let out_p = SafeRawPtrMut(out.as_ptr() as usize);

    if len < ELEMENTWISE_PARALLEL_THRESHOLD {
        for i in 0..len {
            let g = grad[i];
            sum[i] += g * g;
            out[i] = alpha * g / (sum[i].sqrt() + eps);
        }
    } else {
        (0..len).into_par_iter().for_each(|i| unsafe {
            let g_ptr = std::slice::from_raw_parts(grad_p.0 as *const f32, len);
            let s_ptr = std::slice::from_raw_parts_mut(sum_p.0 as *mut f32, len);
            let o_ptr = std::slice::from_raw_parts_mut(out_p.0 as *mut f32, len);

            let g = g_ptr[i];
            s_ptr[i] += g * g;
            o_ptr[i] = alpha * g / (s_ptr[i].sqrt() + eps);
        });
    }
}

#[napi]
pub fn momentum_update_native(
    grad: Float32Array,
    mut v: Float32Array,
    mut out: Float32Array,
    alpha: f64,
    beta: f64,
) {
    let alpha = alpha as f32;
    let beta = beta as f32;
    let len = out.len();
    let grad_p = SafeRawPtr(grad.as_ptr() as usize);
    let v_p = SafeRawPtrMut(v.as_ptr() as usize);
    let out_p = SafeRawPtrMut(out.as_ptr() as usize);

    if len < ELEMENTWISE_PARALLEL_THRESHOLD {
        for i in 0..len {
            v[i] = beta * v[i] + alpha * grad[i];
            out[i] = v[i];
        }
    } else {
        (0..len).into_par_iter().for_each(|i| unsafe {
            let g_ptr = std::slice::from_raw_parts(grad_p.0 as *const f32, len);
            let v_ptr = std::slice::from_raw_parts_mut(v_p.0 as *mut f32, len);
            let o_ptr = std::slice::from_raw_parts_mut(out_p.0 as *mut f32, len);

            v_ptr[i] = beta * v_ptr[i] + alpha * g_ptr[i];
            o_ptr[i] = v_ptr[i];
        });
    }
}

#[napi]
pub fn nag_update_native(
    grad: Float32Array,
    mut v: Float32Array,
    mut out: Float32Array,
    alpha: f64,
    beta: f64,
) {
    let alpha = alpha as f32;
    let beta = beta as f32;
    let len = out.len();
    let grad_p = SafeRawPtr(grad.as_ptr() as usize);
    let v_p = SafeRawPtrMut(v.as_ptr() as usize);
    let out_p = SafeRawPtrMut(out.as_ptr() as usize);

    if len < ELEMENTWISE_PARALLEL_THRESHOLD {
        for i in 0..len {
            let v_old = v[i];
            let v_new = beta * v_old + alpha * (grad[i] - beta * v_old);
            v[i] = v_new;
            out[i] = v_new;
        }
    } else {
        (0..len).into_par_iter().for_each(|i| unsafe {
            let g_ptr = std::slice::from_raw_parts(grad_p.0 as *const f32, len);
            let v_ptr = std::slice::from_raw_parts_mut(v_p.0 as *mut f32, len);
            let o_ptr = std::slice::from_raw_parts_mut(out_p.0 as *mut f32, len);

            let v_old = v_ptr[i];
            let v_new = beta * v_old + alpha * (g_ptr[i] - beta * v_old);
            v_ptr[i] = v_new;
            o_ptr[i] = v_new;
        });
    }
}
