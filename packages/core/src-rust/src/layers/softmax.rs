use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;
use crate::math::{SafeRawPtr, SafeRawPtrMut};

const SOFTMAX_PARALLEL_THRESHOLD: usize = 16 * 1024;
const SOFTMAX_TRANSPOSE_TILE: usize = 64;

fn softmax_row_in_place(row: &mut [f32]) {
    let mut max_val = f32::NEG_INFINITY;
    for value in row.iter().copied() {
        if value > max_val {
            max_val = value;
        }
    }
    if !max_val.is_finite() {
        let uniform = 1.0 / row.len() as f32;
        row.fill(uniform);
        return;
    }

    let mut sum_exp = 0.0f32;
    for value in row.iter_mut() {
        let exp_val = (*value - max_val).exp();
        *value = exp_val;
        sum_exp += exp_val;
    }
    if !sum_exp.is_finite() || sum_exp <= 0.0 {
        let uniform = 1.0 / row.len() as f32;
        row.fill(uniform);
        return;
    }

    for value in row.iter_mut() {
        *value /= sum_exp;
    }
}

fn transpose_copy(input: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    for row_block in (0..rows).step_by(SOFTMAX_TRANSPOSE_TILE) {
        let row_end = (row_block + SOFTMAX_TRANSPOSE_TILE).min(rows);
        for col_block in (0..cols).step_by(SOFTMAX_TRANSPOSE_TILE) {
            let col_end = (col_block + SOFTMAX_TRANSPOSE_TILE).min(cols);
            for row in row_block..row_end {
                let input_offset = row * cols;
                for col in col_block..col_end {
                    out[col * rows + row] = input[input_offset + col];
                }
            }
        }
    }
}

fn softmax_backward_row_in_place(s_row: &[f32], g_row: &[f32], out_row: &mut [f32]) {
    let mut sum_grad_s = 0.0;
    for j in 0..s_row.len() {
        sum_grad_s += s_row[j] * g_row[j];
    }
    for j in 0..s_row.len() {
        out_row[j] = s_row[j] * (g_row[j] - sum_grad_s);
    }
}

#[napi]
pub fn softmax_native_into(
    data: Float32Array,
    rows: u32,
    cols: u32,
    is_row: bool,
    mut out: Float32Array,
) {
    let r = rows as usize;
    let c = cols as usize;
    let data_len = data.len();
    let out_len = out.len();
    let data_p = SafeRawPtr(data.as_ptr() as usize);
    let out_p = SafeRawPtrMut(out.as_ptr() as usize);

    if is_row {
        (0..r).into_par_iter().for_each(|i| {
            unsafe {
                let d_ptr = std::slice::from_raw_parts(data_p.0 as *const f32, data_len);
                let out_ptr = std::slice::from_raw_parts_mut(out_p.0 as *mut f32, out_len);
                let row = &mut out_ptr[i * c..(i + 1) * c];
                row.copy_from_slice(&d_ptr[i * c..(i + 1) * c]);
                softmax_row_in_place(row);
            }
        });
    } else {
        let mut transposed = vec![0.0f32; out.len()];
        transpose_copy(&data, r, c, &mut transposed);
        if transposed.len() < SOFTMAX_PARALLEL_THRESHOLD {
            for row in transposed.chunks_mut(r) {
                softmax_row_in_place(row);
            }
        } else {
            transposed.par_chunks_mut(r).for_each(softmax_row_in_place);
        }
        transpose_copy(&transposed, c, r, &mut out);
    }
}

#[napi]
pub fn softmax_backward_native_into(
    s_data: Float32Array,
    g_data: Float32Array,
    rows: u32,
    cols: u32,
    is_row: bool,
    mut out: Float32Array,
) {
    let r = rows as usize;
    let c = cols as usize;
    let s_slice = &*s_data;
    let g_slice = &*g_data;
    let out_slice = &mut *out;
    if is_row {
        if out_slice.len() < SOFTMAX_PARALLEL_THRESHOLD {
            for i in 0..r {
                let offset = i * c;
                softmax_backward_row_in_place(
                    &s_slice[offset..offset + c],
                    &g_slice[offset..offset + c],
                    &mut out_slice[offset..offset + c],
                );
            }
        } else {
            out_slice
                .par_chunks_mut(c)
                .enumerate()
                .for_each(|(i, out_row)| {
                    let offset = i * c;
                    softmax_backward_row_in_place(
                        &s_slice[offset..offset + c],
                        &g_slice[offset..offset + c],
                        out_row,
                    );
                });
        }
    } else {
        let mut s_transposed = vec![0.0f32; s_slice.len()];
        let mut g_transposed = vec![0.0f32; g_slice.len()];
        let mut out_transposed = vec![0.0f32; out_slice.len()];
        transpose_copy(s_slice, r, c, &mut s_transposed);
        transpose_copy(g_slice, r, c, &mut g_transposed);
        if out_transposed.len() < SOFTMAX_PARALLEL_THRESHOLD {
            for col in 0..c {
                let offset = col * r;
                softmax_backward_row_in_place(
                    &s_transposed[offset..offset + r],
                    &g_transposed[offset..offset + r],
                    &mut out_transposed[offset..offset + r],
                );
            }
        } else {
            out_transposed
                .par_chunks_mut(r)
                .enumerate()
                .for_each(|(col, out_row)| {
                    let offset = col * r;
                    softmax_backward_row_in_place(
                        &s_transposed[offset..offset + r],
                        &g_transposed[offset..offset + r],
                        out_row,
                    );
                });
        }
        transpose_copy(&out_transposed, c, r, out_slice);
    }
}
