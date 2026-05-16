use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;

/// A simple wrapper to pass raw pointers across threads safely.
/// Safety: The caller must ensure that parallel access to the underlying memory 
/// follows Rust's borrowing rules (no data races).
pub struct SafeRawPtr(pub usize);
unsafe impl Send for SafeRawPtr {}
unsafe impl Sync for SafeRawPtr {}

pub struct SafeRawPtrMut(pub usize);
unsafe impl Send for SafeRawPtrMut {}
unsafe impl Sync for SafeRawPtrMut {}

const ELEMENTWISE_PARALLEL_THRESHOLD: usize = 16 * 1024;
const SUM_AXIS_COL_TILE: usize = 64;

#[napi]
pub fn dot_product(
    a_data: Float32Array,
    a_shape: Vec<u32>,
    b_data: Float32Array,
    b_shape: Vec<u32>,
    trans_a: bool,
    trans_b: bool,
) -> Float32Array {
    let a_rows = if trans_a { a_shape[1] } else { a_shape[0] } as usize;
    let b_cols = if trans_b { b_shape[0] } else { b_shape[1] } as usize;
    let result = vec![0.0; a_rows * b_cols];

    let out_array = Float32Array::from(result);
    dot_product_into(
        a_data,
        a_shape,
        b_data,
        b_shape,
        out_array.clone(),
        trans_a,
        trans_b,
    );
    out_array
}

#[napi]
pub fn dot_product_into(
    a_data: Float32Array,
    a_shape: Vec<u32>,
    b_data: Float32Array,
    b_shape: Vec<u32>,
    out_data: Float32Array,
    trans_a: bool,
    trans_b: bool,
) {
    let a_rows_orig = a_shape[0] as usize;
    let a_cols_orig = a_shape[1] as usize;
    let b_rows_orig = b_shape[0] as usize;
    let b_cols_orig = b_shape[1] as usize;
    dot_product_into_impl(
        a_data,
        a_rows_orig,
        a_cols_orig,
        b_data,
        b_rows_orig,
        b_cols_orig,
        out_data,
        trans_a,
        trans_b,
    );
}

#[napi]
pub fn dot_product_into_dims(
    a_data: Float32Array,
    a_rows: u32,
    a_cols: u32,
    b_data: Float32Array,
    b_rows: u32,
    b_cols: u32,
    out_data: Float32Array,
    trans_a: bool,
    trans_b: bool,
) {
    dot_product_into_impl(
        a_data,
        a_rows as usize,
        a_cols as usize,
        b_data,
        b_rows as usize,
        b_cols as usize,
        out_data,
        trans_a,
        trans_b,
    );
}

pub fn dot_product_into_impl(
    a_data: Float32Array,
    a_rows_orig: usize,
    a_cols_orig: usize,
    b_data: Float32Array,
    b_rows_orig: usize,
    b_cols_orig: usize,
    mut out_data: Float32Array,
    trans_a: bool,
    trans_b: bool,
) {
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
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a_data.as_ptr(),
            rsa,
            csa,
            b_data.as_ptr(),
            rsb,
            csb,
            0.0,
            out_data.as_mut_ptr(),
            rsc,
            csc,
        );
    }
}

#[inline(always)]
fn elementwise_op_chunk(a: &[f32], b: &[f32], out: &mut [f32], op: fn(f32, f32) -> f32) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = op(a[i], b[i]);
    }
}

fn elementwise_op_parallel(
    a_slice: &[f32],
    b_slice: &[f32],
    out_slice: &mut [f32],
    op: fn(f32, f32) -> f32,
) {
    const CHUNK: usize = 1024;
    if out_slice.len() < ELEMENTWISE_PARALLEL_THRESHOLD {
        elementwise_op_chunk(a_slice, b_slice, out_slice, op);
    } else {
        out_slice
            .par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let start = chunk_idx * CHUNK;
                let end = start + out_chunk.len();
                elementwise_op_chunk(&a_slice[start..end], &b_slice[start..end], out_chunk, op);
            });
    }
}

#[napi]
pub fn add_matrices_into(a: Float32Array, b: Float32Array, mut out: Float32Array) {
    let a_slice = &*a;
    let b_slice = &*b;
    let out_slice = &mut *out;
    elementwise_op_parallel(a_slice, b_slice, out_slice, |x, y| x + y);
}

#[napi]
pub fn sub_matrices_into(a: Float32Array, b: Float32Array, mut out: Float32Array) {
    let a_slice = &*a;
    let b_slice = &*b;
    let out_slice = &mut *out;
    elementwise_op_parallel(a_slice, b_slice, out_slice, |x, y| x - y);
}

#[napi]
pub fn mul_matrices_into(a: Float32Array, b: Float32Array, mut out: Float32Array) {
    let a_slice = &*a;
    let b_slice = &*b;
    let out_slice = &mut *out;
    elementwise_op_parallel(a_slice, b_slice, out_slice, |x, y| {
        if x == 0.0 || y == 0.0 {
            0.0
        } else {
            x * y
        }
    });
}

#[napi]
pub fn div_matrices_into(a: Float32Array, b: Float32Array, mut out: Float32Array) {
    let a_slice = &*a;
    let b_slice = &*b;
    let out_slice = &mut *out;
    for (idx, &denom) in b_slice.iter().enumerate() {
        if denom == 0.0 {
            panic!("Pembagian dengan nol pada indeks [{}]", idx);
        }
    }
    elementwise_op_parallel(a_slice, b_slice, out_slice, |x, y| {
        if x == 0.0 {
            0.0
        } else {
            x / y
        }
    });
}

#[inline(always)]
fn inplace_op_chunk(a: &mut [f32], b: &[f32], op: fn(f32, f32) -> f32) {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] = op(a[i], b[i]);
    }
}

fn inplace_op_parallel(a_slice: &mut [f32], b_slice: &[f32], op: fn(f32, f32) -> f32) {
    const CHUNK: usize = 1024;
    if a_slice.len() < ELEMENTWISE_PARALLEL_THRESHOLD {
        inplace_op_chunk(a_slice, b_slice, op);
    } else {
        a_slice
            .par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, a_chunk)| {
                let start = chunk_idx * CHUNK;
                let end = start + a_chunk.len();
                inplace_op_chunk(a_chunk, &b_slice[start..end], op);
            });
    }
}

#[napi]
pub fn add_in_place(mut a: Float32Array, b: Float32Array) {
    assert_eq!(
        a.len(),
        b.len(),
        "add_in_place: length mismatch {} != {}",
        a.len(),
        b.len()
    );
    let b_slice = &*b;
    let a_slice = &mut *a;
    inplace_op_parallel(a_slice, b_slice, |x, y| x + y);
}

#[napi]
pub fn sub_in_place(mut a: Float32Array, b: Float32Array) {
    assert_eq!(
        a.len(),
        b.len(),
        "sub_in_place: length mismatch {} != {}",
        a.len(),
        b.len()
    );
    let b_slice = &*b;
    let a_slice = &mut *a;
    inplace_op_parallel(a_slice, b_slice, |x, y| x - y);
}

#[napi]
pub fn mul_in_place(mut a: Float32Array, b: Float32Array) {
    assert_eq!(
        a.len(),
        b.len(),
        "mul_in_place: length mismatch {} != {}",
        a.len(),
        b.len()
    );
    let b_slice = &*b;
    let a_slice = &mut *a;
    inplace_op_parallel(a_slice, b_slice, |x, y| {
        if x == 0.0 || y == 0.0 {
            0.0
        } else {
            x * y
        }
    });
}

#[napi]
pub fn add_bias_native(mut data: Float32Array, bias: Float32Array, rows: u32, cols: u32) {
    let r = rows as usize;
    let c = cols as usize;
    let bias_slice = &*bias;
    let data_slice = &mut *data;
    assert_eq!(
        bias_slice.len(),
        r,
        "add_bias_native: expected bias length {} for shape [rows,1], got {}",
        r,
        bias_slice.len()
    );
    if data_slice.len() < ELEMENTWISE_PARALLEL_THRESHOLD {
        for i in 0..r {
            let bias_value = bias_slice[i];
            let row = &mut data_slice[i * c..(i + 1) * c];
            for value in row.iter_mut() {
                *value += bias_value;
            }
        }
    } else {
        data_slice
            .par_chunks_mut(c)
            .enumerate()
            .for_each(|(i, row)| {
                let bias_value = bias_slice[i];
                for value in row.iter_mut() {
                    *value += bias_value;
                }
            });
    }
}

#[napi]
pub fn sum_axis_native(data: Float32Array, rows: u32, cols: u32, axis: u32, mut out: Float32Array) {
    let r = rows as usize;
    let c = cols as usize;
    let data_slice = &*data;
    let out_slice = &mut *out;
    if axis == 1 {
        // Sum across columns (result is [rows x 1])
        if data_slice.len() < ELEMENTWISE_PARALLEL_THRESHOLD {
            for i in 0..r {
                let mut sum = 0.0;
                for j in 0..c {
                    sum += data_slice[i * c + j];
                }
                out_slice[i] = sum;
            }
        } else {
            out_slice
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, out_val)| {
                    let mut sum = 0.0;
                    let row_offset = i * c;
                    for j in 0..c {
                        sum += data_slice[row_offset + j];
                    }
                    *out_val = sum;
                });
        }
    } else {
        // Sum across rows (result is [1 x cols])
        if data_slice.len() < ELEMENTWISE_PARALLEL_THRESHOLD {
            for j in 0..c {
                let mut sum = 0.0;
                for i in 0..r {
                    sum += data_slice[i * c + j];
                }
                out_slice[j] = sum;
            }
        } else {
            out_slice
                .par_chunks_mut(SUM_AXIS_COL_TILE)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let start_col = chunk_idx * SUM_AXIS_COL_TILE;
                    for (local_col, out_val) in out_chunk.iter_mut().enumerate() {
                        let col = start_col + local_col;
                        let mut sum = 0.0;
                        for i in 0..r {
                            sum += data_slice[i * c + col];
                        }
                        *out_val = sum;
                    }
                });
        }
    }
}
#[napi]
pub fn pow_native(a: Float32Array, n: f64, mut out: Float32Array) {
    let a_slice = &*a;
    let out_slice = &mut *out;
    let n_f32 = n as f32;
    if a_slice.len() < ELEMENTWISE_PARALLEL_THRESHOLD {
        for i in 0..a_slice.len() {
            let x = a_slice[i];
            out_slice[i] = if x == 0.0 { 0.0 } else { x.powf(n_f32) };
        }
    } else {
        out_slice
            .par_iter_mut()
            .zip(a_slice.par_iter())
            .for_each(|(o, &x)| *o = if x == 0.0 { 0.0 } else { x.powf(n_f32) });
    }
}

#[napi]
pub fn absm_native(a: Float32Array, mut out: Float32Array) {
    let a_slice = &*a;
    let out_slice = &mut *out;
    elementwise_op_parallel(a_slice, a_slice, out_slice, |x, _| x.abs());
}

#[napi]
pub fn expm_native(a: Float32Array, mut out: Float32Array) {
    let a_slice = &*a;
    let out_slice = &mut *out;
    elementwise_op_parallel(a_slice, a_slice, out_slice, |x, _| x.exp());
}

#[napi]
pub fn logm_native(a: Float32Array, mut out: Float32Array) {
    let a_slice = &*a;
    let out_slice = &mut *out;
    elementwise_op_parallel(a_slice, a_slice, out_slice, |x, _| {
        if x <= 0.0 {
            1e-15_f32.ln()
        } else {
            x.ln()
        }
    });
}

#[napi]
pub fn transpose_native(a: Float32Array, rows: u32, cols: u32, mut out: Float32Array) {
    let r = rows as usize;
    let c = cols as usize;
    let a_slice = &*a;
    let out_slice = &mut *out;
    if a_slice.len() < ELEMENTWISE_PARALLEL_THRESHOLD {
        for i in 0..r {
            let i_offset = i * c;
            for j in 0..c {
                out_slice[j * r + i] = a_slice[i_offset + j];
            }
        }
    } else {
        out_slice
            .par_chunks_mut(r)
            .enumerate()
            .for_each(|(j, out_col)| {
                for i in 0..r {
                    out_col[i] = a_slice[i * c + j];
                }
            });
    }
}

#[napi]
pub fn dot_sum_native(a: Float32Array) -> f64 {
    let a_slice = &*a;
    let mut sum: f64 = 0.0;
    for &val in a_slice {
        sum += val as f64;
    }
    sum
}

#[napi]
pub fn dot_sub_native(a: Float32Array) -> f64 {
    let a_slice = &*a;
    let mut val: f64 = 0.0;
    for &v in a_slice {
        val -= v as f64;
    }
    val
}

#[napi]
pub fn dot_mul_native(a: Float32Array) -> f64 {
    let a_slice = &*a;
    let mut val: f64 = 1.0;
    for &v in a_slice {
        val *= v as f64;
    }
    val
}

#[napi]
pub fn dot_div_native(a: Float32Array) -> f64 {
    let a_slice = &*a;
    let mut val: f64 = 1.0;
    for &v in a_slice {
        val /= v as f64;
    }
    val
}
