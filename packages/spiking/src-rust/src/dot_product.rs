use napi_derive::napi;
use napi::bindgen_prelude::Float32Array;
use rayon::prelude::*;

#[napi]
pub fn dot_product_add_only_native(
    a_data: Float32Array,
    a_rows_orig: u32,
    a_cols_orig: u32,
    b_data: Float32Array,
    b_rows_orig: u32,
    b_cols_orig: u32,
    trans_a: bool,
    trans_b: bool,
    mut out_data: Float32Array
) {
    let a_rows = if trans_a { a_cols_orig } else { a_rows_orig } as usize;
    let a_cols = if trans_a { a_rows_orig } else { a_cols_orig } as usize;
    let b_cols = if trans_b { b_rows_orig } else { b_cols_orig } as usize;

    let a_slice: &[f32] = &a_data;
    let b_slice: &[f32] = &b_data;
    let out_slice: &mut [f32] = &mut out_data;

    out_slice.par_chunks_mut(b_cols).enumerate().for_each(|(i, out_row)| {
        let a_offset = i * a_cols;
        for k in 0..a_cols {
            let a_val = if trans_a {
                a_slice[k * a_rows + i]
            } else {
                a_slice[a_offset + k]
            };
            
            if a_val > 0.5 {
                let b_offset = k * b_cols;
                for j in 0..b_cols {
                    let b_val = if trans_b {
                        b_slice[j * a_cols + k]
                    } else {
                        b_slice[b_offset + j]
                    };
                    out_row[j] += b_val;
                }
            }
        }
    });
}
