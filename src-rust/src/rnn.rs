use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;

struct SyncPtr(usize);
unsafe impl Send for SyncPtr {}
unsafe impl Sync for SyncPtr {}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers shared by AdaptiveMemoryRNN
// ─────────────────────────────────────────────────────────────────────────────

#[inline(always)]
fn stable_softmax(scores: &[f32], attention: &mut [f32]) {
    let n = scores.len();
    let mut max_v = f32::NEG_INFINITY;
    for &s in scores.iter() { if s > max_v { max_v = s; } }
    let mut denom = 0.0f32;
    for i in 0..n { let v = (scores[i] - max_v).exp(); attention[i] = v; denom += v; }
    if denom == 0.0 || !denom.is_finite() {
        let u = 1.0 / n as f32;
        for i in 0..n { attention[i] = u; }
    } else {
        for i in 0..n { attention[i] /= denom; }
    }
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let z = x.exp(); z / (1.0 + z) }
}

#[napi]
pub fn lstm_forward_native_into(
    wxi: Float32Array, wxf: Float32Array, wxo: Float32Array, wxg: Float32Array,
    whi: Float32Array, whf: Float32Array, who: Float32Array, whg: Float32Array,
    bi: Float32Array, bf: Float32Array, bo: Float32Array, bg: Float32Array,
    x_seq: Float32Array, h0: Float32Array, c0: Float32Array,
    hidden_units: u32, input_units: u32, seq_len: u32, batch_size: u32,
    mut h_seq: Float32Array, mut c_seq: Float32Array,
    mut gi_seq: Float32Array, mut gf_seq: Float32Array, mut go_seq: Float32Array, mut gg_seq: Float32Array,
) {
    let hu = hidden_units as usize;
    let iu = input_units as usize;
    let sl = seq_len as usize;
    let bs = batch_size as usize;
    let total_cols = sl * bs;

    let mut proj = vec![0.0f32; 4 * hu * total_cols];
    let proj_ptr = SyncPtr(proj.as_mut_ptr() as usize);
    let x_ptr = SyncPtr(x_seq.as_ptr() as usize);
    let wx_ptrs = [SyncPtr(wxi.as_ptr() as usize), SyncPtr(wxf.as_ptr() as usize), SyncPtr(wxo.as_ptr() as usize), SyncPtr(wxg.as_ptr() as usize)];
    let b_ptrs = [SyncPtr(bi.as_ptr() as usize), SyncPtr(bf.as_ptr() as usize), SyncPtr(bo.as_ptr() as usize), SyncPtr(bg.as_ptr() as usize)];

    (0..4).into_par_iter().for_each(|g| {
        unsafe {
            let wx = std::slice::from_raw_parts(wx_ptrs[g].0 as *const f32, hu * iu);
            let b = std::slice::from_raw_parts(b_ptrs[g].0 as *const f32, hu);
            let x = std::slice::from_raw_parts(x_ptr.0 as *const f32, iu * total_cols);
            let p_base = proj_ptr.0 as *mut f32;
            let g_off = g * hu * total_cols;
            for r in 0..hu {
                let wx_off = r * iu;
                let b_val = b[r];
                for t_bs in 0..total_cols {
                    let mut s = b_val;
                    for c in 0..iu { s += wx[wx_off + c] * x[c * total_cols + t_bs]; }
                    *p_base.add(g_off + r * total_cols + t_bs) = s;
                }
            }
        }
    });

    for i in 0..(hu * bs) { h_seq[i] = h0[i]; c_seq[i] = c0[i]; }
    let h_s_ptr = SyncPtr(h_seq.as_mut_ptr() as usize);
    let c_s_ptr = SyncPtr(c_seq.as_mut_ptr() as usize);
    let gi_s_ptr = SyncPtr(gi_seq.as_mut_ptr() as usize);
    let gf_s_ptr = SyncPtr(gf_seq.as_mut_ptr() as usize);
    let go_s_ptr = SyncPtr(go_seq.as_mut_ptr() as usize);
    let gg_s_ptr = SyncPtr(gg_seq.as_mut_ptr() as usize);
    let whi_ptr = SyncPtr(whi.as_ptr() as usize);
    let whf_ptr = SyncPtr(whf.as_ptr() as usize);
    let who_ptr = SyncPtr(who.as_ptr() as usize);
    let whg_ptr = SyncPtr(whg.as_ptr() as usize);

    (0..bs).into_par_iter().for_each(|b_idx| {
        unsafe {
            let whi = std::slice::from_raw_parts(whi_ptr.0 as *const f32, hu * hu);
            let whf = std::slice::from_raw_parts(whf_ptr.0 as *const f32, hu * hu);
            let who = std::slice::from_raw_parts(who_ptr.0 as *const f32, hu * hu);
            let whg = std::slice::from_raw_parts(whg_ptr.0 as *const f32, hu * hu);
            let p = std::slice::from_raw_parts(proj_ptr.0 as *const f32, 4 * hu * total_cols);
            let hb = h_s_ptr.0 as *mut f32;
            let cb = c_s_ptr.0 as *mut f32;
            let ib = gi_s_ptr.0 as *mut f32;
            let fb = gf_s_ptr.0 as *mut f32;
            let ob = go_s_ptr.0 as *mut f32;
            let gb = gg_s_ptr.0 as *mut f32;
            for t in 0..sl {
                let prev = t * bs * hu + b_idx;
                let curr = (t + 1) * bs * hu + b_idx;
                let gate = t * bs * hu + b_idx;
                let t_bs = t * bs + b_idx;
                for r in 0..hu {
                    let mut ip = p[0*hu*total_cols + r*total_cols + t_bs];
                    let mut fp = p[1*hu*total_cols + r*total_cols + t_bs];
                    let mut op = p[2*hu*total_cols + r*total_cols + t_bs];
                    let mut gp = p[3*hu*total_cols + r*total_cols + t_bs];
                    let wh_off = r * hu;
                    for c in 0..hu {
                        let hv = *hb.add(prev + c * bs);
                        ip += whi[wh_off+c]*hv; fp += whf[wh_off+c]*hv; op += who[wh_off+c]*hv; gp += whg[wh_off+c]*hv;
                    }
                    let iv = 1.0 / (1.0 + (-ip).exp()); let fv = 1.0 / (1.0 + (-fp).exp()); let ov = 1.0 / (1.0 + (-op).exp()); let gv = gp.tanh();
                    let cp = *cb.add(prev + r * bs); let cc = fv * cp + iv * gv; let hc = ov * cc.tanh();
                    *ib.add(gate + r * bs) = iv; *fb.add(gate + r * bs) = fv; *ob.add(gate + r * bs) = ov; *gb.add(gate + r * bs) = gv;
                    *cb.add(curr + r * bs) = cc; *hb.add(curr + r * bs) = hc;
                }
            }
        }
    });
}

#[napi]
pub fn rnn_forward_native_into(
    wxh: Float32Array, whh: Float32Array, bias: Float32Array,
    x_seq: Float32Array, h0: Float32Array,
    hidden_units: u32, input_units: u32, seq_len: u32, batch_size: u32,
    mut h_seq: Float32Array, mut act_grad: Float32Array,
) {
    let hu = hidden_units as usize; let iu = input_units as usize; let sl = seq_len as usize; let bs = batch_size as usize; let total_cols = sl * bs;
    let mut proj = vec![0.0f32; hu * total_cols];
    let p_ptr = SyncPtr(proj.as_mut_ptr() as usize);
    let wx_ptr = SyncPtr(wxh.as_ptr() as usize); let b_ptr = SyncPtr(bias.as_ptr() as usize); let x_ptr = SyncPtr(x_seq.as_ptr() as usize);
    (0..hu).into_par_iter().for_each(|r| {
        unsafe {
            let wx = std::slice::from_raw_parts(wx_ptr.0 as *const f32, hu * iu);
            let b = std::slice::from_raw_parts(b_ptr.0 as *const f32, hu);
            let x = std::slice::from_raw_parts(x_ptr.0 as *const f32, iu * total_cols);
            let pb = p_ptr.0 as *mut f32;
            let bv = b[r]; let wx_off = r * iu; let p_off = r * total_cols;
            for t_bs in 0..total_cols {
                let mut s = bv; for c in 0..iu { s += wx[wx_off+c] * x[c*total_cols+t_bs]; }
                *pb.add(p_off + t_bs) = s;
            }
        }
    });
    for i in 0..(hu * bs) { h_seq[i] = h0[i]; }
    let h_s_ptr = SyncPtr(h_seq.as_mut_ptr() as usize);
    let d_s_ptr = SyncPtr(act_grad.as_mut_ptr() as usize);
    let whh_ptr = SyncPtr(whh.as_ptr() as usize);
    (0..bs).into_par_iter().for_each(|b_idx| {
        unsafe {
            let whh = std::slice::from_raw_parts(whh_ptr.0 as *const f32, hu * hu);
            let pr = std::slice::from_raw_parts(p_ptr.0 as *const f32, hu * total_cols);
            let hb = h_s_ptr.0 as *mut f32; let db = d_s_ptr.0 as *mut f32;
            for t in 0..sl {
                let prev = t * bs * hu + b_idx; let curr = (t + 1) * bs * hu + b_idx; let t_bs = t * bs + b_idx;
                for r in 0..hu {
                    let mut s = pr[r * total_cols + t_bs];
                    let wh_off = r * hu;
                    for c in 0..hu { s += whh[wh_off+c] * *hb.add(prev + c * bs); }
                    let hv = s.tanh(); *hb.add(curr + r * bs) = hv; *db.add(prev + r * bs) = 1.0 - hv*hv;
                }
            }
        }
    });
}

#[napi]
pub fn rnn_backward_native_into(
    wxh: Float32Array, whh: Float32Array, x_seq: Float32Array, h_seq: Float32Array, d_act: Float32Array, err_h: Float32Array,
    hidden_units: u32, input_units: u32, seq_len: u32, batch_size: u32,
    mut dwxh: Float32Array, mut dwhh: Float32Array, mut dbh: Float32Array, mut dx_out: Float32Array,
) {
    let hu = hidden_units as usize; let iu = input_units as usize; let sl = seq_len as usize; let bs = batch_size as usize; let total_cols = sl * bs;
    let wxh_ptr = SyncPtr(wxh.as_ptr() as usize); let whh_ptr = SyncPtr(whh.as_ptr() as usize); let x_ptr = SyncPtr(x_seq.as_ptr() as usize);
    let h_ptr = SyncPtr(h_seq.as_ptr() as usize); let da_ptr = SyncPtr(d_act.as_ptr() as usize); let eh_ptr = SyncPtr(err_h.as_ptr() as usize);
    let dx_s_ptr = SyncPtr(dx_out.as_mut_ptr() as usize);
    let mut dz_all = vec![0.0f32; sl * bs * hu]; let dz_ptr = SyncPtr(dz_all.as_mut_ptr() as usize);
    (0..bs).into_par_iter().for_each(|b_idx| {
        unsafe {
            let eh = std::slice::from_raw_parts(eh_ptr.0 as *const f32, sl*bs*hu); let da = std::slice::from_raw_parts(da_ptr.0 as *const f32, sl*bs*hu);
            let whh = std::slice::from_raw_parts(whh_ptr.0 as *const f32, hu*hu); let wxh = std::slice::from_raw_parts(wxh_ptr.0 as *const f32, hu*iu);
            let dzb = dz_ptr.0 as *mut f32; let dxb = dx_s_ptr.0 as *mut f32;
            let mut dhn = vec![0.0f32; hu];
            for ti in 0..sl {
                let t = sl - 1 - ti; let gate = t*bs*hu+b_idx;
                for r in 0..hu { let dz = (eh[gate+r*bs] + dhn[r]) * da[gate+r*bs]; *dzb.add(gate+r*bs) = dz; }
                let mut ndh = vec![0.0f32; hu]; for r in 0..hu {
                    let mut s = 0.0f32; for k in 0..hu { s += whh[k*hu+r] * *dzb.add(gate+k*bs); }
                    ndh[r] = s;
                }
                dhn = ndh;
                for j in 0..iu {
                    let mut s = 0.0f32; for k in 0..hu { s += wxh[k*iu+j] * *dzb.add(gate+k*bs); }
                    *dxb.add(j*total_cols + t*bs + b_idx) = s;
                }
            }
        }
    });
    let dwx_ptr = SyncPtr(dwxh.as_mut_ptr() as usize); let dwh_ptr = SyncPtr(dwhh.as_mut_ptr() as usize);
    (0..hu).into_par_iter().for_each(|r| {
        unsafe {
            let dz = std::slice::from_raw_parts(dz_ptr.0 as *const f32, sl*bs*hu); let x = std::slice::from_raw_parts(x_ptr.0 as *const f32, iu*total_cols);
            let h = std::slice::from_raw_parts(h_ptr.0 as *const f32, (sl+1)*bs*hu);
            let dwp = dwx_ptr.0 as *mut f32; let dwhp = dwh_ptr.0 as *mut f32;
            for t in 0..sl {
                let dzo = t*bs*hu+r*bs; let hoff = t*bs*hu; let xoff = t*bs;
                for b in 0..bs {
                    let dv = dz[dzo+b];
                    for c in 0..iu { *dwp.add(r*iu+c) += dv * x[c*total_cols+xoff+b]; }
                    for c in 0..hu { *dwhp.add(r*hu+c) += dv * h[hoff+c*bs+b]; }
                }
            }
        }
    });
    let dbh_s = &mut *dbh; for t in 0..sl { let off = t*bs*hu; for b in 0..bs { for r in 0..hu { dbh_s[r] += dz_all[off+r*bs+b]; } } }
}

#[napi]
pub fn gru_forward_native_into(
    wxr: Float32Array, whr: Float32Array, br: Float32Array, wxz: Float32Array, whz: Float32Array, bz: Float32Array, wxh: Float32Array, whh: Float32Array, bh: Float32Array,
    x_seq: Float32Array, h0: Float32Array, hidden_units: u32, input_units: u32, seq_len: u32, batch_size: u32,
    mut h_seq: Float32Array, mut r_seq: Float32Array, mut z_seq: Float32Array, mut n_seq: Float32Array,
) {
    let hu = hidden_units as usize; let iu = input_units as usize; let sl = seq_len as usize; let bs = batch_size as usize; let total_cols = sl * bs;
    let mut proj = vec![0.0f32; 3 * hu * total_cols]; let p_ptr = SyncPtr(proj.as_mut_ptr() as usize);
    let x_ptr = SyncPtr(x_seq.as_ptr() as usize);
    let wx_ptrs = [SyncPtr(wxr.as_ptr() as usize), SyncPtr(wxz.as_ptr() as usize), SyncPtr(wxh.as_ptr() as usize)];
    let b_ptrs = [SyncPtr(br.as_ptr() as usize), SyncPtr(bz.as_ptr() as usize), SyncPtr(bh.as_ptr() as usize)];
    
    (0..3).into_par_iter().for_each(|g| {
        unsafe {
            let wx = std::slice::from_raw_parts(wx_ptrs[g].0 as *const f32, hu*iu); let b = std::slice::from_raw_parts(b_ptrs[g].0 as *const f32, hu);
            let x = std::slice::from_raw_parts(x_ptr.0 as *const f32, iu*total_cols); let pb = p_ptr.0 as *mut f32;
            for r in 0..hu {
                let bv = b[r]; let wx_off = r * iu; let po = g * hu * total_cols + r * total_cols;
                for t_bs in 0..total_cols {
                    let mut s = bv; for c in 0..iu { s += wx[wx_off+c] * x[c*total_cols+t_bs]; }
                    *pb.add(po + t_bs) = s;
                }
            }
        }
    });
    
    for i in 0..(hu * bs) { h_seq[i] = h0[i]; }
    let h_s_ptr = SyncPtr(h_seq.as_mut_ptr() as usize); let r_s_ptr = SyncPtr(r_seq.as_mut_ptr() as usize);
    let z_s_ptr = SyncPtr(z_seq.as_mut_ptr() as usize); let n_s_ptr = SyncPtr(n_seq.as_mut_ptr() as usize);
    let wr_ptr = SyncPtr(whr.as_ptr() as usize); let wz_ptr = SyncPtr(whz.as_ptr() as usize); let wh_ptr = SyncPtr(whh.as_ptr() as usize);
    
    (0..bs).into_par_iter().for_each(|b_idx| {
        unsafe {
            let wr = std::slice::from_raw_parts(wr_ptr.0 as *const f32, hu*hu); let wz = std::slice::from_raw_parts(wz_ptr.0 as *const f32, hu*hu);
            let wh = std::slice::from_raw_parts(wh_ptr.0 as *const f32, hu*hu); let pr = std::slice::from_raw_parts(p_ptr.0 as *const f32, 3*hu*total_cols);
            let hb = h_s_ptr.0 as *mut f32; let rb = r_s_ptr.0 as *mut f32; let zb = z_s_ptr.0 as *mut f32; let nb = n_s_ptr.0 as *mut f32;
            for t in 0..sl {
                let pre = t*bs*hu+b_idx; let cur = (t+1)*bs*hu+b_idx; let t_bs = t*bs+b_idx;
                for r in 0..hu {
                    let mut rp = pr[0*hu*total_cols + r*total_cols + t_bs];
                    let mut zp = pr[1*hu*total_cols + r*total_cols + t_bs];
                    let wh_off = r * hu;
                    for c in 0..hu { let hv = *hb.add(pre+c*bs); rp += wr[wh_off+c]*hv; zp += wz[wh_off+c]*hv; }
                    let rv = 1.0 / (1.0 + (-rp).exp()); let zv = 1.0 / (1.0 + (-zp).exp());
                    let gate = t*bs*hu+b_idx;
                    *rb.add(gate + r*bs) = rv; *zb.add(gate + r*bs) = zv;
                    let mut np = pr[2*hu*total_cols + r*total_cols + t_bs];
                    for c in 0..hu { np += wh[wh_off+c] * (*rb.add(gate+c*bs)) * *hb.add(pre+c*bs); }
                    let nv = np.tanh(); *nb.add(gate+r*bs) = nv;
                    let hpv = *hb.add(pre+r*bs); *hb.add(cur+r*bs) = (1.0-zv)*nv + zv*hpv;
                }
            }
        }
    });
}

#[napi]
pub fn gru_backward_native_into(
    wxr: Float32Array, whr: Float32Array, wxz: Float32Array, whz: Float32Array, wxh: Float32Array, whh: Float32Array,
    x_seq: Float32Array, h_seq: Float32Array, r_seq: Float32Array, z_seq: Float32Array, n_seq: Float32Array, err_h: Float32Array,
    hidden_units: u32, input_units: u32, seq_len: u32, batch_size: u32,
    mut dwxr: Float32Array, mut dwhr: Float32Array, mut dbr: Float32Array,
    mut dwxz: Float32Array, mut dwhz: Float32Array, mut dbz: Float32Array,
    mut dwxh: Float32Array, mut dwhh: Float32Array, mut dbh: Float32Array, mut dx_out: Float32Array,
) {
    let hu = hidden_units as usize; let iu = input_units as usize; let sl = seq_len as usize; let bs = batch_size as usize; let total_cols = sl * bs;
    let wrp = SyncPtr(whr.as_ptr() as usize); let wzp = SyncPtr(whz.as_ptr() as usize); let whp = SyncPtr(whh.as_ptr() as usize);
    let wxrp = SyncPtr(wxr.as_ptr() as usize); let wxzp = SyncPtr(wxz.as_ptr() as usize); let wxhp = SyncPtr(wxh.as_ptr() as usize);
    let hp = SyncPtr(h_seq.as_ptr() as usize); let rp = SyncPtr(r_seq.as_ptr() as usize); let zp = SyncPtr(z_seq.as_ptr() as usize);
    let np = SyncPtr(n_seq.as_ptr() as usize); let ep = SyncPtr(err_h.as_ptr() as usize); let dxp = SyncPtr(dx_out.as_mut_ptr() as usize);
    let mut dz_all = vec![0.0f32; sl * bs * 3 * hu]; let dz_ptr = SyncPtr(dz_all.as_mut_ptr() as usize);
    
    (0..bs).into_par_iter().for_each(|b_idx| {
        unsafe {
            let wr = std::slice::from_raw_parts(wrp.0 as *const f32, hu*hu); let wz = std::slice::from_raw_parts(wzp.0 as *const f32, hu*hu);
            let wh = std::slice::from_raw_parts(whp.0 as *const f32, hu*hu); let h = std::slice::from_raw_parts(hp.0 as *const f32, (sl+1)*bs*hu);
            let r = std::slice::from_raw_parts(rp.0 as *const f32, sl*bs*hu); let z = std::slice::from_raw_parts(zp.0 as *const f32, sl*bs*hu);
            let n = std::slice::from_raw_parts(np.0 as *const f32, sl*bs*hu); let eh = std::slice::from_raw_parts(ep.0 as *const f32, sl*bs*hu);
            let dzb = dz_ptr.0 as *mut f32; let dxb = dxp.0 as *mut f32;
            let mut dhn = vec![0.0f32; hu];
            for ti in 0..sl {
                let t = sl-1 - ti; let gate = t*bs*hu+b_idx; let pre = t*bs*hu+b_idx; let dzts = t*3*hu*bs;
                for row in 0..hu {
                    let dh = eh[gate+row*bs] + dhn[row]; let zv = z[gate+row*bs]; let nv = n[gate+row*bs]; let hpv = h[pre+row*bs];
                    let dn = dh * (1.0 - zv); let dze = dh * (hpv - nv);
                    let dnp = dn * (1.0 - nv * nv); let dzp = dze * zv * (1.0 - zv);
                    *dzb.add(dzts + 2*hu*bs + row*bs + b_idx) = dnp; *dzb.add(dzts + 1*hu*bs + row*bs + b_idx) = dzp;
                }
                for j in 0..hu {
                    let mut dr_aux = 0.0f32;
                    for i in 0..hu { dr_aux += wh[i*hu + j] * *dzb.add(dzts + 2*hu*bs + i*bs + b_idx); }
                    let dr_val = r[gate + j*bs];
                    *dzb.add(dzts + 0*hu*bs + j*bs + b_idx) = dr_aux * h[pre + j*bs] * dr_val * (1.0 - dr_val);
                }
                let mut dhp = vec![0.0f32; hu]; for row in 0..hu {
                    let mut s = dhn[row] * z[gate+row*bs];
                    for k in 0..hu {
                        s += wr[k*hu+row] * *dzb.add(dzts + 0*hu*bs + k*bs + b_idx);
                        s += wz[k*hu+row] * *dzb.add(dzts + 1*hu*bs + k*bs + b_idx);
                        // dh_prev logic for Candidate: Whh[k, row] * dz_n[k] * r[row]
                        s += wh[k*hu+row] * *dzb.add(dzts + 2*hu*bs + k*bs + b_idx) * r[gate+row*bs];
                    }
                    dhp[row] = s;
                }
                dhn = dhp;
                for j in 0..iu {
                    let mut s = 0.0f32;
                    for k in 0..hu {
                        s += (*wxr_logic(wxrp.0, k, j, iu)) * *dzb.add(dzts + 0*hu*bs + k*bs + b_idx);
                        s += (*wxz_logic(wxzp.0, k, j, iu)) * *dzb.add(dzts + 1*hu*bs + k*bs + b_idx);
                        s += (*wxh_logic(wxhp.0, k, j, iu)) * *dzb.add(dzts + 2*hu*bs + k*bs + b_idx);
                    }
                    *dxb.add(j*total_cols + t*bs + b_idx) = s;
                }
            }
        }
    });
    unsafe fn wxr_logic(p: usize, k: usize, j: usize, iu: usize) -> *const f32 { (p as *const f32).add(k*iu+j) }
    unsafe fn wxz_logic(p: usize, k: usize, j: usize, iu: usize) -> *const f32 { (p as *const f32).add(k*iu+j) }
    unsafe fn wxh_logic(p: usize, k: usize, j: usize, iu: usize) -> *const f32 { (p as *const f32).add(k*iu+j) }

    let dz_s_raw = SyncPtr(dz_all.as_ptr() as usize); let x_s_raw = SyncPtr(x_seq.as_ptr() as usize);
    let h_s_raw = SyncPtr(h_seq.as_ptr() as usize); let r_s_raw = SyncPtr(r_seq.as_ptr() as usize);
    let dwx_ptrs = [SyncPtr(dwxr.as_mut_ptr() as usize), SyncPtr(dwxz.as_mut_ptr() as usize), SyncPtr(dwxh.as_mut_ptr() as usize)];
    let dwh_ptrs = [SyncPtr(dwhr.as_mut_ptr() as usize), SyncPtr(dwhz.as_mut_ptr() as usize), SyncPtr(dwhh.as_mut_ptr() as usize)];
    let db_ptrs = [SyncPtr(dbr.as_mut_ptr() as usize), SyncPtr(dbz.as_mut_ptr() as usize), SyncPtr(dbh.as_mut_ptr() as usize)];

    (0..3).into_par_iter().for_each(|g| {
        unsafe {
            let dwx = dwx_ptrs[g].0 as *mut f32; let dwh = dwh_ptrs[g].0 as *mut f32; let db = db_ptrs[g].0 as *mut f32;
            let dz = std::slice::from_raw_parts(dz_s_raw.0 as *const f32, sl*bs*3*hu);
            let x = std::slice::from_raw_parts(x_s_raw.0 as *const f32, iu*total_cols);
            let h = std::slice::from_raw_parts(h_s_raw.0 as *const f32, (sl+1)*bs*hu);
            let r = std::slice::from_raw_parts(r_s_raw.0 as *const f32, sl*bs*hu);
            for t in 0..sl {
                let dzts = t*3*hu*bs; let goff = g*hu*bs; let hoff = t*bs*hu; let xoff = t*bs;
                for row in 0..hu {
                    let mut s_db = 0.0f32;
                    for b in 0..bs {
                        let dz_v = dz[dzts + goff + row*bs + b]; s_db += dz_v;
                        for c in 0..iu { *dwx.add(row*iu+c) += dz_v * x[c*total_cols + xoff + b]; }
                        for c in 0..hu {
                            let mut h_val = h[hoff + c*bs + b];
                            if g == 2 { h_val *= r[t*bs*hu + c*bs + b]; }
                            *dwh.add(row*hu+c) += dz_v * h_val;
                        }
                    }
                    *db.add(row) += s_db;
                }
            }
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// AdaptiveMemoryRNN – forward pass (fused sequence loop, batched)
//
// Memory layout (time-major, batch-interleaved):
//   x_seq      : [input_units, total_cols]  total_cols = seq_len * batch_size
//   h_seq_out  : [(seq_len+1) * batch_size * hidden_units]  h[0..bs] = h0
//   act_grad   : [seq_len * batch_size * hidden_units]
//   mem_keys   : [batch_size * memory_dim * memory_slots]  (in/out)
//   mem_values : [batch_size * memory_dim * memory_slots]  (in/out)
//   mem_usage  : [batch_size * memory_slots]               (in/out)
//   combined   : [seq_len * (input_units+memory_dim) * batch_size]  (out)
//
// Weights:
//   wq  : [memory_dim, input_units + hidden_units]
//   wm  : [memory_dim, hidden_units]
//   wxh : [hidden_units, input_units + memory_dim]
//   whh : [hidden_units, hidden_units]
//   bh  : [hidden_units]
//   wg  : [memory_dim, input_units + hidden_units + memory_dim]
//   bg  : [memory_dim]
// ─────────────────────────────────────────────────────────────────────────────
#[napi]
#[allow(clippy::too_many_arguments)]
pub fn adaptive_memory_rnn_forward_native_into(
    wq: Float32Array, wm: Float32Array,
    wxh: Float32Array, whh: Float32Array, bh: Float32Array,
    wg: Float32Array, bg: Float32Array,
    x_seq: Float32Array, h0: Float32Array,
    hidden_units: u32, input_units: u32, memory_dim: u32, memory_slots: u32,
    seq_len: u32, batch_size: u32,
    use_relu: bool,
    mut h_seq_out: Float32Array,
    mut act_grad: Float32Array,
    mut mem_keys: Float32Array,
    mut mem_values: Float32Array,
    mut mem_usage: Float32Array,
    mut combined_out: Float32Array,
) {
    let hu = hidden_units as usize;
    let iu = input_units as usize;
    let md = memory_dim as usize;
    let ms = memory_slots as usize;
    let sl = seq_len as usize;
    let bs = batch_size as usize;
    let total_cols = sl * bs;
    let combined_width = iu + md; // width of a single combined vector

    // Initialise h_seq from h0  (h0: [hu * bs])
    for i in 0..(hu * bs) { h_seq_out[i] = h0[i]; }

    // Per-batch sequential recurrence (batches can be parallelised)
    let h_ptr     = SyncPtr(h_seq_out.as_mut_ptr() as usize);
    let da_ptr    = SyncPtr(act_grad.as_mut_ptr() as usize);
    let mk_ptr    = SyncPtr(mem_keys.as_mut_ptr() as usize);
    let mv_ptr    = SyncPtr(mem_values.as_mut_ptr() as usize);
    let mu_ptr    = SyncPtr(mem_usage.as_mut_ptr() as usize);
    let co_ptr    = SyncPtr(combined_out.as_mut_ptr() as usize);
    let x_ptr     = SyncPtr(x_seq.as_ptr() as usize);
    let wq_ptr    = SyncPtr(wq.as_ptr() as usize);
    let wm_ptr    = SyncPtr(wm.as_ptr() as usize);
    let wxh_ptr   = SyncPtr(wxh.as_ptr() as usize);
    let whh_ptr   = SyncPtr(whh.as_ptr() as usize);
    let bh_ptr    = SyncPtr(bh.as_ptr() as usize);
    let wg_ptr    = SyncPtr(wg.as_ptr() as usize);
    let bg_ptr    = SyncPtr(bg.as_ptr() as usize);

    (0..bs).into_par_iter().for_each(|b| {
        unsafe {
            let x     = std::slice::from_raw_parts(x_ptr.0 as *const f32,  iu * total_cols);
            let wq    = std::slice::from_raw_parts(wq_ptr.0 as *const f32,  md * (iu + hu));
            let wm    = std::slice::from_raw_parts(wm_ptr.0 as *const f32,  md * hu);
            let wxh   = std::slice::from_raw_parts(wxh_ptr.0 as *const f32, hu * combined_width);
            let whh   = std::slice::from_raw_parts(whh_ptr.0 as *const f32, hu * hu);
            let bh    = std::slice::from_raw_parts(bh_ptr.0 as *const f32,  hu);
            let wg    = std::slice::from_raw_parts(wg_ptr.0 as *const f32,  md * (iu + hu + md));
            let bg    = std::slice::from_raw_parts(bg_ptr.0 as *const f32,  md);

            let hb    = h_ptr.0 as *mut f32;
            let dab   = da_ptr.0 as *mut f32;
            let mkb   = mk_ptr.0 as *mut f32;
            let mvb   = mv_ptr.0 as *mut f32;
            let mub   = mu_ptr.0 as *mut f32;
            let cob   = co_ptr.0 as *mut f32;

            let mem_off   = b * md * ms;   // offset into keys/values for this sample
            let usage_off = b * ms;        // offset into usage for this sample
            let score_scale = 1.0 / (md as f32).sqrt();

            let mut query  = vec![0.0f32; md];
            let mut scores = vec![0.0f32; ms];
            let mut attn   = vec![0.0f32; ms];
            let mut read   = vec![0.0f32; md];
            let mut gate   = vec![0.0f32; md];
            let mut cand   = vec![0.0f32; md];
            let mut x_t    = vec![0.0f32; iu];
            let mut h_t    = vec![0.0f32; hu];

            for t in 0..sl {
                let t_bs = t * bs + b;

                // Extract x_t  (column-major, time-major)
                for i in 0..iu { x_t[i] = x[i * total_cols + t_bs]; }

                // h_prev: layout  h_seq[(t * bs + b) + r * bs]  →  h[(t*bs*hu) + r*bs + b]
                let h_prev_base = t * bs * hu + b;

                // ── 1. Query projection: query = Wq * [x_t ; h_prev] ──────────
                let qi_len = iu + hu;
                for i in 0..md {
                    let mut s = 0.0f32;
                    let row = i * qi_len;
                    for j in 0..iu  { s += wq[row + j]       * x_t[j]; }
                    for j in 0..hu  { s += wq[row + iu + j]  * *hb.add(h_prev_base + j * bs); }
                    query[i] = s;
                }

                // ── 2. Attention retrieval ─────────────────────────────────────
                let mut best_slot = 0usize;
                let mut best_score = f32::NEG_INFINITY;
                for slot in 0..ms {
                    let mut sc = 0.0f32;
                    for i in 0..md { sc += query[i] * *mkb.add(mem_off + i * ms + slot); }
                    sc *= score_scale;
                    scores[slot] = sc;
                    if sc > best_score { best_score = sc; best_slot = slot; }
                }
                stable_softmax(&scores, &mut attn);
                for i in 0..md {
                    let mut s = 0.0f32;
                    let base = mem_off + i * ms;
                    for slot in 0..ms { s += *mvb.add(base + slot) * attn[slot]; }
                    read[i] = s;
                }

                // ── 3. Combined = [x_t ; read] ────────────────────────────────
                // combined_out layout: [t * combined_width * bs + j * bs + b]
                let co_t_base = t * combined_width * bs;
                for j in 0..iu  { *cob.add(co_t_base + j * bs + b)        = x_t[j]; }
                for j in 0..md  { *cob.add(co_t_base + (iu + j) * bs + b) = read[j]; }

                // ── 4. RNN cell ───────────────────────────────────────────────
                let h_curr_base = (t + 1) * bs * hu + b;
                let da_base     = t * bs * hu + b;
                for i in 0..hu {
                    let mut s = bh[i];
                    let wx_row = i * combined_width;
                    for j in 0..iu { s += wxh[wx_row + j]      * x_t[j]; }
                    for j in 0..md { s += wxh[wx_row + iu + j]  * read[j]; }
                    let wh_row = i * hu;
                    for j in 0..hu { s += whh[wh_row + j] * *hb.add(h_prev_base + j * bs); }
                    if use_relu {
                        if s > 0.0 { h_t[i] = s; *dab.add(da_base + i * bs) = 1.0; }
                        else        { h_t[i] = 0.0; *dab.add(da_base + i * bs) = 0.0; }
                    } else {
                        let tv = s.tanh(); h_t[i] = tv;
                        *dab.add(da_base + i * bs) = 1.0 - tv * tv;
                    }
                    *hb.add(h_curr_base + i * bs) = h_t[i];
                }

                // ── 5. Write gate: gate = sigmoid(Wg * [x_t ; h_t ; read]) ───
                let gi_len = iu + hu + md;
                for i in 0..md {
                    let mut s = bg[i];
                    let row = i * gi_len;
                    for j in 0..iu { s += wg[row + j]           * x_t[j]; }
                    for j in 0..hu { s += wg[row + iu + j]       * h_t[j]; }
                    for j in 0..md { s += wg[row + iu + hu + j]  * read[j]; }
                    gate[i] = sigmoid(s);
                }

                // ── 6. Candidate memory: cand = Wm * h_t ──────────────────────
                for i in 0..md {
                    let mut s = 0.0f32;
                    let row = i * hu;
                    for j in 0..hu { s += wm[row + j] * h_t[j]; }
                    cand[i] = s;
                }

                // ── 7. Select write slot ───────────────────────────────────────
                let mut write_slot = ms; // sentinel: no free slot yet
                for slot in 0..ms {
                    if *mub.add(usage_off + slot) == 0.0 { write_slot = slot; break; }
                }
                if write_slot == ms {
                    // fall back to least-used (favour retrieved slot on tie)
                    write_slot = best_slot;
                    let mut min_u = *mub.add(usage_off + best_slot);
                    for slot in 0..ms {
                        let u = *mub.add(usage_off + slot);
                        if u < min_u { min_u = u; write_slot = slot; }
                    }
                }

                // ── 8. Gated memory update ─────────────────────────────────────
                for i in 0..md {
                    let idx = mem_off + i * ms + write_slot;
                    let g = gate[i];
                    *mkb.add(idx) = (1.0 - g) * *mkb.add(idx) + g * query[i];
                    *mvb.add(idx) = (1.0 - g) * *mvb.add(idx) + g * cand[i];
                }
                *mub.add(usage_off + write_slot) += 1.0;
            }
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// AdaptiveMemoryRNN – legacy partial backward
//
// IMPORTANT:
// This native kernel only covers gradients for Wxh/Whh/bh and does not provide
// parity with the full JS reference backward used by the TypeScript layer.
// The TS wrapper intentionally disables use of this kernel for training until a
// parity-complete native implementation exists.
//
// Input buffers (from forward):
//   combined  : [seq_len * combined_width * batch_size]
//   h_seq     : [(seq_len+1) * batch_size * hidden_units]
//   act_grad  : [seq_len * batch_size * hidden_units]
//   err_h     : [seq_len * batch_size * hidden_units]  (per-step errors, pre-built)
//
// Output gradient buffers (accumulated, must be zeroed by caller):
//   dwxh, dwhh, dbh, dx_out
//
// dx_out layout: [input_units, total_cols]  (only the first `input_units` rows
//                                             of the combined gradient are emitted)
// ─────────────────────────────────────────────────────────────────────────────
#[napi]
#[allow(clippy::too_many_arguments)]
pub fn adaptive_memory_rnn_backward_native_into(
    wxh: Float32Array, whh: Float32Array,
    combined: Float32Array, h_seq: Float32Array,
    act_grad: Float32Array, err_h: Float32Array,
    hidden_units: u32, input_units: u32, memory_dim: u32,
    seq_len: u32, batch_size: u32,
    mut dwxh: Float32Array, mut dwhh: Float32Array, mut dbh: Float32Array,
    mut dx_out: Float32Array,
) {
    let hu = hidden_units as usize;
    let iu = input_units as usize;
    let md = memory_dim as usize;
    let sl = seq_len as usize;
    let bs = batch_size as usize;
    let total_cols = sl * bs;
    let combined_width = iu + md;

    // ── Pass 1: compute dz per (t, b) and propagate dx, accumulate dh_prev ──
    let mut dz_all = vec![0.0f32; sl * bs * hu];
    let dz_ptr  = SyncPtr(dz_all.as_mut_ptr() as usize);
    let wx_ptr  = SyncPtr(wxh.as_ptr() as usize);
    let wh_ptr  = SyncPtr(whh.as_ptr() as usize);
    let co_ptr  = SyncPtr(combined.as_ptr() as usize);
    let hs_ptr  = SyncPtr(h_seq.as_ptr() as usize);
    let da_ptr  = SyncPtr(act_grad.as_ptr() as usize);
    let eh_ptr  = SyncPtr(err_h.as_ptr() as usize);
    let dx_ptr  = SyncPtr(dx_out.as_mut_ptr() as usize);

    (0..bs).into_par_iter().for_each(|b| {
        unsafe {
            let wx  = std::slice::from_raw_parts(wx_ptr.0 as *const f32, hu * combined_width);
            let wh  = std::slice::from_raw_parts(wh_ptr.0 as *const f32, hu * hu);
            let da  = std::slice::from_raw_parts(da_ptr.0 as *const f32, sl * bs * hu);
            let eh  = std::slice::from_raw_parts(eh_ptr.0 as *const f32, sl * bs * hu);
            let dzb = dz_ptr.0 as *mut f32;
            let dxb = dx_ptr.0 as *mut f32;

            let mut dhn = vec![0.0f32; hu];
            for ti in 0..sl {
                let t      = sl - 1 - ti;
                let gate   = t * bs * hu + b;
                // dz[i] = (err_h[gate+i*bs] + dh_next[i]) * act_grad[gate+i*bs]
                for i in 0..hu {
                    let dz = (eh[gate + i * bs] + dhn[i]) * da[gate + i * bs];
                    *dzb.add(gate + i * bs) = dz;
                }
                // dh_prev: W_hh^T * dz
                let mut ndh = vec![0.0f32; hu];
                for j in 0..hu {
                    let mut s = 0.0f32;
                    for k in 0..hu { s += wh[k * hu + j] * *dzb.add(gate + k * bs); }
                    ndh[j] = s;
                }
                dhn = ndh;
                // dx: Wxh[:, 0..iu]^T * dz  (only the input portion of combined)
                for j in 0..iu {
                    let mut s = 0.0f32;
                    for k in 0..hu { s += wx[k * combined_width + j] * *dzb.add(gate + k * bs); }
                    *dxb.add(j * total_cols + t * bs + b) = s;
                }
            }
        }
    });

    // ── Pass 2: weight gradients (parallel over hidden rows) ──────────────────
    let dwx_ptr = SyncPtr(dwxh.as_mut_ptr() as usize);
    let dwh_ptr = SyncPtr(dwhh.as_mut_ptr() as usize);

    (0..hu).into_par_iter().for_each(|r| {
        unsafe {
            let dz = std::slice::from_raw_parts(dz_ptr.0 as *const f32, sl * bs * hu);
            let co = std::slice::from_raw_parts(co_ptr.0 as *const f32, sl * bs * combined_width);
            let hs = std::slice::from_raw_parts(hs_ptr.0 as *const f32, (sl + 1) * bs * hu);
            let dwx = dwx_ptr.0 as *mut f32;
            let dwh = dwh_ptr.0 as *mut f32;
            for t in 0..sl {
                let dz_off  = t * bs * hu;
                let co_off  = t * combined_width * bs;
                let hs_off  = t * bs * hu;
                for b in 0..bs {
                    let dv = dz[dz_off + r * bs + b];
                    for c in 0..combined_width {
                        *dwx.add(r * combined_width + c) += dv * co[co_off + c * bs + b];
                    }
                    for c in 0..hu {
                        *dwh.add(r * hu + c) += dv * hs[hs_off + c * bs + b];
                    }
                }
            }
        }
    });

    // ── Pass 3: bias gradients ─────────────────────────────────────────────────
    let dbh_s = &mut *dbh;
    for t in 0..sl {
        let off = t * bs * hu;
        for b in 0..bs {
            for r in 0..hu { dbh_s[r] += dz_all[off + r * bs + b]; }
        }
    }
}

#[napi]
#[allow(clippy::too_many_arguments)]
pub fn adaptive_memory_rnn_backward_full_native_into(
    wxh: Float32Array,
    whh: Float32Array,
    wq: Float32Array,
    wm: Float32Array,
    wg: Float32Array,
    h_prev: Float32Array,
    h: Float32Array,
    d_act: Float32Array,
    combined: Float32Array,
    read: Float32Array,
    query_input: Float32Array,
    query: Float32Array,
    attention: Float32Array,
    gate_input: Float32Array,
    gate: Float32Array,
    candidate: Float32Array,
    memory_keys_before: Float32Array,
    memory_values_before: Float32Array,
    write_slots: Int32Array,
    err_h: Float32Array,
    hidden_units: u32,
    input_units: u32,
    memory_dim: u32,
    memory_slots: u32,
    seq_len: u32,
    batch_size: u32,
    mut dwxh: Float32Array,
    mut dwhh: Float32Array,
    mut dbh: Float32Array,
    mut dwq: Float32Array,
    mut dwm: Float32Array,
    mut dwg: Float32Array,
    mut dbg: Float32Array,
    mut dx_out: Float32Array,
) {
    let hu = hidden_units as usize;
    let iu = input_units as usize;
    let md = memory_dim as usize;
    let ms = memory_slots as usize;
    let sl = seq_len as usize;
    let bs = batch_size as usize;
    let total_cols = sl * bs;
    let combined_width = iu + md;
    let query_input_width = iu + hu;
    let gate_input_width = iu + hu + md;
    let memory_state_size = md * ms;
    let score_scale = 1.0 / (md as f32).sqrt();

    let wxh_s = &*wxh;
    let whh_s = &*whh;
    let wq_s = &*wq;
    let wm_s = &*wm;
    let wg_s = &*wg;
    let h_prev_s = &*h_prev;
    let h_s = &*h;
    let d_act_s = &*d_act;
    let combined_s = &*combined;
    let _read_s = &*read;
    let query_input_s = &*query_input;
    let query_s = &*query;
    let attention_s = &*attention;
    let gate_input_s = &*gate_input;
    let gate_s = &*gate;
    let candidate_s = &*candidate;
    let memory_keys_before_s = &*memory_keys_before;
    let memory_values_before_s = &*memory_values_before;
    let write_slots_s = &*write_slots;
    let err_h_s = &*err_h;
    let dwxh_s = &mut *dwxh;
    let dwhh_s = &mut *dwhh;
    let dbh_s = &mut *dbh;
    let dwq_s = &mut *dwq;
    let dwm_s = &mut *dwm;
    let dwg_s = &mut *dwg;
    let dbg_s = &mut *dbg;
    let dx_out_s = &mut *dx_out;

    for b in 0..bs {
        let mut dh_next = vec![0.0f32; hu];
        let mut d_memory_keys_next = vec![0.0f32; memory_state_size];
        let mut d_memory_values_next = vec![0.0f32; memory_state_size];

        for ti in 0..sl {
            let t = sl - 1 - ti;
            let step_index = b * sl + t;
            let h_prev_base = step_index * hu;
            let h_base = step_index * hu;
            let d_act_base = step_index * hu;
            let combined_base = step_index * combined_width;
            let _read_base = step_index * md;
            let query_input_base = step_index * query_input_width;
            let query_base = step_index * md;
            let attention_base = step_index * ms;
            let gate_input_base = step_index * gate_input_width;
            let gate_base = step_index * md;
            let candidate_base = step_index * md;
            let memory_base = step_index * memory_state_size;
            let err_base = step_index * hu;
            let write_slot = write_slots_s[step_index] as usize;

            let mut d_query = vec![0.0f32; md];
            let mut d_candidate = vec![0.0f32; md];
            let mut d_gate = vec![0.0f32; md];
            let mut d_read = vec![0.0f32; md];
            let mut d_memory_keys_before = d_memory_keys_next.clone();
            let mut d_memory_values_before = d_memory_values_next.clone();

            for i in 0..md {
                let idx = i * ms + write_slot;
                let gate_val = gate_s[gate_base + i];
                let old_key = memory_keys_before_s[memory_base + idx];
                let old_value = memory_values_before_s[memory_base + idx];
                let d_key_after = d_memory_keys_next[idx];
                let d_value_after = d_memory_values_next[idx];
                d_memory_keys_before[idx] = d_key_after * (1.0 - gate_val);
                d_memory_values_before[idx] = d_value_after * (1.0 - gate_val);
                d_query[i] += d_key_after * gate_val;
                d_candidate[i] += d_value_after * gate_val;
                d_gate[i] += d_key_after * (query_s[query_base + i] - old_key)
                    + d_value_after * (candidate_s[candidate_base + i] - old_value);
            }

            let mut dh = vec![0.0f32; hu];
            for i in 0..hu {
                dh[i] = err_h_s[err_base + i] + dh_next[i];
            }

            for i in 0..md {
                let dc = d_candidate[i];
                let row_off = i * hu;
                for j in 0..hu {
                    dwm_s[row_off + j] += dc * h_s[h_base + j];
                }
            }
            for j in 0..hu {
                let mut sum = 0.0f32;
                for i in 0..md {
                    sum += wm_s[i * hu + j] * d_candidate[i];
                }
                dh[j] += sum;
            }

            let mut d_gate_pre = vec![0.0f32; md];
            for i in 0..md {
                let gate_val = gate_s[gate_base + i];
                let dgp = d_gate[i] * gate_val * (1.0 - gate_val);
                d_gate_pre[i] = dgp;
                dbg_s[i] += dgp;
            }

            for i in 0..md {
                let dgp = d_gate_pre[i];
                let row_off = i * gate_input_width;
                for j in 0..gate_input_width {
                    dwg_s[row_off + j] += dgp * gate_input_s[gate_input_base + j];
                }
            }

            let col = t * bs + b;
            for j in 0..iu {
                let mut sum = 0.0f32;
                for i in 0..md {
                    sum += wg_s[i * gate_input_width + j] * d_gate_pre[i];
                }
                dx_out_s[j * total_cols + col] += sum;
            }
            for j in 0..hu {
                let mut sum = 0.0f32;
                for i in 0..md {
                    sum += wg_s[i * gate_input_width + iu + j] * d_gate_pre[i];
                }
                dh[j] += sum;
            }
            for j in 0..md {
                let mut sum = 0.0f32;
                for i in 0..md {
                    sum += wg_s[i * gate_input_width + iu + hu + j] * d_gate_pre[i];
                }
                d_read[j] += sum;
            }

            let mut dz = vec![0.0f32; hu];
            for i in 0..hu {
                let dzv = dh[i] * d_act_s[d_act_base + i];
                dz[i] = dzv;
                dbh_s[i] += dzv;
            }

            for i in 0..hu {
                let dzv = dz[i];
                let wxh_off = i * combined_width;
                let whh_off = i * hu;
                for j in 0..combined_width {
                    dwxh_s[wxh_off + j] += dzv * combined_s[combined_base + j];
                }
                for j in 0..hu {
                    dwhh_s[whh_off + j] += dzv * h_prev_s[h_prev_base + j];
                }
            }

            for j in 0..iu {
                let mut sum = 0.0f32;
                for i in 0..hu {
                    sum += wxh_s[i * combined_width + j] * dz[i];
                }
                dx_out_s[j * total_cols + col] += sum;
            }
            for j in 0..md {
                let mut sum = 0.0f32;
                for i in 0..hu {
                    sum += wxh_s[i * combined_width + iu + j] * dz[i];
                }
                d_read[j] += sum;
            }

            let mut dh_prev = vec![0.0f32; hu];
            for j in 0..hu {
                let mut sum = 0.0f32;
                for i in 0..hu {
                    sum += whh_s[i * hu + j] * dz[i];
                }
                dh_prev[j] = sum;
            }

            let mut d_attention = vec![0.0f32; ms];
            for slot in 0..ms {
                let mut attn_grad = 0.0f32;
                for i in 0..md {
                    let idx = i * ms + slot;
                    d_memory_values_before[idx] += d_read[i] * attention_s[attention_base + slot];
                    attn_grad += memory_values_before_s[memory_base + idx] * d_read[i];
                }
                d_attention[slot] = attn_grad;
            }

            let mut softmax_inner = 0.0f32;
            for slot in 0..ms {
                softmax_inner += d_attention[slot] * attention_s[attention_base + slot];
            }
            let mut d_scores = vec![0.0f32; ms];
            for slot in 0..ms {
                d_scores[slot] = attention_s[attention_base + slot] * (d_attention[slot] - softmax_inner);
            }

            for slot in 0..ms {
                let score_grad = d_scores[slot] * score_scale;
                for i in 0..md {
                    let idx = i * ms + slot;
                    d_memory_keys_before[idx] += query_s[query_base + i] * score_grad;
                    d_query[i] += memory_keys_before_s[memory_base + idx] * score_grad;
                }
            }

            for i in 0..md {
                let dq = d_query[i];
                let row_off = i * query_input_width;
                for j in 0..query_input_width {
                    dwq_s[row_off + j] += dq * query_input_s[query_input_base + j];
                }
            }

            for j in 0..iu {
                let mut sum = 0.0f32;
                for i in 0..md {
                    sum += wq_s[i * query_input_width + j] * d_query[i];
                }
                dx_out_s[j * total_cols + col] += sum;
            }
            for j in 0..hu {
                let mut sum = 0.0f32;
                for i in 0..md {
                    sum += wq_s[i * query_input_width + iu + j] * d_query[i];
                }
                dh_prev[j] += sum;
            }

            dh_next = dh_prev;
            d_memory_keys_next = d_memory_keys_before;
            d_memory_values_next = d_memory_values_before;
        }
    }
}
