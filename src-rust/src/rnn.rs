use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;

struct SyncPtr(usize);
unsafe impl Send for SyncPtr {}
unsafe impl Sync for SyncPtr {}

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
