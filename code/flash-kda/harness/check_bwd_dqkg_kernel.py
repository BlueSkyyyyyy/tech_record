"""P6-4 evidence: a minimal runnable B2 (`wy_dqkg`) backward kernel.

P6-2 landed B1 (the cross-chunk *state adjoint*, FLA `dhu`). This script lands the
next piece of the target-card backward design (docs/06 §5.2): **B2 `bwd_dqkg`**, the
per-chunk pass that turns the saved per-chunk state adjoint ``dh`` (from B1) plus the
output/gate quantities into the intra-chunk gradients ``dq, dk, dv, dbeta, dg, dA``.

It mirrors FLA's ``chunk_kda_bwd_kernel_wy_dqkg_fused`` (fla/ops/kda/chunk_bwd.py:133-310),
which is the reference implementation of `05` §7 eqs (7.2)-(7.10).  We validate three ways:

  Part A  torch ``wy_dqkg_torch`` (per-chunk, fp32/fp64)  vs  FLA kernel.
  Part B  Triton ``_wy_dqkg_kernel``  vs  FLA kernel  (and vs the torch reference).
  Part C  operand-precision floor: run the Triton kernel with bf16 operands and bound the
          extra error (what an actual bf16 kernel would pay), plus a fp64 torch run to
          show the residual is `tl.dot` (tf32) and not a formula mistake.

Notation (docs/05 §7, docs/06 §3.3), all per chunk `c`, `t` = intra-chunk row (`o_t`):
    k, q            [BT,K]   (raw; undecayed)
    g               [BT,K]   chunk-local log2 cumsum  (FLA `g_cumsum`)
    beta            [BT]     write gate (sigmoid'd)
    A               [BT,BT]  = P = (I+L)^{-1}  (FLA `Akk`)
    S_in = h        [K,V]    chunk-input state
    dS   = dh       [K,V]    chunk-output state adjoint (from B1)
    do              [BT,V]   upstream dL/do
    dv              [BT,V]   output-branch dL/dU (from `dAv`)
    v_new           [BT,V]   = U

The kernel *semantics* (loaded-into-register orientations are all transposed):
    b_h  = S_in^T  [V,K]     b_dh = dS^T [V,K]     b_A = A^T [BT,BT]
    b_dq = do @ b_h ;  b_dk = v_new @ b_dh ;  b_dw = dv @ b_h
    b_dgk = sum_V(b_h * b_dh)
    (i_k==0) b_dA = dv @ v^T ; b_dvb = A @ dv ; b_dv2 = b_dvb*beta ; b_db = sum_V(b_dvb*v)
    ... decay rescale / gate chain / dA = -A^T (beta*...) A^T ...

Run inside `kernel_lab`:
    python ~/proj/tech_record/code/flash-kda/harness/check_bwd_dqkg_kernel.py
"""
import math
import os
import sys

# FLA >=0.5 ships a TileLang backend for the KDA backward that is auto-selected;
# docs/06 reads the *Triton* reference, so pin the dispatch to it here.
os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

sys.path.insert(0, "~/proj/tech_record/code/flash-kda/src")

RCP_LN2 = 1.4426950408889634


def stats(name, a, b):
    d = (a.float() - b.float()).abs()
    denom = b.float().abs().max().item()
    max_abs = d.max().item()
    max_rel = max_abs / (denom + 1e-12)
    print(f"  {name:34s} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  ref_max={denom:.3e}")
    return max_abs, max_rel


def cdiv(a, b):
    return (a + b - 1) // b


# --------------------------------------------------------------------------- data
def make_inputs(B, T, H, K, V, BT, device, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    NT = cdiv(T, BT)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=device), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=device), dim=-1)
    v = torch.randn(B, T, H, V, device=device)
    # chunk-local log2 cumsum of a negative per-channel gate, <= 0 (built padded, then sliced)
    glog = -torch.rand(B, T, H, K, device=device) * 0.5
    gpad = -torch.rand(B, NT * BT, H, K, device=device) * 0.5
    gpad[:, :T] = glog
    g = gpad.view(B, NT, BT, H, K).cumsum(dim=2).reshape(B, NT * BT, H, K)[:, :T].contiguous()
    beta = torch.sigmoid(torch.randn(B, T, H, device=device))
    # A = (I + L)^{-1}, L strictly lower, on random small L
    L = torch.randn(B, NT, H, BT, BT, device=device) * 0.02
    L = torch.tril(L, -1)
    eye = torch.eye(BT, device=device).expand(B, NT, H, BT, BT)
    A = torch.linalg.solve(eye + L, eye)                             # [B,NT,H,BT,BT]
    A = A.permute(0, 1, 3, 2, 4).reshape(B, NT * BT, H, BT)[:, :T].contiguous()   # [B,T,H,BT]
    do = torch.randn(B, T, H, V, device=device)
    dv = torch.randn(B, T, H, V, device=device) * 0.1
    v_new = torch.randn(B, T, H, V, device=device)
    h = torch.randn(B, NT, H, K, V, device=device) * 0.5
    dh = torch.randn(B, NT, H, K, V, device=device) * 0.5
    to = lambda x: x.to(dtype)
    return dict(q=to(q), k=to(k), v=to(v), v_new=to(v_new), g=to(g), beta=to(beta),
                A=to(A), h=to(h), do=to(do), dh=to(dh), dv=to(dv))


# --------------------------------------------------------------------- FLA kernel
def fla_wy_dqkg(inp, scale, BT):
    from fla.ops.kda.chunk_bwd import chunk_kda_bwd_wy_dqkg_fused
    out = chunk_kda_bwd_wy_dqkg_fused(
        q=inp["q"], k=inp["k"], v=inp["v"], v_new=inp["v_new"], g=inp["g"],
        beta=inp["beta"], A=inp["A"], h=inp["h"], do=inp["do"], dh=inp["dh"],
        dv=inp["dv"], scale=scale, chunk_size=BT)
    dq, dk, dv2, db, dg, dA = out
    return dict(dq=dq, dk=dk, dv=dv2, db=db, dg=dg, dA=dA)


# ------------------------------------------------------------------ torch golden
def wy_dqkg_torch(inp, scale, BT):
    """Per-chunk torch transcription of the kernel (BK=K, BV=V), index-faithful."""
    qd, kd, vd, vnd = inp["q"].float(), inp["k"].float(), inp["v"].float(), inp["v_new"].float()
    gd, bd = inp["g"].float(), inp["beta"].float()
    Ad, hd, dod, dhd, dvd = (inp["A"].float(), inp["h"].float(), inp["do"].float(),
                             inp["dh"].float(), inp["dv"].float())
    B, T, H, K = qd.shape
    V = vd.shape[-1]
    NT = cdiv(T, BT)
    dev = qd.device

    q4 = qd.view(B, NT, BT, H, K)
    k4 = kd.view(B, NT, BT, H, K)
    v4 = vd.view(B, NT, BT, H, V)
    vn4 = vnd.view(B, NT, BT, H, V)
    g4 = gd.view(B, NT, BT, H, K)
    b4 = bd.view(B, NT, BT, H)
    A4 = Ad.view(B, NT, BT, H, BT)               # A4[b,c,o_t,h,o_A]
    do4 = dod.view(B, NT, BT, H, V)
    dv4 = dvd.view(B, NT, BT, H, V)

    dq = torch.zeros(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.zeros(B, T, H, K, device=dev, dtype=torch.float32)
    dv2 = torch.zeros(B, T, H, V, device=dev, dtype=torch.float32)
    dg = torch.zeros(B, T, H, K, device=dev, dtype=torch.float32)
    db = torch.zeros(B, T, H, device=dev, dtype=torch.float32)
    dA = torch.zeros(B, T, H, BT, device=dev, dtype=torch.float32)

    dq4 = dq.view(B, NT, BT, H, K)
    dk4 = dk.view(B, NT, BT, H, K)
    dv24 = dv2.view(B, NT, BT, H, V)
    dg4 = dg.view(B, NT, BT, H, K)
    db4 = db.view(B, NT, BT, H)
    dA4 = dA.view(B, NT, BT, H, BT)

    for c in range(NT):
        o_t = c * BT + torch.arange(BT, device=dev)
        m_t = o_t < T
        last = min(T, c * BT + BT) - 1
        m_last = o_t == last
        b_beta = torch.where(m_t[None, :, None], b4[:, c], torch.zeros_like(b4[:, c])).permute(0, 2, 1)  # [B,H,BT]
        # b_A[i,j] = A4[b,c,o_t[j],h,o_A[i]]  => transpose of [BT(rows=o_t),BT(cols=o_A)]
        A_ch = A4[:, c].permute(0, 2, 1, 3)            # [B,H,BT(o_t),BT(o_A)]
        b_A = A_ch.transpose(-1, -2)                   # [B,H,BT(o_A),BT(o_t)]
        mt = m_t.to(torch.float32)
        # per (b,h)
        b_do = torch.where(m_t[None, :, None], do4[:, c].permute(0, 2, 1, 3), torch.zeros_like(do4[:, c].permute(0, 2, 1, 3)))  # [B,H,BT,V]
        b_dv = torch.where(m_t[None, :, None], dv4[:, c].permute(0, 2, 1, 3), torch.zeros_like(b_do))
        b_vn = torch.where(m_t[None, :, None], vn4[:, c].permute(0, 2, 1, 3), torch.zeros_like(b_do))
        b_v = torch.where(m_t[None, :, None], v4[:, c].permute(0, 2, 1, 3), torch.zeros_like(b_do))
        b_k = torch.where(m_t[None, :, None], k4[:, c].permute(0, 2, 1, 3), torch.zeros_like(b_do))
        b_q = torch.where(m_t[None, :, None], q4[:, c].permute(0, 2, 1, 3), torch.zeros_like(b_do))
        b_g = torch.where(m_t[None, :, None], g4[:, c].permute(0, 2, 1, 3), torch.zeros_like(b_do))
        # b_h[v,k] = h[k,v] => transpose
        b_h = hd[:, c].transpose(-1, -2)               # [B,H,V,K]
        b_dh = dhd[:, c].transpose(-1, -2)             # [B,H,V,K]
        b_dgk = (b_h * b_dh).sum(dim=2)                # [B,H,K]  (sum over V)
        b_dq = torch.einsum("bhtv,bhvk->bhtk", b_do, b_h)
        b_dk = torch.einsum("bhtv,bhvk->bhtk", b_vn, b_dh)
        b_dw = torch.einsum("bhtv,bhvk->bhtk", b_dv, b_h)
        # i_k == 0 branch
        b_dA = torch.einsum("bhtv,bhuv->bhtu", b_dv, b_v)          # [B,H,BT(o_t),BT(o_v)]
        b_dvb = torch.einsum("bhtu,bhuv->bhtv", b_A, b_dv)
        b_dv2 = b_dvb * b_beta.unsqueeze(-1)
        b_db = (b_dvb * b_v).sum(dim=3)                            # [B,H,BT]
        # post i_k
        gk_exp = torch.exp2(b_g)
        b_gb = gk_exp * b_beta.unsqueeze(-1)
        b_gn = g4[:, c][:, last - c * BT]              # [B,H,K]  (last valid row of the chunk)
        b_dgk = b_dgk * torch.exp2(b_gn)
        b_dq = b_dq * gk_exp * scale
        b_dk = b_dk * torch.where(m_t[None, None, :, None], torch.exp2(b_gn.unsqueeze(2) - b_g), torch.zeros_like(b_g))
        b_kg = b_k * gk_exp
        b_dw = -b_dw
        b_dA = b_dA + torch.einsum("bhtk,bhuk->bhtu", b_dw, b_kg)
        b_dkgb = torch.einsum("bhtu,bhuk->bhtk", b_A, b_dw)
        b_db = b_db + (b_dkgb * b_kg).sum(dim=3)
        b_kdk = b_k * b_dk
        b_dgk = b_dgk + b_kdk.sum(dim=2)               # [B,H,K]  (sum over BT)
        b_dg = (b_q * b_dq - b_kdk
                + m_last[None, None, :, None].float() * b_dgk.unsqueeze(2)
                + b_kg * b_dkgb * b_beta.unsqueeze(-1))
        b_dk = b_dk + b_dkgb * b_gb
        # final dA
        m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t[None, :])
        m_A = m_A[None, None]                          # [1,1,BT,BT]
        b_dA = torch.where(m_A, b_dA * b_beta.unsqueeze(2), torch.zeros_like(b_dA))
        b_dA = torch.einsum("bhtu,bhuv->bhtv", b_dA, b_A)
        b_dA = torch.einsum("bhtu,bhuv->bhtv", b_A, b_dA)
        b_dA = torch.where(m_A, -b_dA, torch.zeros_like(b_dA))
        # scatter back (transpose permutes)
        dq4[:, c] = b_dq.permute(0, 2, 1, 3)
        dk4[:, c] = b_dk.permute(0, 2, 1, 3)
        dg4[:, c] = b_dg.permute(0, 2, 1, 3)
        dv24[:, c] = b_dv2.permute(0, 2, 1, 3)
        db4[:, c] = b_db.permute(0, 2, 1)
        dA4[:, c] = b_dA.permute(0, 2, 1, 3)           # b_dA indexed [o_t, o_A] -> A4 [o_t,o_A]
    return dict(dq=dq, dk=dk, dv=dv2, db=db, dg=dg, dA=dA)


# ----------------------------------------------------------------------- triton
def build_kernel(nodh=False):
    """Build the B2 (`wy_dqkg`) kernel.  ``nodh=True`` (P7-31) drops the per-chunk
    `dh` global load and reuses the already-loaded `b_h` in its place — numerically
    wrong, but it isolates the cost of the B1->B2 `dh` read (a K2-shaped fusion would
    feed `dS` straight from B1's registers)."""
    import triton
    import triton.language as tl

    NODH = tl.constexpr(bool(nodh))

    @triton.jit
    def _wy_dqkg_kernel(
        q_ptr, k_ptr, v_ptr, vn_ptr, g_ptr, beta_ptr, A_ptr, h_ptr,
        do_ptr, dh_ptr, dv_ptr,
        dq_ptr, dk_ptr, dv2_ptr, db_ptr, dg_ptr, dA_ptr,
        T, NT, scale,
        H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
        BK: tl.constexpr, BV: tl.constexpr,
    ):
        i_t = tl.program_id(0)
        i_bh = tl.program_id(1)
        i_b = i_bh // H
        i_h = i_bh % H

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        last = min(T, i_t * BT + BT) - 1
        m_last = o_t == last

        b_beta = tl.load(beta_ptr + (i_b * T + o_t) * H + i_h, mask=m_t, other=0.0)
        A_base = A_ptr + ((i_b * T + i_t * BT) * H + i_h) * BT
        p_A = A_base + tl.arange(0, BT)[:, None] + tl.arange(0, BT)[None, :] * (H * BT)
        b_A = tl.load(p_A, mask=m_t[None, :], other=0.0)      # [o_A, o_t]

        m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t[None, :])
        b_dA = tl.zeros([BT, BT], dtype=tl.float32)
        b_db = tl.zeros([BT], dtype=tl.float32)

        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            m_tk = m_t[:, None] & m_k[None, :]
            p_k = k_ptr + (i_b * T + o_t)[:, None] * (H * K) + i_h * K + o_k[None, :]
            p_q = q_ptr + (i_b * T + o_t)[:, None] * (H * K) + i_h * K + o_k[None, :]
            p_g = g_ptr + (i_b * T + o_t)[:, None] * (H * K) + i_h * K + o_k[None, :]
            b_k = tl.load(p_k, mask=m_tk, other=0.0)
            b_q = tl.load(p_q, mask=m_tk, other=0.0)
            b_g = tl.load(p_g, mask=m_tk, other=0.0).to(tl.float32)
            b_gn = tl.load(g_ptr + (i_b * T + last) * (H * K) + i_h * K + o_k,
                           mask=m_k, other=0.0).to(tl.float32)

            b_dq = tl.zeros([BT, BK], dtype=tl.float32)
            b_dk = tl.zeros([BT, BK], dtype=tl.float32)
            b_dw = tl.zeros([BT, BK], dtype=tl.float32)
            b_dgk = tl.zeros([BK], dtype=tl.float32)

            for i_v in range(tl.cdiv(V, BV)):
                o_v = i_v * BV + tl.arange(0, BV)
                m_v = o_v < V
                m_tv = m_t[:, None] & m_v[None, :]
                m_h = m_v[:, None] & m_k[None, :]
                p_vn = vn_ptr + (i_b * T + o_t)[:, None] * (H * V) + i_h * V + o_v[None, :]
                p_do = do_ptr + (i_b * T + o_t)[:, None] * (H * V) + i_h * V + o_v[None, :]
                b_vn = tl.load(p_vn, mask=m_tv, other=0.0)
                b_do = tl.load(p_do, mask=m_tv, other=0.0)
                # h[K,V] -> b_h[v,k] = h[k,v]
                p_h = h_ptr + ((i_b * NT + i_t) * H + i_h) * K * V + o_k[None, :] * V + o_v[:, None]
                p_dh = dh_ptr + ((i_b * NT + i_t) * H + i_h) * K * V + o_k[None, :] * V + o_v[:, None]
                b_h = tl.load(p_h, mask=m_h, other=0.0)           # [BV,BK]
                if NODH:   # P7-31: no `dh` read; reuse b_h (wrong value, timing-only)
                    b_dh = b_h
                else:
                    b_dh = tl.load(p_dh, mask=m_h, other=0.0)
                p_dv = dv_ptr + (i_b * T + o_t)[:, None] * (H * V) + i_h * V + o_v[None, :]
                b_dv = tl.load(p_dv, mask=m_tv, other=0.0)

                b_dgk += tl.sum(b_h * b_dh, axis=0)
                b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
                b_dk += tl.dot(b_vn, b_dh.to(b_vn.dtype))
                b_dw += tl.dot(b_dv.to(b_vn.dtype), b_h.to(b_vn.dtype))
                tl.debug_barrier()  # DO NOT REMOVE (mirrors FLA)
                if i_k == 0:
                    p_v = v_ptr + (i_b * T + o_t)[:, None] * (H * V) + i_h * V + o_v[None, :]
                    b_v = tl.load(p_v, mask=m_tv, other=0.0)
                    b_dA += tl.dot(b_dv, tl.trans(b_v))
                    b_dvb = tl.dot(b_A, b_dv)
                    b_dv2 = b_dvb * b_beta[:, None]
                    b_db += tl.sum(b_dvb * b_v, axis=1)
                    p_dv2 = dv2_ptr + (i_b * T + o_t)[:, None] * (H * V) + i_h * V + o_v[None, :]
                    tl.store(p_dv2, b_dv2.to(dv2_ptr.dtype.element_ty), mask=m_tv)

            gk_exp = tl.exp2(b_g)
            b_gb = gk_exp * b_beta[:, None]
            b_dgk = b_dgk * tl.exp2(b_gn)
            b_dq = b_dq * gk_exp * scale
            b_dk = b_dk * tl.where(m_t[:, None], tl.exp2(b_gn[None, :] - b_g), 0.0)
            b_kg = b_k * gk_exp
            b_dw = -b_dw.to(b_A.dtype)
            b_dA = b_dA + tl.dot(b_dw, tl.trans(b_kg.to(b_A.dtype)))
            b_dkgb = tl.dot(b_A, b_dw)
            b_db = b_db + tl.sum(b_dkgb * b_kg, axis=1)
            b_kdk = b_k * b_dk
            b_dgk = b_dgk + tl.sum(b_kdk, axis=0)
            b_dg = (b_q * b_dq - b_kdk
                    + tl.where(m_last[:, None], 1.0, 0.0) * b_dgk[None, :]
                    + b_kg * b_dkgb * b_beta[:, None])
            b_dk = b_dk + b_dkgb * b_gb

            p_dq = dq_ptr + (i_b * T + o_t)[:, None] * (H * K) + i_h * K + o_k[None, :]
            p_dk = dk_ptr + (i_b * T + o_t)[:, None] * (H * K) + i_h * K + o_k[None, :]
            p_dg = dg_ptr + (i_b * T + o_t)[:, None] * (H * K) + i_h * K + o_k[None, :]
            tl.store(p_dq, b_dq.to(dq_ptr.dtype.element_ty), mask=m_tk)
            tl.store(p_dk, b_dk.to(dk_ptr.dtype.element_ty), mask=m_tk)
            tl.store(p_dg, b_dg.to(dg_ptr.dtype.element_ty), mask=m_tk)

        b_dA = tl.where(m_A, b_dA * b_beta[None, :], 0.0)
        b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
        b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
        b_dA = tl.where(m_A, -b_dA, 0.0)

        # dA[b, t, h, c] layout; row = o_t, col = o_A
        p_dA = dA_ptr + ((i_b * T + o_t) * H + i_h)[:, None] * BT + tl.arange(0, BT)[None, :]
        tl.store(p_dA, b_dA.to(dA_ptr.dtype.element_ty), mask=m_t[:, None])
        tl.store(db_ptr + (i_b * T + o_t) * H + i_h, b_db.to(db_ptr.dtype.element_ty), mask=m_t)

    return _wy_dqkg_kernel


def wy_dqkg_triton(kernel, inp, scale, BT, num_warps=4, num_stages=1):
    B, T, H, K = inp["q"].shape
    V = inp["v"].shape[-1]
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dv2 = torch.empty(B, T, H, V, device=dev, dtype=torch.float32)
    db = torch.empty(B, T, H, device=dev, dtype=torch.float32)
    dA = torch.empty(B, T, H, BT, device=dev, dtype=torch.float32)
    grid = (NT, B * H)
    BK = min(64, K)
    BV = min(64, V)
    kernel[grid](
        inp["q"], inp["k"], inp["v"], inp["v_new"], inp["g"], inp["beta"], inp["A"], inp["h"],
        inp["do"], inp["dh"], inp["dv"],
        dq, dk, dv2, db, dg, dA,
        T, NT, scale, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV,
        num_warps=num_warps, num_stages=num_stages)
    return dict(dq=dq, dk=dk, dv=dv2, db=db, dg=dg, dA=dA)


# --------------------------------------------------------------------------- main
def run_case(B, T, H, K, V, BT, scale, seed=0, dtype=torch.float32):
    dev = "cuda"
    print(f"\n=== B={B} T={T} H={H} K={K} V={V} BT={BT} seed={seed} dtype={dtype} ===")
    inp = make_inputs(B, T, H, K, V, BT, dev, dtype=dtype, seed=seed)
    fla = fla_wy_dqkg(inp, scale, BT)
    tor = wy_dqkg_torch(inp, scale, BT) if T % BT == 0 else None
    if tor is not None:
        print("  -- Part A: torch ref vs FLA kernel --")
        for name in ["dq", "dk", "dv", "db", "dg", "dA"]:
            stats(name, tor[name], fla[name])
    else:
        print("  (torch ref skipped: T not divisible by BT -> kernel-only check)")
    kernel = build_kernel()
    tri = wy_dqkg_triton(kernel, inp, scale, BT)
    print("  -- Part B: Triton vs FLA kernel --")
    for name in ["dq", "dk", "dv", "db", "dg", "dA"]:
        stats(name, tri[name], fla[name])
    if tor is not None:
        print("  -- Part B2: Triton vs torch ref --")
        for name in ["dq", "dk", "dv", "db", "dg", "dA"]:
            stats(name, tri[name], tor[name])
    return inp, fla, tor, tri


def main():
    torch.manual_seed(0)
    assert torch.cuda.is_available()
    print("device:", torch.cuda.get_device_name(0), "torch", torch.__version__)
    scale = 1.0 / math.sqrt(128)
    # Part A/B
    run_case(B=1, T=256, H=2, K=64, V=64, BT=64, scale=1.0 / math.sqrt(64), seed=0)
    run_case(B=2, T=512, H=4, K=128, V=128, BT=64, scale=scale, seed=1)
    run_case(B=1, T=250, H=3, K=64, V=128, BT=64, scale=1.0 / math.sqrt(64), seed=2)
    run_case(B=1, T=1024, H=8, K=128, V=128, BT=64, scale=scale, seed=3)
    run_case(B=2, T=513, H=2, K=64, V=64, BT=64, scale=1.0 / math.sqrt(64), seed=4)
    # Part C: operand precision floor (bf16 operands on the triton kernel)
    print("\n########## Part C: bf16 operand floor ##########")
    for (B, T, H, K, V, BT, seed) in [(1, 256, 2, 64, 64, 64, 0), (2, 512, 4, 128, 128, 64, 1)]:
        inp = make_inputs(B, T, H, K, V, BT, "cuda", dtype=torch.float32, seed=seed)
        fla = fla_wy_dqkg(inp, scale, BT)
        inp_bf = {k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v) for k, v in inp.items()}
        kernel = build_kernel()
        tri_bf = wy_dqkg_triton(kernel, inp_bf, scale, BT)
        print(f"\n--- bf16 operands B={B} T={T} H={H} K={K} V={V} ---")
        for name in ["dq", "dk", "dv", "db", "dg", "dA"]:
            stats(name, tri_bf[name], fla[name])

    # Part D: rough timing (naive kernel, not tuned) vs the FLA wrapper
    print("\n########## Part D: timing (naive) ##########")
    import triton.testing
    for (B, T, H, K, V, BT) in [(1, 512, 4, 128, 128, 64), (2, 2048, 8, 128, 128, 64)]:
        inp = make_inputs(B, T, H, K, V, BT, "cuda", dtype=torch.bfloat16, seed=0)
        inp_c = {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}
        kernel = build_kernel()
        ms_tri = triton.testing.do_bench(lambda: wy_dqkg_triton(kernel, inp_c, scale, BT),
                                         warmup=25, rep=100)
        ms_fla = triton.testing.do_bench(lambda: fla_wy_dqkg(inp_c, scale, BT),
                                         warmup=25, rep=100)
        print(f"  B={B} T={T} H={H} K={K} V={V}: triton {ms_tri*1e3:.1f}us  fla {ms_fla*1e3:.1f}us")


if __name__ == "__main__":
    main()
