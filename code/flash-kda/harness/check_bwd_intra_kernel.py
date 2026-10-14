"""P6-5 evidence: a minimal runnable B3 (`intra`) backward kernel.

P6-2 landed B1 (cross-chunk *state adjoint*, FLA `dhu`); P6-4 landed B2 (`wy_dqkg`,
the per-chunk `dq/dk/dv/db/dg/dA`).  The last GEMM piece of the target-card backward
design (docs/06 §5.2, §6.7) is **B3 `bwd_intra`**: given the per-chunk block matrices
``dAqk`` and ``dAkk`` (produced by `dAv` + `wy_dqkg`), back-propagate the *intra-chunk*
attention matrices into ``dq, dk, dbeta, dg``.

It mirrors FLA's ``chunk_kda_bwd_kernel_intra`` (fla/ops/kda/chunk_intra.py:395-667),
the non-``SAFE_GATE`` branch — which is the default (``chunk.py:47`` safe_gate=False) and
the direct, transparent form.  We validate three ways, exactly like P6-4:

  Part A  torch ``intra_torch`` (per chunk, ieee fp32)   vs  FLA Triton kernel.
  Part B  Triton ``_intra_kernel``  vs  FLA Triton kernel (expect bit-exact, same tiling).
  Part C  operand-precision floor: bf16 operands through the Triton kernel.

Notation (docs/05 §7, docs/06 §3.4), per chunk `c`, `t` = intra-chunk row:
    q, k            [BT,K]   raw (undecayed)
    g               [BT,K]   chunk-local log2 cumsum (FLA `g_cumsum`)
    beta            [BT]     write gate
    dAqk            [BT,BT]  = tril(\bar M_qk * scale)      (from `dAv`)
    dAkk            [BT,BT]  = strictly-lower A-matrix adjoint (from `wy_dqkg`)
Outputs: dq, dk [BT,K], db [BT], dg [BT,K] (additive).

Run inside `kernel_lab`:
    python ~/proj/tech_record/code/flash-kda/harness/check_bwd_intra_kernel.py
"""
import math
import os
import sys

# FLA >=0.5 may auto-select a backend via @dispatch('kda'); docs/06 reads the Triton
# reference, so pin the dispatch to it here.
os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

sys.path.insert(0, os.path.expanduser("~/proj/tech_record/code/flash-kda/src"))


def stats(name, a, b):
    d = (a.float() - b.float()).abs()
    denom = b.float().abs().max().item()
    max_abs = d.max().item()
    max_rel = max_abs / (denom + 1e-12)
    print(f"  {name:22s} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  ref_max={denom:.3e}")
    return max_abs, max_rel


def cdiv(a, b):
    return (a + b - 1) // b


# --------------------------------------------------------------------------- data
def make_inputs(B, T, H, K, BT, BC, device, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    NT = cdiv(T, BT)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=device), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=device), dim=-1)
    # chunk-local log2 cumsum of a negative per-channel gate, <= 0 (built padded, then sliced)
    gpad = -torch.rand(B, NT * BT, H, K, device=device) * 0.5
    gpad[:, :T] = -torch.rand(B, T, H, K, device=device) * 0.5
    g = gpad.view(B, NT, BT, H, K).cumsum(dim=2).reshape(B, NT * BT, H, K)[:, :T].contiguous()
    beta = torch.sigmoid(torch.randn(B, T, H, device=device))
    # dAqk = tril (incl diag), dAkk = strict lower, matching what dAv/wy_dqkg emit
    o_t = torch.arange(BT, device=device)
    low_incl = (o_t[:, None] >= o_t[None, :]).float()
    low_strict = (o_t[:, None] > o_t[None, :]).float()
    dAqk = torch.randn(B, NT, H, BT, BT, device=device) * low_incl
    dAkk = torch.randn(B, NT, H, BT, BT, device=device) * low_strict
    dAqk = dAqk.permute(0, 1, 3, 2, 4).reshape(B, NT * BT, H, BT)[:, :T].contiguous()
    dAkk = dAkk.permute(0, 1, 3, 2, 4).reshape(B, NT * BT, H, BT)[:, :T].contiguous()
    to = lambda x: x.to(dtype)
    return dict(q=to(q), k=to(k), g=to(g), beta=to(beta), dAqk=to(dAqk), dAkk=to(dAkk))


# --------------------------------------------------------------------- FLA kernel
def fla_intra(inp, BT, BC, dtype=None):
    from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra
    B, T, H, K = inp["q"].shape
    dev = inp["q"].device
    dt = dtype or inp["q"].dtype
    dq = torch.zeros(B, T, H, K, device=dev, dtype=dt)
    dk = torch.zeros(B, T, H, K, device=dev, dtype=dt)
    db = torch.zeros(B, T, H, device=dev, dtype=dt)
    dg = torch.zeros(B, T, H, K, device=dev, dtype=dt)
    out = chunk_kda_bwd_intra(
        q=inp["q"], k=inp["k"], g=inp["g"], beta=inp["beta"],
        dAqk=inp["dAqk"], dAkk=inp["dAkk"],
        dq=dq, dk=dk, db=db, dg=dg,
        chunk_size=BT, safe_gate=False)
    dq, dk, db, dg = out
    return dict(dq=dq, dk=dk, db=db, dg=dg)


# ------------------------------------------------------------------ torch golden
def intra_torch(inp, BT, BC):
    """Per-chunk torch transcription of the non-SAFE_GATE kernel (BK=K), index-faithful.

    Layout: dAqk/dAkk are [B,T,H,BT], the last axis the *chunk-local* key index; the
    row axis is the query token.  Part A/B read it in that orientation; Parts C/D
    (the ``dk`` branch) read it transposed, exactly as the kernel's pointer arithmetic
    does (``chunk_intra.py:554`` / ``629``).
    """
    qf, kf, gf, bf = (inp["q"].float(), inp["k"].float(), inp["g"].float(), inp["beta"].float())
    Aqf, Akf = inp["dAqk"].float(), inp["dAkk"].float()
    B, T, H, K = qf.shape
    NT = cdiv(T, BT)
    NC = cdiv(BT, BC)
    dev = qf.device

    q4 = qf.view(B, NT, BT, H, K)
    k4 = kf.view(B, NT, BT, H, K)
    g4 = gf.view(B, NT, BT, H, K)
    b4 = bf.view(B, NT, BT, H)
    Aq4 = Aqf.view(B, NT, BT, H, BT)          # [b,c,query_local,h,key_local]
    Ak4 = Akf.view(B, NT, BT, H, BT)

    dq = torch.zeros(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.zeros(B, T, H, K, device=dev, dtype=torch.float32)
    db = torch.zeros(B, T, H, device=dev, dtype=torch.float32)
    dg = torch.zeros(B, T, H, K, device=dev, dtype=torch.float32)
    dq4, dk4, db4, dg4 = (dq.view(B, NT, BT, H, K), dk.view(B, NT, BT, H, K),
                          db.view(B, NT, BT, H), dg.view(B, NT, BT, H, K))

    ar = lambda n: torch.arange(n, device=dev)

    for c in range(NT):
        chunk_T = min(BT, T - c * BT)
        for i_i in range(NC):
            i_ti = i_i * BC
            if i_ti >= chunk_T:
                continue
            bc = min(BC, chunk_T - i_ti)
            oc = i_ti + ar(BC)                                   # query local rows
            m_c = oc < chunk_T
            # [B,H,..] views
            b_g = g4[:, c][:, oc].permute(0, 2, 1, 3)            # [B,H,BC,K]
            b_k = k4[:, c][:, oc].permute(0, 2, 1, 3)            # [B,H,BC,K]
            b_q = q4[:, c][:, oc].permute(0, 2, 1, 3)
            b_b = b4[:, c][:, oc].permute(0, 2, 1)               # [B,H,BC]
            b_Aq = Aq4[:, c].permute(0, 2, 1, 3)                 # [B,H,BT(query),BT(key_local)]
            b_Ak = Ak4[:, c].permute(0, 2, 1, 3)

            b_dq2 = torch.zeros(B, H, BC, K, device=dev, dtype=torch.float32)
            b_dk2 = torch.zeros(B, H, BC, K, device=dev, dtype=torch.float32)

            # ---- Part A: full key sub-blocks strictly before i_i ----
            if i_i > 0:
                b_gn = g4[:, c][:, i_ti]                          # [B,H,K]  (first row of sub-block)
                for i_j in range(i_i):
                    o_j = slice(i_j * BC, (i_j + 1) * BC)
                    b_kj = k4[:, c][:, o_j].permute(0, 2, 1, 3)  # [B,H,BC,K]
                    b_gkj = g4[:, c][:, o_j].permute(0, 2, 1, 3)
                    b_kg = b_kj * torch.exp2(b_gn.unsqueeze(2) - b_gkj)
                    b_dAqk = b_Aq[:, :, oc][:, :, :, o_j]        # [B,H,BC,BC]
                    b_dAkk = b_Ak[:, :, oc][:, :, :, o_j]
                    b_dq2 += torch.einsum("bhqk,bhkl->bhql", b_dAqk, b_kg)
                    b_dk2 += torch.einsum("bhqk,bhkl->bhql", b_dAkk, b_kg)
                b_gqn = torch.exp2(b_g - b_gn.unsqueeze(2))      # [B,H,BC,K]
                b_dq2 = b_dq2 * b_gqn
                b_dk2 = b_dk2 * b_gqn

            # ---- Part B: diagonal sub-block ----
            for j in range(bc):
                b_dAqk = b_Aq[:, :, oc, i_ti + j]                # [B,H,BC]
                b_dAkk = b_Ak[:, :, oc, i_ti + j]
                b_kj = k4[:, c][:, i_ti + j]                      # [B,H,K]
                b_gkj = g4[:, c][:, i_ti + j]
                m_i = (ar(BC) >= j).view(1, 1, BC, 1).float()
                b_gqk = torch.exp2(b_g - b_gkj.unsqueeze(2))     # [B,H,BC,K]
                b_dq2 += m_i * b_dAqk.unsqueeze(-1) * b_kj.unsqueeze(2) * b_gqk
                b_dk2 += m_i * b_dAkk.unsqueeze(-1) * b_kj.unsqueeze(2) * b_gqk

            b_db = (b_dk2 * b_k).sum(dim=-1)                     # [B,H,BC]
            b_dk2 = b_dk2 * b_b.unsqueeze(-1)
            b_dg2 = b_q * b_dq2                                  # \bar g += \bar q^dec \odot q (eq 7.10)

            # store dq / db (rows oc only)
            dq4[:, c][:, oc] = (b_dq2 + dq4[:, c][:, oc].permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            db4[:, c][:, oc] = (b_db + db4[:, c][:, oc].permute(0, 2, 1)).permute(0, 2, 1)

            # ---- Part C: dk branch from query sub-blocks strictly after i_i ----
            b_dkt = torch.zeros(B, H, BC, K, device=dev, dtype=torch.float32)
            NC_eff = min(NC, cdiv(chunk_T, BC))
            if i_i < NC_eff - 1:
                b_gn = g4[:, c][:, min(i_ti + BC, chunk_T) - 1]        # [B,H,K]
                for i_j in range(i_i + 1, NC_eff):
                    o_j = slice(i_j * BC, min((i_j + 1) * BC, chunk_T))
                    n_j = o_j.stop - o_j.start
                    o_jj = i_j * BC + ar(n_j)
                    b_bj = b4[:, c][:, o_jj].permute(0, 2, 1)            # [B,H,n_j]
                    b_qj = q4[:, c][:, o_jj].permute(0, 2, 1, 3)         # [B,H,n_j,K]
                    b_kbj = k4[:, c][:, o_jj].permute(0, 2, 1, 3) * b_bj.unsqueeze(-1)
                    b_gkj = g4[:, c][:, o_jj].permute(0, 2, 1, 3)
                    # b_dAqk[a, o_j] = Aq[query=o_j, key_local=i_ti+a]
                    b_dAqk = b_Aq[:, :, o_jj, i_ti:i_ti + BC].transpose(-1, -2)  # [B,H,BC,n_j]
                    b_dAkk = b_Ak[:, :, o_jj, i_ti:i_ti + BC].transpose(-1, -2)
                    b_gkn = torch.exp2(b_gkj - b_gn.unsqueeze(2))        # [B,H,n_j,K]
                    b_qg = b_qj * b_gkn
                    b_kbg = b_kbj * b_gkn
                    b_dkt += torch.einsum("bhaj,bhjk->bhak", b_dAqk, b_qg)
                    b_dkt += torch.einsum("bhaj,bhjk->bhak", b_dAkk, b_kbg)
                b_dkt = b_dkt * torch.exp2(b_gn.unsqueeze(2) - b_g)

            # ---- Part D: diagonal dk branch ----
            for j in range(bc):
                # b_dAqk[a] = Aq[query=i_ti+j, key_local=i_ti+a]
                b_dAqk = b_Aq[:, :, i_ti + j, i_ti:i_ti + BC]         # [B,H,BC]
                b_dAkk = b_Ak[:, :, i_ti + j, i_ti:i_ti + BC]
                b_qj = q4[:, c][:, i_ti + j]                           # [B,H,K]
                b_bj = b4[:, c][:, i_ti + j]                           # [B,H]
                b_kbj = k4[:, c][:, i_ti + j] * b_bj.unsqueeze(-1)
                b_gkj = g4[:, c][:, i_ti + j]
                m_i = (ar(BC) <= j).view(1, 1, BC, 1).float()
                b_gkq = torch.exp2(b_gkj.unsqueeze(2) - b_g)          # [B,H,BC,K]
                b_dkt += m_i * b_dAqk.unsqueeze(-1) * b_qj.unsqueeze(2) * b_gkq
                b_dkt += m_i * b_dAkk.unsqueeze(-1) * b_kbj.unsqueeze(2) * b_gkq

            b_dg2 = b_dg2 + (b_dk2 - b_dkt) * b_k + dg4[:, c][:, oc].permute(0, 2, 1, 3)
            b_dk2 = b_dk2 + dk4[:, c][:, oc].permute(0, 2, 1, 3) + b_dkt
            dk4[:, c][:, oc] = b_dk2.permute(0, 2, 1, 3)
            dg4[:, c][:, oc] = b_dg2.permute(0, 2, 1, 3)

    return dict(dq=dq, dk=dk, db=db, dg=dg)


# ----------------------------------------------------------------------- triton
def build_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _intra_kernel(
        q_ptr, k_ptr, g_ptr, beta_ptr, dAqk_ptr, dAkk_ptr,
        dq_ptr, dk_ptr, db_ptr, dg_ptr,
        T,
        H: tl.constexpr, K: tl.constexpr,
        BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr,
    ):
        i_kc = tl.program_id(0)
        i_t = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_b = i_bh // H
        i_h = i_bh % H
        i_k = i_kc // NC
        i_i = i_kc % NC

        i_ti = i_t * BT + i_i * BC
        if i_ti >= T:
            return
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        o_i = tl.arange(0, BC)
        o_c = i_ti + o_i
        m_c = o_c < T
        m_ck = m_c[:, None] & m_k[None, :]

        # ---- load query sub-block [BC, BK] ----
        p_q = q_ptr + (i_b * T + o_c)[:, None] * (H * K) + i_h * K + o_k[None, :]
        p_k = k_ptr + (i_b * T + o_c)[:, None] * (H * K) + i_h * K + o_k[None, :]
        p_g = g_ptr + (i_b * T + o_c)[:, None] * (H * K) + i_h * K + o_k[None, :]
        b_q = tl.load(p_q, mask=m_ck, other=0.0)
        b_k = tl.load(p_k, mask=m_ck, other=0.0)
        b_g = tl.load(p_g, mask=m_ck, other=0.0).to(tl.float32)
        b_b = tl.load(beta_ptr + (i_b * T + o_c) * H + i_h, mask=m_c, other=0.0)

        b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
        b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)

        # ---- Part A: key sub-blocks fully before i_i ----
        if i_i > 0:
            p_gn = g_ptr + (i_b * T + i_ti) * (H * K) + i_h * K + o_k
            b_gn = tl.load(p_gn, mask=m_k, other=0.0).to(tl.float32)[None, :]
            for i_j in range(0, i_i):
                o_j = i_t * BT + i_j * BC + o_i
                m_jk = (o_j < T)[:, None] & m_k[None, :]
                p_kj = k_ptr + (i_b * T + o_j)[:, None] * (H * K) + i_h * K + o_k[None, :]
                p_gkj = g_ptr + (i_b * T + o_j)[:, None] * (H * K) + i_h * K + o_k[None, :]
                b_kj = tl.load(p_kj, mask=m_jk, other=0.0)
                b_gkj = tl.load(p_gkj, mask=m_jk, other=0.0).to(tl.float32)
                b_kg = b_kj * tl.exp2(b_gn - b_gkj)
                p_dAqk = dAqk_ptr + ((i_b * T + o_c)[:, None] * H + i_h) * BT + (i_j * BC + o_i)[None, :]
                p_dAkk = dAkk_ptr + ((i_b * T + o_c)[:, None] * H + i_h) * BT + (i_j * BC + o_i)[None, :]
                b_dAqk = tl.load(p_dAqk, mask=m_c[:, None], other=0.0)
                b_dAkk = tl.load(p_dAkk, mask=m_c[:, None], other=0.0)
                b_dq2 += tl.dot(b_dAqk, b_kg.to(b_dAqk.dtype))
                b_dk2 += tl.dot(b_dAkk, b_kg.to(b_dAkk.dtype))
            b_gqn = tl.exp2(b_g - b_gn)
            b_dq2 *= b_gqn
            b_dk2 *= b_gqn

        # ---- Part B: diagonal sub-block ----
        bc = min(BC, T - i_t * BT - i_i * BC)
        for j in range(0, bc):
            col = i_i * BC + j
            p_dAqk = dAqk_ptr + ((i_b * T + o_c) * H + i_h) * BT + col
            p_dAkk = dAkk_ptr + ((i_b * T + o_c) * H + i_h) * BT + col
            b_dAqk = tl.load(p_dAqk, mask=m_c, other=0.0).to(tl.float32)
            b_dAkk = tl.load(p_dAkk, mask=m_c, other=0.0).to(tl.float32)
            p_kj = k_ptr + (i_b * T + i_ti + j) * (H * K) + i_h * K + o_k
            p_gkj = g_ptr + (i_b * T + i_ti + j) * (H * K) + i_h * K + o_k
            b_kj = tl.load(p_kj, mask=m_k, other=0.0).to(tl.float32)
            b_gkj = tl.load(p_gkj, mask=m_k, other=0.0).to(tl.float32)
            m_i = o_i[:, None] >= j
            b_gqk = tl.exp2(b_g - b_gkj[None, :])
            b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kj[None, :] * b_gqk, 0.0)
            b_dk2 += tl.where(m_i, b_dAkk[:, None] * b_kj[None, :] * b_gqk, 0.0)

        b_db = tl.sum(b_dk2 * b_k, axis=1)
        b_dk2 *= b_b[:, None]

        # \bar q^dec branch feeds the decay chain: \bar g += \bar q^dec \odot q
        # (chunk_intra.py:558, docs/05 eq 7.10) -- MUST use the *fresh* dq2, before
        # folding in the upstream dq accumulator.
        b_dg2 = b_q * b_dq2
        p_dq = dq_ptr + (i_b * T + o_c)[:, None] * (H * K) + i_h * K + o_k[None, :]
        b_dq2 = b_dq2 + tl.load(p_dq, mask=m_ck, other=0.0)
        tl.store(p_dq, b_dq2.to(p_dq.dtype.element_ty), mask=m_ck)
        # db is a per-token scalar accumulated across the NK K-blocks -> atomic add
        # (FLA instead writes an [NK, ...] buffer and sums it in the wrapper)
        tl.atomic_add(db_ptr + (i_b * T + o_c) * H + i_h, b_db.to(db_ptr.dtype.element_ty), mask=m_c)

        tl.debug_barrier()
        b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

        # ---- Part C: dk branch from query sub-blocks strictly after i_i ----
        NC_eff = min(NC, tl.cdiv(T - i_t * BT, BC))
        if i_i < NC_eff - 1:
            p_gn = g_ptr + (i_b * T + min(i_ti + BC, T) - 1) * (H * K) + i_h * K + o_k
            b_gn = tl.load(p_gn, mask=m_k, other=0.0).to(tl.float32)[None, :]
            for i_j in range(i_i + 1, NC_eff):
                o_j = i_t * BT + i_j * BC + o_i
                m_j = o_j < T
                m_jk = m_j[:, None] & m_k[None, :]
                p_qj = q_ptr + (i_b * T + o_j)[:, None] * (H * K) + i_h * K + o_k[None, :]
                p_kj = k_ptr + (i_b * T + o_j)[:, None] * (H * K) + i_h * K + o_k[None, :]
                p_gkj = g_ptr + (i_b * T + o_j)[:, None] * (H * K) + i_h * K + o_k[None, :]
                b_bj = tl.load(beta_ptr + (i_b * T + o_j) * H + i_h, mask=m_j, other=0.0)
                b_qj = tl.load(p_qj, mask=m_jk, other=0.0)
                b_kbj = tl.load(p_kj, mask=m_jk, other=0.0) * b_bj[:, None]
                b_gkj = tl.load(p_gkj, mask=m_jk, other=0.0).to(tl.float32)
                # b_dAqk[a, o_j] = dAqk[query=o_j, key_local=i_ti+a]
                p_dAqk = dAqk_ptr + ((i_b * T + o_j)[None, :] * H + i_h) * BT + (i_i * BC + o_i)[:, None]
                p_dAkk = dAkk_ptr + ((i_b * T + o_j)[None, :] * H + i_h) * BT + (i_i * BC + o_i)[:, None]
                b_dAqk = tl.load(p_dAqk, mask=m_j[None, :], other=0.0)
                b_dAkk = tl.load(p_dAkk, mask=m_j[None, :], other=0.0)
                b_gkn = tl.exp2(b_gkj - b_gn)
                b_qg = b_qj * tl.where(m_j[:, None], b_gkn, 0.0)
                b_kbg = b_kbj * tl.where(m_j[:, None], b_gkn, 0.0)
                b_dkt += tl.dot(b_dAqk, b_qg.to(b_dAqk.dtype))
                b_dkt += tl.dot(b_dAkk, b_kbg.to(b_dAkk.dtype))
            b_dkt *= tl.exp2(b_gn - b_g)

        # ---- Part D: diagonal dk branch ----
        for j in range(0, bc):
            p_dAqk = dAqk_ptr + ((i_b * T + i_ti + j) * H + i_h) * BT + (i_i * BC + o_i)
            p_dAkk = dAkk_ptr + ((i_b * T + i_ti + j) * H + i_h) * BT + (i_i * BC + o_i)
            b_dAqk = tl.load(p_dAqk, mask=m_c, other=0.0).to(tl.float32)
            b_dAkk = tl.load(p_dAkk, mask=m_c, other=0.0).to(tl.float32)
            p_qj = q_ptr + (i_b * T + i_ti + j) * (H * K) + i_h * K + o_k
            p_kj = k_ptr + (i_b * T + i_ti + j) * (H * K) + i_h * K + o_k
            p_gkj = g_ptr + (i_b * T + i_ti + j) * (H * K) + i_h * K + o_k
            b_qj = tl.load(p_qj, mask=m_k, other=0.0).to(tl.float32)
            b_bj = tl.load(beta_ptr + (i_b * T + i_ti + j) * H + i_h)
            b_kbj = tl.load(p_kj, mask=m_k, other=0.0).to(tl.float32) * b_bj
            b_gkj = tl.load(p_gkj, mask=m_k, other=0.0).to(tl.float32)
            m_i = o_i[:, None] <= j
            b_gkq = tl.exp2(b_gkj[None, :] - b_g)
            b_dkt += tl.where(m_i, b_dAqk[:, None] * b_qj[None, :] * b_gkq, 0.0)
            b_dkt += tl.where(m_i, b_dAkk[:, None] * b_kbj[None, :] * b_gkq, 0.0)

        p_dk = dk_ptr + (i_b * T + o_c)[:, None] * (H * K) + i_h * K + o_k[None, :]
        p_dg = dg_ptr + (i_b * T + o_c)[:, None] * (H * K) + i_h * K + o_k[None, :]
        b_dg2 = b_dg2 + (b_dk2 - b_dkt) * b_k + tl.load(p_dg, mask=m_ck, other=0.0)
        b_dk2 = b_dk2 + tl.load(p_dk, mask=m_ck, other=0.0) + b_dkt
        tl.store(p_dk, b_dk2.to(p_dk.dtype.element_ty), mask=m_ck)
        tl.store(p_dg, b_dg2.to(p_dg.dtype.element_ty), mask=m_ck)

    return _intra_kernel


def intra_triton(kernel, inp, BT, BC, num_warps=4, num_stages=1):
    import triton
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT)
    NC = cdiv(BT, BC)
    BK = min(32, triton.next_power_of_2(K))
    NK = cdiv(K, BK)
    dev = inp["q"].device
    dq = torch.zeros(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.zeros(B, T, H, K, device=dev, dtype=torch.float32)
    db = torch.zeros(B, T, H, device=dev, dtype=torch.float32)
    dg = torch.zeros(B, T, H, K, device=dev, dtype=torch.float32)
    grid = (NK * NC, NT, B * H)
    kernel[grid](
        inp["q"], inp["k"], inp["g"], inp["beta"], inp["dAqk"], inp["dAkk"],
        dq, dk, db, dg,
        T, H=H, K=K, BT=BT, BC=BC, BK=BK, NC=NC,
        num_warps=num_warps, num_stages=num_stages)
    return dict(dq=dq, dk=dk, db=db, dg=dg)


# --------------------------------------------------------------------------- main
def run_case(B, T, H, K, BT, BC, seed=0, dtype=torch.float32):
    dev = "cuda"
    print(f"\n=== B={B} T={T} H={H} K={K} BT={BT} BC={BC} seed={seed} dtype={dtype} ===")
    inp = make_inputs(B, T, H, K, BT, BC, dev, dtype=dtype, seed=seed)
    fla = fla_intra(inp, BT, BC)
    tor = intra_torch(inp, BT, BC) if T % BT == 0 else None
    if tor is not None:
        print("  -- Part A: torch ref vs FLA kernel --")
        for name in ["dq", "dk", "db", "dg"]:
            stats(name, tor[name], fla[name])
    else:
        print("  (torch ref skipped: T not divisible by BT -> kernel-only check)")
    kernel = build_kernel()
    tri = intra_triton(kernel, inp, BT, BC)
    print("  -- Part B: Triton vs FLA kernel --")
    for name in ["dq", "dk", "db", "dg"]:
        stats(name, tri[name], fla[name])
    if tor is not None:
        print("  -- Part B2: Triton vs torch ref --")
        for name in ["dq", "dk", "db", "dg"]:
            stats(name, tri[name], tor[name])
    return inp, fla, tor, tri


def main():
    torch.manual_seed(0)
    assert torch.cuda.is_available()
    print("device:", torch.cuda.get_device_name(0), "torch", torch.__version__)
    run_case(B=1, T=256, H=2, K=64, BT=64, BC=16, seed=0)
    run_case(B=2, T=512, H=4, K=128, BT=64, BC=16, seed=1)
    run_case(B=1, T=250, H=3, K=64, BT=64, BC=16, seed=2)
    run_case(B=1, T=1024, H=8, K=128, BT=64, BC=16, seed=3)
    run_case(B=2, T=513, H=2, K=64, BT=64, BC=16, seed=4)
    run_case(B=1, T=64, H=1, K=32, BT=64, BC=16, seed=5)

    print("\n########## Part C: bf16 operand floor ##########")
    for (B, T, H, K, BT, seed) in [(1, 256, 2, 64, 64, 0), (2, 512, 4, 128, 64, 1)]:
        inp = make_inputs(B, T, H, K, BT, 16, "cuda", dtype=torch.float32, seed=seed)
        fla = fla_intra(inp, BT, 16)
        inp_bf = {k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v) for k, v in inp.items()}
        kernel = build_kernel()
        tri_bf = intra_triton(kernel, inp_bf, BT, 16)
        print(f"\n--- bf16 operands B={B} T={T} H={H} K={K} ---")
        for name in ["dq", "dk", "db", "dg"]:
            stats(name, tri_bf[name], fla[name])

    print("\n########## Part D: timing (naive) ##########")
    import triton.testing
    for (B, T, H, K, BT) in [(1, 512, 4, 128, 64), (2, 2048, 8, 128, 64)]:
        inp = make_inputs(B, T, H, K, BT, 16, "cuda", dtype=torch.float32, seed=0)
        inp_c = {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}
        kernel = build_kernel()
        ms_tri = triton.testing.do_bench(lambda: intra_triton(kernel, inp_c, BT, 16),
                                         warmup=25, rep=100)
        ms_fla = triton.testing.do_bench(lambda: fla_intra(inp_c, BT, 16),
                                         warmup=25, rep=100)
        print(f"  B={B} T={T} H={H} K={K}: triton {ms_tri*1e3:.1f}us  fla {ms_fla*1e3:.1f}us")


if __name__ == "__main__":
    main()
