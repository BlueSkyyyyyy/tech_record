"""P6-3 evidence: close the P6-2 "non-divisible T" gap for the B1 state-adjoint kernel.

`docs/06` §6.2 noted that the P6-2 B1 prototype was only validated on T divisible by
BT and left *non-divisible T* as 待验证: the wrappers reshaped the per-token tensors
with `.view(B, NT, BT, H, K)`, which silently assumes T % BT == 0.

What actually breaks (and what does not) for non-divisible T:

  * FLA's forward intermediates (`qg/kg/w`, `g_cum`) are returned **unpadded**
    `[B, T, H, *]` (verified: T=250 -> size 250*H*K), and FLA's own `dhu` handles
    non-divisible T by masking.  So FLA is a valid reference here.
  * The *Triton kernel itself* is already T-general: it masks every token load with
    `m_t = o_t < T`, so the padded rows contribute 0 to every `tl.dot`.  The only
    thing that was T-divisible-specific is the **host-side `etot` bookkeeping**:
    `dhu_triton`/`dhu_torch` derived `etot[c] = 2^{g_cum[:, (c+1)*BT-1]}` via the
    `.view` reshape.
  * Fix: the per-chunk total decay `E_tot` is `2^{g_cum}` at the **last valid token**
    of each chunk, i.e. row `min((c+1)*BT, T) - 1`; the partial chunk's padded rows
    are simply dropped (never indexed).  `g_cum` is chunk-local, so the last valid
    row already carries the summed decay over exactly the valid rows.

This script implements the general `etot` gather and re-runs the three-way check
(torch explicit recurrence / Triton kernel / FLA `dhu`) on non-divisible T, plus a
sanitity rerun on a divisible T to show the fix is a no-op there.

Run inside `kernel_lab`:
    python ~/proj/tech_record/code/flash-kda/harness/check_bwd_dhu_boundary.py
"""
import os
import math
import sys

import torch

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
sys.path.insert(0, HARNESS)
sys.path.insert(0, os.path.expanduser("~/proj/tech_record/code/flash-kda/src"))

import check_bwd_dhu_kernel as M  # reuse make_inputs / fla_forward / fla_dhu


def cdiv(a, b):
    return (a + b - 1) // b


def chunk_etot(g_cum, T, BT):
    """Per-chunk total decay `E_tot = 2^{g_cum[last valid token of chunk]}`.

    Works for T divisible or not.  `g_cum` is [B,T,HV,K] and chunk-local, so the
    last *valid* row of a chunk already equals the sum of that chunk's gate logits.
    Returns [B,NT,HV,K] fp32.
    """
    B, Tc, HV, K = g_cum.shape
    assert Tc == T
    NT = cdiv(T, BT)
    idx = torch.clamp((torch.arange(NT, device=g_cum.device) + 1) * BT - 1, max=T - 1)
    return torch.exp2(g_cum.float()[:, idx])                     # [B,NT,HV,K]


# --------------------------------------------------------------- torch golden (general T)
def dhu_torch_nd(fi, do, dv, h0, scale, BT, dtype=torch.float32):
    qg, kg, w, g_cum = fi["qg"], fi["kg"], fi["w"], fi["g_cum"]
    B, T, HV, K = qg.shape
    V = do.shape[-1]
    NT = cdiv(T, BT)
    etot = chunk_etot(g_cum, T, BT)                              # [B,NT,HV,K]

    def pad(x):
        """[B,T,HV,D] -> [B,NT,BT,HV,D], zero pad the tail of the last chunk."""
        D = x.shape[-1]
        out = torch.zeros(B, NT * BT, HV, D, dtype=x.dtype, device=x.device)
        out[:, :T] = x
        return out.view(B, NT, BT, HV, D)

    qdec = pad(qg.float())
    krest = pad(kg.float())
    wc = pad(w.float())
    doc = pad(do.float())
    dvc = pad(dv.float())

    dh = torch.zeros(B, NT, HV, K, V, dtype=torch.float32, device=qg.device)
    dv2 = torch.zeros_like(doc)
    dS = torch.zeros(B, HV, K, V, dtype=torch.float32, device=qg.device)
    for c in range(NT - 1, -1, -1):
        dh[:, c] = dS
        dU = torch.einsum("bthk,bhkv->bthv", krest[:, c], dS) + dvc[:, c]
        dv2[:, c] = dU
        dS = (etot[:, c, :, None] * dS
              + torch.einsum("bthk,bthv->bhkv", qdec[:, c], doc[:, c]) * scale
              - torch.einsum("bthk,bthv->bhkv", wc[:, c], dU))
    return dh, dS, dv2.view(B, NT * BT, HV, V)[:, :T].contiguous()


# ------------------------------------------------------------------- triton kernel
def build_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _dhu_kernel(
        q_ptr, k_ptr, w_ptr, do_ptr, dv_ptr, etot_ptr,
        dh_ptr, dh0_ptr, dv2_ptr,
        T, NT, scale,
        HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
        BT: tl.constexpr, BV: tl.constexpr,
    ):
        i_v = tl.program_id(0)
        i_nh = tl.program_id(1)
        i_n = i_nh // HV
        i_h = i_nh % HV
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        o_k = tl.arange(0, K)
        b_dh = tl.zeros([K, BV], dtype=tl.float32)
        for c in range(NT - 1, -1, -1):
            p_dh = dh_ptr + ((i_n * NT + c) * HV + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
            tl.store(p_dh, b_dh.to(dh_ptr.dtype.element_ty), mask=m_v[None, :])
            o_t = c * BT + tl.arange(0, BT)
            m_t = o_t < T
            base = (i_n * T + o_t)[:, None]
            p_k = k_ptr + base * (HV * K) + i_h * K + o_k[None, :]
            p_q = q_ptr + base * (HV * K) + i_h * K + o_k[None, :]
            p_w = w_ptr + base * (HV * K) + i_h * K + o_k[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None], other=0.0)
            b_q = tl.load(p_q, mask=m_t[:, None], other=0.0)
            b_w = tl.load(p_w, mask=m_t[:, None], other=0.0)
            p_do = do_ptr + base * (HV * V) + i_h * V + o_v[None, :]
            p_dv = dv_ptr + base * (HV * V) + i_h * V + o_v[None, :]
            m_dv = m_t[:, None] & m_v[None, :]
            b_do = tl.load(p_do, mask=m_dv, other=0.0)
            b_dv = tl.load(p_dv, mask=m_dv, other=0.0)
            p_e = etot_ptr + ((i_n * NT + c) * HV + i_h) * K + o_k
            b_e = tl.load(p_e)
            b_du = tl.dot(b_k, b_dh) + b_dv
            p_dv2 = dv2_ptr + base * (HV * V) + i_h * V + o_v[None, :]
            tl.store(p_dv2, b_du.to(dv2_ptr.dtype.element_ty), mask=m_dv)
            b_dh = (b_dh * b_e[:, None]
                    + tl.dot(tl.trans(b_q), b_do) * scale
                    - tl.dot(tl.trans(b_w), b_du))
        p_dh0 = dh0_ptr + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dh.to(dh0_ptr.dtype.element_ty), mask=m_v[None, :])

    return _dhu_kernel


def dhu_triton_nd(kernel, fi, do, dv, h0, scale, BT, BV=32):
    qg, kg, w, g_cum = fi["qg"], fi["kg"], fi["w"], fi["g_cum"]
    B, T, HV, K = qg.shape
    V = do.shape[-1]
    NT = cdiv(T, BT)
    etot = chunk_etot(g_cum, T, BT).contiguous()                 # [B,NT,HV,K]
    dh = torch.empty(B, NT, HV, K, V, dtype=qg.dtype, device=qg.device)
    dh0 = torch.empty(B, HV, K, V, dtype=torch.float32, device=qg.device)
    dv2 = torch.empty(B, T, HV, V, dtype=qg.dtype, device=qg.device)
    grid = (cdiv(V, BV), B * HV)
    kernel[grid](
        qg.contiguous(), kg.contiguous(), w.contiguous(), do.contiguous(),
        dv.contiguous(), etot, dh, dh0, dv2, T, NT, scale,
        HV=HV, K=K, V=V, BT=BT, BV=BV, num_warps=4, num_stages=1,
    )
    return dh, dh0, dv2


def run_case(B, T, H, K, V, BT, BV=32, seed=0):
    dev = "cuda"
    scale = K ** -0.5
    q, k, v, g_log2, beta, h0 = M.make_inputs(B, T, H, K, V, BT, dev, torch.float32)
    torch.manual_seed(123)
    do = torch.randn(B, T, H, V, device=dev)
    fi = M.fla_forward(q, k, v, g_log2, beta, h0, scale, BT)
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
    from fla.ops.kda.chunk_bwd import chunk_kda_bwd_dAv
    dAqk, dv = chunk_kda_bwd_dAv(q=q, k=k, v=fi["v_new"], do=do, A=fi["Aqk"],
                                 scale=scale, chunk_size=BT)
    dh, dh0, dv2 = chunk_gated_delta_rule_bwd_dhu(
        q=fi["qg"], k=fi["kg"], w=fi["w"], gk=fi["g_cum"], h0=h0, dht=None,
        do=do, dv=dv, scale=scale, chunk_size=BT)

    print("#" * 78)
    print(f"config: B={B} T={T} H={H} K={K} V={V} BT={BT} BV={BV} "
          f"NT={cdiv(T,BT)} tail={T % BT} dtype=fp32")
    kernel = build_kernel()
    dht, dh0t, dv2t = dhu_torch_nd(fi, do, dv, h0, scale, BT)
    dhk, dh0k, dv2k = dhu_triton_nd(kernel, fi, do, dv, h0, scale, BT, BV)

    print(f"  FLA dh shape={tuple(dh.shape)}  ours dh shape={tuple(dht.shape)}")
    print(f"  FLA dv2 shape={tuple(dv2.shape)}  ours dv2 shape={tuple(dv2t.shape)}")
    M.stats("dh   torch vs FLA", dht, dh.float())
    M.stats("dh   triton vs FLA", dhk, dh.float())
    M.stats("dv2  triton vs FLA", dv2k, dv2.float())
    M.stats("dh0  triton vs FLA", dh0k, dh0.float())
    print()


if __name__ == "__main__":
    assert torch.cuda.is_available(), "need GPU"
    # "before": the P6-2 wrappers used a plain `.view(B, NT, BT, H, K)`, which only
    # works when T % BT == 0.  Reproduce the crash on a non-divisible T so the gap is
    # documented, not merely asserted.
    print("=" * 78)
    print("'before' (P6-2 wrappers, `.view`-based): non-divisible T")
    print("=" * 78)
    try:
        gc = torch.zeros(1, 250, 2, 128, dtype=torch.float32, device="cuda")
        gc.view(1, 4, 64, 2, 128)
        print("  no error (unexpected)")
    except RuntimeError as e:
        print(f"  RuntimeError: {e}")
    print("  -> `.view` cannot absorb the 58-row tail; the tail's `E_tot` was never")
    print("     addressed at all.  The fix gathers `E_tot[c]` at the last *valid* row.")
    print()

    # divisible T: fix must be a no-op vs the P6-2 numbers
    run_case(1, 256, 2, 128, 128, 64)
    # non-divisible T, both BT, several tails (incl. tail=1 and tail=2)
    for (B, T, H, BT) in [(1, 250, 2, 64), (1, 100, 2, 64), (1, 250, 2, 32),
                          (2, 130, 3, 64), (1, 513, 4, 64), (1, 31, 1, 32)]:
        run_case(B, T, H, 128, 128, BT)
    print("DONE")
