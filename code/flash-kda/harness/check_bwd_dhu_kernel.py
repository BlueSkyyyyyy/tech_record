"""P6-2 evidence: a minimal runnable state-adjoint (B1 / FLA `dhu`) backward kernel.

Writes a small Triton kernel that reproduces the *cross-chunk state-adjoint* pass of the
KDA backward (docs/06 §3.2, docs/05 §7.3) and validates it three ways on the GPU:

  Part A  torch `dhu_torch`  vs  FLA `chunk_gated_delta_rule_bwd_dhu` (the real `dhu`).
  Part B  Triton `_dhu_kernel`  vs  FLA `dhu`  (and vs `dhu_torch`).
  Part C  controlled ablations: (i) bf16 *carry* of the running `dS`; (ii) bf16 *storage*
          of the per-chunk adjoint `dh` (what FLA actually does) -> magnitude bound for
          the `dk/dbeta` error the B2 consumer would see.

Notation (docs/05 §7, docs/06 Part A mapping, all per chunk `c`, `t` = intra-chunk row):

    q_dec   = qg        [BT,K]   (q * 2^{g_cum} * scale)
    k_rest  = kg        [BT,K]   (k * 2^{g_tot - g_cum})
    w       = P diag(beta) k_dec [BT,K]
    E_tot   = 2^{g_tot} [K]       (per key, chunk-local log2 cumsum)
    do      = upstream dL/do    [BT,V]
    dv_in   = dL/dU from the output branch (`dAv`)  [BT,V]
    dS      = adjoint of the chunk-OUTPUT state S_out  [K,V]

    dh[c] = dS                                    (saved for B2, FLA `dh`)
    dU    = k_rest @ dS + dv_in                    (FLA `dv2`)
    dS   <- E_tot[:,None] * dS
            + q_dec^T @ do * scale                 (output branch)
            - w^T @ dU                             (delta / predict branch)
    dh0   = dS after chunk 0                       (adjoint of the initial state)

Run inside `kernel_lab`:
    python ~/proj/tech_record/code/flash-kda/harness/check_bwd_dhu_kernel.py
"""
import math
import sys

import torch

sys.path.insert(0, "~/proj/tech_record/code/flash-kda/src")

RCP_LN2 = 1.4426950408889634


def stats(name, a, b):
    d = (a.float() - b.float()).abs()
    denom = b.float().abs().max().item()
    max_abs = d.max().item()
    max_rel = max_abs / (denom + 1e-12)
    print(f"  {name:34s} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  "
          f"ref_max={denom:.3e}")
    return max_abs, max_rel


# ----------------------------------------------------------------------------- data
def make_inputs(B, T, H, K, V, BT, device, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=device), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=device), dim=-1)
    v = torch.randn(B, T, H, V, device=device)
    g_log2 = -torch.rand(B, T, H, K, device=device) * 0.5
    beta = torch.sigmoid(torch.randn(B, T, H, device=device))
    h0 = torch.randn(B, H, K, V, device=device) * 0.1
    return q.to(dtype), k.to(dtype), v.to(dtype), g_log2.to(dtype), beta.to(dtype), h0.to(dtype)


def fla_forward(q, k, v, g_log2, beta, h0, scale, BT):
    """Same as check_bwd_kernel_map.fla_forward_intermediates (P6-1 Part A)."""
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
    from fla.ops.utils import chunk_local_cumsum

    g_fla = g_log2 * math.log(2.0)
    g_cum = chunk_local_cumsum(g_fla, scale=RCP_LN2, chunk_size=BT)
    w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(
        q=q, k=k, v=v, gk=g_cum, beta=beta, scale=scale, chunk_size=BT,
        disable_recompute=True)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, gk=g_cum, initial_state=h0, output_final_state=True,
        chunk_size=BT)
    return dict(g_cum=g_cum, w=w, u=u, qg=qg, kg=kg, Aqk=Aqk, Akk=Akk,
                v_new=v_new, h=h, final_state=final_state)


def fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do):
    """Run the exact sequence the KDA backward uses up to and including FLA's `dhu`."""
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
    from fla.ops.kda.chunk_bwd import chunk_kda_bwd_dAv

    fi = fla_forward(q, k, v, g_log2, beta, h0, scale, BT)
    dAqk, dv = chunk_kda_bwd_dAv(q=q, k=k, v=fi["v_new"], do=do, A=fi["Aqk"],
                                 scale=scale, chunk_size=BT)
    dh, dh0, dv2 = chunk_gated_delta_rule_bwd_dhu(
        q=fi["qg"], k=fi["kg"], w=fi["w"], gk=fi["g_cum"], h0=h0, dht=None,
        do=do, dv=dv, scale=scale, chunk_size=BT)
    return dict(fi=fi, dAqk=dAqk, dv=dv, dh=dh, dh0=dh0, dv2=dv2)


# ----------------------------------------------------------------------- torch golden
def dhu_torch(fi, do, dv, h0, scale, BT, carry_bf16=False):
    """Explicit chunk-reverse state-adjoint recurrence (the `dhu` math)."""
    qg, kg, w, g_cum = fi["qg"], fi["kg"], fi["w"], fi["g_cum"]
    B, T, HV, K = qg.shape
    V = do.shape[-1]
    NT = triton_cdiv(T, BT)
    dt = qg.dtype
    qdec = qg.view(B, NT, BT, HV, K)
    krest = kg.view(B, NT, BT, HV, K)
    wc = w.view(B, NT, BT, HV, K)
    doc = do.view(B, NT, BT, HV, V).to(dt)
    dvc = dv.view(B, NT, BT, HV, V).to(dt)
    gc = g_cum.float().view(B, NT, BT, HV, K)
    etot = torch.exp2(gc[:, :, -1])                       # [B,NT,HV,K]

    dh = torch.zeros(B, NT, HV, K, V, dtype=torch.float32, device=qg.device)
    dv2 = torch.zeros_like(doc)
    # adjoint of the chunk-OUTPUT state chain starts from the final-state gradient
    # (dht); here there is no final-state upstream, so it is zero.  ``h0`` is the
    # *initial* state and only enters B2 (detot), never this recurrence.
    dS = torch.zeros(B, HV, K, V, dtype=torch.float32, device=qg.device)

    for c in range(NT - 1, -1, -1):
        dh[:, c] = dS
        dU = torch.einsum("bthk,bhkv->bthv", krest[:, c].float(), dS) + dvc[:, c].float()
        dv2[:, c] = dU
        dS = (etot[:, c, :, None] * dS
              + torch.einsum("bthk,bthv->bhkv", qdec[:, c].float(), doc[:, c].float()) * scale
              - torch.einsum("bthk,bthv->bhkv", wc[:, c].float(), dU))
        if carry_bf16:
            dS = dS.to(torch.bfloat16).to(torch.float32)
    return dh, dS, dv2.view(B, T, HV, V)


def triton_cdiv(a, b):
    return (a + b - 1) // b


# ------------------------------------------------------------------------ triton kernel
def build_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _dhu_kernel(
        q_ptr, k_ptr, w_ptr, do_ptr, dv_ptr, etot_ptr,
        dh_ptr, dh0_ptr, dv2_ptr,
        T, NT, scale,
        HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
        BT: tl.constexpr, BV: tl.constexpr, CARRY_BF16: tl.constexpr,
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
            if CARRY_BF16:
                b_dh = b_dh.to(tl.bfloat16).to(tl.float32)
        p_dh0 = dh0_ptr + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dh.to(dh0_ptr.dtype.element_ty), mask=m_v[None, :])

    return _dhu_kernel


def dhu_triton(kernel, fi, do, dv, h0, scale, BT, BV=32, carry_bf16=False):
    qg, kg, w, g_cum = fi["qg"], fi["kg"], fi["w"], fi["g_cum"]
    B, T, HV, K = qg.shape
    V = do.shape[-1]
    NT = triton_cdiv(T, BT)
    gc = g_cum.float().view(B, NT, BT, HV, K)
    etot = torch.exp2(gc[:, :, -1]).contiguous()          # [B,NT,HV,K]
    dh = torch.empty(B, NT, HV, K, V, dtype=qg.dtype, device=qg.device)
    dh0 = torch.empty(B, HV, K, V, dtype=torch.float32, device=qg.device)
    dv2 = torch.empty_like(do)
    grid = (triton_cdiv(V, BV), B * HV)
    kernel[grid](
        qg.contiguous(), kg.contiguous(), w.contiguous(), do.contiguous(),
        dv.contiguous(), etot, dh, dh0, dv2, T, NT, scale,
        HV=HV, K=K, V=V, BT=BT, BV=BV, CARRY_BF16=carry_bf16,
        num_warps=4, num_stages=1,
    )
    return dh, dh0, dv2


# ------------------------------------------------------------------------------- parts
def part_a(fi, do, dv, h0, scale, BT):
    print("=" * 78)
    print("Part A: torch explicit dhu recurrence  vs  FLA `chunk_gated_delta_rule_bwd_dhu`")
    print("=" * 78)
    dh, dh0, dv2 = dhu_torch(fi, do, dv, h0, scale, BT)
    stats("dh   (per-chunk adjoint)", dh, FLA["dh"].float())
    stats("dv2  (dU)", dv2, FLA["dv2"].float())
    stats("dh0  (init-state grad)", dh0, FLA["dh0"].float())
    print()


def part_a2(fi, q, k, v, g_log2, beta, h0, do, dv, scale, BT):
    """Cross-check the B1 recurrence's dh0 against `src/kda_bwd_ref.py` (head 0).

    `kda_chunked_backward` carries the same chunk-to-chunk state adjoint; its returned
    ``dh0`` is the adjoint of the initial state, which is exactly the B1 output after
    the last chunk.  This ties the isolated kernel to the P5-4 reference implementation.
    """
    print("=" * 78)
    print("Part A2: B1 dh0  vs  src/kda_bwd_ref.kda_chunked_backward dh0 (head 0)")
    print("=" * 78)
    from kda_bwd_ref import kda_chunked_backward

    h = 0
    ref = kda_chunked_backward(q[0, :, h], k[0, :, h], v[0, :, h], g_log2[0, :, h],
                               beta[0, :, h], scale, do[0, :, h], h0=h0[0, h],
                               dfinal_state=None, chunk=BT)
    dh, dh0, dv2 = dhu_torch(fi, do, dv, h0, scale, BT)
    stats("dh0  B1 vs kda_bwd_ref", dh0[0, h], ref["dh0"])
    print()


def part_b(kernel, fi, do, dv, h0, scale, BT):
    print("=" * 78)
    print("Part B: Triton dhu kernel  vs  FLA `dhu`  (fp32)")
    print("=" * 78)
    dh, dh0, dv2 = dhu_triton(kernel, fi, do, dv, h0, scale, BT)
    stats("dh   vs FLA", dh, FLA["dh"].float())
    stats("dv2  vs FLA", dv2, FLA["dv2"].float())
    stats("dh0  vs FLA", dh0, FLA["dh0"].float())
    # torch golden uses full fp32 (ieee) matmul, FLA/Triton use tf32 tl.dot, so the
    # three-way gap is dominated by tf32; Triton lands *closer* to FLA (same tf32).
    dht, dh0t, dv2t = dhu_torch(fi, do, dv, h0, scale, BT)
    stats("dh   vs torch(fp32-ieee)", dh, dht)
    print("  [note] FLA dhu uses tf32 `tl.dot`; the torch golden uses ieee fp32, so")
    print("         torch-vs-FLA ~1e-3 is tf32 rounding, not a formula difference.")
    # timing
    import triton
    qg, kg, w, g_cum = fi["qg"], fi["kg"], fi["w"], fi["g_cum"]
    B, T, HV, K = qg.shape
    V = do.shape[-1]
    NT = triton_cdiv(T, BT)
    gc = g_cum.float().view(B, NT, BT, HV, K)
    etot = torch.exp2(gc[:, :, -1]).contiguous()
    dh_t = torch.empty(B, NT, HV, K, V, dtype=qg.dtype, device=qg.device)
    dh0_t = torch.empty(B, HV, K, V, dtype=torch.float32, device=qg.device)
    dv2_t = torch.empty_like(do)
    grid = (triton_cdiv(V, 32), B * HV)
    fn = lambda: kernel[grid](qg, kg, w, do, dv, etot, dh_t, dh0_t, dv2_t, T, NT, scale,
                              HV=HV, K=K, V=V, BT=BT, BV=32, CARRY_BF16=False, num_warps=4, num_stages=1)
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    print(f"  Triton dhu: B={B} T={T} H={HV} K={K} V={V} BT={BT} -> {ms*1e3:.1f} us/call")
    print()


def part_c(kernel, fi, do, dv, h0, scale, BT):
    print("=" * 78)
    print("Part C: controlled ablations of state-adjoint storage precision")
    print("=" * 78)
    dh32, dh0_32, dv2_32 = dhu_torch(fi, do, dv, h0, scale, BT, carry_bf16=False)
    # (i) worst case: carry the running dS in bf16 every chunk
    dh_q, dh0_q, dv2_q = dhu_torch(fi, do, dv, h0, scale, BT, carry_bf16=True)
    print("  (i) bf16 *carry* of the running dS (worst case; FLA keeps it fp32):")
    stats("dh   bf16-carry vs fp32", dh_q, dh32)
    stats("dv2  bf16-carry vs fp32", dv2_q, dv2_32)
    stats("dh0  bf16-carry vs fp32", dh0_q, dh0_32)
    # (ii) what FLA really does: keep dS fp32, store per-chunk dh in bf16 for B2.
    dh_bf16 = dh32.to(torch.bfloat16).float()
    print("  (ii) bf16 *storage* of the per-chunk dh (B2 consumer input; FLA default):")
    stats("dh   bf16-store vs fp32", dh_bf16, dh32)
    print("       -> B2 is linear in dh, so this is the relative error it inherits.")
    # Triton kernel variant with bf16 carry for cross-check
    dh_tq, dh0_tq, dv2_tq = dhu_triton(kernel, fi, do, dv, h0, scale, BT, carry_bf16=True)
    stats("dh   triton bf16-carry vs torch", dh_tq, dh_q)
    stats("dh0  triton bf16-carry vs torch", dh0_tq, dh0_q)
    print()


if __name__ == "__main__":
    assert torch.cuda.is_available(), "need GPU"
    dev = "cuda"
    global FLA
    for (B, T, H, K, V, BT) in [(1, 256, 2, 128, 128, 64), (1, 512, 4, 128, 128, 32)]:
        scale = K ** -0.5
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H, K, V, BT, dev, torch.float32)
        torch.manual_seed(123)
        do = torch.randn(B, T, H, V, device=dev)
        print("#" * 78)
        print(f"config: B={B} T={T} H={H} K={K} V={V} BT={BT} dtype=fp32")
        FLA = fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do)
        kernel = build_kernel()
        part_a(FLA["fi"], do, FLA["dv"], h0, scale, BT)
        part_a2(FLA["fi"], q, k, v, g_log2, beta, h0, do, FLA["dv"], scale, BT)
        part_b(kernel, FLA["fi"], do, FLA["dv"], h0, scale, BT)
        part_c(kernel, FLA["fi"], do, FLA["dv"], h0, scale, BT)
    print("DONE")
