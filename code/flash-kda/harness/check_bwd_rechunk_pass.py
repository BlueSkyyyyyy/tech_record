"""P7-11: fuse P7-10's 16->64 re-chunk *recovery* into a real Triton pass.

P7-10 (`check_bwd_b1_rechunk.py`, docs/07 §13) showed that B1@64 is correct and 2.1-2.3x
faster than B1@16, but the recovery that builds the 64-chunk operands from FlashKDA's
16-chunk K1 workspace was a **torch prototype costing ~2-5.7 ms** (far more than B1 itself).
Its own conclusion: "must be fused into a K1-style Triton pass".  This script does that.

The recovery decomposes into:
  (a) a per-channel **rescale** of qd/kd/kr into 64-chunk qg64/kg64/g_cum64 (bounded), and
  (b) the 64-block intra matrix `A64 = kd16 @ k_inv^T` (off-diagonal blocks only, folded
      per-pair so no intermediate exceeds ~|k|), whose strict-lower `L` feeds the WY solve
      `P64 = (I+L64)^{-1}` and then `w64 = P64 diag(beta) kd64`.

Key observation that makes (b) cheap: the *diagonal* 16x16 blocks of `L64` are exactly the
within-16 blocks K1 already inverted and stored as `INV` (= P16).  So `P64` needs **no new
16x16 inversion** -- it is assembled from workspace `P16` by the standard 4x4 block
forward-substitution (same structure as FLA's `merge_16x16_to_64x64_inverse_kernel`).

`_recover64_kernel` does *all* of (a)+(b) in one program per (64-group, B*H): rescale,
6 off-diagonal `A64` blocks, the 6 merge formulas, and `w64`.  Everything is compared
against the torch prototype (`recover_operands_64`) and FLA's own forward@64.

Run inside `kernel_lab` (GPU, FlashKDA built, FLA installed), idle card:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_rechunk_pass.py
    RP_MODE=correctness|timing
"""
import math
import os
import statistics
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch
import triton
import triton.language as tl

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_ws_reuse import (make_inputs, flash_forward, D, CHUNK, LOG2E,   # noqa: E402
                                cdiv, stats)
from check_bwd_b1_rechunk import (b0_gate, recover_operands_64, prep, BIG)      # noqa: E402

S = CHUNK        # 16
BT = BIG         # 64


# --------------------------------------------------------------------------- Triton pass
@triton.jit
def _ld_tok(ptr, i_b, i_h, t0, o_s, o_d, T,
            H: tl.constexpr, D: tl.constexpr, Tp: tl.constexpr):
    o_t = t0 + o_s
    m = o_t < T
    base = (i_b * Tp + o_t)[:, None] * (H * D) + i_h * D
    v = tl.load(ptr + base + o_d[None, :], mask=m[:, None], other=0.0).to(tl.float32)
    return v, m


@triton.jit
def _st_tok(ptr, i_b, i_h, t0, o_s, o_d, val, mm,
            H: tl.constexpr, D: tl.constexpr, Tp: tl.constexpr):
    o_t = t0 + o_s
    base = (i_b * Tp + o_t)[:, None] * (H * D) + i_h * D
    tl.store(ptr + base + o_d[None, :], val, mask=mm[:, None])


@triton.jit
def _ld_beta(ptr, i_b, i_h, t0, o_s, T, H: tl.constexpr, Tp: tl.constexpr):
    o_t = t0 + o_s
    return tl.load(ptr + (i_b * Tp + o_t) * H + i_h, mask=o_t < T, other=0.0).to(tl.float32)


@triton.jit
def _ld_scal(ptr, i_b, i_h, i_g, i_s, o_d,
             H: tl.constexpr, D: tl.constexpr, SG: tl.constexpr, NG: tl.constexpr):
    gbase = ((i_b * NG + i_g) * SG + i_s) * (H * D) + i_h * D
    return tl.load(ptr + gbase + o_d).to(tl.float32)


@triton.jit
def _ld_inv(ptr, i_b, i_h, i_g, i_s, o_a,
            H: tl.constexpr, S: tl.constexpr, SG: tl.constexpr, NSUB: tl.constexpr):
    c = i_g * SG + i_s
    p = ptr + (((i_b * NSUB + c) * H + i_h) * S * S + o_a[:, None] * S + o_a[None, :])
    return tl.load(p).to(tl.float32)


@triton.jit
def _recover64_kernel(
    qd_ptr, kd_ptr, kr_ptr, inv_ptr, beta_ptr, gcum_ptr, G16_ptr, off_ptr,
    qg_ptr, kg_ptr, w_ptr, gcum64_ptr,
    T, rscale,
    H: tl.constexpr, D: tl.constexpr, Tp: tl.constexpr, NSUB: tl.constexpr,
    NG: tl.constexpr, S: tl.constexpr, BT: tl.constexpr, IP: tl.constexpr,
):
    """One program per (64-group, b*H).  ``Tp`` = padded token count, ``NSUB`` = Tp/S,
    ``NG`` = Tp/BT.  All field pointers are token-major [B, Tp, H, D] (inv: [B,NSUB,H,S,S])."""
    i_g = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    SG: tl.constexpr = BT // S     # 4 sub-chunks per group

    o_s = tl.arange(0, S)
    o_d = tl.arange(0, D)
    o_a = tl.arange(0, S)

    t0 = i_g * BT
    kd0, m0 = _ld_tok(kd_ptr, i_b, i_h, t0 + 0 * S, o_s, o_d, T, H, D, Tp)
    kd1, m1 = _ld_tok(kd_ptr, i_b, i_h, t0 + 1 * S, o_s, o_d, T, H, D, Tp)
    kd2, m2 = _ld_tok(kd_ptr, i_b, i_h, t0 + 2 * S, o_s, o_d, T, H, D, Tp)
    kd3, m3 = _ld_tok(kd_ptr, i_b, i_h, t0 + 3 * S, o_s, o_d, T, H, D, Tp)
    qd0, _ = _ld_tok(qd_ptr, i_b, i_h, t0 + 0 * S, o_s, o_d, T, H, D, Tp)
    qd1, _ = _ld_tok(qd_ptr, i_b, i_h, t0 + 1 * S, o_s, o_d, T, H, D, Tp)
    qd2, _ = _ld_tok(qd_ptr, i_b, i_h, t0 + 2 * S, o_s, o_d, T, H, D, Tp)
    qd3, _ = _ld_tok(qd_ptr, i_b, i_h, t0 + 3 * S, o_s, o_d, T, H, D, Tp)
    kr0, _ = _ld_tok(kr_ptr, i_b, i_h, t0 + 0 * S, o_s, o_d, T, H, D, Tp)
    kr1, _ = _ld_tok(kr_ptr, i_b, i_h, t0 + 1 * S, o_s, o_d, T, H, D, Tp)
    kr2, _ = _ld_tok(kr_ptr, i_b, i_h, t0 + 2 * S, o_s, o_d, T, H, D, Tp)
    kr3, _ = _ld_tok(kr_ptr, i_b, i_h, t0 + 3 * S, o_s, o_d, T, H, D, Tp)
    b0 = _ld_beta(beta_ptr, i_b, i_h, t0 + 0 * S, o_s, T, H, Tp)
    b1 = _ld_beta(beta_ptr, i_b, i_h, t0 + 1 * S, o_s, T, H, Tp)
    b2 = _ld_beta(beta_ptr, i_b, i_h, t0 + 2 * S, o_s, T, H, Tp)
    b3 = _ld_beta(beta_ptr, i_b, i_h, t0 + 3 * S, o_s, T, H, Tp)
    g0, _ = _ld_tok(gcum_ptr, i_b, i_h, t0 + 0 * S, o_s, o_d, T, H, D, Tp)
    g1, _ = _ld_tok(gcum_ptr, i_b, i_h, t0 + 1 * S, o_s, o_d, T, H, D, Tp)
    g2v, _ = _ld_tok(gcum_ptr, i_b, i_h, t0 + 2 * S, o_s, o_d, T, H, D, Tp)
    g3, _ = _ld_tok(gcum_ptr, i_b, i_h, t0 + 3 * S, o_s, o_d, T, H, D, Tp)
    G0 = _ld_scal(G16_ptr, i_b, i_h, i_g, 0, o_d, H, D, SG, NG)
    G1 = _ld_scal(G16_ptr, i_b, i_h, i_g, 1, o_d, H, D, SG, NG)
    G2 = _ld_scal(G16_ptr, i_b, i_h, i_g, 2, o_d, H, D, SG, NG)
    G3 = _ld_scal(G16_ptr, i_b, i_h, i_g, 3, o_d, H, D, SG, NG)
    o0 = _ld_scal(off_ptr, i_b, i_h, i_g, 0, o_d, H, D, SG, NG)
    o1 = _ld_scal(off_ptr, i_b, i_h, i_g, 1, o_d, H, D, SG, NG)
    o2 = _ld_scal(off_ptr, i_b, i_h, i_g, 2, o_d, H, D, SG, NG)
    o3 = _ld_scal(off_ptr, i_b, i_h, i_g, 3, o_d, H, D, SG, NG)
    P0 = _ld_inv(inv_ptr, i_b, i_h, i_g, 0, o_a, H, S, SG, NSUB)
    P1 = _ld_inv(inv_ptr, i_b, i_h, i_g, 1, o_a, H, S, SG, NSUB)
    P2 = _ld_inv(inv_ptr, i_b, i_h, i_g, 2, o_a, H, S, SG, NSUB)
    P3 = _ld_inv(inv_ptr, i_b, i_h, i_g, 3, o_a, H, S, SG, NSUB)

    # ---- (a) rescale: qg / kg / g_cum64 ----
    G64 = o3 + G3                                       # [D] block gate total (log2)
    _st_tok(qg_ptr, i_b, i_h, t0 + 0 * S, o_s, o_d, qd0 * tl.exp2(o0)[None, :] * rscale, m0, H, D, Tp)
    _st_tok(qg_ptr, i_b, i_h, t0 + 1 * S, o_s, o_d, qd1 * tl.exp2(o1)[None, :] * rscale, m1, H, D, Tp)
    _st_tok(qg_ptr, i_b, i_h, t0 + 2 * S, o_s, o_d, qd2 * tl.exp2(o2)[None, :] * rscale, m2, H, D, Tp)
    _st_tok(qg_ptr, i_b, i_h, t0 + 3 * S, o_s, o_d, qd3 * tl.exp2(o3)[None, :] * rscale, m3, H, D, Tp)
    _st_tok(kg_ptr, i_b, i_h, t0 + 0 * S, o_s, o_d, kr0 * tl.exp2(G64[None, :] - o0[None, :] - G0[None, :]), m0, H, D, Tp)
    _st_tok(kg_ptr, i_b, i_h, t0 + 1 * S, o_s, o_d, kr1 * tl.exp2(G64[None, :] - o1[None, :] - G1[None, :]), m1, H, D, Tp)
    _st_tok(kg_ptr, i_b, i_h, t0 + 2 * S, o_s, o_d, kr2 * tl.exp2(G64[None, :] - o2[None, :] - G2[None, :]), m2, H, D, Tp)
    _st_tok(kg_ptr, i_b, i_h, t0 + 3 * S, o_s, o_d, kr3 * tl.exp2(G64[None, :] - o3[None, :] - G3[None, :]), m3, H, D, Tp)
    # g_cum is written **unmasked**: the pad rows carry the last valid 16-local value (fed in
    # via gc_pad), which B1 needs for `etot` at the padded chunk end (P7-10 §13.4 carry).
    mfull = o_s < S
    _st_tok(gcum64_ptr, i_b, i_h, t0 + 0 * S, o_s, o_d, g0 + o0[None, :], mfull, H, D, Tp)
    _st_tok(gcum64_ptr, i_b, i_h, t0 + 1 * S, o_s, o_d, g1 + o1[None, :], mfull, H, D, Tp)
    _st_tok(gcum64_ptr, i_b, i_h, t0 + 2 * S, o_s, o_d, g2v + o2[None, :], mfull, H, D, Tp)
    _st_tok(gcum64_ptr, i_b, i_h, t0 + 3 * S, o_s, o_d, g3 + o3[None, :], mfull, H, D, Tp)

    # ---- (b1) off-diagonal raw A64 blocks, folded per pair (exponent <= 0 into kr) ----
    # A[i,j] = kd_i @ (kr_j * 2^{sum_{s=j+1}^{i-1} G16_s})^T ,  i > j
    A10 = tl.dot(kd1, tl.trans(kr0), input_precision=IP)
    A20 = tl.dot(kd2, tl.trans(kr0 * tl.exp2(G1)[None, :]), input_precision=IP)
    A21 = tl.dot(kd2, tl.trans(kr1), input_precision=IP)
    A30 = tl.dot(kd3, tl.trans(kr0 * tl.exp2(G1 + G2)[None, :]), input_precision=IP)
    A31 = tl.dot(kd3, tl.trans(kr1 * tl.exp2(G2)[None, :]), input_precision=IP)
    A32 = tl.dot(kd3, tl.trans(kr2), input_precision=IP)

    # L_ij = A_ij * beta_j (columns u of sub-chunk j)
    L10 = A10 * b0[None, :]
    L20 = A20 * b0[None, :]
    L21 = A21 * b1[None, :]
    L30 = A30 * b0[None, :]
    L31 = A31 * b1[None, :]
    L32 = A32 * b2[None, :]

    # ---- (b2) merge: P64 from workspace P16 (diagonal) + off-diagonal L (FLA structure) ----
    P10 = -tl.dot(P1, tl.dot(L10, P0, input_precision=IP), input_precision=IP)
    P21 = -tl.dot(P2, tl.dot(L21, P1, input_precision=IP), input_precision=IP)
    P32 = -tl.dot(P3, tl.dot(L32, P2, input_precision=IP), input_precision=IP)
    P20 = -tl.dot(P2, tl.dot(L20, P0, input_precision=IP)
                  + tl.dot(L21, P10, input_precision=IP), input_precision=IP)
    P31 = -tl.dot(P3, tl.dot(L31, P1, input_precision=IP)
                  + tl.dot(L32, P21, input_precision=IP), input_precision=IP)
    P30 = -tl.dot(P3, tl.dot(L30, P0, input_precision=IP)
                  + tl.dot(L31, P10, input_precision=IP)
                  + tl.dot(L32, P20, input_precision=IP), input_precision=IP)

    # ---- (b3) w64 = P64 diag(beta) kd64 ----
    kb0 = kd0 * tl.exp2(o0)[None, :] * b0[:, None]
    kb1 = kd1 * tl.exp2(o1)[None, :] * b1[:, None]
    kb2 = kd2 * tl.exp2(o2)[None, :] * b2[:, None]
    kb3 = kd3 * tl.exp2(o3)[None, :] * b3[:, None]
    w0 = tl.dot(P0, kb0, input_precision=IP)
    w1 = tl.dot(P10, kb0, input_precision=IP) + tl.dot(P1, kb1, input_precision=IP)
    w2 = (tl.dot(P20, kb0, input_precision=IP)
          + tl.dot(P21, kb1, input_precision=IP) + tl.dot(P2, kb2, input_precision=IP))
    w3 = (tl.dot(P30, kb0, input_precision=IP)
          + tl.dot(P31, kb1, input_precision=IP)
          + tl.dot(P32, kb2, input_precision=IP) + tl.dot(P3, kb3, input_precision=IP))
    _st_tok(w_ptr, i_b, i_h, t0 + 0 * S, o_s, o_d, w0, m0, H, D, Tp)
    _st_tok(w_ptr, i_b, i_h, t0 + 1 * S, o_s, o_d, w1, m1, H, D, Tp)
    _st_tok(w_ptr, i_b, i_h, t0 + 2 * S, o_s, o_d, w2, m2, H, D, Tp)
    _st_tok(w_ptr, i_b, i_h, t0 + 3 * S, o_s, o_d, w3, m3, H, D, Tp)


def recover64_pass(fw, inp, lower_bound, scale, BT=BIG):
    """Run the fused Triton pass; returns operands dict in FLA `fi` convention (token-major)."""
    B, T, H, _ = inp["q"].shape
    dev = inp["q"].device
    NT = fw["NT"]
    NG = cdiv(NT, BT // CHUNK)
    Tp = NG * BT
    NSUB = Tp // CHUNK

    def tok(x, last):
        # [B,NT,H,16,D] -> [B,NT*16,H,D]
        return x.permute(0, 1, 3, 2, 4).reshape(B, NT * CHUNK, H, *last)

    def pad0(x):
        if x.shape[1] == Tp:
            return x
        z = torch.zeros(B, Tp, *x.shape[2:], device=dev, dtype=x.dtype)
        z[:, :x.shape[1]] = x
        return z

    qdT = pad0(tok(fw["qd"], (D,))).contiguous()
    kdT = pad0(tok(fw["kd"], (D,))).contiguous()
    krT = pad0(tok(fw["kr"], (D,))).contiguous()
    invT = torch.zeros(B, NSUB, H, CHUNK, CHUNK, device=dev)
    invT[:, :NT] = fw["inv"].float()

    gc16, beta16 = b0_gate(inp, lower_bound)                # [B,T,H,D], [B,T,H]
    beta_pad = torch.zeros(B, Tp, H, device=dev)
    beta_pad[:, :T] = beta16.float()
    gc_pad = torch.zeros(B, Tp, H, D, device=dev)
    gc_pad[:, :T] = gc16.float()
    # carry the last 16-local g_cum only up to the K1-padded tile end NT*CHUNK; tokens beyond
    # (the extra re-chunk pad) stay 0 so their sub-chunk gate totals G16 are 0 (NOT the carry).
    if NT * CHUNK > T:
        gc_pad[:, T:NT * CHUNK] = gc16.float()[:, T - 1:T]

    gc_sub = gc_pad.view(B, NSUB, CHUNK, H, D)
    G16_full = gc_sub[:, :, -1]                             # [B,NSUB,H,D]
    G16g = G16_full.view(B, NG, BT // CHUNK, H, D)
    offg = torch.zeros_like(G16g)
    offg[:, :, 1:] = G16g[:, :, :-1].cumsum(2)
    off = offg.reshape(B, NSUB, H, D)
    G16 = G16_full
    gcum64_ref = (gc_sub + off.view(B, NSUB, 1, H, D)).reshape(B, Tp, H, D)

    qg = torch.zeros(B, Tp, H, D, device=dev)
    kg = torch.zeros(B, Tp, H, D, device=dev)
    w = torch.zeros(B, Tp, H, D, device=dev)
    gcm = torch.zeros(B, Tp, H, D, device=dev)

    grid = (NG, B * H)
    _recover64_kernel[grid](
        qdT, kdT, krT, invT.contiguous(), beta_pad, gc_pad, G16.contiguous(), off.contiguous(),
        qg, kg, w, gcm,
        T, 1.0 / scale,
        H=H, D=D, Tp=Tp, NSUB=NSUB, NG=NG, S=CHUNK, BT=BT,
        IP=os.environ.get("RP_PREC", "ieee"), num_warps=4,
    )
    return dict(qg=qg, kg=kg, w=w, g_cum=gcm), dict(gcum64_ref=gcum64_ref, Tp=Tp)


# --------------------------------------------------------------------------- timing
def wall_us(fn, reps=30, rounds=3, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    med = []
    for _ in range(rounds):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        torch.cuda.synchronize()
        med.append(s.elapsed_time(e) / reps)
    return statistics.median(med) * 1e3


# --------------------------------------------------------------------------- parts
def part_a(cases, scale, lower_bound):
    print("=" * 92)
    print("Part A: fused Triton recovery vs torch prototype (recover_operands_64) & FLA fwd@64")
    print("=" * 92)
    from check_bwd_b1_rechunk import fla_forward
    worst = {}
    for (name, B, T, H, seed, gs) in cases:
        inp, fw, qn, kn, beta_act, g_fla, do, dv = prep(B, T, H, scale, lower_bound, seed, gs)
        rec, diag = recover_operands_64(fw, inp, lower_bound, scale, with_intra=True)
        pas, _ = recover64_pass(fw, inp, lower_bound, scale)
        qnb = qn.to(torch.bfloat16)
        knb = kn.to(torch.bfloat16)
        g2 = (g_fla * 1.4426950408889634).to(torch.bfloat16)
        F = fla_forward(qnb, knb, inp["v"].to(torch.bfloat16), g2,
                        beta_act.to(torch.bfloat16), inp["h0"], scale, BIG)
        print(f"\n# {name}  B={B} T={T} H={H}  |strict-lower A64|max={diag['A64_Lmax']:.3e}")
        for nm in ["qg", "kg", "w", "g_cum"]:
            margs = dict(pred=pas[nm][:, :T], gold=rec[nm][:, :T])
            _, mr = stats(f"pass vs torch   {nm:6s}", margs["pred"], margs["gold"])
            worst["pass." + nm] = max(worst.get("pass." + nm, 0.0), mr)
            _, mr = stats(f"pass vs FLA64   {nm:6s}", pas[nm][:, :T], F[nm].float()[:, :T])
            worst["fla." + nm] = max(worst.get("fla." + nm, 0.0), mr)
    print("\nworst-case max_rel across cases:")
    for nm in ["qg", "kg", "w", "g_cum"]:
        print(f"  {nm:6s}  pass-vs-torch {worst['pass.' + nm]:.3e}   pass-vs-FLA64 {worst['fla.' + nm]:.3e}")


def part_c(cases, scale, lower_bound):
    """Close the loop: feed the *pass-produced* 64-chunk operands to the bf16 B1 kernel and
    compare against FLA `dhu`@64 — the same check P7-10 did, but on the fused pass."""
    print("\n" + "=" * 92)
    print("Part C: bf16 B1@64 on the fused-pass operands vs FLA `dhu`@64")
    print("=" * 92)
    from check_bwd_b1_bf16_e2e import build_dhu_bf16, dhu_bf16_triton
    from check_bwd_b1_rechunk import b1_64, fla_b1_64
    kernel = build_dhu_bf16()
    worst = {}
    for (name, B, T, H, seed, gs) in cases:
        inp, fw, qn, kn, beta_act, g_fla, do, dv = prep(B, T, H, scale, lower_bound, seed, gs)
        pas, _ = recover64_pass(fw, inp, lower_bound, scale)
        qnb = qn.to(torch.bfloat16)
        knb = kn.to(torch.bfloat16)
        g2 = (g_fla * 1.4426950408889634).to(torch.bfloat16)
        fb = fla_b1_64(inp["v"].to(torch.bfloat16), qnb, knb, beta_act.to(torch.bfloat16), g2,
                       inp["h0"], scale, do.to(torch.bfloat16), dv.to(torch.bfloat16))
        dh, dh0, dv2 = b1_64(kernel, pas, do, dv, scale, dht=None)
        print(f"\n# {name}  B={B} T={T} H={H}")
        _, mr = stats("dh  pass@64 vs FLA64", dh, fb["dh"].float())
        worst["dh"] = max(worst.get("dh", 0.0), mr)
        _, mr = stats("dv2 pass@64 vs FLA64", dv2[:, :T], fb["dv2"].float()[:, :T])
        worst["dv2"] = max(worst.get("dv2", 0.0), mr)
        _, mr = stats("dh0 pass@64 vs FLA64", dh0, fb["dh0"].float())
        worst["dh0"] = max(worst.get("dh0", 0.0), mr)
    print("\nworst-case max_rel:")
    for nm in ["dh", "dv2", "dh0"]:
        print(f"  {nm:5s} {worst[nm]:.3e}")


def part_b(cases, scale, lower_bound):
    print("\n" + "=" * 92)
    print("Part B: timing — fused pass vs torch prototype (recovery only)")
    print("=" * 92)
    print(f"  {'case':18s} | {'pass dev':>9s} {'pass wall':>9s} | {'torch rescale':>13s} "
          f"{'torch full':>11s} | {'pass/torch':>10s}")
    for (name, B, T, H, seed, gs) in cases:
        inp, fw, qn, kn, beta_act, g_fla, do, dv = prep(B, T, H, scale, lower_bound, seed, gs)

        def run_pass():
            return recover64_pass(fw, inp, lower_bound, scale)

        def run_rescale():
            return recover_operands_64(fw, inp, lower_bound, scale, dt=torch.float32,
                                       with_intra=False)

        def run_full():
            return recover_operands_64(fw, inp, lower_bound, scale, dt=torch.float32)

        from torch.profiler import ProfilerActivity, profile
        for _ in range(5):
            run_pass()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(30):
                run_pass()
            torch.cuda.synchronize()
        dev = sum(ev.self_device_time_total for ev in prof.key_averages()
                  if "_recover64" in ev.key) / 30

        tpass = wall_us(run_pass)
        tres = wall_us(run_rescale)
        tfull = wall_us(run_full)
        print(f"  {name:18s} | {dev:9.1f} {tpass:9.1f} | {tres:13.1f} {tfull:11.1f} | "
              f"{tpass / tfull:10.2f}")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0
    cases = [
        ("T=256 H=2", 1, 256, 2, 0, 1.0),
        ("T=250 H=3 non-div", 1, 250, 3, 2, 1.0),
        ("B2 T=513 H=2 non-div", 2, 513, 2, 4, 1.0),
        ("T=192 H=2 g=0", 1, 192, 2, 5, 0.0),
    ]
    tcases = [
        ("T=512 H=4", 1, 512, 4, 0, 1.0),
        ("B2 T=1024 H=8", 2, 1024, 8, 1, 1.0),
        ("B8 T=2048 H=8", 8, 2048, 8, 2, 1.0),
        ("T=250 H=3 nondiv", 1, 250, 3, 3, 1.0),
    ]
    mode = os.environ.get("RP_MODE", "")
    if mode != "timing":
        part_a(cases, scale, lower_bound)
        part_c(cases, scale, lower_bound)
    if mode != "correctness":
        part_b(tcases, scale, lower_bound)
    print("\nDONE")


if __name__ == "__main__":
    main()
