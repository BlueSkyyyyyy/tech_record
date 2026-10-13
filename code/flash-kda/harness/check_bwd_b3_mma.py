"""P7-24: B3 `intra` structural rewrite -- MMA-ize the *diagonal* sub-block loop.

Motivation (P7-23, `docs/07` §26)
----------------------------------
At the *deployed* CHUNK=16 granularity B3 (`intra`) builds its grid as
`(NK*NC, NT, B*H)` with `BC=BT=16` -> `NC=1`.  With `i_i=0` every program skips
Parts A/C (the off-diagonal sub-blocks, which *do* use `tl.dot`) and executes **only**
the diagonal loops (Parts B/D) -- `bc=16` iterations of rank-1 elementwise outer
products.  That is exactly why P7-23 measured B3 at `tensor 0.05%`, `SM 82%`,
`L1/TEX 71%`: the TensorCore sits completely idle while the FMA/LSU pipes do all the work.

The diagonal loops are algebraically a small matmul though.  Part B computes, for query
row `a`, channel `l` (docs/05 eq 7.10, chunk_intra.py:471-500):

    dq2[a,l] = sum_{j<=a} dAqk[a,j] * k[j,l] * exp2(g[a,l] - g[j,l])
             = exp2(g[a,l]) * sum_j [ (a>=j) * dAqk[a,j] ] * ( k[j,l] * exp2(-g[j,l]) )

i.e. `Mq @ KG` with the gate exponential *factored out of the contraction* (the sign
split is exact, and numerically stable: for `j<=a` the largest term is `j=a` where the
two factors cancel).  Likewise Part D (chunk_intra.py:626-650) is `emg * (S @ QG)` with
`S[a,j] = (a<=j) * dAqk[j,a]`.  So both diagonal loops collapse to a single `tl.dot`
each, moving the work back onto the TensorCores.

This script is the controlled experiment that P7-23 §26.5 called for ("B3 MMA 化"):

  Part A  isolated kernel: MMA-diagonal B3 vs the elementwise baseline (bt-exact-ish,
          different accumulation) and vs FLA / the per-chunk torch golden -- does the
          factored form stay at the tf32 floor, incl. non-divisible T and g=0?
  Part B  isolated timing, baseline vs MMA, BT=16 (deployment) and BT=64, + nw sweep.
  Part C  land MMA B3 into the assembled chain, verify the six gradients vs FLA autograd
          and vs the baseline chain, and time chain-level (dev, CUPTI).
  Part D  bound analysis lives in `probe_b3_mma_ncu.out.txt`: tensor%, Duration, regs.

Run inside `kernel_lab` (GPU, FlashKDA built, FLA installed), idle card:
    CUDA_VISIBLE_DEVICES=1 python .../check_bwd_b3_mma.py
    B3M_MODE=correctness | timing | chain
"""
import math
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch
import triton
import triton.language as tl

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_ws_reuse import D, CHUNK, stats, cdiv                                    # noqa: E402
from check_bwd_intra_kernel import (make_inputs, fla_intra, intra_torch,                # noqa: E402
                                    build_kernel as build_intra_base,
                                    intra_triton as intra_triton_base)
from check_bwd_dqkg_kernel import build_kernel as build_dqkg, wy_dqkg_triton            # noqa: E402
from check_bwd_b1_pipe_e2e import build_dhu_bf16_pipe, dhu_bf16_pipe_triton             # noqa: E402
from check_bwd_e2e import build_dav_kernel, dav_triton                                  # noqa: E402
from check_bwd_b1_bf16_e2e import (prep, fla_golden, time_wall, profile_dev)            # noqa: E402
from fla.ops.utils import chunk_local_cumsum                                            # noqa: E402


# ------------------------------------------------------------------- MMA-diagonal kernel
def build_intra_mma():
    """Same kernel as `check_bwd_intra_kernel.build_kernel` (FLA chunk_intra.py:395-667,
    non-SAFE_GATE), except Parts B/D -- the diagonal sub-blocks -- are a single `tl.dot`
    each instead of a `bc`-iteration elementwise loop.  Parts A/C (off-diagonal) are
    unchanged (they already use `tl.dot`)."""

    @triton.jit
    def _intra_kernel_mma(
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

        p_q = q_ptr + (i_b * T + o_c)[:, None] * (H * K) + i_h * K + o_k[None, :]
        p_k = k_ptr + (i_b * T + o_c)[:, None] * (H * K) + i_h * K + o_k[None, :]
        p_g = g_ptr + (i_b * T + o_c)[:, None] * (H * K) + i_h * K + o_k[None, :]
        b_q = tl.load(p_q, mask=m_ck, other=0.0)
        b_k = tl.load(p_k, mask=m_ck, other=0.0)
        b_g = tl.load(p_g, mask=m_ck, other=0.0).to(tl.float32)
        b_b = tl.load(beta_ptr + (i_b * T + o_c) * H + i_h, mask=m_c, other=0.0)

        b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
        b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)

        # ---- Part A: key sub-blocks fully before i_i (unchanged; uses tl.dot) ----
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

        # ---- Part B (MMA): diagonal sub-block as exp2(g) * (Mq @ KG) ----
        # diagonal dAqk / dAkk tile [BC,BC]: rows = query o_c (absolute token),
        # cols = the *chunk-local* key index i_i*BC+o_i (dAqk's last axis is chunk-local).
        m_jd = (i_ti + o_i) < T
        m_diag = m_c[:, None] & m_jd[None, :]
        p_dAqd = dAqk_ptr + ((i_b * T + o_c)[:, None] * H + i_h) * BT + (i_i * BC + o_i)[None, :]
        p_dAkkd = dAkk_ptr + ((i_b * T + o_c)[:, None] * H + i_h) * BT + (i_i * BC + o_i)[None, :]
        b_dAqd = tl.load(p_dAqd, mask=m_diag, other=0.0).to(tl.float32)
        b_dAkkd = tl.load(p_dAkkd, mask=m_diag, other=0.0).to(tl.float32)
        # key rows j = i_ti + o_i: k and g live on the *same* rows as the query tile, but
        # keep them as [BC,BK] with the j-row mask (identical to o_c mask here).
        m_jk = m_jd[:, None] & m_k[None, :]
        p_kd = k_ptr + (i_b * T + (i_ti + o_i))[:, None] * (H * K) + i_h * K + o_k[None, :]
        p_gd = g_ptr + (i_b * T + (i_ti + o_i))[:, None] * (H * K) + i_h * K + o_k[None, :]
        b_kd = tl.load(p_kd, mask=m_jk, other=0.0)
        b_gd = tl.load(p_gd, mask=m_jk, other=0.0).to(tl.float32)
        m_ge = o_i[:, None] >= o_i[None, :]
        b_Mq = tl.where(m_ge, b_dAqd, 0.0)
        b_Mk = tl.where(m_ge, b_dAkkd, 0.0)
        b_KG = b_kd * tl.exp2(-b_gd)
        b_eg = tl.exp2(b_g)
        b_dq2 += b_eg * tl.dot(b_Mq, b_KG)
        b_dk2 += b_eg * tl.dot(b_Mk, b_KG)

        b_db = tl.sum(b_dk2 * b_k, axis=1)
        b_dk2 *= b_b[:, None]

        # \bar q^dec branch feeds the decay chain: \bar g += \bar q^dec \odot q
        # (chunk_intra.py:558, docs/05 eq 7.10) -- MUST use the *fresh* dq2.
        b_dg2 = b_q * b_dq2
        p_dq = dq_ptr + (i_b * T + o_c)[:, None] * (H * K) + i_h * K + o_k[None, :]
        b_dq2 = b_dq2 + tl.load(p_dq, mask=m_ck, other=0.0)
        tl.store(p_dq, b_dq2.to(p_dq.dtype.element_ty), mask=m_ck)
        tl.atomic_add(db_ptr + (i_b * T + o_c) * H + i_h, b_db.to(db_ptr.dtype.element_ty), mask=m_c)

        tl.debug_barrier()
        b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

        # ---- Part C: dk branch from query sub-blocks strictly after i_i (unchanged) ----
        NC_eff = min(NC, tl.cdiv(T - i_t * BT, BC))
        if i_i < NC_eff - 1:
            p_gn = g_ptr + (i_b * T + min(i_ti + BC, T) - 1) * (H * K) + i_h * K + o_k
            b_gn = tl.load(p_gn, mask=m_k, other=0.0).to(tl.float32)[None, :]
            for i_j in range(i_i + 1, NC_eff):
                o_j = i_t * BT + i_j * BC + o_i
                m_j = o_j < T
                m_jk2 = m_j[:, None] & m_k[None, :]
                p_qj = q_ptr + (i_b * T + o_j)[:, None] * (H * K) + i_h * K + o_k[None, :]
                p_kj = k_ptr + (i_b * T + o_j)[:, None] * (H * K) + i_h * K + o_k[None, :]
                p_gkj = g_ptr + (i_b * T + o_j)[:, None] * (H * K) + i_h * K + o_k[None, :]
                b_bj = tl.load(beta_ptr + (i_b * T + o_j) * H + i_h, mask=m_j, other=0.0)
                b_qj = tl.load(p_qj, mask=m_jk2, other=0.0)
                b_kbj = tl.load(p_kj, mask=m_jk2, other=0.0) * b_bj[:, None]
                b_gkj = tl.load(p_gkj, mask=m_jk2, other=0.0).to(tl.float32)
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

        # ---- Part D (MMA): diagonal dk branch as exp2(-g) * (Sq @ QG + Sk @ kbg) ----
        m_le = o_i[:, None] <= o_i[None, :]
        b_Sq = tl.where(m_le, tl.trans(b_dAqd), 0.0)
        b_Sk = tl.where(m_le, tl.trans(b_dAkkd), 0.0)
        b_QG = b_q * tl.exp2(b_g)
        b_KGd = (b_k * b_b[:, None]) * tl.exp2(b_g)
        b_emg = tl.exp2(-b_g)
        b_dkt += b_emg * tl.dot(b_Sq, b_QG)
        b_dkt += b_emg * tl.dot(b_Sk, b_KGd)

        p_dk = dk_ptr + (i_b * T + o_c)[:, None] * (H * K) + i_h * K + o_k[None, :]
        p_dg = dg_ptr + (i_b * T + o_c)[:, None] * (H * K) + i_h * K + o_k[None, :]
        b_dg2 = b_dg2 + (b_dk2 - b_dkt) * b_k + tl.load(p_dg, mask=m_ck, other=0.0)
        b_dk2 = b_dk2 + tl.load(p_dk, mask=m_ck, other=0.0) + b_dkt
        tl.store(p_dk, b_dk2.to(p_dk.dtype.element_ty), mask=m_ck)
        tl.store(p_dg, b_dg2.to(p_dg.dtype.element_ty), mask=m_ck)

    return _intra_kernel_mma


def intra_triton_mma(kernel, inp, BT, BC, num_warps=4, num_stages=1):
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


# --------------------------------------------------------------------------------- parts
CASES = [
    ("T=128 H=2", 1, 128, 2, 0, 1.0),
    ("T=256 H=4", 1, 256, 4, 1, 1.0),
    ("T=250 H=3 (non-div)", 1, 250, 3, 2, 1.0),
    ("T=512 H=4 g=2", 1, 512, 4, 3, 2.0),
    ("B2 T=513 H=2 (non-div)", 2, 513, 2, 4, 1.0),
    ("T=192 H=2 g=0", 1, 192, 2, 5, 0.0),
]
TCASES = [
    ("T=512 H=4", 1, 512, 4, 0, 1.0),
    ("B2 T=1024 H=8", 2, 1024, 8, 1, 1.0),
    ("B8 T=2048 H=8", 8, 2048, 8, 2, 1.0),
    ("T=250 H=3 nondiv", 1, 250, 3, 3, 1.0),
]


def part_correctness():
    print("=" * 78)
    print("Part A: isolated B3 -- MMA-diagonal vs elementwise baseline, FLA, torch golden")
    print("=" * 78)
    base_k = build_intra_base()
    mma_k = build_intra_mma()
    worst = {}
    for (B, T, H, K, BT, BC, seed) in [
        (1, 256, 2, 64, 64, 16, 0), (2, 512, 4, 128, 64, 16, 1),
        (1, 250, 3, 64, 64, 16, 2), (1, 1024, 8, 128, 64, 16, 3),
        (2, 513, 2, 64, 64, 16, 4), (1, 64, 1, 32, 64, 16, 5),
        (1, 128, 2, 64, 16, 16, 6), (1, 250, 3, 128, 16, 16, 7),  # BT=16 deployment
    ]:
        print(f"\n=== B={B} T={T} H={H} K={K} BT={BT} BC={BC} seed={seed} ===")
        inp = make_inputs(B, T, H, K, BT, BC, "cuda", dtype=torch.float32, seed=seed)
        fla = fla_intra(inp, BT, BC)
        tri_b = intra_triton_base(base_k, inp, BT, BC)
        tri_m = intra_triton_mma(mma_k, inp, BT, BC)
        print("  -- MMA vs baseline --")
        for name in ["dq", "dk", "db", "dg"]:
            _, mr = stats(f"  mma {name} vs base", tri_m[name], tri_b[name])
            worst[f"vsbase_{name}"] = max(worst.get(f"vsbase_{name}", 0.0), mr)
        print("  -- MMA vs FLA --")
        for name in ["dq", "dk", "db", "dg"]:
            _, mr = stats(f"  mma {name} vs FLA", tri_m[name], fla[name])
            worst[f"vsfla_{name}"] = max(worst.get(f"vsfla_{name}", 0.0), mr)
    print("\nworst-case max_rel (MMA B3):")
    for kk in sorted(worst):
        print(f"  {kk:16s}  {worst[kk]:.3e}")


def part_timing():
    print("\n" + "=" * 78)
    print("Part B: isolated B3 timing -- baseline vs MMA, BT=16 (deploy) and BT=64")
    print("=" * 78)
    base_k = build_intra_base()
    mma_k = build_intra_mma()
    for (B, T, H, K, BT) in [(1, 512, 4, 128, 64), (2, 2048, 8, 128, 64),
                             (1, 512, 4, 128, 16), (2, 2048, 8, 128, 16),
                             (8, 2048, 8, 128, 16)]:
        inp = make_inputs(B, T, H, K, BT, 16, "cuda", dtype=torch.float32, seed=0)
        inp = {k2: (v if v.is_contiguous() else v.contiguous()) for k2, v in inp.items()}
        print(f"\n# B={B} T={T} H={H} K={K} BT={BT}")
        for nw in (2, 4):
            fb = lambda: intra_triton_base(base_k, inp, BT, 16, num_warps=nw)
            fm = lambda: intra_triton_mma(mma_k, inp, BT, 16, num_warps=nw)
            fb(); fm(); torch.cuda.synchronize()
            db_, _ = profile_dev(fb, reps=30)
            dm_, _ = profile_dev(fm, reps=30)
            wb = time_wall(fb, reps=50)
            wm = time_wall(fm, reps=50)
            print(f"  nw={nw}: base dev={db_:7.1f} wall={wb:7.1f} | "
                  f"mma dev={dm_:7.1f} wall={wm:7.1f} | dev {dm_/db_:.2f}x wall {wm/wb:.2f}x")


# --------------------------------------------------------------------------- chain
def _make_chain_mma(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, nw_b3, mma):
    dhu_k = build_dhu_bf16_pipe()
    dqkg_k = build_dqkg()
    dav_k = build_dav_kernel()
    intra_k = build_intra_mma() if mma else build_intra_base()
    intra_fn = intra_triton_mma if mma else intra_triton_base
    v_f = inp["v"].float()

    def run():
        dAqk, dv = dav_triton(dav_k, fi_b["v_new"], fi_b["Aqk"], do, scale, CHUNK)
        dh, dh0, dv2 = dhu_bf16_pipe_triton(dhu_k, fi_b, do_b, dv, scale, CHUNK,
                                            NS=NS, dht=dfin)
        inp2 = dict(q=qn, k=kn, v=v_f, v_new=fi_b["v_new"], g=fi_b["g_cum"],
                    beta=fi_b["beta"], A=fi_b["Akk"], h=fi_b["h"], do=do, dh=dh, dv=dv2)
        b2 = wy_dqkg_triton(dqkg_k, inp2, scale, CHUNK)
        inp3 = dict(q=qn, k=kn, g=fi_b["g_cum"], beta=fi_b["beta"],
                    dAqk=dAqk, dAkk=b2["dA"])
        b3 = intra_fn(intra_k, inp3, CHUNK, CHUNK, num_warps=nw_b3)
        dg = chunk_local_cumsum(b2["dg"] + b3["dg"], chunk_size=CHUNK, reverse=True)
        return dict(dq=b2["dq"] + b3["dq"], dk=b2["dk"] + b3["dk"],
                    dv=b2["dv"], dg=dg, dbeta=b2["db"] + b3["db"], dh0=dh0,
                    dh=dh, dv2=dv2)

    return run


def _prep_b(fi, inp, do):
    fi_b = dict(fi)
    for kk in ("qg", "kg", "w"):
        fi_b[kk] = fi[kk].to(torch.bfloat16)
    return fi_b, do.to(torch.bfloat16)


def part_chain(cases, scale, lower_bound):
    from fla.ops.kda import chunk_kda
    print("\n" + "=" * 78)
    print("Part C: land MMA B3 into the assembled chain -- correctness vs FLA and baseline")
    print("=" * 78)
    NS = 3
    worst = {}
    for (name, B, T, H, seed, gs) in cases:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        fi_b, do_b = _prep_b(fi, inp, do)
        ch_base = _make_chain_mma(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, 4, mma=False)
        ch_mma = _make_chain_mma(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, 4, mma=True)
        r_base, r_mma = ch_base(), ch_mma()
        gold = fla_golden(qn, kn, inp["v"].float(), beta_act, g_fla, inp["h0"], scale, do, dfin)
        print(f"\n# {name}")
        print("  -- MMA chain vs FLA --")
        for fld, gk in [("dq", "q"), ("dk", "k"), ("dv", "v"), ("dbeta", "beta"), ("dg", "g")]:
            _, mr = stats(f"  mma {fld} vs FLA", r_mma[fld], gold[gk])
            worst[fld] = max(worst.get(fld, 0.0), mr)
        _, mr = stats("  mma dh0 vs FLA", r_mma["dh0"], gold["h0"].transpose(-1, -2))
        worst["dh0"] = max(worst.get("dh0", 0.0), mr)
        print("  -- MMA chain vs baseline chain (the rewrite's own delta) --")
        for fld in ["dq", "dk", "dv", "dbeta", "dg"]:
            _, mr = stats(f"  delta {fld}", r_mma[fld], r_base[fld])
            worst[f"delta_{fld}"] = max(worst.get(f"delta_{fld}", 0.0), mr)
    print("\nworst-case max_rel (MMA chain):")
    for kk in sorted(worst):
        print(f"  {kk:16s}  {worst[kk]:.3e}")


def part_chain_timing(cases, scale, lower_bound):
    from fla.ops.kda import chunk_kda
    print("\n" + "=" * 78)
    print("Part C2: chain timing -- baseline B3 vs MMA B3 (dev CUPTI + wall), vs FLA")
    print("=" * 78)
    NS = 3
    print(f"  {'case':18s} | {'FLA dev':>8s} {'base dev':>8s} {'mma dev':>8s} "
          f"{'mma/base':>8s} | {'base wall':>9s} {'mma wall':>8s}")
    for (name, B, T, H, seed, gs) in cases:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        fi_b, do_b = _prep_b(fi, inp, do)
        ch_base = _make_chain_mma(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, 4, mma=False)
        ch_mma = _make_chain_mma(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, 4, mma=True)
        qa = qn.clone().requires_grad_(True)
        ka = kn.clone().requires_grad_(True)
        va = inp["v"].float().clone().requires_grad_(True)
        ba = beta_act.clone().requires_grad_(True)
        ga = g_fla.clone().requires_grad_(True)
        ha = inp["h0"].clone().requires_grad_(True)
        o, s = chunk_kda(qa, ka, va, ga, ba, scale=scale,
                         initial_state=ha.float().transpose(-1, -2), output_final_state=True,
                         use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                         use_beta_sigmoid_in_kernel=False)
        loss = (o.float() * do).sum() + (s.float() * dfin).sum()
        holders = (qa, ka, va, ba, ga, ha)

        def fla_bwd():
            loss.backward(retain_graph=True)
            for t in holders:
                t.grad = None

        ch_base(); ch_mma(); fla_bwd()
        torch.cuda.synchronize()
        df, _ = profile_dev(fla_bwd)
        db_, _ = profile_dev(ch_base)
        dm_, _ = profile_dev(ch_mma)
        wb = time_wall(ch_base, reps=50)
        wm = time_wall(ch_mma, reps=50)
        print(f"  {name:18s} | {df:8.1f} {db_:8.1f} {dm_:8.1f} {dm_/db_:7.2f}x | "
              f"{wb:9.1f} {wm:8.1f}")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}  "
          f"triton={triton.__version__}")
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0
    mode = os.environ.get("B3M_MODE", "")
    if mode in ("", "correctness"):
        part_correctness()
    if mode in ("", "timing"):
        part_timing()
    if mode in ("", "chain"):
        part_chain(CASES[:int(os.environ.get("B3M_CASES", len(CASES)))], scale, lower_bound)
        part_chain_timing(TCASES, scale, lower_bound)
    print("\nDONE")


if __name__ == "__main__":
    main()
