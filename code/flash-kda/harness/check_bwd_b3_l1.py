"""P7-29: B3 `intra` L1 locality -- kill the redundant per-program reads.

Motivation (P7-24 §27.7 / P7-28 §31.3)
---------------------------------------
P7-24 MMA-ized the diagonal sub-blocks of B3 (the *whole* kernel at the deployed
`BC=BT=16 -> NC=1` granularity), giving `tensor 0.05% -> 2.61%`, `Duration 527->257us`
but leaving B3 **L2/访存 bound** (`L2 74%`, `DRAM 57%`, `L1 hit rate 62%->20%`,
`long_scoreboard 3.22->16.71`).  §27.7/§31.3 left the next lever as "reduce the diagonal
tiling's repeated reads / improve its L1 locality".

Two concrete redundant reads live in the P7-24 kernel at NC=1:

 (R1) **`k`/`g` are loaded twice per program.**  Part B/D read the diagonal key rows
      `j = i_ti + o_i` -- but `o_c = i_ti + o_i`, so `p_kd`/`p_gd`
      (`_intra_kernel_mma:142-143`) are *byte-identical addresses* to `p_k`/`p_g`
      (`:98-99`), with the same mask `m_jk == m_ck`.  The kernel re-issues those loads.

 (R2) **the diagonal `dAqk`/`dAkk` tiles are re-read once per `k`-block.**  The grid is
      `(NK*NC, NT, B*H)` with `BK=min(32,K)`, so `NK=ceil(K/32)=4` programs per chunk
      each load the same `[BC,BC]` diagonal tile (`:135-138`), which depends only on
      `(chunk, head)`, not on the `k`-block.  With `BK=K` (`NK=1`) each program reads it
      exactly once.

This script is the controlled experiment:

  Part A  isolated correctness: every variant == the P7-24 MMA kernel where the algebra
          is identical (bit-exact expected for `reuse`/`bk`), vs elementwise baseline.
  Part B  isolated timing sweep: `reuse_kg` x `BK in {32,64,128}` x `nw in {4,8}`.
  Part C  land the winner into the assembled chain: correctness vs FLA + timing.
  Part D  bound analysis in `probe_b3_l1_ncu.out.txt` (ncu).

Run inside `kernel_lab` (GPU, FlashKDA built, FLA installed), idle card:
    CUDA_VISIBLE_DEVICES=1 python .../check_bwd_b3_l1.py
    B3L_MODE=correctness | timing | chain
"""
import math
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch
import triton
import triton.language as tl

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_ws_reuse import D, CHUNK, stats, cdiv                                    # noqa: E402
from check_bwd_intra_kernel import (make_inputs, fla_intra,                            # noqa: E402
                                    build_kernel as build_intra_base,
                                    intra_triton as intra_triton_base)
from check_bwd_b3_mma import (build_intra_mma, intra_triton_mma, _make_chain_mma,        # noqa: E402
                              CASES, TCASES, build_dhu_bf16_pipe, build_dqkg,
                              build_dav_kernel)
from check_bwd_b1_pipe_e2e import dhu_bf16_pipe_triton                                  # noqa: E402
from check_bwd_dqkg_kernel import wy_dqkg_triton                                        # noqa: E402
from check_bwd_e2e import dav_triton                                                    # noqa: E402
from fla.ops.utils import chunk_local_cumsum                                            # noqa: E402
from check_bwd_b1_bf16_e2e import prep, fla_golden, time_wall, profile_dev              # noqa: E402


# --------------------------------------------------------------- parameterized MMA kernel
def build_intra_mma_v2(reuse_kg=True, hoist=False):
    """The P7-24 MMA kernel with the NC=1 redundant reads removed.

    reuse_kg : do not reload k/g for the diagonal key rows (R1) -- `b_kd=b_k`, `b_gd=b_g`.
    hoist    : pull the per-`k`-block-invariant diagonal `dAqk`/`dAkk`/beta loads *and* the
               `i_k` loop into a single programinator loop (grid `(NC,NT,B*H)`, R2 by
               explicit reuse).  When False, `dAqk`/`dAkk` are still re-read per program
               unless the caller launches with `BK=K` (NK=1).
    """
    @triton.jit
    def _intra_kernel_mma_v2(
        q_ptr, k_ptr, g_ptr, beta_ptr, dAqk_ptr, dAkk_ptr,
        dq_ptr, dk_ptr, db_ptr, dg_ptr,
        T,
        H: tl.constexpr, K: tl.constexpr,
        BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr,
        REUSE_KG: tl.constexpr,
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
        m_jd = (i_ti + o_i) < T
        m_diag = m_c[:, None] & m_jd[None, :]
        p_dAqd = dAqk_ptr + ((i_b * T + o_c)[:, None] * H + i_h) * BT + (i_i * BC + o_i)[None, :]
        p_dAkkd = dAkk_ptr + ((i_b * T + o_c)[:, None] * H + i_h) * BT + (i_i * BC + o_i)[None, :]
        b_dAqd = tl.load(p_dAqd, mask=m_diag, other=0.0).to(tl.float32)
        b_dAkkd = tl.load(p_dAkkd, mask=m_diag, other=0.0).to(tl.float32)
        if REUSE_KG:
            # p_kd/p_gd (P7-24 `:142-143`, rows i_ti+o_i == o_c, mask m_jk == m_ck) are
            # byte-identical to p_k/p_g -> reuse instead of reloading.
            b_kd = b_k
            b_gd = b_g
        else:
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

    _REUSE_KG[_intra_kernel_mma_v2] = reuse_kg
    return _intra_kernel_mma_v2


def build_intra_loop(bk=32):
    """NC=1 (deployment) B3 diagonal kernel with the k-blocks *looped* instead of
    gridded, so `tl.range(..., num_stages)` software-pipelines the q/k/g loads across
    k-blocks (the direct attack on the P7-24 `long_scoreboard 16.71` / `L1 hit 20%`).
    The diagonal `dAqk`/`dAkk` tiles and beta are loop-invariant -> loaded once (R2)."""
    @triton.jit
    def _intra_kernel_loop(
        q_ptr, k_ptr, g_ptr, beta_ptr, dAqk_ptr, dAkk_ptr,
        dq_ptr, dk_ptr, db_ptr, dg_ptr,
        T,
        H: tl.constexpr, K: tl.constexpr,
        BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, NK: tl.constexpr,
        NS: tl.constexpr,
    ):
        i_t = tl.program_id(0)
        i_bh = tl.program_id(1)
        i_b = i_bh // H
        i_h = i_bh % H
        i_ti = i_t * BT
        o_i = tl.arange(0, BC)
        o_c = i_ti + o_i
        m_c = o_c < T
        m_jd = (i_ti + o_i) < T
        m_diag = m_c[:, None] & m_jd[None, :]

        # loop-invariant diagonal dAqk/dAkk [BC,BC] + beta (R2: read once, not per k-block)
        p_dAqd = dAqk_ptr + ((i_b * T + o_c)[:, None] * H + i_h) * BT + o_i[None, :]
        p_dAkkd = dAkk_ptr + ((i_b * T + o_c)[:, None] * H + i_h) * BT + o_i[None, :]
        b_dAqd = tl.load(p_dAqd, mask=m_diag, other=0.0).to(tl.float32)
        b_dAkkd = tl.load(p_dAkkd, mask=m_diag, other=0.0).to(tl.float32)
        b_b = tl.load(beta_ptr + (i_b * T + o_c) * H + i_h, mask=m_c, other=0.0)
        m_ge = o_i[:, None] >= o_i[None, :]
        m_le = o_i[:, None] <= o_i[None, :]
        b_Mq = tl.where(m_ge, b_dAqd, 0.0)
        b_Mk = tl.where(m_ge, b_dAkkd, 0.0)
        b_Sq = tl.where(m_le, tl.trans(b_dAqd), 0.0)
        b_Sk = tl.where(m_le, tl.trans(b_dAkkd), 0.0)

        for i_k in tl.range(0, NK, num_stages=NS):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            m_ck = m_c[:, None] & m_k[None, :]
            p_q = q_ptr + (i_b * T + o_c)[:, None] * (H * K) + i_h * K + o_k[None, :]
            p_k = k_ptr + (i_b * T + o_c)[:, None] * (H * K) + i_h * K + o_k[None, :]
            p_g = g_ptr + (i_b * T + o_c)[:, None] * (H * K) + i_h * K + o_k[None, :]
            b_q = tl.load(p_q, mask=m_ck, other=0.0)
            b_k = tl.load(p_k, mask=m_ck, other=0.0)
            b_g = tl.load(p_g, mask=m_ck, other=0.0).to(tl.float32)
            b_eg = tl.exp2(b_g)
            b_KG = b_k * tl.exp2(-b_g)
            b_dq2 = b_eg * tl.dot(b_Mq, b_KG)
            b_dk2 = b_eg * tl.dot(b_Mk, b_KG)
            b_db = tl.sum(b_dk2 * b_k, axis=1)
            b_dk2 *= b_b[:, None]
            b_dg2 = b_q * b_dq2
            p_dq = dq_ptr + (i_b * T + o_c)[:, None] * (H * K) + i_h * K + o_k[None, :]
            b_dq2 = b_dq2 + tl.load(p_dq, mask=m_ck, other=0.0)
            tl.store(p_dq, b_dq2.to(p_dq.dtype.element_ty), mask=m_ck)
            tl.atomic_add(db_ptr + (i_b * T + o_c) * H + i_h,
                          b_db.to(db_ptr.dtype.element_ty), mask=m_c)
            b_QG = b_q * b_eg
            b_KGd = (b_k * b_b[:, None]) * b_eg
            b_emg = tl.exp2(-b_g)
            b_dkt = b_emg * tl.dot(b_Sq, b_QG)
            b_dkt += b_emg * tl.dot(b_Sk, b_KGd)
            p_dk = dk_ptr + (i_b * T + o_c)[:, None] * (H * K) + i_h * K + o_k[None, :]
            p_dg = dg_ptr + (i_b * T + o_c)[:, None] * (H * K) + i_h * K + o_k[None, :]
            b_dg2 = b_dg2 + (b_dk2 - b_dkt) * b_k + tl.load(p_dg, mask=m_ck, other=0.0)
            b_dk2 = b_dk2 + tl.load(p_dk, mask=m_ck, other=0.0) + b_dkt
            tl.store(p_dk, b_dk2.to(p_dk.dtype.element_ty), mask=m_ck)
            tl.store(p_dg, b_dg2.to(p_dg.dtype.element_ty), mask=m_ck)

    return _intra_kernel_loop


def intra_triton_loop(kernel, inp, BT, BC, bk=32, num_warps=4, num_stages=1):
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT)
    assert BC == BT, "loop variant is NC=1 only"
    BK = bk
    NK = cdiv(K, bk)
    dev = inp["q"].device
    dq = torch.zeros(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.zeros(B, T, H, K, device=dev, dtype=torch.float32)
    db = torch.zeros(B, T, H, device=dev, dtype=torch.float32)
    dg = torch.zeros(B, T, H, K, device=dev, dtype=torch.float32)
    grid = (NT, B * H)
    kernel[grid](
        inp["q"], inp["k"], inp["g"], inp["beta"], inp["dAqk"], inp["dAkk"],
        dq, dk, db, dg,
        T, H=H, K=K, BT=BT, BC=BC, BK=BK, NK=NK, NS=num_stages,
        num_warps=num_warps, num_stages=num_stages)
    return dict(dq=dq, dk=dk, db=db, dg=dg)


_REUSE_KG = {}


def intra_triton_v2(kernel, inp, BT, BC, bk=None, num_warps=4, num_stages=1):
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT)
    NC = cdiv(BT, BC)
    if bk is None:
        bk = min(32, triton.next_power_of_2(K))
    NK = cdiv(K, bk)
    dev = inp["q"].device
    dq = torch.zeros(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.zeros(B, T, H, K, device=dev, dtype=torch.float32)
    db = torch.zeros(B, T, H, device=dev, dtype=torch.float32)
    dg = torch.zeros(B, T, H, K, device=dev, dtype=torch.float32)
    grid = (NK * NC, NT, B * H)
    kernel[grid](
        inp["q"], inp["k"], inp["g"], inp["beta"], inp["dAqk"], inp["dAkk"],
        dq, dk, db, dg,
        T, H=H, K=K, BT=BT, BC=BC, BK=bk, NC=NC, REUSE_KG=_REUSE_KG[kernel],
        num_warps=num_warps, num_stages=num_stages)
    return dict(dq=dq, dk=dk, db=db, dg=dg)


# --------------------------------------------------------------------------------- parts
def part_correctness():
    print("=" * 78)
    print("Part A: isolated B3 -- v2(reuse) / v2(noreuse) / v2(BK=K) vs P7-24 MMA & base")
    print("=" * 78)
    base_k = build_intra_base()
    mma_k = build_intra_mma()
    v2_reuse = build_intra_mma_v2(reuse_kg=True)
    v2_noreuse = build_intra_mma_v2(reuse_kg=False)
    loop_k = build_intra_loop()
    worst = {}
    for (B, T, H, K, BT, BC, seed) in [
        (1, 256, 2, 64, 64, 16, 0), (2, 512, 4, 128, 64, 16, 1),
        (1, 250, 3, 64, 64, 16, 2), (1, 1024, 8, 128, 64, 16, 3),
        (2, 513, 2, 64, 64, 16, 4), (1, 64, 1, 32, 64, 16, 5),
        (1, 128, 2, 64, 16, 16, 6), (1, 250, 3, 128, 16, 16, 7),
    ]:
        print(f"\n=== B={B} T={T} H={H} K={K} BT={BT} BC={BC} seed={seed} ===")
        inp = make_inputs(B, T, H, K, BT, BC, "cuda", dtype=torch.float32, seed=seed)
        fla = fla_intra(inp, BT, BC)
        ref = intra_triton_mma(mma_k, inp, BT, BC)
        r_reuse = intra_triton_v2(v2_reuse, inp, BT, BC)
        r_nore = intra_triton_v2(v2_noreuse, inp, BT, BC)
        r_bkK = intra_triton_v2(v2_reuse, inp, BT, BC, bk=triton.next_power_of_2(K))
        r_loop = (intra_triton_loop(loop_k, inp, BT, BC, bk=32, num_stages=2)
                  if BC == BT else None)
        variants = [("reuse", r_reuse), ("noreuse", r_nore), ("bkK", r_bkK)]
        if r_loop is not None:
            variants.append(("loop_ns2", r_loop))
        for tag, r in variants:
            for name in ["dq", "dk", "db", "dg"]:
                _, mr = stats(f"  {tag} {name} vs P7-24", r[name], ref[name])
                worst[f"{tag}_{name}"] = max(worst.get(f"{tag}_{name}", 0.0), mr)
        # sanity: reuse/noreuse/bkK vs FLA
        for tag, r in [("reuse", r_reuse), ("bkK", r_bkK)]:
            for name in ["dq", "dk", "db", "dg"]:
                _, mr = stats(f"  {tag} {name} vs FLA", r[name], fla[name])
                worst[f"vsfla_{name}"] = max(worst.get(f"vsfla_{name}", 0.0), mr)
    print("\nworst-case max_rel (B3 v2):")
    for kk in sorted(worst):
        print(f"  {kk:20s}  {worst[kk]:.3e}")


def part_timing():
    print("\n" + "=" * 78)
    print("Part B: isolated B3 timing -- reuse_kg x BK x nw, BT=16 (deploy) & BT=64")
    print("=" * 78)
    base_k = build_intra_base()
    mma_k = build_intra_mma()
    v2_reuse = build_intra_mma_v2(reuse_kg=True)
    v2_noreuse = build_intra_mma_v2(reuse_kg=False)
    loop_k = build_intra_loop()
    for (B, T, H, K, BT) in [(1, 512, 4, 128, 64), (2, 2048, 8, 128, 64),
                             (1, 512, 4, 128, 16), (2, 2048, 8, 128, 16),
                             (8, 2048, 8, 128, 16)]:
        inp = make_inputs(B, T, H, K, BT, 16, "cuda", dtype=torch.float32, seed=0)
        inp = {k2: (v if v.is_contiguous() else v.contiguous()) for k2, v in inp.items()}
        print(f"\n# B={B} T={T} H={H} K={K} BT={BT}")
        fbf = lambda: intra_triton_base(base_k, inp, BT, 16, num_warps=4)
        fmf = lambda: intra_triton_mma(mma_k, inp, BT, 16, num_warps=4)
        fbf(); fmf(); torch.cuda.synchronize()
        dbf, _ = profile_dev(fbf, reps=30)
        dmf, _ = profile_dev(fmf, reps=30)
        print(f"  baseline dev={dbf:7.1f} | P7-24 mma dev={dmf:7.1f}")
        for nw in (4, 8):
            for bk in (32, 64, 128):
                fr = lambda: intra_triton_v2(v2_reuse, inp, BT, 16, bk=bk, num_warps=nw)
                fn = lambda: intra_triton_v2(v2_noreuse, inp, BT, 16, bk=bk, num_warps=nw)
                fr(); fn(); torch.cuda.synchronize()
                dr, _ = profile_dev(fr, reps=30)
                dn, _ = profile_dev(fn, reps=30)
                print(f"  nw={nw} bk={bk:3d}: reuse dev={dr:7.1f} ({dr/dmf:.2f}x mma, "
                      f"{dr/dbf:.2f}x base) | noreuse dev={dn:7.1f} ({dn/dr:.2f}x reuse)")
        if BT == 16:
            for nw in (4, 8):
                for bk in (32, 64):
                    for ns in (1, 2, 3):
                        fl = lambda: intra_triton_loop(loop_k, inp, BT, 16, bk=bk,
                                                       num_warps=nw, num_stages=ns)
                        fl(); torch.cuda.synchronize()
                        dl, _ = profile_dev(fl, reps=30)
                        print(f"  LOOP nw={nw} bk={bk:3d} ns={ns}: dev={dl:7.1f} "
                              f"({dl/dmf:.2f}x mma, {dl/dbf:.2f}x base)")


# --------------------------------------------------------------------------------- chain
def _make_chain_v2(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, nw_b3, bk,
                   b3_loop=False, b3_ns=2):
    dhu_k = build_dhu_bf16_pipe()
    dqkg_k = build_dqkg()
    dav_k = build_dav_kernel()
    intra_k = build_intra_loop(32) if b3_loop else build_intra_mma_v2(reuse_kg=True)
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
        if b3_loop:
            b3 = intra_triton_loop(intra_k, inp3, CHUNK, CHUNK, bk=bk, num_warps=nw_b3,
                                   num_stages=b3_ns)
        else:
            b3 = intra_triton_v2(intra_k, inp3, CHUNK, CHUNK, bk=bk, num_warps=nw_b3)
        dg = chunk_local_cumsum(b2["dg"] + b3["dg"], chunk_size=CHUNK, reverse=True)
        return dict(dq=b2["dq"] + b3["dq"], dk=b2["dk"] + b3["dk"],
                    dv=b2["dv"], dg=dg, dbeta=b2["db"] + b3["db"], dh0=dh0,
                    dh=dh, dv2=dv2)

    return run


def part_chain(cases, scale, lower_bound):
    print("\n" + "=" * 78)
    print("Part C: land B3 v2 into the assembled chain -- correctness vs FLA + timing")
    print("=" * 78)
    from fla.ops.kda import chunk_kda
    NS = 3
    worst = {}
    for (name, B, T, H, seed, gs) in cases:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        fi_b, do_b = _prep_b(fi, inp, do)
        ch_ref = _make_chain_mma(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, 2, mma=True)
        ch_v2 = _make_chain_v2(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, 2, bk=64)
        ch_loop = _make_chain_v2(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, 2, bk=64,
                                 b3_loop=True, b3_ns=2)
        r_ref, r_v2, r_loop = ch_ref(), ch_v2(), ch_loop()
        gold = fla_golden(qn, kn, inp["v"].float(), beta_act, g_fla, inp["h0"], scale, do, dfin)
        print(f"\n# {name}")
        for tag, r in [("v2", r_v2), ("loop", r_loop)]:
            for fld, gk in [("dq", "q"), ("dk", "k"), ("dv", "v"), ("dbeta", "beta"), ("dg", "g")]:
                _, mr = stats(f"  {tag} {fld} vs FLA", r[fld], gold[gk])
                worst[f"{tag}_{fld}"] = max(worst.get(f"{tag}_{fld}", 0.0), mr)
        for tag, r in [("v2", r_v2), ("loop", r_loop)]:
            _, mr = stats(f"  {tag} dh0 vs FLA", r["dh0"], gold["h0"].transpose(-1, -2))
            worst[f"{tag}_dh0"] = max(worst.get(f"{tag}_dh0", 0.0), mr)
        for tag, r in [("v2", r_v2), ("loop", r_loop)]:
            print(f"  -- {tag} chain vs P7-24 chain (own delta) --")
            for fld in ["dq", "dk", "dv", "dbeta", "dg"]:
                _, mr = stats(f"  delta {fld}", r[fld], r_ref[fld])
                worst[f"delta_{tag}_{fld}"] = max(worst.get(f"delta_{tag}_{fld}", 0.0), mr)
    print("\nworst-case max_rel (v2 chain):")
    for kk in sorted(worst):
        print(f"  {kk:16s}  {worst[kk]:.3e}")


def part_chain_timing(cases, scale, lower_bound):
    print("\n" + "=" * 78)
    print("Part C2: chain timing -- P7-24 B3 vs v2 B3 (dev CUPTI + wall), vs FLA")
    print("=" * 78)
    from fla.ops.kda import chunk_kda
    NS = 3
    print(f"  {'case':18s} | {'FLA dev':>8s} {'ref dev':>8s} {'v2 dev':>8s} {'loop dev':>8s} "
          f"{'v2/ref':>7s} {'lp/ref':>7s}")
    for (name, B, T, H, seed, gs) in cases:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        fi_b, do_b = _prep_b(fi, inp, do)
        ch_ref = _make_chain_mma(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, 2, mma=True)
        ch_v2 = _make_chain_v2(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, 2, bk=64)
        ch_loop = _make_chain_v2(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, 2, bk=64,
                                 b3_loop=True, b3_ns=2)
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

        ch_ref(); ch_v2(); ch_loop(); fla_bwd()
        torch.cuda.synchronize()
        df, _ = profile_dev(fla_bwd)
        dr, _ = profile_dev(ch_ref)
        dv, _ = profile_dev(ch_v2)
        dl, _ = profile_dev(ch_loop)
        print(f"  {name:18s} | {df:8.1f} {dr:8.1f} {dv:8.1f} {dl:8.1f} "
              f"{dv/dr:6.2f}x {dl/dr:6.2f}x")


def _prep_b(fi, inp, do):
    fi_b = dict(fi)
    for kk in ("qg", "kg", "w"):
        fi_b[kk] = fi[kk].to(torch.bfloat16)
    return fi_b, do.to(torch.bfloat16)


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}  "
          f"triton={triton.__version__}")
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0
    mode = os.environ.get("B3L_MODE", "")
    if mode in ("", "correctness"):
        part_correctness()
    if mode in ("", "timing"):
        part_timing()
    if mode in ("", "chain"):
        part_chain(CASES[:int(os.environ.get("B3L_CASES", len(CASES)))], scale, lower_bound)
        part_chain_timing(TCASES, scale, lower_bound)
    print("\nDONE")


if __name__ == "__main__":
    main()
