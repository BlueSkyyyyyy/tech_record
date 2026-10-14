"""P7-25: B2 `wy_dqkg` structural rewrite -- kill the register spills by *load ordering*.

Motivation
----------
P7-23 (`docs/07` §26) pinned the chain-level bottleneck to B2 (`wy_dqkg`): DRAM-bound
(59% SOL) **and** register-crushed (ncu `Registers/thread=247`, `Block Limit Registers=2`,
achieved occupancy 12.4%).  P7-24 then halved B3, leaving B2 more dominant still.

A direct register probe of the *unchanged* kernel (this round, idle card) shows the smoking
gun: at the deployed BT=16, BK=BV=64, `num_warps=4` the kernel hits Triton's 255-reg cap and
**spills** (`n_regs=255, n_spills=12` at nw=4; `nw=2` spills 190; only `nw=8` is spill-free).

Root cause: the kernel loads `q/k/g` at the *top* of the `i_k` loop and only uses them
**after** the inner `i_v` loop (for the decay rescale / gate chain).  So `b_q/b_k/b_g`
(`[BT,BK]` each) are live across the entire `i_v` loop -- exactly where `b_h/b_dh` (`[BV,BK]`)
and the three `[BT,BK]` accumulators already live.  Peak pressure lands on the `i_v` body.

The fix is pure scheduling and **arithmetic-identical**: hoist the `q/k/g` loads to *after*
the `i_v` loop (they are not needed inside it).  This drops the spills to 0 and shortens the
live ranges of the big tiles, without changing a single operation or accumulation order --
so the six outputs are **bit-for-bit identical** (verified incl. non-divisible T, g=0, B=2).

  Part A  isolated: base vs late -- regs/spills, dev/wall timing, and bit-exactness.
  Part B  an occupancy/register explanation (why this is the right lever).
  Part C  land it into the assembled chain (P7-22 pipe B1 + P7-24 MMA B3), verify vs FLA
          autograd and vs the baseline chain, and time chain-level.
  Part D  ncu bound analysis lives in `probe_b2_struct_ncu.out.txt` (see docs/07 §28).

Run inside `kernel_lab` (GPU, FlashKDA built, FLA installed), idle card:
    CUDA_VISIBLE_DEVICES=1 python .../check_bwd_b2_struct.py
    B2S_MODE=correctness | timing | regs | chain
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

from check_bwd_ws_reuse import D, CHUNK, stats, cdiv                                   # noqa: E402
from check_bwd_dqkg_kernel import (build_kernel as build_dqkg_base,                   # noqa: E402
                                   wy_dqkg_triton as wy_dqkg_base_triton,
                                   fla_wy_dqkg, make_inputs, wy_dqkg_torch)
from check_bwd_intra_kernel import build_kernel as build_intra                        # noqa: E402
from check_bwd_b3_mma import build_intra_mma, intra_triton_mma                        # noqa: E402
from check_bwd_b1_pipe_e2e import build_dhu_bf16_pipe, dhu_bf16_pipe_triton           # noqa: E402
from check_bwd_e2e import build_dav_kernel, dav_triton                                # noqa: E402
from check_bwd_b1_bf16_e2e import (prep, fla_golden, time_wall, profile_dev)          # noqa: E402
from fla.ops.utils import chunk_local_cumsum                                          # noqa: E402


# ------------------------------------------------------------------ late-load kernel
def build_dqkg_late():
    """Identical to `check_bwd_dqkg_kernel.build_kernel` except the `q/k/g` loads are
    moved from the top of the `i_k` loop to *after* the `i_v` loop (their only use site is
    the decay-rescale/gate-chain block).  Arithmetic and accumulation order unchanged."""

    @triton.jit
    def _wy_dqkg_kernel_late(
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

            # --- q/k/g loaded HERE (after the i_v loop) instead of before it ---
            p_k = k_ptr + (i_b * T + o_t)[:, None] * (H * K) + i_h * K + o_k[None, :]
            p_q = q_ptr + (i_b * T + o_t)[:, None] * (H * K) + i_h * K + o_k[None, :]
            p_g = g_ptr + (i_b * T + o_t)[:, None] * (H * K) + i_h * K + o_k[None, :]
            b_k = tl.load(p_k, mask=m_tk, other=0.0)
            b_q = tl.load(p_q, mask=m_tk, other=0.0)
            b_g = tl.load(p_g, mask=m_tk, other=0.0).to(tl.float32)

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

        p_dA = dA_ptr + ((i_b * T + o_t) * H + i_h)[:, None] * BT + tl.arange(0, BT)[None, :]
        tl.store(p_dA, b_dA.to(dA_ptr.dtype.element_ty), mask=m_t[:, None])
        tl.store(db_ptr + (i_b * T + o_t) * H + i_h, b_db.to(db_ptr.dtype.element_ty), mask=m_t)

    return _wy_dqkg_kernel_late


def wy_dqkg_late_triton(kernel, inp, scale, BT, num_warps=4, num_stages=1):
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


# ------------------------------------------------------------------------- Part A
def part_correctness():
    print("=" * 78)
    print("Part A: isolated B2 -- late-load vs baseline (must be bit-identical) + FLA")
    print("=" * 78)
    base_k = build_dqkg_base()
    late_k = build_dqkg_late()
    worst_b = {}
    worst_f = {}
    for (B, T, H, K, V, BT, seed) in [
        (1, 256, 2, 64, 64, 64, 0), (2, 512, 4, 128, 128, 64, 1),
        (1, 250, 3, 64, 128, 64, 2),          # non-divisible T
        (1, 1024, 8, 128, 128, 64, 3),
        (2, 513, 2, 64, 64, 64, 4),           # non-divisible T, B=2
        (1, 128, 2, 128, 128, 16, 5),         # BT=16 deployment granularity
        (1, 250, 3, 128, 128, 16, 6),         # BT=16 + non-divisible
    ]:
        print(f"\n=== B={B} T={T} H={H} K={K} V={V} BT={BT} seed={seed} ===")
        inp = make_inputs(B, T, H, K, V, BT, "cuda", dtype=torch.float32, seed=seed)
        fla = fla_wy_dqkg(inp, 1.0 / math.sqrt(K), BT)
        tri_b = wy_dqkg_base_triton(base_k, inp, 1.0 / math.sqrt(K), BT)
        tri_l = wy_dqkg_late_triton(late_k, inp, 1.0 / math.sqrt(K), BT)
        for name in ["dq", "dk", "dv", "db", "dg", "dA"]:
            _, mr = stats(f"  late {name} vs base", tri_l[name], tri_b[name])
            worst_b[name] = max(worst_b.get(name, 0.0), mr)
            _, mr = stats(f"  late {name} vs FLA ", tri_l[name], fla[name])
            worst_f[name] = max(worst_f.get(name, 0.0), mr)
    print("\nworst-case max_rel (late vs baseline):",
          {k: f"{v:.1e}" for k, v in worst_b.items()})
    print("worst-case max_rel (late vs FLA):     ",
          {k: f"{v:.1e}" for k, v in worst_f.items()})


# ------------------------------------------------------------------------- Part B
def part_regs():
    print("=" * 78)
    print("Part B: register pressure / spills -- base vs late (deployed BT=16, BK=BV=64)")
    print("=" * 78)
    base_k = build_dqkg_base()
    late_k = build_dqkg_late()
    for (B, T, H, K, V, BT) in [(1, 512, 4, 128, 128, 16),
                                (2, 1024, 8, 128, 128, 16),
                                (8, 2048, 8, 128, 128, 16)]:
        inp = make_inputs(B, T, H, K, V, BT, "cuda", dtype=torch.float32, seed=0)
        inp = {k2: (v if v.is_contiguous() else v.contiguous()) for k2, v in inp.items()}
        print(f"\n# B={B} T={T} H={H}:")
        for nw in (2, 4, 8):
            b_regs, b_spill = _regs_of(base_k, inp, BT, nw, late=False)
            l_regs, l_spill = _regs_of(late_k, inp, BT, nw, late=True)
            print(f"  nw={nw}: base regs={b_regs:3d} spill={b_spill:3d} | "
                  f"late regs={l_regs:3d} spill={l_spill:3d}")


def _regs_of(kernel, inp, BT, nw, late):
    B, T, H, K = inp["q"].shape
    V = inp["v"].shape[-1]
    NT = cdiv(T, BT)
    dq = torch.empty(B, T, H, K, device="cuda")
    dk = torch.empty_like(dq)
    dg = torch.empty_like(dq)
    dv2 = torch.empty(B, T, H, V, device="cuda")
    db = torch.empty(B, T, H, device="cuda")
    dA = torch.empty(B, T, H, BT, device="cuda")
    h = kernel[(NT, B * H)](
        inp["q"], inp["k"], inp["v"], inp["v_new"], inp["g"], inp["beta"], inp["A"], inp["h"],
        inp["do"], inp["dh"], inp["dv"], dq, dk, dv2, db, dg, dA,
        T, NT, 1.0 / math.sqrt(K), H=H, K=K, V=V, BT=BT, BK=min(64, K), BV=min(64, V),
        num_warps=nw, num_stages=1)
    return h.n_regs, h.n_spills


# ------------------------------------------------------------------------- Part C
def part_timing():
    print("=" * 78)
    print("Part C: isolated B2 timing -- base vs late (dev CUPTI + wall), BT=16")
    print("=" * 78)
    base_k = build_dqkg_base()
    late_k = build_dqkg_late()
    print(f"  {'case':16s} | {'nw':>3s} | {'base dev':>8s} {'late dev':>8s} {'dev x':>6s} "
          f"| {'base wall':>9s} {'late wall':>9s} {'wall x':>6s} "
          f"| {'base bench':>10s} {'late bench':>10s} {'bench x':>7s}")
    for (name, B, T, H, K, V) in [("T=512 H=4", 1, 512, 4, 128, 128),
                                  ("B2 T=1024 H=8", 2, 1024, 8, 128, 128),
                                  ("B8 T=2048 H=8", 8, 2048, 8, 128, 128),
                                  ("T=250 H=3 nondiv", 1, 250, 3, 128, 128)]:
        inp = make_inputs(B, T, H, K, V, 16, "cuda", dtype=torch.float32, seed=0)
        inp = {k2: (v if v.is_contiguous() else v.contiguous()) for k2, v in inp.items()}
        for nw in (4,):
            fb = lambda: wy_dqkg_base_triton(base_k, inp, 1.0 / math.sqrt(K), 16, num_warps=nw)
            fl = lambda: wy_dqkg_late_triton(late_k, inp, 1.0 / math.sqrt(K), 16, num_warps=nw)
            fb(); fl(); torch.cuda.synchronize()
            db_, _ = profile_dev(fb, reps=30)
            dl_, _ = profile_dev(fl, reps=30)
            wb = time_wall(fb, reps=50)
            wl = time_wall(fl, reps=50)
            bb_ = triton.testing.do_bench(fb, warmup=25, rep=200) * 1e3
            bl_ = triton.testing.do_bench(fl, warmup=25, rep=200) * 1e3
            print(f"  {name:16s} | {nw:3d} | {db_:8.1f} {dl_:8.1f} {dl_/db_:5.2f}x "
                  f"| {wb:9.1f} {wl:9.1f} {wl/wb:5.2f}x | {bb_:10.1f} {bl_:10.1f} {bl_/bb_:6.2f}x")


# --------------------------------------------------------------------------- chain
def _make_chain(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, nw_b3, b2_late):
    dhu_k = build_dhu_bf16_pipe()
    dqkg_base_k = build_dqkg_base()
    dqkg_late_k = build_dqkg_late()
    dav_k = build_dav_kernel()
    intra_k = build_intra_mma()
    v_f = inp["v"].float()

    def run():
        dAqk, dv = dav_triton(dav_k, fi_b["v_new"], fi_b["Aqk"], do, scale, CHUNK)
        dh, dh0, dv2 = dhu_bf16_pipe_triton(dhu_k, fi_b, do_b, dv, scale, CHUNK,
                                            NS=NS, dht=dfin)
        inp2 = dict(q=qn, k=kn, v=v_f, v_new=fi_b["v_new"], g=fi_b["g_cum"],
                    beta=fi_b["beta"], A=fi_b["Akk"], h=fi_b["h"], do=do, dh=dh, dv=dv2)
        if b2_late:
            b2 = wy_dqkg_late_triton(dqkg_late_k, inp2, scale, CHUNK)
        else:
            b2 = wy_dqkg_base_triton(dqkg_base_k, inp2, scale, CHUNK)
        inp3 = dict(q=qn, k=kn, g=fi_b["g_cum"], beta=fi_b["beta"],
                    dAqk=dAqk, dAkk=b2["dA"])
        b3 = intra_triton_mma(intra_k, inp3, CHUNK, CHUNK, num_warps=nw_b3)
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


def part_chain(cases, scale, lower_bound):
    print("\n" + "=" * 78)
    print("Part D: land late B2 into the assembled chain -- correctness vs FLA and baseline")
    print("=" * 78)
    NS = 3
    worst = {}
    for (name, B, T, H, seed, gs) in cases:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        fi_b, do_b = _prep_b(fi, inp, do)
        ch_base = _make_chain(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, 4, b2_late=False)
        ch_late = _make_chain(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, 4, b2_late=True)
        r_base, r_late = ch_base(), ch_late()
        gold = fla_golden(qn, kn, inp["v"].float(), beta_act, g_fla, inp["h0"], scale, do, dfin)
        print(f"\n# {name}")
        print("  -- late chain vs FLA --")
        for fld, gk in [("dq", "q"), ("dk", "k"), ("dv", "v"), ("dbeta", "beta"), ("dg", "g")]:
            _, mr = stats(f"  late {fld} vs FLA", r_late[fld], gold[gk])
            worst[fld] = max(worst.get(fld, 0.0), mr)
        _, mr = stats("  late dh0 vs FLA", r_late["dh0"], gold["h0"].transpose(-1, -2))
        worst["dh0"] = max(worst.get("dh0", 0.0), mr)
        print("  -- late chain vs baseline chain (rewrite's own delta) --")
        for fld in ["dq", "dk", "dv", "dbeta", "dg"]:
            _, mr = stats(f"  delta {fld}", r_late[fld], r_base[fld])
            worst[f"delta_{fld}"] = max(worst.get(f"delta_{fld}", 0.0), mr)
    print("\nworst-case max_rel (late chain):")
    for kk in sorted(worst):
        print(f"  {kk:16s}  {worst[kk]:.3e}")


def part_chain_timing(cases, scale, lower_bound):
    print("\n" + "=" * 78)
    print("Part D2: chain timing -- baseline B2 vs late B2 (dev CUPTI + wall), vs FLA")
    print("=" * 78)
    NS = 3
    print(f"  {'case':18s} | {'FLA dev':>8s} {'base dev':>8s} {'late dev':>8s} "
          f"{'late/base':>8s} | {'base wall':>9s} {'late wall':>9s}")
    for (name, B, T, H, seed, gs) in cases:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        fi_b, do_b = _prep_b(fi, inp, do)
        ch_base = _make_chain(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, 4, b2_late=False)
        ch_late = _make_chain(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, 4, b2_late=True)
        qa = qn.clone().requires_grad_(True)
        ka = kn.clone().requires_grad_(True)
        va = inp["v"].float().clone().requires_grad_(True)
        ba = beta_act.clone().requires_grad_(True)
        ga = g_fla.clone().requires_grad_(True)
        ha = inp["h0"].clone().requires_grad_(True)
        o, s = _chunk_kda(qa, ka, va, ga, ba, scale, ha, do, dfin)
        holders = (qa, ka, va, ba, ga, ha)

        def fla_bwd():
            s.backward(retain_graph=True)
            for t in holders:
                t.grad = None

        ch_base(); ch_late(); fla_bwd()
        torch.cuda.synchronize()
        df, _ = profile_dev(fla_bwd)
        db_, _ = profile_dev(ch_base)
        dl_, _ = profile_dev(ch_late)
        wb = time_wall(ch_base, reps=50)
        wl = time_wall(ch_late, reps=50)
        print(f"  {name:18s} | {df:8.1f} {db_:8.1f} {dl_:8.1f} {dl_/db_:7.2f}x | "
              f"{wb:9.1f} {wl:9.1f}")


def _chunk_kda(qa, ka, va, ga, ba, scale, ha, do, dfin):
    from fla.ops.kda import chunk_kda
    o, s = chunk_kda(qa, ka, va, ga, ba, scale=scale,
                     initial_state=ha.float().transpose(-1, -2), output_final_state=True,
                     use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                     use_beta_sigmoid_in_kernel=False)
    global _LOSS
    _LOSS = (o.float() * do).sum() + (s.float() * dfin).sum()
    return o, _LOSS


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}  "
          f"triton={triton.__version__}")
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0
    mode = os.environ.get("B2S_MODE", "")
    if mode in ("", "correctness"):
        part_correctness()
    if mode in ("", "regs"):
        part_regs()
    if mode in ("", "timing"):
        part_timing()
    if mode in ("", "chain"):
        part_chain(CASES[:int(os.environ.get("B2S_CASES", len(CASES)))], scale, lower_bound)
        part_chain_timing(TCASES, scale, lower_bound)
    print("\nDONE")


if __name__ == "__main__":
    main()
