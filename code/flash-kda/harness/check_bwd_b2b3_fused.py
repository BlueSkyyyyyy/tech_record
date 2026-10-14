"""P7-30: B2 (`wy_dqkg`) + B3 (`intra`) fusion -- keep the `dq/dk/dg` accumulators
in registers instead of round-tripping them through DRAM.

Motivation (§31.3 / §32.6 / §30.5)
----------------------------------
After P7-28 (B3 consumes B2's buffers in place) the assembled backward chain still
pays a full `[B,T,H,K]` fp32 round trip for `dq/dk/dg`: B2 `wy_dqkg` **writes** them,
B3 `intra` **reads** them, adds the intra-chunk contributions, and **writes them back**.
At the deployed `BT=BC=CHUNK=16 -> NC=1` granularity B3 has *only* the diagonal
sub-blocks (Parts B/D; the off-diagonal Parts A/C are skipped) -- see §27.1.  So the
whole B3 can be folded into B2's `(NT, B*H)` program: B2 already computes per-chunk
`dAkk` (its stored `dA`) and has `q/k/g/beta` loaded, and it can load `dAqk` (the `dAv`
output) once.

This script is the controlled experiment that §32.6 / §31.3 called for:

  Part A  isolated kernel: fused B2+B3 vs the unfused "B2 then B3 (MMA diagonal)" pair
          on *identical* operands, and vs FLA `wy_dqkg`+`intra` where applicable --
          exact algebra, differs only by fp32 accumulation ordering.
  Part B  isolated kernel timing: fused vs (B2 + B3) unfused (CUPTI dev + wall).
  Part C  land the fused kernel into the assembled chain (pipe B1 NS=3): six gradients
          vs FLA autograd + vs the unfused MMA chain, and chain timing.
  Part D  static register/occupancy read (no GPU needed): nw sweep.

Run inside `kernel_lab` (GPU, FlashKDA built, FLA installed), idle card:
    CUDA_VISIBLE_DEVICES=1 python .../check_bwd_b2b3_fused.py
    F23_MODE=correctness | timing | chain
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
from check_bwd_intra_kernel import (make_inputs as intra_make_inputs, fla_intra,        # noqa: E402
                                    build_kernel as build_intra_base,
                                    intra_triton as intra_triton_base)
from check_bwd_dqkg_kernel import (make_inputs as dqkg_make_inputs, fla_wy_dqkg,        # noqa: E402
                                   build_kernel as build_dqkg, wy_dqkg_triton)
from check_bwd_b3_mma import (build_intra_mma, intra_triton_mma,                        # noqa: E402
                              CASES, TCASES, build_dhu_bf16_pipe, build_dav_kernel)
from check_bwd_b3_accum import make_chain as make_chain_ip                            # noqa: E402
from check_bwd_b1_pipe_e2e import dhu_bf16_pipe_triton                                  # noqa: E402
from check_bwd_e2e import dav_triton                                                    # noqa: E402
from check_bwd_b1_bf16_e2e import prep, fla_golden, time_wall, profile_dev              # noqa: E402
from fla.ops.utils import chunk_local_cumsum                                            # noqa: E402


# --------------------------------------------------------------------------- fused kernel
def build_fused_b2b3():
    """B2 `wy_dqkg` (check_bwd_dqkg_kernel:212-322) with B3 `intra`'s *diagonal* Parts
    B/D (check_bwd_b3_mma:130-210) folded in before the stores.

    BK = K so the `dq/dk/dg` accumulators stay live `[BT, K]` registers across the whole
    program; the intra diagonal contribution is added to them, then stored once.  Also
    stores `dv2`/`db` (+ accumlutes the intra `db`) exactly as the two unfused kernels do.
    `Akk = dA` is used only internally and is *not* written.
    """

    @triton.jit
    def _fused_b2b3_kernel(
        q_ptr, k_ptr, v_ptr, vn_ptr, g_ptr, beta_ptr, A_ptr, h_ptr,
        do_ptr, dh_ptr, dv_ptr, dAqk_ptr,
        dq_ptr, dk_ptr, dv2_ptr, db_ptr, dg_ptr,
        T, NT, scale,
        H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
        BV: tl.constexpr,
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

        o_k = tl.arange(0, K)
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

        b_dq = tl.zeros([BT, K], dtype=tl.float32)
        b_dk = tl.zeros([BT, K], dtype=tl.float32)
        b_dw = tl.zeros([BT, K], dtype=tl.float32)
        b_dgk = tl.zeros([K], dtype=tl.float32)

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
            b_h = tl.load(p_h, mask=m_h, other=0.0)           # [BV,K]
            b_dh = tl.load(p_dh, mask=m_h, other=0.0)
            p_dv = dv_ptr + (i_b * T + o_t)[:, None] * (H * V) + i_h * V + o_v[None, :]
            b_dv = tl.load(p_dv, mask=m_tv, other=0.0)

            b_dgk += tl.sum(b_h * b_dh, axis=0)
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
            b_dk += tl.dot(b_vn, b_dh.to(b_vn.dtype))
            b_dw += tl.dot(b_dv.to(b_vn.dtype), b_h.to(b_vn.dtype))
            tl.debug_barrier()  # DO NOT REMOVE (mirrors FLA)
            # BK=K -> the single k-block is always `i_k == 0`, so this V-dependent branch
            # runs for *every* i_v (it accumulates b_dA/b_db over V and stores dv2 per i_v).
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

        # ----- finalize dAkk = Akk-adjoint, in-memory layout [o_t, o_A] -----
        b_dA = tl.where(m_A, b_dA * b_beta[None, :], 0.0)
        b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
        b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
        b_dA = tl.where(m_A, -b_dA, 0.0)

        # ------------------- intra diagonal (B3 Parts B/D, NC=1) -------------------
        # diagonal tiles: rows = query o_t, cols = chunk-local key index o_i = 0..BC-1
        o_i = tl.arange(0, BT)
        m_jd = (i_t * BT + o_i) < T
        m_diag = m_t[:, None] & m_jd[None, :]
        p_dAqd = dAqk_ptr + ((i_b * T + o_t)[:, None] * H + i_h) * BT + o_i[None, :]
        b_dAqd = tl.load(p_dAqd, mask=m_diag, other=0.0).to(tl.float32)
        b_dAkkd = b_dA                                   # [o_t, o_A=chunk-local col]
        # key rows j = o_t here, so k/g are the already-loaded b_k/b_g (R1 reuse)
        m_ge = o_i[:, None] >= o_i[None, :]
        b_Mq = tl.where(m_ge, b_dAqd, 0.0)
        b_Mk = tl.where(m_ge, b_dAkkd, 0.0)
        b_KG = b_k * tl.exp2(-b_g)
        b_eg = tl.exp2(b_g)
        b_dq2 = b_eg * tl.dot(b_Mq, b_KG)                # [BT,K]
        b_dk2 = b_eg * tl.dot(b_Mk, b_KG)
        b_db = b_db + tl.sum(b_dk2 * b_k, axis=1)
        b_dk2 = b_dk2 * b_beta[:, None]
        b_dg2 = b_q * b_dq2                              # \bar g decay-chain term
        b_dq = b_dq + b_dq2

        m_le = o_i[:, None] <= o_i[None, :]
        b_Sq = tl.where(m_le, tl.trans(b_dAqd), 0.0)
        b_Sk = tl.where(m_le, tl.trans(b_dAkkd), 0.0)
        b_QG = b_q * tl.exp2(b_g)
        b_KGd = (b_k * b_beta[:, None]) * tl.exp2(b_g)
        b_emg = tl.exp2(-b_g)
        b_dkt = b_emg * tl.dot(b_Sq, b_QG) + b_emg * tl.dot(b_Sk, b_KGd)

        b_dg = b_dg + b_dg2 + (b_dk2 - b_dkt) * b_k
        b_dk = b_dk + b_dk2 + b_dkt

        p_dq = dq_ptr + (i_b * T + o_t)[:, None] * (H * K) + i_h * K + o_k[None, :]
        p_dk = dk_ptr + (i_b * T + o_t)[:, None] * (H * K) + i_h * K + o_k[None, :]
        p_dg = dg_ptr + (i_b * T + o_t)[:, None] * (H * K) + i_h * K + o_k[None, :]
        tl.store(p_dq, b_dq.to(dq_ptr.dtype.element_ty), mask=m_tk)
        tl.store(p_dk, b_dk.to(dk_ptr.dtype.element_ty), mask=m_tk)
        tl.store(p_dg, b_dg.to(dg_ptr.dtype.element_ty), mask=m_tk)
        tl.store(db_ptr + (i_b * T + o_t) * H + i_h, b_db.to(db_ptr.dtype.element_ty), mask=m_t)

    return _fused_b2b3_kernel


def fused_b2b3_triton(kernel, inp, dAqk, scale, BT, BV=128, num_warps=8, num_stages=1):
    B, T, H, K = inp["q"].shape
    V = inp["v"].shape[-1]
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dv2 = torch.empty(B, T, H, V, device=dev, dtype=torch.float32)
    db = torch.empty(B, T, H, device=dev, dtype=torch.float32)
    grid = (NT, B * H)
    kernel[grid](
        inp["q"], inp["k"], inp["v"], inp["v_new"], inp["g"], inp["beta"], inp["A"], inp["h"],
        inp["do"], inp["dh"], inp["dv"], dAqk,
        dq, dk, dv2, db, dg,
        T, NT, scale, H=H, K=K, V=V, BT=BT, BV=min(BV, V),
        num_warps=num_warps, num_stages=num_stages)
    return dict(dq=dq, dk=dk, dv2=dv2, db=db, dg=dg)


# --------------------------------------------------------------------------- isolated parts
def _unfused_pair(inp, dAqk, scale, BT, nw_b2=4, nw_b3=4):
    """Reference pair on identical operands: existing B2 then existing MMA B3 (in-place)."""
    dqkg_k = build_dqkg()
    mma_k = build_intra_mma()
    b2 = wy_dqkg_triton(dqkg_k, inp, scale, BT, num_warps=nw_b2)
    inp3 = dict(q=inp["q"], k=inp["k"], g=inp["g"], beta=inp["beta"],
                dAqk=dAqk, dAkk=b2["dA"])
    b3 = intra_triton_mma(mma_k, inp3, BT, BT, num_warps=nw_b3)
    return dict(dq=b2["dq"] + b3["dq"], dk=b2["dk"] + b3["dk"],
                dv2=b2["dv"], db=b2["db"] + b3["db"], dg=b2["dg"] + b3["dg"])


def part_a():
    print("=" * 78)
    print("Part A: isolated fused B2+B3 vs unfused (B2 then MMA-diagonal B3), vs FLA")
    print("=" * 78)
    fused_k = build_fused_b2b3()
    worst = {}
    for (B, T, H, K, V, BT, seed) in [
        (1, 256, 2, 64, 64, 16, 0), (2, 512, 4, 128, 128, 16, 1),
        (1, 250, 3, 128, 128, 16, 2), (1, 512, 4, 128, 128, 16, 3),
        (2, 513, 2, 64, 128, 16, 4), (1, 192, 2, 64, 64, 16, 5),
    ]:
        print(f"\n=== B={B} T={T} H={H} K={K} V={V} BT={BT} seed={seed} ===")
        inp = dqkg_make_inputs(B, T, H, K, V, BT, "cuda", dtype=torch.float32, seed=seed)
        torch.manual_seed(seed + 100)
        dAqk = torch.randn(B, T, H, BT, device="cuda") * 0.05
        pair = _unfused_pair(inp, dAqk, 1.0 / math.sqrt(K), BT)
        for nw in (4, 8):
            fu = fused_b2b3_triton(fused_k, inp, dAqk, 1.0 / math.sqrt(K), BT, num_warps=nw)
            print(f"  -- fused(nw={nw}) vs unfused --")
            for name in ["dq", "dk", "dv2", "db", "dg"]:
                ref = pair["dv2"] if name == "dv2" else pair[name]
                _, mr = stats(f"  fused {name} vs pair", fu[name], ref)
                worst[f"{name}"] = max(worst.get(f"{name}", 0.0), mr)
        # vs FLA on its own operands is done in Part C (needs real dAqk from dAv)
    print("\nworst-case max_rel (fused vs unfused pair):")
    for kk in sorted(worst):
        print(f"  {kk:8s}  {worst[kk]:.3e}")


def part_b():
    print("\n" + "=" * 78)
    print("Part B: isolated timing -- fused vs unfused (B2 + B3), CUPTI dev + wall")
    print("=" * 78)
    fused_k = build_fused_b2b3()
    dqkg_k = build_dqkg()
    mma_k = build_intra_mma()
    # fp32 operands: this matches the *deployed* chain (P7-26 §29.1: only q/k/w are cast
    # to bf16 for B1; B2/B3 operands are all fp32).  bf16 here would be a different,
    # non-deployed regime.
    for (B, T, H, K, V, BT) in [(1, 512, 4, 128, 128, 16), (2, 1024, 8, 128, 128, 16),
                                (8, 2048, 8, 128, 128, 16), (1, 250, 3, 128, 128, 16)]:
        inp = dqkg_make_inputs(B, T, H, K, V, BT, "cuda", dtype=torch.float32, seed=0)
        inp = {k2: (v if v.is_contiguous() else v.contiguous()) for k2, v in inp.items()}
        torch.manual_seed(1)
        dAqk = torch.randn(B, T, H, BT, device="cuda") * 0.05
        print(f"\n# B={B} T={T} H={H} K={K} V={V}  (fp32 operands)")
        # deployed baseline: B2 at nw=4 (the chain's `wy_dqkg_triton` default), B3 at nw=4.
        b2 = wy_dqkg_triton(dqkg_k, inp, 1.0 / math.sqrt(K), BT, num_warps=4)
        inp3 = dict(q=inp["q"], k=inp["k"], g=inp["g"], beta=inp["beta"],
                    dAqk=dAqk, dAkk=b2["dA"])

        def fb2():
            wy_dqkg_triton(dqkg_k, inp, 1.0 / math.sqrt(K), BT, num_warps=4)

        def fb3():
            intra_triton_mma(mma_k, inp3, BT, BT, num_warps=4)

        (fb2(), fb3())
        torch.cuda.synchronize()
        d2, _ = profile_dev(fb2, reps=30)
        d3, _ = profile_dev(fb3, reps=30)
        w2 = time_wall(fb2, reps=50)
        w3 = time_wall(fb3, reps=50)
        print(f"  baseline (B2 nw=4 + B3 nw=4): dev={d2 + d3:7.1f} wall={w2 + w3:7.1f}")
        for (BV, nw) in [(64, 8), (128, 4), (128, 8)]:
            def ffus():
                fused_b2b3_triton(fused_k, inp, dAqk, 1.0 / math.sqrt(K), BT,
                                  BV=BV, num_warps=nw)
            ffus()
            torch.cuda.synchronize()
            df, _ = profile_dev(ffus, reps=30)
            wf = time_wall(ffus, reps=50)
            fen = build_fused_b2b3()
            fused_b2b3_triton(fen, inp, dAqk, 1.0 / math.sqrt(K), BT, BV=BV, num_warps=nw)
            cache = list(fen.device_caches.values())[0][0]
            c = next(iter(cache.values()))
            print(f"  fused BV={BV:3d} nw={nw}: dev={df:7.1f} ({df/(d2+d3):.2f}x base) "
                  f"wall={wf:7.1f} ({wf/(w2+w3):.2f}x) regs/spill={c.n_regs}/{c.n_spills}")


def part_d():
    print("\n" + "=" * 78)
    print("Part D: static register/spill read of the fused kernel (fp32, nw sweep)")
    print("=" * 78)
    B, T, H, K, V, BT = 8, 2048, 8, 128, 128, 16
    inp = dqkg_make_inputs(B, T, H, K, V, BT, "cuda", dtype=torch.float32, seed=0)
    torch.manual_seed(1)
    dAqk = torch.randn(B, T, H, BT, device="cuda") * 0.05
    for nw in (4, 8):
        fen = build_fused_b2b3()
        fused_b2b3_triton(fen, inp, dAqk, 1.0 / math.sqrt(K), BT, num_warps=nw)
        b2k = build_dqkg()
        wy_dqkg_triton(b2k, inp, 1.0 / math.sqrt(K), BT, num_warps=nw)

        def regs(kern):
            try:
                cache = list(kern.device_caches.values())[0][0]
                compiled = next(iter(cache.values()))
                return compiled.n_regs, compiled.n_spills
            except Exception as e:  # noqa: BLE001
                return None, str(e)

        print(f"  nw={nw}: B2 regs/spills={regs(b2k)}  fused regs/spills={regs(fen)}")


# --------------------------------------------------------------------------- chain
def make_chain_fused(fi, inp, qn, kn, do, dfin, scale, NS, nw_f=8):
    dhu_k = build_dhu_bf16_pipe()
    dav_k = build_dav_kernel()
    fused_k = build_fused_b2b3()
    fi_b = dict(fi)
    for kk in ("qg", "kg", "w"):
        fi_b[kk] = fi[kk].to(torch.bfloat16)
    do_b = do.to(torch.bfloat16)
    v_f = inp["v"].float()

    def run():
        dAqk, dv = dav_triton(dav_k, fi_b["v_new"], fi_b["Aqk"], do, scale, CHUNK)
        dh, dh0, dv2 = dhu_bf16_pipe_triton(dhu_k, fi_b, do_b, dv, scale, CHUNK,
                                            NS=NS, dht=dfin)
        inp2 = dict(q=qn, k=kn, v=v_f, v_new=fi_b["v_new"], g=fi_b["g_cum"],
                    beta=fi_b["beta"], A=fi_b["Akk"], h=fi_b["h"], do=do, dh=dh, dv=dv2)
        b = fused_b2b3_triton(fused_k, inp2, dAqk, scale, CHUNK, num_warps=nw_f)
        dg = chunk_local_cumsum(b["dg"], chunk_size=CHUNK, reverse=True)
        return dict(dq=b["dq"], dk=b["dk"], dv=b["dv2"], dg=dg, dbeta=b["db"],
                    dh0=dh0, dh=dh, dv2=dv2)

    return run


def _prep_b(fi):
    fi_b = dict(fi)
    for kk in ("qg", "kg", "w"):
        fi_b[kk] = fi[kk].to(torch.bfloat16)
    return fi_b


def part_c(cases, scale, lower_bound):
    print("\n" + "=" * 78)
    print("Part C: land fused B2+B3 into the assembled chain -- correctness vs FLA/baseline")
    print("=" * 78)
    NS = 3
    worst = {}
    for (name, B, T, H, seed, gs) in cases:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        fi_b = _prep_b(fi)
        do_b = do.to(torch.bfloat16)
        ch_base = make_chain_ip(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, True)
        ch_fus = make_chain_fused(fi, inp, qn, kn, do, dfin, scale, NS)
        r_base, r_fus = ch_base(), ch_fus()
        gold = fla_golden(qn, kn, inp["v"].float(), beta_act, g_fla, inp["h0"], scale, do, dfin)
        print(f"\n# {name}")
        print("  -- fused chain vs FLA --")
        for fld, gk in [("dq", "q"), ("dk", "k"), ("dv", "v"), ("dbeta", "beta"), ("dg", "g")]:
            _, mr = stats(f"  fused {fld} vs FLA", r_fus[fld], gold[gk])
            worst[fld] = max(worst.get(fld, 0.0), mr)
        _, mr = stats("  fused dh0 vs FLA", r_fus["dh0"], gold["h0"].transpose(-1, -2))
        worst["dh0"] = max(worst.get("dh0", 0.0), mr)
        print("  -- fused chain vs baseline (own delta) --")
        for fld in ["dq", "dk", "dv", "dbeta", "dg"]:
            _, mr = stats(f"  delta {fld}", r_fus[fld], r_base[fld])
            worst[f"delta_{fld}"] = max(worst.get(f"delta_{fld}", 0.0), mr)
    print("\nworst-case max_rel (fused chain):")
    for kk in sorted(worst):
        print(f"  {kk:16s}  {worst[kk]:.3e}")


def part_c_timing(cases, scale, lower_bound):
    from fla.ops.kda import chunk_kda
    print("\n" + "=" * 78)
    print("Part C2: chain timing -- baseline MMA B3 chain vs fused chain, dev + wall, vs FLA")
    print("=" * 78)
    NS = 3
    print(f"  {'case':18s} | {'FLA dev':>8s} {'base dev':>8s} {'fus dev':>8s} "
          f"{'fus/base':>8s} | {'base wall':>9s} {'fus wall':>9s}")
    for (name, B, T, H, seed, gs) in cases:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        fi_b = _prep_b(fi)
        do_b = do.to(torch.bfloat16)
        ch_base = make_chain_ip(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, True)
        ch_fus = make_chain_fused(fi, inp, qn, kn, do, dfin, scale, NS)
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

        ch_base(); ch_fus(); fla_bwd()
        torch.cuda.synchronize()
        df, _ = profile_dev(fla_bwd)
        db_, pb = profile_dev(ch_base)
        dq_, pq = profile_dev(ch_fus)
        wb = time_wall(ch_base, reps=50)
        wq = time_wall(ch_fus, reps=50)
        print(f"  {name:18s} | {df:8.1f} {db_:8.1f} {dq_:8.1f} {dq_/db_:7.2f}x | "
              f"{wb:9.1f} {wq:9.1f}")
        for tag, prof in (("base", pb), ("fus", pq)):
            tot = {"wy_dqkg": 0.0, "intra": 0.0, "fused": 0.0, "dhu": 0.0, "dav": 0.0}
            for ev in prof.key_averages():
                for kk in tot:
                    if f"_{kk}_kernel" in ev.key or (kk in ev.key and "kernel" in ev.key):
                        tot[kk] += ev.self_device_time_total
            print(f"      {tag} per-kernel dev: " +
                  " ".join(f"{k}={v:.1f}" for k, v in tot.items()))


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}  "
          f"triton={triton.__version__}")
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0
    mode = os.environ.get("F23_MODE", "")
    if mode in ("", "correctness"):
        part_a()
    if mode in ("", "timing"):
        part_b()
    if mode in ("", "regs"):
        part_d()
    if mode in ("", "chain"):
        part_c(CASES[:int(os.environ.get("F23_CASES", len(CASES)))], scale, lower_bound)
        part_c_timing(TCASES, scale, lower_bound)
    print("\nDONE")


if __name__ == "__main__":
    main()
