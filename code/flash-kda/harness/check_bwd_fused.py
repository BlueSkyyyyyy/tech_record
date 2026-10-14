"""P7-27: B1->B2 fusion prototype -- keep the cross-chunk state adjoint `dh` in registers,
never spill it to DRAM (the last "true structural item" left by docs/07 §29).

Motivation
----------
P7-26 (`docs/07` §29) closed every cheap/半-cheap lever on the B2 `wy_dqkg` consumer and
left exactly one structural item: **B1->B2 fusion** -- today B1 writes the per-chunk state
adjoint `dh` [B,NT,H,K,V] to global and B2 reads it back (plus reads the forward state `h`).
The B1 program already *holds* `dS == dh[c]` in registers as it sweeps chunks in reverse; if
B2's per-chunk work ran in the same program, `dh` would never touch DRAM.

Design (why BV = V here)
------------------------
B1 partitions over `V` (grid `(cdiv(V,BV), B*H)`), B2 partitions over chunks
(grid `(NT, B*H)`).  B2 contracts over all of `V`, so a V-sliced program could only produce
*partial* `dq/dk/dw` -- a cross-V reduction (atomics) would be needed before B2's nonlinear
decay finish.  The clean, atomic-free prototype is therefore **BV = V** (one program per
`(b,h)`), which is also exactly the forward K2 shape (head-parallel, serial over T).  The
fused kernel:
  * sweeps chunks c = NT-1 .. 0, carrying `b_dS [K,V]` in registers (B1),
  * writes B1's `dv2` output,
  * immediately runs B2's full per-chunk pass on `dh = b_dS` and `h[c]` (no store/load),
  * updates `b_dS` for the next (earlier) chunk.
Grid `(B*H,)`; `num_warps`/`K`/`V` constexpr; `dh`, `h` are never materialised.

Parts
-----
  Part A  correctness: fused kernel's `dq/dk/dv/dbeta/dg/dh0` fed through the *same* B3 +
          reverse-cumsum epilogue as the deployed chain, vs FLA `torch.autograd` (P6-8
          setup: same activated operands, loss includes final-state term).
  Part B  timing: fused `dev` (CUPTI) vs the unfused `B1 + B2` kernel pair, and the whole
          `dAv -> B1 -> B2 -> B3 -> cumsum` chain, at the deployment granularity CHUNK=16.
  Part C  register/occupancy + DRAM bytes via `ncu` (optional; separate driver).

Run inside `kernel_lab` (GPU, FlashKDA built, FLA installed), idle card:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_fused.py
    B12F_MODE=timing ...   # skip Part A
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

from check_bwd_ws_reuse import D, CHUNK, stats, cdiv                              # noqa: E402
from check_bwd_b1_bf16_e2e import (prep, fla_golden, time_wall, profile_dev,     # noqa: E402
                                   make_chain_fn, build_dhu_bf16, dhu_bf16_triton)
from check_bwd_dqkg_kernel import build_kernel as build_dqkg, wy_dqkg_triton      # noqa: E402
from check_bwd_intra_kernel import build_kernel as build_intra, intra_triton      # noqa: E402
from check_bwd_e2e import build_dav_kernel, dav_triton                            # noqa: E402
from fla.ops.utils import chunk_local_cumsum                                      # noqa: E402


# --------------------------------------------------------------------------------- kernel
def build_fused():
    """Single Triton kernel fusing B1 (state adjoint) + B2 (`wy_dqkg`); grid `(B*HV,)`."""

    @triton.jit
    def _fused_b12_kernel(
        qg_ptr, kg_ptr, w_ptr,          # q_dec, k_rest, w  (bf16, B1 mma operands)
        q_ptr, k_ptr,                    # raw normalized q/k (fp32, B2 operands)
        h_ptr,                           # forward chunk state S_in [B,NT,HV,K,V] fp32
        do_ptr, dvin_ptr,                # do, dL/dU(output branch) [B,T,HV,V] fp32
        v_ptr, vn_ptr,                   # raw v, v_new=U [B,T,HV,V] fp32
        g_ptr, beta_ptr, A_ptr, etot_ptr,
        dht_ptr,
        dq_ptr, dk_ptr, dvg_ptr, db_ptr, dg_ptr, dA_ptr, dh0_ptr,
        T, NT, scale,
        HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
        HAS_DHT: tl.constexpr,
    ):
        i_nh = tl.program_id(0)
        i_n = i_nh // HV
        i_h = i_nh % HV
        o_k = tl.arange(0, K)
        o_v = tl.arange(0, V)
        o_bt = tl.arange(0, BT)

        if HAS_DHT:
            p_dht = dht_ptr + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
            b_dS = tl.load(p_dht).to(tl.float32)
        else:
            b_dS = tl.zeros([K, V], dtype=tl.float32)

        for cc in range(0, NT):
            c = NT - 1 - cc
            o_t = c * BT + o_bt
            m_t = o_t < T
            last = min(T, c * BT + BT) - 1
            m_last = o_t == last
            base_t = (i_n * T + o_t)[:, None]
            m_tk = m_t[:, None]
            m_tv = m_t[:, None]

            # ---- B1 operands (bf16 mma)
            p_qg = qg_ptr + base_t * (HV * K) + i_h * K + o_k[None, :]
            p_kg = kg_ptr + base_t * (HV * K) + i_h * K + o_k[None, :]
            p_w = w_ptr + base_t * (HV * K) + i_h * K + o_k[None, :]
            b_qg = tl.load(p_qg, mask=m_tk, other=0.0)
            b_kg = tl.load(p_kg, mask=m_tk, other=0.0)
            b_w = tl.load(p_w, mask=m_tk, other=0.0)
            # ---- B2 raw operands (fp32)
            p_q = q_ptr + base_t * (HV * K) + i_h * K + o_k[None, :]
            p_k = k_ptr + base_t * (HV * K) + i_h * K + o_k[None, :]
            p_g = g_ptr + base_t * (HV * K) + i_h * K + o_k[None, :]
            b_q = tl.load(p_q, mask=m_tk, other=0.0).to(tl.float32)
            b_k = tl.load(p_k, mask=m_tk, other=0.0).to(tl.float32)
            b_g = tl.load(p_g, mask=m_tk, other=0.0).to(tl.float32)
            b_gn = tl.load(g_ptr + (i_n * T + last) * (HV * K) + i_h * K + o_k).to(tl.float32)

            base_v = (i_n * T + o_t)[:, None]
            p_do = do_ptr + base_v * (HV * V) + i_h * V + o_v[None, :]
            p_dvi = dvin_ptr + base_v * (HV * V) + i_h * V + o_v[None, :]
            p_v = v_ptr + base_v * (HV * V) + i_h * V + o_v[None, :]
            p_vn = vn_ptr + base_v * (HV * V) + i_h * V + o_v[None, :]
            b_do = tl.load(p_do, mask=m_tv, other=0.0).to(tl.float32)
            b_dvi = tl.load(p_dvi, mask=m_tv, other=0.0).to(tl.float32)
            b_v = tl.load(p_v, mask=m_tv, other=0.0).to(tl.float32)
            b_vn = tl.load(p_vn, mask=m_tv, other=0.0).to(tl.float32)

            b_beta = tl.load(beta_ptr + (i_n * T + o_t) * HV + i_h,
                             mask=m_t, other=0.0).to(tl.float32)
            A_base = A_ptr + ((i_n * T + c * BT) * HV + i_h) * BT
            p_A = A_base + o_bt[:, None] + o_bt[None, :] * (HV * BT)
            b_A = tl.load(p_A, mask=m_t[None, :], other=0.0).to(tl.float32)   # [o_A,o_t]
            p_h = h_ptr + ((i_n * NT + c) * HV + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
            b_h = tl.load(p_h).to(tl.float32)                                # [K,V]
            b_e = tl.load(etot_ptr + ((i_n * NT + c) * HV + i_h) * K + o_k).to(tl.float32)

            # =============================== B1: state adjoint ===============================
            dt = b_kg.dtype
            b_dU = tl.dot(b_kg, b_dS.to(dt)) + b_dvi.to(dt)                  # [BT,V]  (B1 dU/dv2)
            b_dS_next = (b_dS * b_e[:, None]
                         + tl.dot(tl.trans(b_qg), b_do.to(dt)) * scale
                         - tl.dot(tl.trans(b_w), b_dU.to(dt)))

            # =============================== B2: wy_dqkg (dh = b_dS) =========================
            b_hT = tl.trans(b_h)          # [V,K]
            b_dhT = tl.trans(b_dS)        # [V,K]
            b_dgk = tl.sum(b_hT * b_dhT, axis=0)                              # [K]
            b_dq = tl.dot(b_do, b_hT)                                         # [BT,K]
            b_dk = tl.dot(b_vn, b_dhT)
            b_dw = tl.dot(b_dU, b_hT)
            b_dA = tl.dot(b_dU, tl.trans(b_v))                                # [BT,BT]
            b_dvb = tl.dot(b_A, b_dU)                                         # [BT,V]
            b_dv2 = b_dvb * b_beta[:, None]
            b_db = tl.sum(b_dvb * b_v, axis=1)

            gk_exp = tl.exp2(b_g)
            b_gb = gk_exp * b_beta[:, None]
            b_dgk = b_dgk * tl.exp2(b_gn)
            b_dq = b_dq * gk_exp * scale
            b_dk = b_dk * tl.where(m_t[:, None], tl.exp2(b_gn[None, :] - b_g), 0.0)
            b_kgx = b_k * gk_exp
            b_dw = -b_dw
            b_dA = b_dA + tl.dot(b_dw, tl.trans(b_kgx))
            b_dkgb = tl.dot(b_A, b_dw)
            b_db = b_db + tl.sum(b_dkgb * b_kgx, axis=1)
            b_kdk = b_k * b_dk
            b_dgk = b_dgk + tl.sum(b_kdk, axis=0)
            b_dg = (b_q * b_dq - b_kdk
                    + tl.where(m_last[:, None], 1.0, 0.0) * b_dgk[None, :]
                    + b_kgx * b_dkgb * b_beta[:, None])
            b_dk = b_dk + b_dkgb * b_gb

            m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t[None, :])
            b_dA = tl.where(m_A, b_dA * b_beta[None, :], 0.0)
            b_dA = tl.dot(b_dA, b_A)
            b_dA = tl.dot(b_A, b_dA)
            b_dA = tl.where(m_A, -b_dA, 0.0)

            p_dq = dq_ptr + base_t * (HV * K) + i_h * K + o_k[None, :]
            p_dk = dk_ptr + base_t * (HV * K) + i_h * K + o_k[None, :]
            p_dg = dg_ptr + base_t * (HV * K) + i_h * K + o_k[None, :]
            tl.store(p_dq, b_dq.to(dq_ptr.dtype.element_ty), mask=m_tk)
            tl.store(p_dk, b_dk.to(dk_ptr.dtype.element_ty), mask=m_tk)
            tl.store(p_dg, b_dg.to(dg_ptr.dtype.element_ty), mask=m_tk)
            p_dvg = dvg_ptr + base_v * (HV * V) + i_h * V + o_v[None, :]
            tl.store(p_dvg, b_dv2.to(dvg_ptr.dtype.element_ty), mask=m_tv)
            p_db = db_ptr + (i_n * T + o_t) * HV + i_h
            tl.store(p_db, b_db.to(db_ptr.dtype.element_ty), mask=m_t)
            p_dA = dA_ptr + ((i_n * T + o_t) * HV + i_h)[:, None] * BT + o_bt[None, :]
            tl.store(p_dA, b_dA.to(dA_ptr.dtype.element_ty), mask=m_t[:, None])

            b_dS = b_dS_next

        p_dh0 = dh0_ptr + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dS.to(dh0_ptr.dtype.element_ty))

    return _fused_b12_kernel


def fused_b12_triton(kernel, fi, qn, kn, do, dvin, scale, BT, dht=None, num_warps=8):
    """Launcher.  B1 needs `dv2` (its dU output) *and* the chain needs B2's `dv` (the v
    gradient); they are different tensors, so we keep two buffers: `dv2` = B1 output (B2
    has already consumed it in-register here), `dvg` = B2's v-gradient output."""
    B, T, HV, K = fi["qg"].shape
    V = do.shape[-1]
    NT = cdiv(T, BT)
    dev = do.device
    qg = fi["qg"].to(torch.bfloat16).contiguous()
    kg = fi["kg"].to(torch.bfloat16).contiguous()
    w = fi["w"].to(torch.bfloat16).contiguous()
    last_idx = torch.clamp((torch.arange(NT, device=dev) + 1) * BT - 1, max=T - 1)
    etot = torch.exp2(fi["g_cum"].float()[:, last_idx]).contiguous()
    dq = torch.empty(B, T, HV, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, HV, K, device=dev, dtype=torch.float32)
    dg = torch.empty(B, T, HV, K, device=dev, dtype=torch.float32)
    db = torch.empty(B, T, HV, device=dev, dtype=torch.float32)
    dA = torch.empty(B, T, HV, BT, device=dev, dtype=torch.float32)
    dvg = torch.empty(B, T, HV, V, device=dev, dtype=torch.float32)
    dh0 = torch.empty(B, HV, K, V, device=dev, dtype=torch.float32)
    has_dht = dht is not None
    dht_arg = dht.contiguous() if has_dht else dh0
    grid = (B * HV,)
    kernel[grid](
        qg, kg, w, qn.contiguous(), kn.contiguous(), fi["h"].contiguous(),
        do.contiguous(), dvin.contiguous(), fi["v_raw"].contiguous(), fi["v_new"].contiguous(),
        fi["g_cum"].contiguous(), fi["beta"].contiguous(), fi["Akk"].contiguous(), etot,
        dht_arg, dq, dk, dvg, db, dg, dA, dh0, T, NT, scale,
        HV=HV, K=K, V=V, BT=BT, HAS_DHT=has_dht, num_warps=num_warps, num_stages=1)
    return dict(dq=dq, dk=dk, dv=dvg, db=db, dg=dg, dA=dA, dh0=dh0)


# ----------------------------------------------------------------------------- chain (fused)
def make_fused_chain(fi, inp, qn, kn, do, dfin, scale):
    dav_k = build_dav_kernel()
    fused_k = build_fused()
    intra_k = build_intra()

    # raw v (not U) is needed by B2's dA/db branch; the forward workspace gives U=v_new
    fi2 = dict(fi)
    fi2["v_raw"] = inp["v"].float()

    def run():
        dAqk, dvin = dav_triton(dav_k, fi["v_new"], fi["Aqk"], do, scale, CHUNK)
        b2 = fused_b12_triton(fused_k, fi2, qn, kn, do, dvin, scale, CHUNK, dht=dfin)
        inp3 = dict(q=qn, k=kn, g=fi["g_cum"], beta=fi["beta"],
                    dAqk=dAqk, dAkk=b2["dA"])
        b3 = intra_triton(intra_k, inp3, CHUNK, CHUNK)
        dg = chunk_local_cumsum(b2["dg"] + b3["dg"], chunk_size=CHUNK, reverse=True)
        return dict(dq=b2["dq"] + b3["dq"], dk=b2["dk"] + b3["dk"], dv=b2["dv"],
                    dg=dg, dbeta=b2["db"] + b3["db"], dh0=b2["dh0"])

    return run


def make_unfused_b12(fi, inp, qn, kn, do, dfin, scale):
    """The deployed B1+B2 pair (bf16 B1, fp32 B2), for an apples-to-apples timing/overlap
    check against the fused kernel."""
    dav_k = build_dav_kernel()
    dhu_k = build_dhu_bf16()
    dqkg_k = build_dqkg()
    fi_b = dict(fi)
    for kk in ("qg", "kg", "w"):
        fi_b[kk] = fi[kk].to(torch.bfloat16)
    v_f = inp["v"].float()

    def run():
        dAqk, dv = dav_triton(dav_k, fi["v_new"], fi["Aqk"], do, scale, CHUNK)
        dh, dh0, dv2 = dhu_bf16_triton(dhu_k, fi_b, do, dv, scale, CHUNK, dht=dfin)
        inp2 = dict(q=qn, k=kn, v=v_f, v_new=fi["v_new"], g=fi["g_cum"], beta=fi["beta"],
                    A=fi["Akk"], h=fi["h"], do=do, dh=dh, dv=dv2)
        b2 = wy_dqkg_triton(dqkg_k, inp2, scale, CHUNK)
        return dict(dq=b2["dq"], dk=b2["dk"], dv=b2["dv"], db=b2["db"], dg=b2["dg"],
                    dA=b2["dA"], dh0=dh0)

    return run


# ----------------------------------------------------------------------------------- parts
CASES = [
    ("T=512 H=4", 1, 512, 4, 0, 1.0),
    ("B2 T=1024 H=8", 2, 1024, 8, 1, 1.0),
    ("B8 T=2048 H=8", 8, 2048, 8, 2, 1.0),
    ("T=250 H=3 nondiv", 1, 250, 3, 3, 1.0),
]


def part_a(cases, scale, lower_bound):
    print("=" * 78)
    print("Part A: fused chain (B1+B2 fused, then B3+cumsum) vs FLA torch.autograd")
    print("=" * 78)
    worst = {}
    for (name, B, T, H, seed, gs) in cases:
        print(f"\n# {name}  B={B} T={T} H={H} seed={seed}")
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        fi2 = dict(fi)
        fi2["v_raw"] = inp["v"].float()
        fused_k = build_fused()
        dav_k = build_dav_kernel()
        ch = make_fused_chain(fi, inp, qn, kn, do, dfin, scale)
        unf = make_unfused_b12(fi, inp, qn, kn, do, dfin, scale)
        r = ch()
        ru = unf()
        dAqk, dvin = dav_triton(dav_k, fi["v_new"], fi["Aqk"], do, scale, CHUNK)
        # isolated fused *B1+B2 only* (no B3), for an apples-to-apples delta vs unfused B1+B2
        rf = fused_b12_triton(fused_k, fi2, qn, kn, do, dvin, scale, CHUNK, dht=dfin)
        gold = fla_golden(qn, kn, inp["v"].float(), beta_act, g_fla, inp["h0"], scale, do, dfin)
        print("  -- fused chain (B1+B2 fused + B3) vs FLA --")
        for fld, gk in [("dq", "q"), ("dk", "k"), ("dv", "v"),
                        ("dbeta", "beta"), ("dg", "g")]:
            _, mr = stats(f"fused {fld} vs FLA", r[fld], gold[gk])
            worst[fld] = max(worst.get(fld, 0.0), mr)
        _, mr = stats("fused dh0 vs FLA", r["dh0"], gold["h0"].transpose(-1, -2))
        worst["dh0"] = max(worst.get("dh0", 0.0), mr)
        print("  -- fused B1+B2 (isolated) vs unfused B1+B2, same operands --")
        for a, b in [("dq", "dq"), ("dk", "dk"), ("dv", "dv"), ("db", "db"),
                     ("dg", "dg"), ("dh0", "dh0")]:
            stats(f"  {a} fused vs unfused", rf[a], ru[b])
    print("\nworst-case max_rel vs FLA across cases:")
    for fld in ["dq", "dk", "dv", "dbeta", "dg", "dh0"]:
        print(f"  {fld:6s}  {worst[fld]:.3e}")


def part_b(cases, scale, lower_bound):
    print("\n" + "=" * 78)
    print("Part B: timing -- fused kernel vs unfused B1+B2 vs full B1+B2+B3 chain")
    print("=" * 78)
    # pre-build kernels used for the isolated pair
    dhu_k = build_dhu_bf16()
    dqkg_k = build_dqkg()
    dav_k = build_dav_kernel()
    fused_k = build_fused()
    print(f"  {'case':18s} | {'B1+B2 dev':>10s} {'fused dev':>10s} {'ratio':>7s} | "
          f"{'chain unfused':>13s} {'chain fused':>12s} {'ratio':>7s}")
    for (name, B, T, H, seed, gs) in cases:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        fi_b = dict(fi)
        for kk in ("qg", "kg", "w"):
            fi_b[kk] = fi[kk].to(torch.bfloat16)
        fi2 = dict(fi)
        fi2["v_raw"] = inp["v"].float()
        v_f = inp["v"].float()
        dAqk, dvin = dav_triton(dav_k, fi["v_new"], fi["Aqk"], do, scale, CHUNK)

        def unfused_pair():
            dh, dh0, dv2 = dhu_bf16_triton(dhu_k, fi_b, do, dvin, scale, CHUNK, dht=dfin)
            inp2 = dict(q=qn, k=kn, v=v_f, v_new=fi["v_new"], g=fi["g_cum"],
                        beta=fi["beta"], A=fi["Akk"], h=fi["h"], do=do, dh=dh, dv=dv2)
            wy_dqkg_triton(dqkg_k, inp2, scale, CHUNK)

        def fused_pair():
            fused_b12_triton(fused_k, fi2, qn, kn, do, dvin, scale, CHUNK, dht=dfin)

        unfused_pair()
        fused_pair()
        torch.cuda.synchronize()
        d_u, _ = profile_dev(unfused_pair)
        d_f, _ = profile_dev(fused_pair)
        ch_unf = make_chain_fn(fi, inp, qn, kn, do, dfin, scale, b1_bf16=True)
        ch_fus = make_fused_chain(fi, inp, qn, kn, do, dfin, scale)
        ch_unf()
        ch_fus()
        torch.cuda.synchronize()
        c_u, _ = profile_dev(ch_unf)
        c_f, _ = profile_dev(ch_fus)
        print(f"  {name:18s} | {d_u:10.1f} {d_f:10.1f} {d_f/d_u:6.2f}x | "
              f"{c_u:13.1f} {c_f:12.1f} {c_f/c_u:6.2f}x")


def part_c(cases, scale, lower_bound):
    """Decompose the parallelism penalty: B1 alone at BV in {32 (deployed), 64, V (the
    partition fusion forces)} x NS in {1,3}, CUPTI device us/call."""
    from check_bwd_b1_bf16_e2e import build_dhu_bf16
    from check_bwd_b1_pipe_e2e import build_dhu_bf16_pipe, dhu_bf16_pipe_triton
    print("\n" + "=" * 78)
    print("Part C: B1 alone -- V-partition sweep (BV=32 deployed vs BV=V fusion forces)")
    print("=" * 78)
    dhu_k = build_dhu_bf16()
    pipe_k = build_dhu_bf16_pipe()
    print(f"  {'case':18s} {'BV':>4s} {'NS':>3s} {'dev':>9s} {'grid':>10s}")
    for (name, B, T, H, seed, gs) in cases:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        fi_b = dict(fi)
        for kk in ("qg", "kg", "w"):
            fi_b[kk] = fi[kk].to(torch.bfloat16)
        dvin = torch.randn_like(do)
        for BV in (32, 64, D):
            for NS in (1, 3):
                if BV == 32 and NS == 1:
                    fn = lambda: dhu_bf16_triton(dhu_k, fi_b, do, dvin, scale, CHUNK,
                                                 dht=dfin)
                else:
                    fn = lambda: dhu_bf16_pipe_triton(pipe_k, fi_b, do, dvin, scale, CHUNK,
                                                      BV=BV, dht=dfin, NS=NS)
                try:
                    fn()
                    torch.cuda.synchronize()
                    d, _ = profile_dev(fn, reps=15)
                    grid = (cdiv(D, BV), B * H)
                    print(f"  {name:18s} {BV:4d} {NS:3d} {d:9.1f} {str(grid):>10s}")
                except triton.runtime.errors.OutOfResources:
                    torch.cuda.synchronize()
                    print(f"  {name:18s} {BV:4d} {NS:3d} {'OOM':>9s}")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}  "
          f"triton={triton.__version__}")
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0
    mode = os.environ.get("B12F_MODE", "")
    if mode not in ("timing", "b"):
        part_a(CASES[:int(os.environ.get("B12F_CASES", len(CASES)))], scale, lower_bound)
    if mode in ("", "b", "timing"):
        part_b(CASES[:int(os.environ.get("B12F_TCASES", len(CASES)))], scale, lower_bound)
    if mode in ("", "c", "timing"):
        part_c(CASES[:int(os.environ.get("B12F_TCASES", len(CASES)))], scale, lower_bound)
    print("\nDONE")


if __name__ == "__main__":
    main()
