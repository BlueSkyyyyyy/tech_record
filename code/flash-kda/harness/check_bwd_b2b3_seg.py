"""P7-45: is the fused B2/B3 Triton kernel's ~1.9x gap (§46/§47) in **bytes** or in
**compute/issue**?  A controlled byte-ablation of the *same* fused kernel -- same
`tl.dot`s, same program shape, same stores -- where only selected **input loads** are
replaced by `tl.zeros` (so the contraction runs on zeros and no `tl.dot`/load layout
changes, but the DRAM traffic drops).

Motivation (docs/07 §47.5 / §47.6)
----------------------------------
P7-43 (§46) showed the fused B2/B3 kernel runs ~1.9x above a same-traffic, no-`tl.dot`
kernel and is *issue-starved* (1 block/SM from 255 regs + 114 KB smem).  P7-44 (§47)
falsified the "drop the tensor core" reading (SIMT GEMM core 3-5x slower, L1/TEX bound)
and relocated the headroom to the **non-GEMM segment + `h/dh` traffic**, where `h/dh`
(fp32, chunk-granular `[B,NT,H,K,V]`) is the single largest stream (~1.07 GB of the
~1.96 GB).  §47.6 admits the split was *inferred from a bytes/time difference*, never
measured directly.

A first attempt to gate the `tl.dot`s off failed: substituting the dots with shape-
preserving reductions forced cross-lane layout conversions and produced 4964 spills (a
compiler pathology, not a measurement).  So instead we gate **bytes**: replace an input
load with `tl.zeros`, keeping every `tl.dot` intact.  This isolates *how much time is
actually attributable to a given byte stream* without perturbing the compute graph.

    full    no loads gated (deployed fused B2/B3)
    no_hdh  h/dh loads -> zeros   (drops ~1.07 GB = 55% of traffic)
    no_qkg  q/k/g/bn loads -> zeros
    no_odv  do/v_new/dv/v loads -> zeros
    no_all  all four big operand groups -> zeros

    Part A  correctness: `full` vs the untouched fused kernel (bit-identical), incl.
            non-divisible T and K=64.
    Part B  timing (CUPTI dev + wall) per variant x shapes: the byte-sensitivity and
            the residual (compute/issue) floor.
    Part C  static register/spill/smem per variant.
    Part D  (ncu driver) confirms the DRAM/L2 byte deltas and the SOL per variant.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../check_bwd_b2b3_seg.py > .../check_bwd_b2b3_seg.out.txt 2>&1
    SEG_MODE=a | b | c
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

from check_bwd_ws_reuse import D, stats, cdiv                                    # noqa: E402
from check_bwd_dqkg_kernel import make_inputs as dqkg_make_inputs                 # noqa: E402
from check_bwd_b2b3_fused import build_fused_b2b3, fused_b2b3_triton              # noqa: E402
from check_bwd_b1_bf16_e2e import time_wall, profile_dev                          # noqa: E402


def build_seg_kernel():
    @triton.jit
    def _seg_b2b3_kernel(
        q_ptr, k_ptr, v_ptr, vn_ptr, g_ptr, beta_ptr, A_ptr, h_ptr,
        do_ptr, dh_ptr, dv_ptr, dAqk_ptr,
        dq_ptr, dk_ptr, dv2_ptr, db_ptr, dg_ptr,
        T, NT, scale,
        H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
        BV: tl.constexpr,
        ZD_HDH: tl.constexpr, ZD_QKG: tl.constexpr, ZD_ODV: tl.constexpr,
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
        if ZD_QKG:
            b_k = tl.zeros([BT, K], dtype=b_k.dtype)
            b_q = tl.zeros([BT, K], dtype=b_q.dtype)
            b_g = tl.zeros([BT, K], dtype=tl.float32)
            b_gn = tl.zeros([K], dtype=tl.float32)

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
            if ZD_HDH:
                b_h = tl.zeros([BV, K], dtype=b_h.dtype)
                b_dh = tl.zeros([BV, K], dtype=b_dh.dtype)
            if ZD_ODV:
                b_vn = tl.zeros([BT, BV], dtype=b_vn.dtype)
                b_do = tl.zeros([BT, BV], dtype=b_do.dtype)
                b_dv = tl.zeros([BT, BV], dtype=b_dv.dtype)

            b_dgk += tl.sum(b_h.to(tl.float32) * b_dh.to(tl.float32), axis=0)
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
            b_dk += tl.dot(b_vn, b_dh.to(b_vn.dtype))
            b_dw += tl.dot(b_dv.to(b_vn.dtype), b_h.to(b_vn.dtype))
            tl.debug_barrier()  # DO NOT REMOVE (mirrors FLA)
            p_v = v_ptr + (i_b * T + o_t)[:, None] * (H * V) + i_h * V + o_v[None, :]
            b_v = tl.load(p_v, mask=m_tv, other=0.0)
            if ZD_ODV:
                b_v = tl.zeros([BT, BV], dtype=b_v.dtype)
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
        o_i = tl.arange(0, BT)
        m_jd = (i_t * BT + o_i) < T
        m_diag = m_t[:, None] & m_jd[None, :]
        p_dAqd = dAqk_ptr + ((i_b * T + o_t)[:, None] * H + i_h) * BT + o_i[None, :]
        b_dAqd = tl.load(p_dAqd, mask=m_diag, other=0.0).to(tl.float32)
        b_dAkkd = b_dA                                   # [o_t, o_A=chunk-local col]
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

    return _seg_b2b3_kernel


VARIANTS = {
    "full":     dict(ZD_HDH=0, ZD_QKG=0, ZD_ODV=0, HDH_BF16=0),
    "no_hdh":   dict(ZD_HDH=1, ZD_QKG=0, ZD_ODV=0, HDH_BF16=0),
    "no_qkg":   dict(ZD_HDH=0, ZD_QKG=1, ZD_ODV=0, HDH_BF16=0),
    "no_odv":   dict(ZD_HDH=0, ZD_QKG=0, ZD_ODV=1, HDH_BF16=0),
    "bf16_hdh": dict(ZD_HDH=0, ZD_QKG=0, ZD_ODV=0, HDH_BF16=1),
    "no_all":   dict(ZD_HDH=1, ZD_QKG=1, ZD_ODV=1, HDH_BF16=0),
}


def seg_triton(kernel, inp, dAqk, scale, BT, variant, BV=128, num_warps=8, num_stages=1):
    B, T, H, K = inp["q"].shape
    V = inp["v"].shape[-1]
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dv2 = torch.empty(B, T, H, V, device=dev, dtype=torch.float32)
    db = torch.empty(B, T, H, device=dev, dtype=torch.float32)
    f = VARIANTS[variant]
    h = inp["h"]
    dh = inp["dh"]
    if f.get("HDH_BF16"):
        h = h.to(torch.bfloat16)
        dh = dh.to(torch.bfloat16)
    grid = (NT, B * H)
    kernel[grid](
        inp["q"], inp["k"], inp["v"], inp["v_new"], inp["g"], inp["beta"], inp["A"], h,
        inp["do"], dh, inp["dv"], dAqk,
        dq, dk, dv2, db, dg,
        T, NT, scale, H=H, K=K, V=V, BT=BT, BV=min(BV, V),
        ZD_HDH=f["ZD_HDH"], ZD_QKG=f["ZD_QKG"], ZD_ODV=f["ZD_ODV"],
        num_warps=num_warps, num_stages=num_stages)
    return dict(dq=dq, dk=dk, dv2=dv2, db=db, dg=dg)


CASES = [
    ("B8 T2048 H8", 8, 2048, 8, 128, 128, 16),
    ("B2 T1024 H8", 2, 1024, 8, 128, 128, 16),
    ("T512 H4", 1, 512, 4, 128, 128, 16),
    ("T250 H3 (non-div)", 1, 250, 3, 128, 128, 16),
    ("K64V128 T513", 2, 513, 2, 64, 128, 16),
]

# analytic DRAM bytes per variant for the deployment shapes (fp32, contiguous)
def byte_model(B, T, H, K, V, BT, variant):
    NT = cdiv(T, BT)
    q = B * T * H * K * 4
    dv = B * T * H * V * 4
    h = B * NT * H * K * V * 4
    store = 3 * q + dv + B * T * H * 4
    read = 0
    if not variant["ZD_QKG"]:
        read += 3 * q                      # q,k,g
    if not variant["ZD_HDH"]:
        read += (h if variant.get("HDH_BF16") else 2 * h)   # h,dh (bf16 halves it)
    if not variant["ZD_ODV"]:
        read += 3 * dv                     # do,v_new,dv  (+ v)
    return read + store, read, store


def make(B, T, H, K, V, BT, seed=0):
    inp = dqkg_make_inputs(B, T, H, K, V, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def part_a():
    print("=" * 92)
    print("Part A: correctness -- gated `full` variant vs untouched fused kernel (bit-identical)")
    print("=" * 92)
    worst = {}
    for (name, B, T, H, K, V, BT) in CASES:
        inp = make(B, T, H, K, V, BT, seed=3)
        torch.manual_seed(7)
        dAqk = torch.randn(B, T, H, BT, device="cuda") * 0.05
        scale = 1.0 / math.sqrt(K)
        ref = fused_b2b3_triton(build_fused_b2b3(), inp, dAqk, scale, BT, num_warps=8)
        got = seg_triton(build_seg_kernel(), inp, dAqk, scale, BT, "full", num_warps=8)
        print(f"\n# {name}  K={K} V={V}")
        for fld in ("dq", "dk", "dv2", "db", "dg"):
            _, mr = stats(f"  full {fld} vs fused", got[fld], ref[fld])
            worst[fld] = max(worst.get(fld, 0.0), mr)
    print("\nworst max_rel (gated full vs untouched fused):")
    for kk in sorted(worst):
        print(f"  {kk:6s} {worst[kk]:.3e}")


def part_b():
    print("\n" + "=" * 92)
    print("Part B: timing -- byte-ablation variants, CUPTI dev + wall (fp32 operands, deployed mix)")
    print("=" * 92)
    k = build_seg_kernel()
    order = ["full", "no_hdh", "bf16_hdh", "no_qkg", "no_odv", "no_all"]
    for (name, B, T, H, K, V, BT) in CASES:
        inp = make(B, T, H, K, V, BT, seed=0)
        torch.manual_seed(1)
        dAqk = torch.randn(B, T, H, BT, device="cuda") * 0.05
        scale = 1.0 / math.sqrt(K)
        gb, rd, st = byte_model(B, T, H, K, V, BT, VARIANTS["full"])
        print(f"\n# {name}  K={K} V={V}   (full bytes={gb/1e6:.0f}MB read={rd/1e6:.0f} "
              f"store={st/1e6:.0f})")
        res = {}
        for var in order:
            def fn(v=var):
                seg_triton(k, inp, dAqk, scale, BT, v, num_warps=8)
            fn()
            torch.cuda.synchronize()
            d, _ = profile_dev(fn, reps=30)
            w = time_wall(fn, reps=50)
            res[var] = (d, w)
        dfull = res["full"][0]
        print(f"  {'variant':9s} {'bytes[MB]':>9s} {'dev[us]':>9s} {'wall[us]':>9s} "
              f"{'dev/full':>9s} {'dBytes[MB]':>11s} {'dDev[us]':>9s}")
        for var in order:
            d, w = res[var]
            vb, _, _ = byte_model(B, T, H, K, V, BT, VARIANTS[var])
            print(f"  {var:9s} {vb/1e6:9.0f} {d:9.1f} {w:9.1f} {d/dfull:8.2f}x "
                  f"{(gb-vb)/1e6:11.0f} {dfull-d:9.1f}")


def part_c():
    print("\n" + "=" * 92)
    print("Part C: static register/spill/smem per variant (B8 T2048 H8, nw=8)")
    print("=" * 92)
    B, T, H, K, V, BT = 8, 2048, 8, 128, 128, 16
    inp = make(B, T, H, K, V, BT, seed=0)
    torch.manual_seed(1)
    dAqk = torch.randn(B, T, H, BT, device="cuda") * 0.05
    scale = 1.0 / math.sqrt(K)
    print(f"  {'variant':9s} {'regs':>6s} {'spills':>7s} {'smem[KB]':>9s}")
    for var in ["full", "no_hdh", "bf16_hdh", "no_qkg", "no_odv", "no_all"]:
        kk = build_seg_kernel()
        seg_triton(kk, inp, dAqk, scale, BT, var, num_warps=8)
        try:
            cache = list(kk.device_caches.values())[0][0]
            c = next(iter(cache.values()))
            print(f"  {var:9s} {int(c.n_regs):6d} {int(c.n_spills):7d} "
                  f"{int(c.metadata.shared) / 1024:9.1f}")
        except Exception as e:  # noqa: BLE001
            print(f"  {var:9s} {e}")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}  "
          f"triton={triton.__version__}")
    mode = os.environ.get("SEG_MODE", "")
    if mode in ("", "a"):
        part_a()
    if mode in ("", "b"):
        part_b()
    if mode in ("", "c"):
        part_c()
    print("\nDONE")


if __name__ == "__main__":
    main()
