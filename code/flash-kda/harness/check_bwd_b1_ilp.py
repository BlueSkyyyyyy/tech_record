"""P7-14: validate the two §16.5 design inputs for a target-card CUDA B1 **before**
committing to the CUDA rewrite.

docs/07 §16 used `ncu` to show that our B1 (`dhu`) is *latency / parallelism* bound,
not bound by any hardware unit, and pinned the 2.14x gap to FLA `dhu` (same BT=64) on
two structural differences of ours:

  (1) no software pipeline over the reversed chunk loop (`num_stages=1` -> 47-55% of
      issue intervals are `long_scoreboard`, i.e. waiting on the global loads of
      k/q/w/do/dv);
  (2) a *single* `[K=128, BV]` running accumulator (one dependency chain), whereas FLA
      splits K=128 into two 64-chains (`fla/ops/common/chunk_delta_h.py:401-415`).

§16.5 then proposed, as design inputs for a target-card CUDA kernel, to (1) pipeline the
loads (mirror K2's TMA input pipeline) and (2) split the D=128 state into >=2 independent
accumulator chains.  This script tests both levers **on the existing bf16 Triton B1
prototype** (cheap, no target hardware needed) and measures how much of the gap each one
closes:

  V0  single chain,        num_stages=1   (= P7-8 baseline, `_dhu_kernel_bf16`)
  V1  single chain,        loop-pipelined (tl.range num_stages = NS)
  V2  split-K (2x64),      num_stages=1
  V3  split-K (2x64),      loop-pipelined

Part A  correctness of every variant vs FLA `dhu` (bf16, same BT), incl. non-divisible T.
Part B  kernel-only CUPTI timing matrix at BT=16 / BT=64, variants vs FLA.

Run inside `kernel_lab` (idle card):
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b1_ilp.py
"""
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

from check_bwd_dhu_dtype import (                                          # noqa: E402
    cuda_kernel_us, stats, make_inputs, as_dtype, fla_dhu, cdiv,
)
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu     # noqa: E402

D = 128


# ------------------------------------------------------- parametrised bf16-mma kernel
def build_kernel_ilp():
    @triton.jit
    def _dhu_ilp(
        q_ptr, k_ptr, w_ptr, do_ptr, dv_ptr, etot_ptr,
        dh_ptr, dh0_ptr, dv2_ptr,
        T, NT, scale,
        HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
        BT: tl.constexpr, BV: tl.constexpr,
        SPLIT_K: tl.constexpr, NS: tl.constexpr,
    ):
        i_v = tl.program_id(0)
        i_nh = tl.program_id(1)
        i_n = i_nh // HV
        i_h = i_nh % HV
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        dt = q_ptr.dtype.element_ty
        K2: tl.constexpr = K // 2
        o_k = tl.arange(0, K)
        if SPLIT_K:
            o_k1 = tl.arange(0, K2)
            o_k2 = K2 + tl.arange(0, K2)
            b_dh1 = tl.zeros([K2, BV], dtype=tl.float32)
            b_dh2 = tl.zeros([K2, BV], dtype=tl.float32)
            for i in tl.range(0, NT, num_stages=NS):
                c = NT - 1 - i
                p_dh = dh_ptr + ((i_n * NT + c) * HV + i_h) * K * V + o_v[None, :]
                tl.store(p_dh + o_k1[:, None] * V,
                         b_dh1.to(dh_ptr.dtype.element_ty), mask=m_v[None, :])
                tl.store(p_dh + o_k2[:, None] * V,
                         b_dh2.to(dh_ptr.dtype.element_ty), mask=m_v[None, :])
                o_t = c * BT + tl.arange(0, BT)
                m_t = o_t < T
                base = (i_n * T + o_t)[:, None]
                p_k = k_ptr + base * (HV * K) + i_h * K
                p_q = q_ptr + base * (HV * K) + i_h * K
                p_w = w_ptr + base * (HV * K) + i_h * K
                b_k1 = tl.load(p_k + o_k1[None, :], mask=m_t[:, None], other=0.0)
                b_k2 = tl.load(p_k + o_k2[None, :], mask=m_t[:, None], other=0.0)
                b_q1 = tl.load(p_q + o_k1[None, :], mask=m_t[:, None], other=0.0)
                b_q2 = tl.load(p_q + o_k2[None, :], mask=m_t[:, None], other=0.0)
                b_w1 = tl.load(p_w + o_k1[None, :], mask=m_t[:, None], other=0.0)
                b_w2 = tl.load(p_w + o_k2[None, :], mask=m_t[:, None], other=0.0)
                p_do = do_ptr + base * (HV * V) + i_h * V + o_v[None, :]
                p_dv = dv_ptr + base * (HV * V) + i_h * V + o_v[None, :]
                m_dv = m_t[:, None] & m_v[None, :]
                b_do = tl.load(p_do, mask=m_dv, other=0.0)
                b_dv = tl.load(p_dv, mask=m_dv, other=0.0)
                p_e = etot_ptr + ((i_n * NT + c) * HV + i_h) * K
                b_e1 = tl.load(p_e + o_k1)
                b_e2 = tl.load(p_e + o_k2)
                b_du = (tl.dot(b_k1, b_dh1.to(dt)) + tl.dot(b_k2, b_dh2.to(dt))
                        + b_dv)
                p_dv2 = dv2_ptr + base * (HV * V) + i_h * V + o_v[None, :]
                tl.store(p_dv2, b_du.to(dv2_ptr.dtype.element_ty), mask=m_dv)
                b_dh1 = (b_dh1 * b_e1[:, None]
                         + tl.dot(tl.trans(b_q1), b_do.to(dt)) * scale
                         - tl.dot(tl.trans(b_w1), b_du.to(dt)))
                b_dh2 = (b_dh2 * b_e2[:, None]
                         + tl.dot(tl.trans(b_q2), b_do.to(dt)) * scale
                         - tl.dot(tl.trans(b_w2), b_du.to(dt)))
            p_dh0 = dh0_ptr + i_nh * K * V + o_v[None, :]
            tl.store(p_dh0 + o_k1[:, None] * V,
                     b_dh1.to(dh0_ptr.dtype.element_ty), mask=m_v[None, :])
            tl.store(p_dh0 + o_k2[:, None] * V,
                     b_dh2.to(dh0_ptr.dtype.element_ty), mask=m_v[None, :])
        else:
            b_dh = tl.zeros([K, BV], dtype=tl.float32)
            for i in tl.range(0, NT, num_stages=NS):
                c = NT - 1 - i
                p_dh = (dh_ptr + ((i_n * NT + c) * HV + i_h) * K * V
                        + o_k[:, None] * V + o_v[None, :])
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
                b_du = tl.dot(b_k, b_dh.to(dt)) + b_dv
                p_dv2 = dv2_ptr + base * (HV * V) + i_h * V + o_v[None, :]
                tl.store(p_dv2, b_du.to(dv2_ptr.dtype.element_ty), mask=m_dv)
                b_dh = (b_dh * b_e[:, None]
                        + tl.dot(tl.trans(b_q), b_do.to(dt)) * scale
                        - tl.dot(tl.trans(b_w), b_du.to(dt)))
            p_dh0 = dh0_ptr + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
            tl.store(p_dh0, b_dh.to(dh0_ptr.dtype.element_ty), mask=m_v[None, :])

    return _dhu_ilp


def dhu_ilp(kernel, fi, do, dv, scale, BT, BV=32, split_k=False, ns=1):
    qg, kg, w, g_cum = fi["qg"], fi["kg"], fi["w"], fi["g_cum"]
    B, T, HV, Kk = qg.shape
    V = do.shape[-1]
    NT = cdiv(T, BT)
    last_idx = torch.clamp((torch.arange(NT, device=qg.device) + 1) * BT - 1, max=T - 1)
    etot = torch.exp2(g_cum.float()[:, last_idx]).contiguous()
    dh = torch.empty(B, NT, HV, Kk, V, dtype=qg.dtype, device=qg.device)
    dh0 = torch.empty(B, HV, Kk, V, dtype=torch.float32, device=qg.device)
    dv2 = torch.empty_like(do)
    grid = (cdiv(V, BV), B * HV)
    kernel[grid](qg.contiguous(), kg.contiguous(), w.contiguous(), do.contiguous(),
                 dv.contiguous(), etot, dh, dh0, dv2, T, NT, scale,
                 HV=HV, K=Kk, V=V, BT=BT, BV=BV, SPLIT_K=split_k, NS=ns,
                 num_warps=4, num_stages=1)
    return dh, dh0, dv2


VARIANTS = [
    ("V0 base   ns=1",              dict(split_k=False, ns=1)),
    ("V1 pipe   ns=2",              dict(split_k=False, ns=2)),
    ("V1 pipe   ns=3",              dict(split_k=False, ns=3)),
    ("V2 splitK ns=1",              dict(split_k=True,  ns=1)),
    ("V3 splitK ns=2",              dict(split_k=True,  ns=2)),
    ("V3 splitK ns=3",              dict(split_k=True,  ns=3)),
]


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    print(f"triton: {triton.__version__}")
    k_ilp = build_kernel_ilp()
    scale = D ** -0.5

    def run_ours(fi, do, dv, BT, cfg):
        return dhu_ilp(k_ilp, fi, do, dv, scale, BT, **cfg)

    # ---------------------------------------------------------------- Part A
    print("\n" + "=" * 88)
    print("Part A: every variant vs FLA dhu (bf16, same BT) — correctness")
    print("=" * 88)
    for (B, T, H) in [(1, 512, 4), (2, 1024, 8), (1, 250, 3)]:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        F = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)
        print(f"  --- B={B} T={T} H={H} (FLA BT=64) ---")
        for name, cfg in VARIANTS:
            dh, dh0, dv2 = run_ours(F["fi"], do, F["dv"], 64, cfg)
            da = (dh.float() - F["dh"].float()).abs().max().item()
            db = (dv2.float() - F["dv2"].float()).abs().max().item()
            dc = (dh0.float() - F["dh0"].float()).abs().max().item()
            print(f"    {name:18s} dh_max_abs={da:.3e}  dv2_max_abs={db:.3e}  "
                  f"dh0_max_abs={dc:.3e}")

    # ---------------------------------------------------------------- Part B
    print("\n" + "=" * 88)
    print("Part B: kernel-only CUPTI us/call vs FLA dhu at FLA-supported BT=32/64")
    print("=" * 88)
    shapes = [(1, 512, 4), (2, 1024, 8), (8, 2048, 8)]
    for (B, T, H) in shapes:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        for BT in (32, 64):
            F = fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do)
            fla_us = cuda_kernel_us(lambda F=F, BT=BT: chunk_gated_delta_rule_bwd_dhu(
                q=F["fi"]["qg"], k=F["fi"]["kg"], w=F["fi"]["w"], gk=F["fi"]["g_cum"],
                h0=h0, dht=None, do=do, dv=F["dv"], scale=scale, chunk_size=BT),
                "bwd_kernel_dhu")
            base_us = None
            print(f"  B{B} T{T} H{H}  BT={BT}   FLA dhu = {fla_us:7.1f}us")
            for name, cfg in VARIANTS:
                us = cuda_kernel_us(lambda F=F, BT=BT, cfg=cfg: run_ours(
                    F["fi"], do, F["dv"], BT, cfg), "_dhu_ilp")
                if base_us is None:
                    base_us = us
                print(f"      {name:18s} {us:8.1f}us   xFLA={us/fla_us:5.2f}  "
                      f"xV0={us/base_us:5.2f}")

    # ---------------------------------------------------------------- Part C
    print("\n" + "=" * 88)
    print("Part C: deployed granularity BT=16 — operands from FlashKDA K1 workspace")
    print("        (P6-8 reconstruct, cast to bf16; FLA intra has no BT=16 kernel)")
    print("=" * 88)
    from check_bwd_ws_reuse import make_inputs as ws_in, flash_forward, reconstruct, CHUNK
    lb = -5.0
    for (B, T, H) in shapes[:2]:
        inp = ws_in(B, T, H, 0, "cuda", 1.0)
        fw = flash_forward(inp, scale, lb, "cuda")
        _, _, fi16 = reconstruct(fw, inp, lb, scale)
        fi16 = {kk: (vv.to(torch.bfloat16) if vv.dtype == torch.float32 else vv)
                for kk, vv in fi16.items()}
        torch.manual_seed(77)
        do = torch.randn(B, T, H, D, device="cuda").bfloat16()
        dv = torch.randn(B, T, H, D, device="cuda").bfloat16()
        base_us = None
        print(f"  B{B} T{T} H{H}  BT={CHUNK}")
        for name, cfg in VARIANTS:
            us = cuda_kernel_us(lambda cfg=cfg: dhu_ilp(
                k_ilp, fi16, do, dv, scale, CHUNK, **cfg), "_dhu_ilp")
            if base_us is None:
                base_us = us
            print(f"      {name:18s} {us:8.1f}us   xV0={us/base_us:5.2f}")
    print("\nDONE")


if __name__ == "__main__":
    main()
