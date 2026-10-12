"""P7-8: re-attribute the P7-7 claim that "B1 (`dhu`) is 3.9x slower than FLA".

P7-7 (`bench_bwd.out.txt`) reported our B1 `_dhu_kernel` at 3.9x FLA's
`chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` and concluded B1 is *the* backward
bottleneck that "should mirror K2's warp specialization".  That comparison is **not
apples-to-apples**: the two sides differ along two axes at once

  (a) chunk granularity  ours BT=16 (FlashKDA CHUNK)   vs  FLA BT=64;
  (b) operand dtype/mma  ours fp32 operands -> tf32 `tl.dot` (state kept fp32)
                         vs  FLA bf16 operands -> bf16 `tl.dot` (state rounded to bf16).

The P7-7 "ours" side is in fp32 only because `reconstruct` (P6-8) casts the whole
workspace to float; the actual FlashKDA workspace fields are bf16, and FLA's own chain is
bf16, so (b) is an artifact of the harness, not of the kernel.

This script controls for each axis.  It builds two kernels — the original
(`check_bwd_dhu_kernel._dhu_kernel`, fp32 state / tf32) and a **bf16-mma** clone that
mirrors FLA exactly (`tl.dot(..., b_dh.to(operand_dtype))`: the state is rounded to bf16
inside the dot but accumulated in fp32) — and measures kernel-only times (CUPTI):

  Part A  correctness of the bf16-mma kernel vs FLA `dhu` (bf16), same BT.
  Part B  timing matrix {ours, FLA} x {fp32/tf32, bf16} x {BT=32, BT=64} on matching
          operands (isolates the implementation gap at fixed BT and dtype).
  Part C  the P7-7-style number (ours B1 fed the fp32 workspace at BT=16 vs FLA bf16
          BT=64) and its decomposition: fp32->bf16 operand switch.

Run inside `kernel_lab` (GPU, FlashKDA built, FLA installed):
    python ~/proj/tech_record/code/flash-kda/harness/check_bwd_dhu_dtype.py
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

import check_bwd_dhu_kernel as K                                            # noqa: E402
from check_bwd_dhu_kernel import build_kernel as build_dhu_fp32, dhu_triton  # noqa: E402

D = 128


def cdiv(a, b):
    return (a + b - 1) // b


# --------------------------------------------------------------- bf16-mma kernel clone
def build_kernel_bf16():
    """Same recurrence as `_dhu_kernel`, but every `tl.dot` casts its operands to the
    operand dtype (bf16) — exactly what FLA does (`b_dh1.to(b_k.dtype)`).  The running
    state `b_dh` stays fp32; only the mma operands are bf16, so it hits the bf16 tensor
    cores at 2x the tf32 rate and halves the loaded bytes."""
    @triton.jit
    def _dhu_kernel_bf16(
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
        bo_t = dh_ptr.dtype.element_ty
        for c in range(NT - 1, -1, -1):
            p_dh = dh_ptr + ((i_n * NT + c) * HV + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
            tl.store(p_dh, b_dh.to(bo_t), mask=m_v[None, :])
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
            dt = b_k.dtype
            b_du = tl.dot(b_k, b_dh.to(dt)) + b_dv
            p_dv2 = dv2_ptr + base * (HV * V) + i_h * V + o_v[None, :]
            tl.store(p_dv2, b_du.to(dv2_ptr.dtype.element_ty), mask=m_dv)
            b_dh = (b_dh * b_e[:, None]
                    + tl.dot(tl.trans(b_q), b_do.to(dt)) * scale
                    - tl.dot(tl.trans(b_w), b_du.to(dt)))
        p_dh0 = dh0_ptr + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dh.to(dh0_ptr.dtype.element_ty), mask=m_v[None, :])

    return _dhu_kernel_bf16


def dhu_triton_bf16(kernel, fi, do, dv, scale, BT, BV=32):
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
                 HV=HV, K=Kk, V=V, BT=BT, BV=BV, num_warps=4, num_stages=1)
    return dh, dh0, dv2


# ------------------------------------------------------------------------------- timing
def cuda_kernel_us(fn, sub, reps=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    return sum(ev.self_device_time_total for ev in prof.key_averages()
               if sub in ev.key) / reps


def stats(name, a, b):
    d = (a.float() - b.float()).abs()
    denom = b.float().abs().max().item()
    print(f"  {name:30s} max_abs={d.max().item():.3e}  "
          f"max_rel={d.max().item()/(denom+1e-12):.3e}  ref_max={denom:.3e}")


# ------------------------------------------------------------------------------- data
def make_inputs(B, T, H, seed=0):
    torch.manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, D), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, D), dim=-1)
    v = torch.randn(B, T, H, D)
    g_log2 = -torch.rand(B, T, H, D) * 0.5
    beta = torch.sigmoid(torch.randn(B, T, H))
    h0 = torch.randn(B, H, D, D) * 0.1
    return q, k, v, g_log2, beta, h0


def as_dtype(x, dt):
    return x.to(dt).cuda()


def fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do):
    return K.fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do)


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    k_fp32 = build_dhu_fp32()
    k_bf16 = build_kernel_bf16()
    scale = D ** -0.5
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu

    def run_ours(fi, do, dv, scale, BT, dt):
        if dt == torch.bfloat16:
            return dhu_triton_bf16(k_bf16, fi, do, dv, scale, BT)
        return K.dhu_triton(k_fp32, fi, do, dv, None, scale, BT)

    shapes = [(1, 512, 4), (2, 1024, 8), (8, 2048, 8)]

    print("\n" + "=" * 84)
    print("Part A: bf16-mma B1 vs FLA dhu (bf16, same BT=64) — correctness")
    print("=" * 84)
    for (B, T, H) in shapes:
        q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
        q, k, v, g_log2, beta, h0 = (as_dtype(x, torch.bfloat16)
                                     for x in (q, k, v, g_log2, beta, h0))
        torch.manual_seed(99)
        do = as_dtype(torch.randn(B, T, H, D), torch.bfloat16)
        F = fla_dhu(q, k, v, g_log2, beta, h0, scale, 64, do)
        dh, dh0, dv2 = run_ours(F["fi"], do, F["dv"], scale, 64, torch.bfloat16)
        print(f"  B={B} T={T} H={H}")
        stats("dh  ours-bf16 vs FLA-bf16", dh, F["dh"])
        stats("dv2 ours-bf16 vs FLA-bf16", dv2, F["dv2"])
        stats("dh0 ours-bf16 vs FLA-bf16", dh0, F["dh0"])

    print("\n" + "=" * 84)
    print("Part B: timing matrix (kernel-only, CUPTI us/call), matching operands per BT")
    print("=" * 84)
    print(f"  {'shape':14s} | {'ours32':>8s} {'FLA32':>8s} {'o/F':>5s} | "
          f"{'ours64':>8s} {'FLA64':>8s} {'o/F':>5s} | dtype")
    for (B, T, H) in shapes:
        for dt in (torch.float32, torch.bfloat16):
            q, k, v, g_log2, beta, h0 = make_inputs(B, T, H)
            q, k, v, g_log2, beta, h0 = (as_dtype(x, dt)
                                         for x in (q, k, v, g_log2, beta, h0))
            torch.manual_seed(99)
            do = as_dtype(torch.randn(B, T, H, D), dt)
            row = {}
            for BT in (32, 64):
                F = fla_dhu(q, k, v, g_log2, beta, h0, scale, BT, do)
                fi = F["fi"]
                fla_fn = lambda F=F, BT=BT: chunk_gated_delta_rule_bwd_dhu(
                    q=F["fi"]["qg"], k=F["fi"]["kg"], w=F["fi"]["w"], gk=F["fi"]["g_cum"],
                    h0=h0, dht=None, do=do, dv=F["dv"], scale=scale, chunk_size=BT)
                row[BT] = (cuda_kernel_us(fla_fn, "bwd_kernel_dhu"),
                           cuda_kernel_us(lambda F=F, BT=BT: run_ours(
                               F["fi"], do, F["dv"], scale, BT, dt), "_dhu_kernel"))
            dtn = "bf16" if dt == torch.bfloat16 else "fp32"
            o32, f32 = row[32][1], row[32][0]
            o64, f64 = row[64][1], row[64][0]
            print(f"  B{B} T{T} H{H:<8} | {o32:8.1f} {f32:8.1f} {o32/f32:5.2f} | "
                  f"{o64:8.1f} {f64:8.1f} {o64/f64:5.2f} | {dtn}")

    print("\n" + "=" * 84)
    print("Part C: the P7-7-style number — ours B1 fed the fp32 workspace @BT=16")
    print("=" * 84)
    from check_bwd_ws_reuse import make_inputs as ws_in, flash_forward, reconstruct, CHUNK
    for (B, T, H) in shapes[:2]:
        lb = -5.0
        inp = ws_in(B, T, H, 0, "cuda", 1.0)
        fw = flash_forward(inp, scale, lb, "cuda")
        _, _, fi16 = reconstruct(fw, inp, lb, scale)   # fp32 (P6-8 casts to float)
        torch.manual_seed(77)
        do = torch.randn(B, T, H, D, device="cuda")
        dv = torch.randn(B, T, H, D, device="cuda")
        h0T = inp["h0"].float().permute(0, 1, 3, 2).contiguous()
        us_fp32 = cuda_kernel_us(
            lambda: K.dhu_triton(k_fp32, fi16, do, dv, h0T, scale, CHUNK), "_dhu_kernel")
        fi16_bf = {kk: (vv.to(torch.bfloat16) if vv.dtype == torch.float32 else vv)
                   for kk, vv in fi16.items()}
        us_bf16 = cuda_kernel_us(
            lambda: dhu_triton_bf16(k_bf16, fi16_bf, do.bfloat16(), dv.bfloat16(),
                                    scale, CHUNK), "_dhu_kernel")
        print(f"  B={B} T={T} H={H}:  ours-B1 (P7-7 fp32/tf32, BT=16) = {us_fp32:7.1f}us"
              f"   ours-B1 (bf16 workspace, BT=16) = {us_bf16:7.1f}us   "
              f"speedup {us_fp32/us_bf16:.2f}x")
    print("\nDONE")


if __name__ == "__main__":
    main()
