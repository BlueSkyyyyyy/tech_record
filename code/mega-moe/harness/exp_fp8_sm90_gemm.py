#!/usr/bin/env python3
"""P22-1: DeepGEMM 的 SM90 FP8 dense GEMM 主 kernel：1D1D vs 1D2D 精读 + H100 实测。

参考源码（~/github/DeepGEMM，commit 057ca59）：
  csrc/apis/gemm.hpp:117-132          —— fp8_fp4_gemm_nt 的 arch9 分派（gran_n==1 → 1d1d，
                                          否则 1d2d），默认 recipe 由 arch9 决定
  csrc/utils/layout.hpp:86-98         —— get_default_recipe: SM90 唯一 (1,128,128) → 1d2d
  csrc/apis/layout.hpp:11-58          —— transform_sf_into_required_layout 四分派
  csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp —— host 入口（Normal 需 c 且 d=float）
  csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp —— host 入口（Normal no c、d=bf16）
  deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh —— kernel（WGMMA 64xNx32，
                                          fp32 D，TMA_REDUCE_ADD，K 分组改 tensormap）
  deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh —— kernel（bf16 D，SFB 2D block）
  deep_gemm/include/deep_gemm/mma/sm90.cuh:42-60 —— FP8MMASelector（M=64/K=32）
  deep_gemm/include/deep_gemm/scheduler/gemm.cuh —— SF_K 取址

本脚本做的事（全部 H100 可跑，SM90 真实设备代码）：
  1. 分派：默认 recipe → 1d2d；显式 recipe=(1,1,128)+accumulate → 1d1d；抓真实 kernel 名。
  2. 契约：1d1d 必须 accumulate 且 D=float；1d2d 必须 no-accumulate 且 D=bf16。
  3. 语义：两种 SF 粒度各自把 A（per-token 1D）与 B（1d2d=per-128 块 2D / 1d1d=per-token 1D）
     反量化后 fp32 matmul 对拍。
  4. 交叉验证：把 1d2d 的 per-128 块 scale 广播成 per-column，喂给 1d1d，二者应数值一致
     —— 坐实「同一数学 D = (A·sfa)·(Bᵀ·sfb)，只是 SFB 粒度不同」。
  5. 性能：同形状 1d2d(bf16) vs 1d1d(fp32+accumulate) 的 TFLOPS。

运行：bash harness/exp_fp8_sm90_gemm.sh > harness/exp_fp8_sm90_gemm.out.txt
"""
import sys

import torch

try:
    import deep_gemm
    from deep_gemm.utils import (
        per_token_cast_to_fp8, per_block_cast_to_fp8,
    )
except Exception as e:  # pragma: no cover
    print("FATAL: cannot import deep_gemm:", e)
    sys.exit(1)

DEV = "cuda"
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False

print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
      f"cc={torch.cuda.get_device_capability(0)} | device_num_sms={deep_gemm.get_num_sms()}")
from deep_gemm.testing import get_arch_major  # noqa: E402
print(f"deep_gemm arch_major = {get_arch_major()}")
print("=" * 100)


# ------------------------------------------------------------------ helpers
def relerr(x, ref):
    x = x.double()
    ref = ref.double()
    return ((x - ref).norm() / ref.norm().clamp_min(1e-30)).item()


def expand_sf_token(sf, k, gran_k=128):
    # sf: [mn, k/gran_k] -> [mn, k] 逐 k 块广播
    return sf.repeat_interleave(gran_k, dim=1)[:, :k]


def dequant_1d(x_fp8, sf, gran_k=128):
    # x_fp8 [mn, k]，sf [mn, k/gran_k]
    return x_fp8.float() * expand_sf_token(sf, x_fp8.size(1), gran_k)


def bench(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2] / 1e3


def kernel_names(fn):
    import torch.profiler as profiler
    fn()
    torch.cuda.synchronize()
    with profiler.profile(activities=[profiler.ProfilerActivity.CUDA]) as prof:
        fn()
    torch.cuda.synchronize()
    return sorted(e.key for e in prof.key_averages()
                  if "gemm" in e.key.lower() or "nvjet" in e.key.lower())


def short_name(names):
    if not names:
        return "<none>"
    n = names[0]
    for tag in ("sm90_fp8_gemm_1d1d_impl", "sm90_fp8_gemm_1d2d_impl"):
        if tag in n:
            return tag + n.split(tag)[-1][:90]
    return n[:110]


# ================================================================== 1. 分派 & 契约
print("\n[1] 分派与契约（SM90：默认 recipe=(1,128,128) → 1d2d；recipe=(1,1,128) → 1d1d）")
m, n, k = 4096, 4096, 4096
deep_gemm.use_deterministic_algorithms(True)


def make_1d2d(mm, nn, kk):
    a = torch.randn((mm, kk), device=DEV, dtype=torch.bfloat16)
    b = torch.randn((nn, kk), device=DEV, dtype=torch.bfloat16)
    a_fp8, sfa = per_token_cast_to_fp8(a, use_ue8m0=False)
    b_fp8, sfb = per_block_cast_to_fp8(b, use_ue8m0=False)  # [n/128, k/128]
    d = torch.empty((mm, nn), device=DEV, dtype=torch.bfloat16)
    return a, b, (a_fp8, sfa), (b_fp8, sfb), sfa, sfb, d


def make_1d1d(mm, nn, kk, accumulate=True):
    a = torch.randn((mm, kk), device=DEV, dtype=torch.bfloat16)
    b = torch.randn((nn, kk), device=DEV, dtype=torch.bfloat16)
    a_fp8, sfa = per_token_cast_to_fp8(a, use_ue8m0=False)
    b_fp8, sfb = per_token_cast_to_fp8(b, use_ue8m0=False)  # [n, k/128]
    d = (torch.randn((mm, nn), device=DEV, dtype=torch.float) * 4 if accumulate
         else torch.empty((mm, nn), device=DEV, dtype=torch.float))
    return a, b, (a_fp8, sfa), (b_fp8, sfb), sfa, sfb, d


# --- 1d2d 默认
a, b, A, B, sfa, sfb, d = make_1d2d(m, n, k)
deep_gemm.fp8_fp4_gemm_nt(A, B, d)
print(f"  默认 dense FP8 -> {short_name(kernel_names(lambda: deep_gemm.fp8_fp4_gemm_nt(A, B, d)))}")
print(f"    (sfb shape={tuple(sfb.shape)} = [n/128, k/128]，per-128 块 2D scale)")

# --- 1d1d 显式
a1, b1, A1, B1, sfa1, sfb1, d1 = make_1d1d(m, n, k, accumulate=True)
deep_gemm.fp8_fp4_gemm_nt(A1, B1, d1, c=d1, recipe=(1, 1, 128))
print(f"  recipe=(1,1,128)+acc -> {short_name(kernel_names(lambda: deep_gemm.fp8_fp4_gemm_nt(A1, B1, d1, c=d1, recipe=(1, 1, 128))))}")
print(f"    (sfb shape={tuple(sfb1.shape)} = [n, k/128]，per-token/per-channel 1D scale)")

# --- 契约反例
print("\n  契约反例：")
try:
    d_bad = torch.empty((m, n), device=DEV, dtype=torch.bfloat16)
    deep_gemm.fp8_fp4_gemm_nt(A1, B1, d_bad, recipe=(1, 1, 128))
    print("    [1d1d 无 accumulate/D=bf16] 未拒绝  <-- 异常！")
except Exception as e:
    print(f"    1d1d 无 c / D=bf16 被拒: {type(e).__name__}: {str(e).strip().splitlines()[-1][:90]}")
try:
    d2 = torch.empty((m, n), device=DEV, dtype=torch.bfloat16)
    deep_gemm.fp8_fp4_gemm_nt(A, B, d2, c=d2)  # 1d2d + accumulate
    print("    [1d2d + accumulate] 未拒绝  <-- 异常！")
except Exception as e:
    print(f"    1d2d + accumulate 被拒: {type(e).__name__}: {str(e).strip().splitlines()[-1][:90]}")

# ================================================================== 2. 语义（隔离量化）
print("\n[2] 语义对拍（反量化 A/B → fp32 matmul 参考）")
print(f"{'m':>6} {'n':>6} {'k':>6} | {'1d2d relerr':>12} | {'1d1d relerr':>12}")
for (mm, nn, kk) in [(1024, 1024, 1024), (2048, 4096, 2048), (4096, 7168, 7168),
                     (4096, 4096, 4096), (128, 4096, 4096)]:
    # 1d2d：A per-token，B per-128 块
    pa = torch.randn((mm, kk), device=DEV, dtype=torch.bfloat16)
    pb = torch.randn((nn, kk), device=DEV, dtype=torch.bfloat16)
    a_fp8, sfa_ = per_token_cast_to_fp8(pa, use_ue8m0=False)
    b_fp8, sfb_ = per_block_cast_to_fp8(pb, use_ue8m0=False)
    d2 = torch.empty((mm, nn), device=DEV, dtype=torch.bfloat16)
    deep_gemm.fp8_fp4_gemm_nt((a_fp8, sfa_), (b_fp8, sfb_), d2)
    # 参考：A_dq @ B_dq^T，B 的块 scale 按 [128,128] 广播
    a_dq = dequant_1d(a_fp8, sfa_)
    b_dq = dequant_1d(
        b_fp8,
        sfb_.repeat_interleave(128, dim=0)[:nn],  # [n/128,k/128] -> [n,k/128]
    )
    ref2 = a_dq @ b_dq.t()
    e2 = relerr(d2, ref2.to(torch.bfloat16))

    # 1d1d：A/B 都 per-token
    a_fp8, sfa_ = per_token_cast_to_fp8(pa, use_ue8m0=False)
    b_fp8, sfb_ = per_token_cast_to_fp8(pb, use_ue8m0=False)
    d1 = torch.zeros((mm, nn), device=DEV, dtype=torch.float)
    deep_gemm.fp8_fp4_gemm_nt((a_fp8, sfa_), (b_fp8, sfb_), d1, c=d1, recipe=(1, 1, 128))
    a_dq = dequant_1d(a_fp8, sfa_)
    b_dq = dequant_1d(b_fp8, sfb_)
    ref1 = a_dq @ b_dq.t()
    e1 = relerr(d1, ref1)
    print(f"{mm:6d} {nn:6d} {kk:6d} | {e2:12.3e} | {e1:12.3e}")

# ================================================================== 3. 交叉验证（SFB 粒度）
print("\n[3] 交叉验证：1d2d 的块 scale 广播成 per-column 后喂 1d1d（同一数学）")
print(f"{'m':>6} {'n':>6} {'k':>6} | {'1d2d vs 1d1d(广播) relerr':>26}")
for (mm, nn, kk) in [(1024, 1024, 1024), (4096, 4096, 4096), (2048, 4096, 2048)]:
    pa = torch.randn((mm, kk), device=DEV, dtype=torch.bfloat16)
    pb = torch.randn((nn, kk), device=DEV, dtype=torch.bfloat16)
    a_fp8, sfa = per_token_cast_to_fp8(pa, use_ue8m0=False)
    b_fp8, sfb_block = per_block_cast_to_fp8(pb, use_ue8m0=False)  # [n/128,k/128]
    # 1d2d 输出（bf16）
    d2 = torch.empty((mm, nn), device=DEV, dtype=torch.bfloat16)
    deep_gemm.fp8_fp4_gemm_nt((a_fp8, sfa), (b_fp8, sfb_block), d2)
    # 1d2d 的块 scale 广播到每列 → 作为 1d1d 的 per-token SFB
    sfb_col = sfb_block.repeat_interleave(128, dim=0)[:nn].contiguous()
    d1 = torch.zeros((mm, nn), device=DEV, dtype=torch.float)
    deep_gemm.fp8_fp4_gemm_nt((a_fp8, sfa), (b_fp8, sfb_col), d1, c=d1, recipe=(1, 1, 128))
    # d2 是 bf16 舍入后的 d1
    print(f"{mm:6d} {nn:6d} {kk:6d} | {relerr(d2, d1.to(torch.bfloat16)):26.3e}")

# ================================================================== 4. 性能
print("\n[4] 性能（SM90 1d2d bf16-out vs 1d1d fp32-out + accumulate）")
print(f"{'m':>6} {'n':>6} {'k':>6} | {'1d2d us':>9} {'1d2d TF':>8} | {'1d1d us':>9} {'1d1d TF':>8}")
for (mm, nn, kk) in [(4096, 7168, 7168), (4096, 4096, 4096), (8192, 4096, 4096),
                     (2048, 4096, 2048), (128, 7168, 2048)]:
    pa = torch.randn((mm, kk), device=DEV, dtype=torch.bfloat16)
    pb = torch.randn((nn, kk), device=DEV, dtype=torch.bfloat16)
    a_fp8, sfa = per_token_cast_to_fp8(pa, use_ue8m0=False)
    b_b_fp8, sfb_block = per_block_cast_to_fp8(pb, use_ue8m0=False)
    b_t_fp8, sfb_tok = per_token_cast_to_fp8(pb, use_ue8m0=False)
    d2 = torch.empty((mm, nn), device=DEV, dtype=torch.bfloat16)
    d1 = torch.zeros((mm, nn), device=DEV, dtype=torch.float)

    f2 = lambda: deep_gemm.fp8_fp4_gemm_nt((a_fp8, sfa), (b_b_fp8, sfb_block), d2)
    f1 = lambda: deep_gemm.fp8_fp4_gemm_nt((a_fp8, sfa), (b_t_fp8, sfb_tok), d1, c=d1, recipe=(1, 1, 128))
    t2, t1 = bench(f2), bench(f1)
    print(f"{mm:6d} {nn:6d} {kk:6d} | {t2*1e6:9.1f} {2*mm*nn*kk/t2/1e12:8.0f} | "
          f"{t1*1e6:9.1f} {2*mm*nn*kk/t1/1e12:8.0f}")

print("\n完成。")
