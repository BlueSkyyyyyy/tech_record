#!/usr/bin/env python3
"""DeepGEMM 的 cuBLASLt 回退路径与选核实测（P17-1，H100 真机）。

参考源码（~/github/DeepGEMM，commit 057ca59）：
  csrc/apis/gemm.hpp             —— bf16_gemm_nt / fp8_fp4_gemm_nt / m_grouped_* 的分派
  csrc/apis/einsum.hpp           —— bhr_hdr_bhd / bhd_hdr_bhr / bhd_bhr_hdr 的分派
  csrc/runtime/runtime.hpp       —— is_cublaslt_available / workspace / handle
  csrc/jit_kernels/impls/smxx_cublaslt.hpp —— 列主序操作数交换、FAST_ACCUM、SM_COUNT_TARGET
  csrc/jit_kernels/impls/epilogue_class.hpp —— get_alpha() → epilogue_alpha.has_value()

结论先行（由本脚本实测）：
  * 稠密 BF16 `bf16_gemm_nt` 默认走 **cuBLASLt**（plain-store epilogue 的 alpha=1.0）；
  * 打开 deterministic 或 `set_num_sms(<device)` 后改走 DeepGEMM 自己的 SM90 kernel；
  * FP8 dense 永远不走 cuBLASLt；einsum `bhd,bhr->hdr` 永远走 cuBLASLt。

运行：bash harness/exp_cublaslt_fallback.sh > harness/exp_cublaslt_fallback.out.txt
"""
import os
import sys
import time

import torch
import torch.profiler as profiler

try:
    import deep_gemm
except Exception as e:  # pragma: no cover
    print("FATAL: cannot import deep_gemm:", e)
    sys.exit(1)

DEV = "cuda"
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False

DEVICE_SMS = deep_gemm.get_num_sms()
print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
      f"sm_util={torch.cuda.get_device_capability(0)} | device_num_sms={DEVICE_SMS}")

# ------------------------------------------------------------------ helpers
def reset_state():
    deep_gemm.set_num_sms(DEVICE_SMS)
    deep_gemm.use_deterministic_algorithms(False)


def run_and_capture(fn):
    """Run fn once under the profiler; return sorted unique CUDA kernel names."""
    fn()  # warmup (JIT compile / cublasLt heuristic)
    torch.cuda.synchronize()
    with profiler.profile(activities=[profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = set()
    for ev in prof.events():
        if ev.device_type == profiler.DeviceType.CUDA and ev.name:
            n = ev.name
            if any(t in n for t in ("gemm", "nvjet", "sm90", "sm100", "xmma",
                                    "cutlass", "cublas", "void ", "kernel")):
                names.add(n)
    return sorted(names)


def classify(names):
    """Heuristic classifier for reporting."""
    joined = " | ".join(names)
    tags = []
    if any("nvjet" in n for n in names):
        tags.append("cuBLASLt(nvjet)")
    if any("sm90_bf16_gemm" in n or "sm90_fp8_gemm" in n for n in names):
        tags.append("DeepGEMM-SM90")
    if any("sm100" in n for n in names):
        tags.append("DeepGEMM-SM100")
    return ",".join(tags) if tags else "(unknown)", joined


def tflops(m, n, k, ms):
    return 2.0 * m * n * k / (ms * 1e-3) / 1e12


def rel(x, ref):
    x = x.float().flatten().double()
    ref = ref.float().flatten().double()
    return (x - ref).norm().item() / (ref.norm().item() + 1e-30)


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


# =================================================================== A. 稠密 BF16
print("\n" + "=" * 84)
print("A. 稠密 BF16 `bf16_gemm_nt`：默认 / deterministic / 限制 num_sms 的选核")
print("=" * 84)
M, N, K = 4096, 7168, 7168
a = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
b = torch.randn(N, K, dtype=torch.bfloat16, device=DEV)
d = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
ref = (a.float() @ b.float().t()).bfloat16()


def run() -> str:
    tag, names = classify(run_and_capture(lambda: deep_gemm.bf16_gemm_nt(a, b, d)))
    err = rel(d, ref)
    return f"{tag:22s} | relerr={err:.2e} | {names[:120]}"


reset_state()
print(f"  默认                : {run()}")
print(f"  deterministic=True  : ", end="")
deep_gemm.use_deterministic_algorithms(True)
print(run())
reset_state()
print(f"  num_sms={DEVICE_SMS-8:4d}   : ", end="")
deep_gemm.set_num_sms(DEVICE_SMS - 8)
print(run())
reset_state()
# alpha 走 AlphaEpilogue，get_alpha()=0.5 → 仍可被 cuBLASLt 覆盖
d2 = torch.empty_like(d)
tag, names = classify(run_and_capture(lambda: deep_gemm.bf16_gemm_nt(a, b, d2, alpha=0.5)))
print(f"  alpha=0.5           : {tag:22s} | relerr_vs_0.5*ref={rel(d2, 0.5 * ref):.2e} | {names[:120]}")

# =================================================================== B. 自定义 epilogue
print("\n" + "=" * 84)
print("B. 自定义 epilogue：Identity/Alpha 可回退，FP8Quantization 在 SM90 被拒（需 SM100）")
print("=" * 84)
reset_state()
d3 = torch.empty_like(d)
tag, names = classify(run_and_capture(
    lambda: deep_gemm.bf16_gemm_nt(a, b, d3, epilogue=deep_gemm.epilogue.Alpha(2.0))))
print(f"  epilogue.Alpha(2.0) : {tag:22s} | {names[:120]}")
try:
    from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8
    _fa, _sfa = per_token_cast_to_fp8(a, use_ue8m0=False)
    _fb, _sfb = per_block_cast_to_fp8(b, use_ue8m0=False)
    sfd = torch.empty(M, N // 128, dtype=torch.int, device=DEV)
    dfp8 = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
    deep_gemm.fp8_gemm_nt((_fa, _sfa), (_fb, _sfb), dfp8,
                          epilogue=deep_gemm.epilogue.FP8Quantization(sfd))
    print("  epilogue.FP8Quant    : (unexpected) no error")
except Exception as e:
    print(f"  epilogue.FP8Quant    : raised {type(e).__name__}: {str(e)[:100]}")

# =================================================================== C. FP8 dense
print("\n" + "=" * 84)
print("C. 稠密 FP8 `fp8_gemm_nt`：永远走 DeepGEMM（无 cuBLASLt 回退）")
print("=" * 84)
reset_state()
from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8
fa, sfa = per_token_cast_to_fp8(a, use_ue8m0=False)
fb, sfb = per_block_cast_to_fp8(b, use_ue8m0=False)
fd = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
# 反量化参考：a ≈ a_fp8 * sfa（per-token 128），b ≈ b_fp8 * sfb（per-128x128 block）
_a_deq = fa.float() * sfa.float().repeat_interleave(128, dim=1)
_b_deq = fb.float() * sfb.float().repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
ref_fp8 = _a_deq @ _b_deq.t()
tag, names = classify(run_and_capture(
    lambda: deep_gemm.fp8_gemm_nt((fa, sfa), (fb, sfb), fd)))
print(f"  fp8_gemm_nt default : {tag:22s} | relerr={rel(fd, ref_fp8):.2e} | {names[:120]}")
deep_gemm.use_deterministic_algorithms(True)
tag, names = classify(run_and_capture(
    lambda: deep_gemm.fp8_gemm_nt((fa, sfa), (fb, sfb), fd)))
print(f"  fp8_gemm_nt determ  : {tag:22s} | {names[:120]}")
reset_state()

# =================================================================== D. einsum
print("\n" + "=" * 84)
print("D. einsum（按头 batched BF16）：`bhr,hdr->bhd` 默认回退，`bhd,hdr->bhr` 同理；")
print("   `bhd,bhr->hdr` 无条件走 cuBLASLt（源码 TODO determinism）")
print("=" * 84)
B, H, R, Dd = 16, 16, 512, 512
A4 = torch.randn(B, H, R, dtype=torch.bfloat16, device=DEV)
B4 = torch.randn(H, Dd, R, dtype=torch.bfloat16, device=DEV)
D4 = torch.empty(B, H, Dd, dtype=torch.bfloat16, device=DEV)
reset_state()
tag, names = classify(run_and_capture(lambda: deep_gemm.einsum("bhr,hdr->bhd", A4, B4, D4)))
print(f"  bhr,hdr->bhd 默认   : {tag:22s} | {names[:120]}")
deep_gemm.use_deterministic_algorithms(True)
tag, names = classify(run_and_capture(lambda: deep_gemm.einsum("bhr,hdr->bhd", A4, B4, D4)))
print(f"  bhr,hdr->bhd determ : {tag:22s} | {names[:120]}")
reset_state()

A6 = torch.randn(B, H, Dd, dtype=torch.bfloat16, device=DEV)   # b,h,d
B6 = torch.randn(H, Dd, R, dtype=torch.bfloat16, device=DEV)   # h,d,r
D6 = torch.empty(B, H, R, dtype=torch.bfloat16, device=DEV)    # b,h,r
tag, names = classify(run_and_capture(lambda: deep_gemm.einsum("bhd,hdr->bhr", A6, B6, D6)))
print(f"  bhd,hdr->bhr 默认   : {tag:22s} | {names[:120]}")
deep_gemm.use_deterministic_algorithms(True)
tag, names = classify(run_and_capture(lambda: deep_gemm.einsum("bhd,hdr->bhr", A6, B6, D6)))
print(f"  bhd,hdr->bhr determ : {tag:22s} | {names[:120]}")
reset_state()

A5 = torch.randn(B, H, Dd, dtype=torch.bfloat16, device=DEV)   # b,h,d
B5 = torch.randn(B, H, R, dtype=torch.bfloat16, device=DEV)    # b,h,r
D5 = torch.empty(H, Dd, R, dtype=torch.float, device=DEV)      # h,d,r
try:
    tag, names = classify(run_and_capture(lambda: deep_gemm.einsum("bhd,bhr->hdr", A5, B5, D5)))
    print(f"  bhd,bhr->hdr 默认   : {tag:22s} | {names[:120]}")
except Exception as e:
    print(f"  bhd,bhr->hdr        : raised {type(e).__name__}: {str(e)[:100]}")

# =================================================================== E. 显式 cublaslt API
print("\n" + "=" * 84)
print("E. 显式 `cublaslt_gemm_nt/tt` API：正确性与 C 累加（beta=1）")
print("=" * 84)
reset_state()
dd = torch.empty_like(d)
deep_gemm.cublaslt_gemm_nt(a, b, dd)
print(f"  cublaslt_gemm_nt     : relerr={rel(dd, ref):.2e}")
dd2 = torch.ones_like(d)
deep_gemm.cublaslt_gemm_nt(a, b, dd2, c=torch.ones_like(d))
print(f"  cublaslt_gemm_nt+C   : relerr(acc)={rel(dd2, ref.float() + 1):.2e}")
aT = a.t().contiguous()  # [K,M]
dd3 = torch.empty_like(d)
deep_gemm.cublaslt_gemm_tt(aT, b, dd3)
print(f"  cublaslt_gemm_tt     : relerr={rel(dd3, ref):.2e}")

# batched symm/syrk（strided-batched 回退）
print("  -- strided-batched --")
NB = 8
ab = torch.randn(NB, 256, 256, dtype=torch.bfloat16, device=DEV)
db = torch.empty(NB, 256, 256, dtype=torch.bfloat16, device=DEV)
deep_gemm.batched_syrk(ab, db)
ref_syrk = torch.bmm(ab.float(), ab.float().transpose(1, 2))
print(f"  batched_syrk         : relerr={rel(db, ref_syrk):.2e}")
bb = torch.randn(NB, 256, 256, dtype=torch.bfloat16, device=DEV)
db2 = torch.empty_like(bb)
deep_gemm.batched_symm(ab, bb, db2)
ref_symm = torch.bmm(ab.float(), bb.float())
print(f"  batched_symm         : relerr={rel(db2, ref_symm):.2e}")

# =================================================================== F. 性能对比
print("\n" + "=" * 84)
print("F. 性能对比：cuBLASLt（默认 bf16_gemm_nt）vs DeepGEMM SM90 kernel（deterministic）")
print("=" * 84)
shape_list = [
    (1, 7168, 7168), (16, 7168, 7168), (128, 7168, 7168),
    (1024, 7168, 7168), (4096, 7168, 7168), (8192, 7168, 7168),
    (4096, 14336, 7168), (4096, 7168, 2048),
]
print(f"{'M':>6}{'N':>7}{'K':>7} | {'cuBLASLt TFLOPS':>16}{'ms':>9} | "
      f"{'DeepGEMM TFLOPS':>16}{'ms':>9} | {'speedup':>8}")
for (m, n, k) in shape_list:
    aa = torch.randn(m, k, dtype=torch.bfloat16, device=DEV)
    bbb = torch.randn(n, k, dtype=torch.bfloat16, device=DEV)
    ddd = torch.empty(m, n, dtype=torch.bfloat16, device=DEV)
    reset_state()
    t_cu = bench(lambda: deep_gemm.bf16_gemm_nt(aa, bbb, ddd))
    deep_gemm.use_deterministic_algorithms(True)
    t_dg = bench(lambda: deep_gemm.bf16_gemm_nt(aa, bbb, ddd))
    reset_state()
    print(f"{m:>6}{n:>7}{k:>7} | {tflops(m,n,k,t_cu):>16.1f}{t_cu:>9.3f} | "
          f"{tflops(m,n,k,t_dg):>16.1f}{t_dg:>9.3f} | {t_dg/t_cu:>7.2f}x")

print("\ndone.")
