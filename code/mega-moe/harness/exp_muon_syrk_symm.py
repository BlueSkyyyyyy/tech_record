#!/usr/bin/env python3
"""DeepGEMM 的 Muon / 对称矩阵 batched 算子实测（P35-1，H100 真机）。

参考源码（~/github/DeepGEMM，commit 057ca59）：
  csrc/apis/gemm.hpp:791-794          get_cublaslt_batched_view（2D→3D unsqueeze）
  csrc/apis/gemm.hpp:796-810          batched_syrk  （D = A @ A.mT）
  csrc/apis/gemm.hpp:812-832          batched_symm  （D = A @ B，A 对称，内部 B 转置）
  csrc/apis/gemm.hpp:991/993          pybind 注册
  csrc/apis/einsum.hpp:128-148        bhd,bhr->hdr  → cublaslt_bhd_bhr_hdr（恒 cuBLASLt）
  csrc/jit_kernels/impls/smxx_cublaslt.hpp:14-25   get_cublaslt_layout（strided-batch 属性）
  csrc/jit_kernels/impls/smxx_cublaslt.hpp:137-164 cublaslt_batched_gemm（列主序交换）
  csrc/jit_kernels/impls/smxx_cublaslt.hpp:223-234 cublaslt_bhd_bhr_hdr
  csrc/utils/layout.hpp:19-40         major_check / get_major_type_ab / check_major_type_cd
  tests/generators.py:176-191         enumerate_batched_syrk_symm（真实 Muon 形状）
  tests/test_bf16.py:247-320          test_cublaslt_batched_syrk / _symm

结论先行（由本脚本实测）：
  * `batched_syrk` / `batched_symm` 是 **cuBLASLt-only** 的算子（无 DeepGEMM 自研 kernel），
    面向 Muon 优化器的 Newton-Schulz 迭代：`A@Aᵀ`（syrk）与 `A@B`（A 对称，symm）。
  * 输入天然是 **strided-batch**：`get_major_type_ab<false>` 放宽了 batch 连续性要求，
    cuBLASLt 用 `stride(0)` 作 batch offset，所以「带 padding 的非连续 batch」也能跑。
  * 2D 与 3D 共用同一实现（2D 先 unsqueeze 成 batch=1）。

运行：bash harness/exp_muon_syrk_symm.sh > harness/exp_muon_syrk_symm.out.txt
"""
import sys

import torch
import torch.profiler as profiler

try:
    import deep_gemm
except Exception as e:  # pragma: no cover
    print("FATAL: cannot import deep_gemm:", e)
    sys.exit(1)

from deep_gemm.testing import bench_kineto, calc_diff, count_bytes

DEV = "cuda"
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False

print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
      f"cc={torch.cuda.get_device_capability(0)} | device_num_sms={deep_gemm.get_num_sms()}")
print()


def sec(title):
    print("=" * 78)
    print(title)
    print("=" * 78)


def rel(x, ref):
    x = x.float().flatten().double()
    ref = ref.float().flatten().double()
    d = (x - ref).norm() / (ref.norm() + 1e-30)
    return d.item()


# 真实 Muon 参数形状（tests/generators.py:180-189，5120/7168 模型；batch 合并上限 16）
B_M_K_LIST = [
    (16, 4608, 5120),
    (16, 1536, 5120),
    (16, 576, 5120),
    (8, 24576, 1536),
    (8, 32768, 512),
    (4, 5120, 16384),
    (8, 4096, 7168),
    (8, 7168, 2048),
]


def make_operand(batch_shape, rows, cols, dtype, k_major, pad):
    """按 tests/test_bf16.py:255-258 的方式造 operand：先带 padding，再切片/转置。"""
    if k_major:
        shape = (rows, cols)
    else:
        shape = (cols, rows)
    a = torch.randn(*batch_shape, shape[0] + pad[0], shape[1] + pad[1],
                    device=DEV, dtype=dtype)[..., :shape[0], :shape[1]]
    if not k_major:
        a = a.mT
    return a


def make_d(batch_shape, rows, cols, dtype, pad):
    return torch.empty(*batch_shape, rows + pad[0], cols + pad[1],
                       device=DEV, dtype=dtype)[..., :rows, :cols]


# ---------------------------------------------------------------- A) syrk 真实形状
sec("A) batched_syrk：真实 Muon 形状（D = A @ A.mT，cuBLASLt-only）")
print(f"{'batch':>6} {'rows':>7} {'cols':>7} {'dtype':>7} | {'calc_diff':>11} {'relerr':>11} | "
      f"{'us':>9} {'TFLOPS':>8} {'GB/s':>7}")
for dtype in (torch.bfloat16, torch.float):
    for b, m, k in B_M_K_LIST:
        rows, cols = min(m, k), max(m, k)
        a = make_operand((b,), rows, cols, dtype, k_major=True, pad=(0, 0))
        d = make_d((b,), rows, rows, dtype, pad=(0, 0))
        deep_gemm.batched_syrk(a, d)
        ref = (a.float() @ a.float().mT).to(dtype)
        cd = calc_diff(d, ref).item()
        re = rel(d, ref)
        t = bench_kineto(lambda: deep_gemm.batched_syrk(a, d),
                         ('nvjet', 'gemv', 'gemm'), suppress_kineto_output=True)
        t = sum(t) if isinstance(t, tuple) else t
        flops = 2.0 * b * rows * rows * cols
        gb = count_bytes(a, d) / 1e9
        print(f"{b:>6} {rows:>7} {cols:>7} {str(dtype).split('.')[-1]:>7} | "
              f"{cd:>11.3e} {re:>11.3e} | {t*1e6:>9.1f} {flops/t/1e12:>8.0f} {gb/t:>7.0f}")
print()

# ---------------------------------------------------------------- B) symm 真实形状
sec("B) batched_symm：真实 Muon 形状（D = A @ B，A 对称，cuBLASLt-only）")
print(f"{'batch':>6} {'rows':>7} {'cols':>7} {'dtype':>7} | {'calc_diff':>11} {'relerr':>11} | "
      f"{'us':>9} {'TFLOPS':>8} {'GB/s':>7}")
for dtype in (torch.bfloat16, torch.float):
    for b, m, k in B_M_K_LIST:
        rows, cols = min(m, k), max(m, k)
        a = make_operand((b,), rows, rows, dtype, k_major=True, pad=(0, 0))
        a = a + a.mT  # 对称：A = A^T
        bb = make_operand((b,), rows, cols, dtype, k_major=True, pad=(0, 0))
        d = make_d((b,), rows, cols, dtype, pad=(0, 0))
        deep_gemm.batched_symm(a, bb, d)
        ref = (a.float() @ bb.float()).to(dtype)
        cd = calc_diff(d, ref).item()
        re = rel(d, ref)
        t = bench_kineto(lambda: deep_gemm.batched_symm(a, bb, d),
                         ('nvjet', 'gemv', 'gemm'), suppress_kineto_output=True)
        t = sum(t) if isinstance(t, tuple) else t
        flops = 2.0 * b * rows * rows * cols
        gb = count_bytes(a, bb, d) / 1e9
        print(f"{b:>6} {rows:>7} {cols:>7} {str(dtype).split('.')[-1]:>7} | "
              f"{cd:>11.3e} {re:>11.3e} | {t*1e6:>9.1f} {flops/t/1e12:>8.0f} {gb/t:>7.0f}")
print()

# ---------------------------------------------------------------- C) major / 2D / padding
sec("C) 布局与形态：major、2D vs 3D、padding、非连续 batch（get_major_type_ab<false>）")
b, rows, cols = 8, 4096, 7168
dtype = torch.bfloat16
for k_major in (True, False):
    a = make_operand((b,), rows, rows, dtype, k_major=k_major, pad=(0, 0))
    d = make_d((b,), rows, rows, dtype, pad=(0, 0))
    deep_gemm.batched_syrk(a, d)
    ref = (a.float() @ a.float().mT).to(dtype)
    major_opt = "N" if a.stride(-1) == 1 else "T"
    print(f"  syrk major={major_opt:<1} stride=({a.stride(-2)},{a.stride(-1)})  "
          f"calc_diff={calc_diff(d, ref).item():.3e}")

# 2D（无 batch 维）与 3D batch=1 等价
a2 = make_operand((), rows, cols, dtype, k_major=True, pad=(0, 0))
d2 = make_d((), rows, rows, dtype, pad=(0, 0))
deep_gemm.batched_syrk(a2, d2)
a3 = a2.unsqueeze(0).contiguous()
d3 = torch.empty(1, rows, rows, device=DEV, dtype=dtype)
deep_gemm.batched_syrk(a3, d3)
print(f"  2D vs 3D(batch=1)：d2 对称={torch.allclose(d2, d2.mT, atol=1e-2)}，"
      f"d2/d3 calc_diff={calc_diff(d2, d3[0]).item():.3e}（unsqueeze 同一实现）")

# 非连续 batch：batch 维给 padding（stride(0) != rows*cols）
for pad in ((0, 0), (3, 5)):
    a = make_operand((b,), rows, rows, dtype, k_major=True, pad=pad)
    d = make_d((b,), rows, rows, dtype, pad=pad)
    contig = a.stride(0) == a.size(-2) * a.size(-1)
    deep_gemm.batched_syrk(a, d)
    ref = (a.float() @ a.float().mT).to(dtype)
    tag = "连续 batch" if contig else "非连续 batch"
    print(f"  pad={str(pad):<8} a.stride(0)={a.stride(0)} {tag:<10}  "
          f"calc_diff={calc_diff(d, ref).item():.3e}  ← 两者都被接受")
print()

# ---------------------------------------------------------------- D) 契约检查
sec("D) 契约检查：D 必须行主序、形状/dtype 一致（gemm.hpp:803-806）")
b, rows, cols = 4, 1024, 2048
a = torch.randn(b, rows, cols, device=DEV, dtype=torch.bfloat16)
d = torch.empty(b, rows, rows, device=DEV, dtype=torch.bfloat16)

cases = []
# D 非行主序（.mT）
try:
    deep_gemm.batched_syrk(a, d.mT)
    cases.append(("D 非行主序(.mT)", "未拒绝"))
except Exception as e:
    cases.append(("D 非行主序(.mT)", f"拒绝: {type(e).__name__}"))
# 形状不符
try:
    deep_gemm.batched_syrk(a, torch.empty(b, rows, rows + 8, device=DEV, dtype=torch.bfloat16))
    cases.append(("D 形状不符", "未拒绝"))
except Exception as e:
    cases.append(("D 形状不符", f"拒绝: {type(e).__name__}"))
# dtype 不符
try:
    deep_gemm.batched_syrk(a, torch.empty(b, rows, rows, device=DEV, dtype=torch.float))
    cases.append(("a/d dtype 不符", "未拒绝"))
except Exception as e:
    cases.append(("a/d dtype 不符", f"拒绝: {type(e).__name__}"))
# 3D a 与 3D d 的 batch 数不符
try:
    deep_gemm.batched_syrk(a, torch.empty(b + 1, rows, rows, device=DEV, dtype=torch.bfloat16))
    cases.append(("batch 数不符", "未拒绝"))
except Exception as e:
    cases.append(("batch 数不符", f"拒绝: {type(e).__name__}"))
for name, res in cases:
    print(f"  {name:<22} → {res}")
print()

# ---------------------------------------------------------------- E) 真实 kernel 名
sec("E) 抓真实发射的 kernel 名（torch.profiler）")


def kernel_names(fn):
    fn()
    torch.cuda.synchronize()
    with profiler.profile(activities=[profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = set()
    for ev in prof.events():
        if ev.device_type == profiler.DeviceType.CUDA and ev.name:
            if any(t in ev.name for t in ("nvjet", "gemv", "gemm", "cublas", "xmma", "cutlass")):
                names.add(ev.name)
    return sorted(names)


b, rows, cols = 8, 4096, 7168
a = torch.randn(b, rows, cols, device=DEV, dtype=torch.bfloat16)
d = torch.empty(b, rows, rows, device=DEV, dtype=torch.bfloat16)
print("  batched_syrk:")
for n in kernel_names(lambda: deep_gemm.batched_syrk(a, d)):
    print(f"    {n}")
print("  batched_symm:")
a_sq = torch.randn(b, rows, rows, device=DEV, dtype=torch.bfloat16)
a_sym = (a_sq + a_sq.mT).contiguous()
b2 = torch.randn(b, rows, cols, device=DEV, dtype=torch.bfloat16)
d2 = torch.empty(b, rows, cols, device=DEV, dtype=torch.bfloat16)
for n in kernel_names(lambda: deep_gemm.batched_symm(a_sym, b2, d2)):
    print(f"    {n}")
print()

# ---------------------------------------------------------------- F) bhd,bhr->hdr
sec("F) einsum `bhd,bhr->hdr`（跨 batch 权重梯度，恒 cuBLASLt，FP32 输出）")
b, h, r, d_ = 8, 16, 4096, 2048
A = torch.randn(b, h, d_, device=DEV, dtype=torch.bfloat16)
B = torch.randn(b, h, r, device=DEV, dtype=torch.bfloat16)
D = torch.empty(h, d_, r, device=DEV, dtype=torch.float32)
deep_gemm.einsum("bhd,bhr->hdr", A, B, D)
ref = torch.einsum("bhd,bhr->hdr", A.float(), B.float())
print(f"  calc_diff={calc_diff(D, ref).item():.3e}  relerr={rel(D, ref):.3e}  "
      f"dtype={D.dtype}")
D2 = torch.empty_like(D)
deep_gemm.einsum("bhd,bhr->hdr", A, B, D2)
bitwise = torch.equal(D.view(torch.int32), D2.view(torch.int32))
print(f"  两次运行逐位相等={bitwise}（源码 einsum.hpp:146 注释「TODO: investigate cuBLAS determinism」）")
t = bench_kineto(lambda: deep_gemm.einsum("bhd,bhr->hdr", A, B, D),
                 ('nvjet', 'gemv', 'gemm'), suppress_kineto_output=True)
t = sum(t) if isinstance(t, tuple) else t
flops = 2.0 * b * h * r * d_
print(f"  性能：{t*1e6:.1f} us  {flops/t/1e12:.0f} TFLOPS  "
      f"{count_bytes(A, B, D)/1e9/t:.0f} GB/s")
print()

print("全域完成。")
