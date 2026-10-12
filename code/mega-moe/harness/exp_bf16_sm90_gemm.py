#!/usr/bin/env python3
"""P19-1: DeepGEMM 的 SM90 BF16 dense GEMM 主 kernel（sm90_bf16_gemm.cuh/.hpp）精读 + H100 实测。

参考源码（~/github/DeepGEMM，commit 057ca59）：
  csrc/jit_kernels/impls/sm90_bf16_gemm.hpp  —— 六个 host 入口（normal / m-grouped contiguous /
                                                 masked / k-grouped / 两个 batched MLA 投影）
  deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh —— 唯一的 BF16 SM90 kernel（WGMMA 64xNx16）
  deep_gemm/include/deep_gemm/mma/sm90.cuh   —— BF16MMASelector / advance_gmma_desc_lo
  deep_gemm/include/deep_gemm/common/tma_copy.cuh —— tma::copy 2D/3D + multicast
  csrc/jit_kernels/heuristics/sm90.hpp       —— BLOCK_K=64 等选核（见 P18-1）

本脚本做的事：
  1. 稠密 `bf16_gemm_nt`（out=bf16 / fp32，含 accumulate）vs torch fp32 参考；
     并抓真实发射的 kernel 名（确认走 DeepGEMM 自家 kernel 而非 cuBLASLt，见 P17-1）。
  2. 别名入口 nn/tn/tt 与 nt 数值一致。
  3. m-grouped contiguous / masked（`GemmType::MGroupedContiguous/Masked`）。
  4. k-grouped bf16 tn contiguous（`GemmType::KGroupedContiguous`，major MN/MN，k%128）。
  5. batched MLA 投影 `bhr,hdr->bhd` / `bhd,hdr->bhr`（`GemmType::Batched`，3D TMA store）。
  6. 性能：若干形状的 TFLOPS。

运行：bash harness/exp_bf16_sm90_gemm.sh > harness/exp_bf16_sm90_gemm.out.txt
"""
import sys
import time

import torch

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
      f"cc={torch.cuda.get_device_capability(0)} | device_num_sms={DEVICE_SMS}")
from deep_gemm.testing import get_arch_major, bench_kineto
print(f"deep_gemm arch_major = {get_arch_major()}")
print("=" * 100)


# ------------------------------------------------------------------ helpers
def relerr(x, ref):
    x = x.double()
    ref = ref.double()
    return ((x - ref).norm() / ref.norm().clamp_min(1e-30)).item()


def bench(fn, kernel=None, warmup=20, iters=100):
    # 每次调用前后 sync（真实单次延迟，含 launch）；取中位数去抖动。
    # 注：连续背靠背计时会因 PDL 让相邻 launch 重叠而系统性地低估延迟，故这里显式 sync。
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


def kernel_name_of(fn):
    import torch.profiler as profiler
    fn()
    torch.cuda.synchronize()
    with profiler.profile(activities=[profiler.ProfilerActivity.CUDA]) as prof:
        fn()
    torch.cuda.synchronize()
    names = {e.key for e in prof.key_averages()}
    return sorted(n for n in names if "gemm" in n.lower() or "nvjet" in n.lower())


# ================================================================== 1. dense
print("\n[1] Dense bf16_gemm_nt（NT，out=bf16/fp32，含 accumulate）")
print(f"{'m':>7} {'n':>7} {'k':>7} {'out':>8} {'acc':>4} | {'relerr':>10} {'TFLOPS':>8}")
dense_shapes = [(1024, 1024, 1024), (2048, 4096, 4096), (4096, 7168, 7168),
                (8192, 8192, 2048), (128, 4096, 4096), (256, 7168, 2048),
                (128, 128, 512)]
dense_tensors = {}
for (m, n, k) in dense_shapes:
    for out_dtype in (torch.bfloat16, torch.float):
        for accumulate in (False, True):
            deep_gemm.use_deterministic_algorithms(True)  # 强制走 DeepGEMM 自家 kernel
            a = torch.randn((m, k), device=DEV, dtype=torch.bfloat16)
            b = torch.randn((n, k), device=DEV, dtype=torch.bfloat16)
            d = (torch.randn((m, n), device=DEV, dtype=out_dtype) * 4 if accumulate
                 else torch.empty((m, n), device=DEV, dtype=out_dtype))
            c = d if accumulate else None
            ref = (a.float() @ b.float().t() + (c.float() if accumulate else 0.0)).to(out_dtype)
            deep_gemm.bf16_gemm_nt(a, b, d, c=c)
            err = relerr(d, ref)
            t = bench(lambda: deep_gemm.bf16_gemm_nt(a, b, d, c=c))
            tf = 2 * m * n * k / t / 1e12
            dense_tensors[(m, n, k, str(out_dtype), accumulate)] = (a, b, d, c)
            print(f"{m:7d} {n:7d} {k:7d} {str(out_dtype)[6:]:>8} {int(accumulate):>4} | "
                  f"{err:10.3e} {tf:8.0f}")

print("\n  真实发射的 kernel 模板实参（deterministic=on，取几个代表形状，验证选核/合并 stage）：")
for key in [(1024, 1024, 1024, str(torch.bfloat16), False),
            (4096, 7168, 7168, str(torch.bfloat16), True),
            (4096, 7168, 7168, str(torch.float), False),
            (256, 7168, 2048, str(torch.float), False),
            (128, 128, 512, str(torch.bfloat16), False)]:
    m, n, k, _, acc = key
    a, b, d, c = dense_tensors[key]
    kn = kernel_name_of(lambda: deep_gemm.bf16_gemm_nt(a, b, d, c=c))
    short = kn[0].split("sm90_bf16_gemm_impl")[-1] if kn else "?"
    print(f"   ({m},{n},{k},out={key[3][6:]},acc={int(acc)}): sm90_bf16_gemm_impl{short[:120]}")

print("\n  同一形状 deterministic=off 时的 kernel 名（对照 P17-1 的 cuBLASLt 回退）：")
m, n, k = 4096, 7168, 7168
a = torch.randn((m, k), device=DEV, dtype=torch.bfloat16)
b = torch.randn((n, k), device=DEV, dtype=torch.bfloat16)
d = torch.empty((m, n), device=DEV, dtype=torch.bfloat16)
deep_gemm.use_deterministic_algorithms(False)
print("   deterministic=off ->", [x.split("sm90_bf16_gemm_impl")[0] or x for x in
                                  kernel_name_of(lambda: deep_gemm.bf16_gemm_nt(a, b, d))])
deep_gemm.use_deterministic_algorithms(True)
print("   deterministic=on  ->", ["sm90_bf16_gemm_impl" for _ in
                                  kernel_name_of(lambda: deep_gemm.bf16_gemm_nt(a, b, d))])
deep_gemm.use_deterministic_algorithms(False)


# ================================================================== 2. aliases
# 四个入口只是把参数转置后转调 nt（gemm.hpp:544/554/564），本身是同一个 kernel。
print("\n[2] 别名入口 nn/tn/tt（都转调 nt，m=n=k=1024）")
m = n = k = 1024
deep_gemm.use_deterministic_algorithms(True)
a_mk = torch.randn((m, k), device=DEV, dtype=torch.bfloat16)
a_km = torch.randn((k, m), device=DEV, dtype=torch.bfloat16)
b_nk = torch.randn((n, k), device=DEV, dtype=torch.bfloat16)
b_kn = torch.randn((k, n), device=DEV, dtype=torch.bfloat16)
d = torch.empty((m, n), device=DEV, dtype=torch.bfloat16)
# nt: [m,k] @ [n,k].T
deep_gemm.bf16_gemm_nt(a_mk, b_nk, d)
print(f"   nt  [m,k]@[n,k]^T  relerr={relerr(d, a_mk.float() @ b_nk.float().t()):.3e}")
# nn: [m,k] @ [k,n]
deep_gemm.bf16_gemm_nn(a_mk, b_kn, d)
print(f"   nn  [m,k]@[k,n]    relerr={relerr(d, a_mk.float() @ b_kn.float()):.3e}")
# tn: [k,m]^T @ [k,n]
deep_gemm.bf16_gemm_tn(a_km, b_kn, d)
print(f"   tn  [k,m]^T@[k,n]  relerr={relerr(d, a_km.float().t() @ b_kn.float()):.3e}")
# tt: [k,m]^T @ [n,k]^T
deep_gemm.bf16_gemm_tt(a_km, b_nk, d)
print(f"   tt  [k,m]^T@[n,k]^T relerr={relerr(d, a_km.float().t() @ b_nk.float().t()):.3e}")
deep_gemm.use_deterministic_algorithms(False)


# ================================================================== 3. m-grouped
print("\n[3] m-grouped contiguous / masked（GemmType::MGrouped*）")

alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
print(f"   contiguous mk_alignment = {alignment}")


def align(x, a):
    return (x + a - 1) // a * a


# --- 3a contiguous
G = 8
exp_m = 512
n, k = 4096, 4096
actual_ms = [int(exp_m * (0.7 + 0.6 * (i / max(G - 1, 1)))) for i in range(G)]
aligned_ms = [align(x, alignment) for x in actual_ms]
m_tot = sum(aligned_ms)
grouped = torch.full((m_tot,), -1, device=DEV, dtype=torch.int32)
a = torch.zeros((m_tot, k), device=DEV, dtype=torch.bfloat16)
b = torch.randn((G, n, k), device=DEV, dtype=torch.bfloat16)
ref = torch.zeros((m_tot, n), device=DEV, dtype=torch.bfloat16)
s = 0
for i, (am, alm) in enumerate(zip(actual_ms, aligned_ms)):
    a[s:s + am] = torch.randn((am, k), device=DEV, dtype=torch.bfloat16)
    grouped[s:s + am] = i
    ref[s:s + alm] = (a[s:s + alm].float() @ b[i].float().t()).to(torch.bfloat16)
    s += alm
valid = grouped >= 0
d = torch.empty((m_tot, n), device=DEV, dtype=torch.bfloat16)
deep_gemm.use_deterministic_algorithms(True)
deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, grouped)
err = relerr(d[valid], ref[valid])
print(f"   contiguous: G={G} m_total={m_tot} n={n} k={k}  relerr={err:.3e}")

# --- 3b masked
max_m = 768
mass_ms = [int(exp_m * (0.7 + 0.6 * (i / max(G - 1, 1)))) for i in range(G)]
a_m = torch.randn((G, max_m, k), device=DEV, dtype=torch.bfloat16)
d_m = torch.empty((G, max_m, n), device=DEV, dtype=torch.bfloat16)
masked_m = torch.tensor(mass_ms, device=DEV, dtype=torch.int32)
ref_m = torch.einsum('gmk,gnk->gmn', a_m.float(), b.float()).to(torch.bfloat16)
deep_gemm.m_grouped_bf16_gemm_nt_masked(a_m, b, d_m, masked_m, max_m)
vm = torch.arange(max_m, device=DEV)[None, :] < masked_m[:, None]
err_m = relerr(d_m[vm], ref_m[vm])
print(f"   masked:     G={G} max_m={max_m} n={n} k={k}  relerr={err_m:.3e}")

# perf for contiguous
t = bench(lambda: deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, grouped))
ops = 2 * int(valid.sum().item()) * n * k
print(f"   contiguous perf: {t*1e6:.0f} us | {ops/t/1e12:.0f} TFLOPS")
deep_gemm.use_deterministic_algorithms(False)


# ================================================================== 4. k-grouped
print("\n[4] k-grouped bf16 tn contiguous（GemmType::KGroupedContiguous，major MN/MN，k%128）")
Gk = 4
m, n = 4096, 7168
logical_ks = [1024, 2048, 512, 1536]
host_ks = [align(x, 128) for x in logical_ks]
total_k = sum(host_ks)
a = torch.zeros((total_k, m), device=DEV, dtype=torch.bfloat16)
b = torch.zeros((total_k, n), device=DEV, dtype=torch.bfloat16)
d = torch.zeros((Gk, m, n), device=DEV, dtype=torch.float)
ref = torch.empty_like(d)
starts = []
p = 0
for i, lk in enumerate(logical_ks):
    starts.append(p)
    a[p:p + lk] = torch.randn((lk, m), device=DEV, dtype=torch.bfloat16)
    b[p:p + lk] = torch.randn((lk, n), device=DEV, dtype=torch.bfloat16)
    ref[i] = a[p:p + lk].float().T @ b[p:p + lk].float()
    p += host_ks[i]
ks_cpu = host_ks
grouped_layout = torch.tensor(ks_cpu, device=DEV, dtype=torch.int32)
# SM90 强制 accumulate（gemm.hpp:707-710: c.has_value()），无 direct output
d0 = torch.randn_like(d)
d_acc = d0.clone()
deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a, b, d_acc, ks_cpu, grouped_layout, d_acc)
print(f"   G={Gk} m={m} n={n} logical_k={logical_ks} host_k={host_ks}  "
      f"accumulate (D=C+AᵀB) relerr={relerr(d_acc, ref + d0):.3e}")

# contract: k not multiple of 128 must be rejected
try:
    bad = [1024, 1000, 512, 1536]
    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a, b, d_acc, bad, grouped_layout, d_acc)
    print("   k%128 校验: 未拒绝（意外）")
except RuntimeError as e:
    print(f"   k%128 校验: 被拒 -> {str(e).splitlines()[0][:80]}")

t = bench(lambda: deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a, b, d_acc, ks_cpu, grouped_layout, d_acc))
print(f"   perf: {t*1e6:.0f} us | {2*m*n*sum(logical_ks)/t/1e12:.0f} TFLOPS")


# ================================================================== 5. batched MLA
print("\n[5] batched MLA 投影（GemmType::Batched，3D TMA store）")
for (bb, h, r, dd) in [(128, 128, 512, 512), (4096, 8, 4096, 1024)]:
    x = torch.randn((bb, h, r), device=DEV, dtype=torch.bfloat16)
    fy = torch.randn((h, dd, r + 128), device=DEV, dtype=torch.bfloat16)
    y = fy[:, :, :r]
    ref_z = torch.einsum('bhr,hdr->bhd', x.float(), y.float()).to(torch.bfloat16)
    z = torch.empty((bb, h, dd), device=DEV, dtype=torch.bfloat16)
    deep_gemm.use_deterministic_algorithms(True)
    deep_gemm.einsum('bhr,hdr->bhd', x, y, z)
    deep_gemm.use_deterministic_algorithms(False)
    t = bench(lambda: deep_gemm.einsum('bhr,hdr->bhd', x, y, z), kernel="gemm")
    print(f"   bhr,hdr->bhd b={bb} h={h} r={r} d={dd}: relerr={relerr(z, ref_z):.3e} "
          f"| {2*bb*h*r*dd/t/1e12:.0f} TFLOPS")

# bhd,hdr->bhr
for (bb, h, r, dd) in [(128, 128, 512, 512), (4096, 8, 4096, 1024)]:
    x = torch.randn((bb, h, dd), device=DEV, dtype=torch.bfloat16)
    y = torch.randn((h, dd, r), device=DEV, dtype=torch.bfloat16)
    ref_z = torch.einsum('bhd,hdr->bhr', x.float(), y.float()).to(torch.bfloat16)
    z = torch.empty((bb, h, r), device=DEV, dtype=torch.bfloat16)
    deep_gemm.use_deterministic_algorithms(True)
    deep_gemm.einsum('bhd,hdr->bhr', x, y, z)
    deep_gemm.use_deterministic_algorithms(False)
    print(f"   bhd,hdr->bhr b={bb} h={h} r={r} d={dd}: relerr={relerr(z, ref_z):.3e}")


print("\n" + "=" * 100)
print("DONE")
