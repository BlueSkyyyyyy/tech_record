#!/usr/bin/env python3
# P8-2 实验：DeepGEMM einsum 的两个 MLA 投影 `bhr,hdr->bhd` 与 `bhd,hdr->bhr`
# 在 H100(sm90) 上的语义/映射/契约/性能实测。
#
# 数学（对每个头 h 一个独立 GEMM，按 h 分 batch）：
#   bhr,hdr->bhd : D[b,h,d] = sum_r A[b,h,r] * B[h,d,r] = A[:,h,:] @ B[h].T
#   bhd,hdr->bhr : D[b,h,r] = sum_d A[b,h,d] * B[h,d,r] = A[:,h,:] @ B[h]
#
# 实现：SM90 `sm90_bf16_gemm.cuh` 的 `GemmType::Batched` 分支
#       （3D TMA + batch_idx，scheduler/gemm.cuh 的 Batched 路径）。
#       host 侧 sm90_bf16_gemm.hpp:332 / :386；einsum 分派 csrc/apis/einsum.hpp:60 / :94。
#
# 关键论点：
#   1) 它是「每个头一个 GEMM、头当 batch」的 batched GEMM，不是 grouped GEMM；
#   2) 两条表达式只是 (m,n,k) 的角色互换 + B 的 major（K / MN）不同；
#   3) `num_groups=h` 的 group 只是 3D TMA 的第三维，调度器把它当第 0 维枚举；
#   4) 带 C 时走 TMA reduce-add 3D（D=C+A@B）；
#   5) B 的最后一维可以带 padding（stride > r），3D TMA 用 outer stride 表达。
#
# 运行：bash harness/exp_einsum_mla.sh > harness/exp_einsum_mla.out.txt
import torch
import deep_gemm

torch.manual_seed(0)
DEV = "cuda"
SMS = torch.cuda.get_device_properties(0).multi_processor_count


def reldiff(x, ref):
    denom = ref.abs().max().item()
    return (x.double() - ref.double()).abs().max().item() / (denom if denom > 0 else 1.0)


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us


print("=" * 78)
print(f"H100 sm{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}  "
      f"num_sms={SMS}  torch={torch.__version__}  deep_gemm={deep_gemm.__path__}")
print("=" * 78)

# ---------------------------------------------------------------------------
# A. 语义：两条表达式 vs torch.einsum（bf16，逐位）与 fp64 参考（真误差）
# ---------------------------------------------------------------------------
print("\n[A] 语义 vs torch.einsum(bf16, 逐位) 与 fp64 参考(relerr)")
print(f"{'expr':>14} {'b':>5} {'h':>4} {'r':>5} {'d':>5} | {'bitwise':>8} {'relerr(fp64)':>13}")
for (h, r, d) in ((128, 512, 128), (8, 4096, 1024)):
    for b in (4, 32, 128, 4096):
        x = torch.randn((b, h, r), dtype=torch.bfloat16, device=DEV)
        y = torch.randn((h, d, r + 128), dtype=torch.bfloat16, device=DEV)[:, :, :r]  # 带 padding 的 B
        z = torch.empty((b, h, d), dtype=torch.bfloat16, device=DEV)
        deep_gemm.einsum("bhr,hdr->bhd", x, y, z)
        ref = torch.einsum("bhr,hdr->bhd", x, y)
        ref64 = torch.einsum("bhr,hdr->bhd", x.double(), y.double())
        print(f"{'bhr,hdr->bhd':>14} {b:>5} {h:>4} {r:>5} {d:>5} | "
              f"{str(bool(torch.equal(z, ref))):>8} {reldiff(z, ref64):>13.3e}")

        x2 = torch.randn((b, h, d), dtype=torch.bfloat16, device=DEV)
        z2 = torch.empty((b, h, r), dtype=torch.bfloat16, device=DEV)
        deep_gemm.einsum("bhd,hdr->bhr", x2, y, z2)
        ref2 = torch.einsum("bhd,hdr->bhr", x2, y)
        ref264 = torch.einsum("bhd,hdr->bhr", x2.double(), y.double())
        print(f"{'bhd,hdr->bhr':>14} {b:>5} {h:>4} {r:>5} {d:>5} | "
              f"{str(bool(torch.equal(z2, ref2))):>8} {reldiff(z2, ref264):>13.3e}")

# ---------------------------------------------------------------------------
# B. 「每个头一个 GEMM」：等价于逐头 torch bmm（头=独立 batch）
# ---------------------------------------------------------------------------
print("\n[B] 等价于「逐头 GEMM」（h 个独立的小 GEMM）：")
b, h, r, d = 32, 128, 512, 128
x = torch.randn((b, h, r), dtype=torch.bfloat16, device=DEV)
y = torch.randn((h, d, r), dtype=torch.bfloat16, device=DEV)
z = torch.empty((b, h, d), dtype=torch.bfloat16, device=DEV)
deep_gemm.einsum("bhr,hdr->bhd", x, y, z)
# ref[b,h,d] = x[b,h,:] @ y[h].T  -> 用 bmm: x.permute(1,0,2) [h,b,r] @ y.mT [h,r,d]
ref_bmm = torch.bmm(x.permute(1, 0, 2).float(), y.permute(0, 2, 1).float()).permute(1, 0, 2)
print(f"  bhr,hdr->bhd: reldiff(vs 逐头 bmm fp32->bf16) = "
      f"{reldiff(z, ref_bmm.bfloat16()):.3e}  (逐头 bmm 与 einsum 语义一致)")

# 独立性验证：把第 0 头的 B 置零 -> 输出第 0 头应恰好为零，其余头不变
z_ref = z.clone()
y0 = y.clone(); y0[0] = 0
z2 = torch.empty_like(z)
deep_gemm.einsum("bhr,hdr->bhd", x, y0, z2)
head0_zero = bool((z2[:, 0, :] == 0).all().item())
others_same = bool(torch.equal(z2[:, 1:, :], z_ref[:, 1:, :]))
print(f"  头 0 的 B 置零 -> 输出头 0 全 0 = {head0_zero}；其余头逐位不变 = {others_same}")

# ---------------------------------------------------------------------------
# C. 累加（C 提供）：D = C + A @ B，走 TMA reduce-add 3D
# ---------------------------------------------------------------------------
print("\n[C] 累加语义（with_accumulation -> TMA REDUCE_ADD_3D）：D = C + A@B")
b, h, r, d = 32, 8, 4096, 1024
x = torch.randn((b, h, r), dtype=torch.bfloat16, device=DEV)
y = torch.randn((h, d, r), dtype=torch.bfloat16, device=DEV)
c = torch.randn((b, h, d), dtype=torch.bfloat16, device=DEV) * 4
z = c.clone()
deep_gemm.einsum("bhr,hdr->bhd", x, y, z, c=z)
ref = torch.einsum("bhr,hdr->bhd", x.double(), y.double()) + c.double()
print(f"  bhr,hdr->bhd  D=C+A@B: reldiff(vs fp64) = {reldiff(z, ref):.3e}")

x2 = torch.randn((b, h, d), dtype=torch.bfloat16, device=DEV)
z2 = torch.empty((b, h, r), dtype=torch.bfloat16, device=DEV)
c2 = torch.randn((b, h, r), dtype=torch.bfloat16, device=DEV) * 4
z2.copy_(c2)
deep_gemm.einsum("bhd,hdr->bhr", x2, y, z2, c=z2)
ref2 = torch.einsum("bhd,hdr->bhr", x2.double(), y.double()) + c2.double()
print(f"  bhd,hdr->bhr  D=C+A@B: reldiff(vs fp64) = {reldiff(z2, ref2):.3e}")

# ---------------------------------------------------------------------------
# D. B 带 padding（stride(2)=1 但 stride(1) > r）：3D TMA 用 outer stride 表达
# ---------------------------------------------------------------------------
print("\n[D] B 的 K/N 维带 padding：fy[:, :, :r] 是 stride 视图（3D TMA outer stride）")
b, h, r, d = 128, 8, 4096, 1024
x = torch.randn((b, h, r), dtype=torch.bfloat16, device=DEV)
fy = torch.randn((h, d, r + 128), dtype=torch.bfloat16, device=DEV)
y = fy[:, :, :r]
assert y.stride(2) == 1 and y.stride(1) == r + 128
z = torch.empty((b, h, d), dtype=torch.bfloat16, device=DEV)
deep_gemm.einsum("bhr,hdr->bhd", x, y, z)
ref = torch.einsum("bhr,hdr->bhd", x, y)
yc = y.contiguous()
zc = torch.empty_like(z)
deep_gemm.einsum("bhr,hdr->bhd", x, yc, zc)
print(f"  padded B vs 连续 B 逐位相等 = {bool(torch.equal(z, zc))}；"
      f"vs torch.einsum 逐位相等 = {bool(torch.equal(z, ref))}")

# ---------------------------------------------------------------------------
# E. 确定性开关与 cuBLASLt 回退（nvjet）：用 profiler 看实际 kernel 名
# ---------------------------------------------------------------------------
print("\n[E] use_deterministic_algorithms：True->自家 kernel，False->cuBLASLt(nvjet)")
b, h, r, d = 128, 128, 512, 128
x = torch.randn((b, h, r), dtype=torch.bfloat16, device=DEV)
y = torch.randn((h, d, r), dtype=torch.bfloat16, device=DEV)
ref64 = torch.einsum("bhr,hdr->bhd", x.double(), y.double())

def top_kernels(fn, n=3):
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
    evts = sorted(prof.key_averages(), key=lambda e: -e.device_time_total)
    return [e.key for e in evts[:n] if e.device_time_total > 0]

for det in (True, False):
    deep_gemm.use_deterministic_algorithms(det)
    z = torch.empty((b, h, d), dtype=torch.bfloat16, device=DEV)
    deep_gemm.einsum("bhr,hdr->bhd", x, y, z)
    t = bench(lambda: deep_gemm.einsum("bhr,hdr->bhd", x, y, z))
    ks = top_kernels(lambda: deep_gemm.einsum("bhr,hdr->bhd", x, y, z))
    print(f"  deterministic={det}: reldiff(fp64)={reldiff(z, ref64):.3e}  {t:8.1f} us  kernels={ks}")
deep_gemm.use_deterministic_algorithms(True)

# ---------------------------------------------------------------------------
# F. host 契约（DG_HOST_ASSERT）
# ---------------------------------------------------------------------------
print("\n[F] host 契约：A/B/D 必须 stride(2)==1、dtype 为 bf16、形状一致")
def try_expr(label, expr, a, bb, dd, cc=None):
    try:
        deep_gemm.einsum(expr, a, bb, dd, c=cc)
        print(f"  {label}: 未报错（意外）")
    except Exception as ex:
        print(f"  {label}: 被拒 -> {str(ex).splitlines()[0][:90]}")

b, h, r, d = 4, 8, 128, 64
a_ok = torch.randn((b, h, r), dtype=torch.bfloat16, device=DEV)
b_ok = torch.randn((h, d, r), dtype=torch.bfloat16, device=DEV)
d_ok = torch.empty((b, h, d), dtype=torch.bfloat16, device=DEV)
# A 非 K-major（stride(2)!=1）
a_bad = torch.randn((b, r, h), dtype=torch.bfloat16, device=DEV).permute(0, 2, 1)
try_expr("A.stride(2)!=1", "bhr,hdr->bhd", a_bad, b_ok, d_ok)
# fp32 输入
try_expr("A 为 fp32", "bhr,hdr->bhd", a_ok.float(), b_ok, d_ok)
# 形状不一致（h 不匹配）
b_badh = torch.randn((h + 1, d, r), dtype=torch.bfloat16, device=DEV)
try_expr("B 的 h 不匹配", "bhr,hdr->bhd", a_ok, b_badh, d_ok)
# 不支持的表达式
try_expr("未知表达式", "abc->a", a_ok, b_ok, d_ok)

# ---------------------------------------------------------------------------
# G. 性能：einsum vs torch.einsum，扫描 b（对齐 DeepGEMM test 的用例）
# ---------------------------------------------------------------------------
print("\n[G] 性能（FLOPs = 2*b*h*r*d）")
for expr in ("bhr,hdr->bhd", "bhd,hdr->bhr"):
    print(f"  --- {expr} ---")
    print(f"  {'b':>6} {'h':>4} {'r':>5} {'d':>5} | {'deep_gemm us':>12} {'TFLOPS':>8} "
          f"{'GB/s':>8} | {'torch us':>10} {'x':>6}")
    for (h, r, d) in ((128, 512, 128), (8, 4096, 1024)):
        for b in (4, 32, 128, 4096, 8192):
            if expr == "bhr,hdr->bhd":
                x = torch.randn((b, h, r), dtype=torch.bfloat16, device=DEV)
                y = torch.randn((h, d, r), dtype=torch.bfloat16, device=DEV)
                z = torch.empty((b, h, d), dtype=torch.bfloat16, device=DEV)
            else:
                x = torch.randn((b, h, d), dtype=torch.bfloat16, device=DEV)
                y = torch.randn((h, d, r), dtype=torch.bfloat16, device=DEV)
                z = torch.empty((b, h, r), dtype=torch.bfloat16, device=DEV)
            run_dg = lambda: deep_gemm.einsum(expr, x, y, z)
            run_t = lambda: torch.einsum(expr, x, y)
            t_dg = bench(run_dg)
            t_t = bench(run_t)
            flops = 2 * b * h * r * d
            nbytes = x.numel() * 2 + y.numel() * 2 + z.numel() * 2
            print(f"  {b:>6} {h:>4} {r:>5} {d:>5} | {t_dg:>12.1f} {flops/t_dg/1e6:>8.0f} "
                  f"{nbytes/t_dg/1e3:>8.0f} | {t_t:>10.1f} {t_t/t_dg:>6.2f}")

print("\n完成。")
