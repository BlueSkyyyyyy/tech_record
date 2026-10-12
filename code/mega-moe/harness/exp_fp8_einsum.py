#!/usr/bin/env python3
# P11-1 实验：DeepGEMM FP8 einsum 投影 `bhr,hdr->bhd`（MLA 吸收投影的 FP8 版）
# 在 H100(sm90) 上的语义 / SF 布局 / 分派契约 / 性能实测。
#
# 数学（对每个头 h 一个独立的 FP8 GEMM，头当 batch）：
#   bhr,hdr->bhd : D[b,h,d] = sum_r A[b,h,r] * B[h,d,r] = A[:,h,:] @ B[h].T
#   代码里把 (b,h) permute 成 (batch_size,m,n,k) = (h, b, d, r)，见 csrc/apis/einsum.hpp:262-276。
#
# 实现：SM90 `sm90_fp8_gemm_1d2d.cuh` 的 `GemmType::Batched` 分支
#       （3D TMA + batch_idx，1D 的 A per-token scale + 2D 的 B block scale）。
#       host 侧 sm90_fp8_bmm：csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:283-352；
#       fp8_bmm 分派：csrc/apis/einsum.hpp:181-249；fp8_einsum：:251-298。
#
# 关键论点：
#   1) FP8 投影 = 「每头一个 GEMM、头当 batch」的 batched GEMM，复用前向 `1d2d` 骨架；
#   2) 量化布局是硬约束：A 走 per-token(gran 128, 1D) SF，B 走 per-128x128 block(2D) SF，
#      SM90 分支强制 gran_k_a == gran_k_b == 128（einsum.hpp:246）；
#   3) SM90 只出 BF16 D，FP8 输出对 (d,sfd) 与自定义 epilogue 是 SM100 专属（einsum.hpp:224/244）；
#   4) SFB 逐 batch 的基址由 `get_global_idx<true, SF_K>` 给出（scheduler/gemm.cuh:169-172）；
#   5) 'bhd,hdr->bhr' / 'bhd,bhr->hdr' 的 FP8 版仅 arch10（einsum.hpp:277/287），SM90 直接不可达。
#
# 运行：bash harness/exp_fp8_einsum.sh > harness/exp_fp8_einsum.out.txt
import torch
import deep_gemm
from deep_gemm.utils.math import per_token_cast_to_fp8, per_block_cast_to_fp8, ceil_div

torch.manual_seed(0)
DEV = "cuda"
SMS = torch.cuda.get_device_properties(0).multi_processor_count
ARCH = torch.cuda.get_device_capability()[0]


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


def quantize(x, y, b, h, r, d):
    """按 DeepGEMM 测试的布局量化：A per-token(gran 128)，B per-128x128 block。"""
    xf = per_token_cast_to_fp8(x.reshape(-1, r), use_ue8m0=True)
    xf = (xf[0].view(b, h, r), xf[1].view(b, h, ceil_div(r, 128)))
    yf = (torch.empty_like(y, dtype=torch.float8_e4m3fn),
          torch.empty((h, ceil_div(d, 128), ceil_div(r, 128)), device=DEV, dtype=torch.float))
    for i in range(h):
        yf[0][i], yf[1][i] = per_block_cast_to_fp8(y[i], use_ue8m0=True)
    return xf, yf


def dequant(xf, yf, r, d):
    """把量化后的 A/B 还原成 fp32，用于「隔离量化」的参考。"""
    xq, xs = xf
    yq, ys = yf
    dx = xq.float() * xs.float().repeat_interleave(128, dim=-1)[..., :r]
    dy = yq.float() * ys.float().repeat_interleave(128, dim=-2).repeat_interleave(128, dim=-1)[..., :d, :r]
    return dx, dy


print("=" * 84)
print(f"H100 sm{ARCH}{torch.cuda.get_device_capability()[1]}  num_sms={SMS}  "
      f"torch={torch.__version__}  deep_gemm={deep_gemm.__path__}")
print("=" * 84)

# ---------------------------------------------------------------------------
# A. 语义：fp8_einsum(BF16 出) vs torch.einsum(真值) 与「隔离量化」模拟参考
# ---------------------------------------------------------------------------
print("\n[A] 语义：BF16 输出 vs 真值 / vs 隔离量化的 fp32 模拟（relerr）")
print(f"{'b':>5} {'h':>4} {'r':>5} {'d':>5} | {'vs 真值(bf16)':>15} {'vs 量化模拟':>12} {'量化误差':>10}")
for (h, r, d) in ((128, 512, 128), (8, 4096, 1024)):
    for b in (4, 32, 128, 4096):
        x = torch.randn((b, h, r), dtype=torch.bfloat16, device=DEV)
        y = torch.randn((h, d, r), dtype=torch.bfloat16, device=DEV)
        xf, yf = quantize(x, y, b, h, r, d)
        z = torch.empty((b, h, d), dtype=torch.bfloat16, device=DEV)
        deep_gemm.fp8_einsum("bhr,hdr->bhd", xf, yf, z)

        ref = torch.einsum("bhr,hdr->bhd", x.double(), y.double())
        dx, dy = dequant(xf, yf, r, d)
        ref_sim = torch.einsum("bhr,hdr->bhd", dx, dy)      # fp32，只保留量化误差
        print(f"{b:>5} {h:>4} {r:>5} {d:>5} | {reldiff(z, ref):>15.3e} "
              f"{reldiff(z, ref_sim):>12.3e} {reldiff(ref_sim, ref):>10.3e}")

# ---------------------------------------------------------------------------
# B. 「每个头一个 GEMM」：等价于逐头 FP8 bmm + 头独立性
# ---------------------------------------------------------------------------
print("\n[B] 每头独立 GEMM 语义（头=独立 batch）")
b, h, r, d = 32, 128, 512, 128
x = torch.randn((b, h, r), dtype=torch.bfloat16, device=DEV)
y = torch.randn((h, d, r), dtype=torch.bfloat16, device=DEV)
xf, yf = quantize(x, y, b, h, r, d)
z = torch.empty((b, h, d), dtype=torch.bfloat16, device=DEV)
deep_gemm.fp8_einsum("bhr,hdr->bhd", xf, yf, z)
dx, dy = dequant(xf, yf, r, d)
ref_bmm = torch.bmm(dx.permute(1, 0, 2), dy.permute(0, 2, 1)).permute(1, 0, 2)  # [h,b,r]@[h,r,d]
print(f"  vs 逐头 fp32 bmm(量化后) relerr = {reldiff(z, ref_bmm):.3e}（仅 bf16 输出舍入）")

# 独立性：把第 0 头的 B 全置零 -> 输出第 0 头全 0，其余头逐位不变
z_ref = z.clone()
y_q0 = yf[0].clone(); y_q0[0] = 0
yf0 = (y_q0, yf[1])
z2 = torch.empty_like(z)
deep_gemm.fp8_einsum("bhr,hdr->bhd", xf, yf0, z2)
head0_zero = bool((z2[:, 0, :] == 0).all().item())
others_same = bool(torch.equal(z2[:, 1:, :], z_ref[:, 1:, :]))
print(f"  头 0 的 B 置零 -> 输出头 0 全 0 = {head0_zero}；其余头逐位不变 = {others_same}")

# ---------------------------------------------------------------------------
# C. SF 布局硬约束：A per-token(1D)、B per-block(2D) 的形状
# ---------------------------------------------------------------------------
print("\n[C] SF 布局（量化器给出的形状，即 host 侧契约）")
b, h, r, d = 4, 8, 1024, 512
x = torch.randn((b, h, r), dtype=torch.bfloat16, device=DEV)
y = torch.randn((h, d, r), dtype=torch.bfloat16, device=DEV)
xf, yf = quantize(x, y, b, h, r, d)
print(f"  A fp8={tuple(xf[0].shape)} SF={tuple(xf[1].shape)}  "
      f"-> per-token: 每 (b,h) 行沿 r 每 128 一个 UE8M0")
print(f"  B fp8={tuple(yf[0].shape)} SF={tuple(yf[1].shape)}  "
      f"-> per-block: 每头 (d/128, r/128) 一个 SF")
# SFB 逐 batch 基址 = current_group_idx * (shape_n_sfb*shape_k_scales)  (scheduler/gemm.cuh:171)
shape_n_sfb, shape_k_scales = ceil_div(d, 128), ceil_div(r, 128)
print(f"  逐头 SFB 基址 stride = shape_n_sfb*shape_k_scales = "
      f"{shape_n_sfb}*{shape_k_scales} = {shape_n_sfb*shape_k_scales}（batch=h 的第 3 维偏移）")

# ---------------------------------------------------------------------------
# D. host 契约（DG_HOST_ASSERT）——SM90 专属拒绝项
# ---------------------------------------------------------------------------
print("\n[D] host 契约拒绝项")
b, h, r, d = 4, 8, 128, 64
x = torch.randn((b, h, r), dtype=torch.bfloat16, device=DEV)
y = torch.randn((h, d, r), dtype=torch.bfloat16, device=DEV)
xf, yf = quantize(x, y, b, h, r, d)
z = torch.empty((b, h, d), dtype=torch.bfloat16, device=DEV)


def try_call(label, fn):
    try:
        fn()
        print(f"  {label}: 未报错（意外）")
    except Exception as ex:
        print(f"  {label}: 被拒 -> {str(ex).splitlines()[0][:100]}")


# 1) SM90 不支持 FP8 (d,sfd) 输出对
try_call("FP8 输出对 on sm90 (einsum.hpp:224)",
         lambda: deep_gemm.fp8_einsum(
             "bhr,hdr->bhd", xf, yf,
             (torch.empty((b, h, d), dtype=torch.float8_e4m3fn, device=DEV),
              torch.empty_strided((b, ceil_div(h * d, 128)), (1, 4), dtype=torch.int32, device=DEV))))
# 2) A 非 K-major（stride(-1) != 1）
a_bad = xf[0].transpose(1, 2).contiguous().transpose(1, 2)
try_call("A 非 K-major (einsum.hpp:206/291)",
         lambda: deep_gemm.fp8_einsum("bhr,hdr->bhd", (a_bad, xf[1].transpose(1, 2).contiguous().transpose(1, 2)), yf, z))
# 3) gran_k != 128：SF 形状与 recipe 不符（先被 layout.hpp:119 拦），
#    或真正造出 gran-64 的 SF 会被 layout.hpp:57「Unknown SF transformation」拦；
#    SM90 bmm 另有 gran_k==128 硬断言（einsum.hpp:246）
try_call("recipe gran_k=64 (layout.hpp:57/119, einsum.hpp:246)",
         lambda: deep_gemm.fp8_einsum("bhr,hdr->bhd", xf, yf, z, recipe=(1, 64, 64)))
# 4) 形状不一致（B 的 h 与 A 不同）
y_badh = torch.randn((h + 1, d, r), dtype=torch.bfloat16, device=DEV)
yf_badh = (torch.empty_like(y_badh, dtype=torch.float8_e4m3fn),
           torch.empty((h + 1, ceil_div(d, 128), ceil_div(r, 128)), device=DEV, dtype=torch.float))
for i in range(h + 1):
    yf_badh[0][i], yf_badh[1][i] = per_block_cast_to_fp8(y_badh[i], use_ue8m0=True)
try_call("B 的 h 不匹配 (einsum.hpp:66)",
         lambda: deep_gemm.fp8_einsum("bhr,hdr->bhd", xf, yf_badh, z))
# 5) arch10 专属表达式
try_call("bhd,hdr->bhr on sm90 (einsum.hpp:296)",
         lambda: deep_gemm.fp8_einsum("bhd,hdr->bhr", xf, yf, z))

# ---------------------------------------------------------------------------
# E. 确定性开关：True -> 自家 gemm kernel；False -> cuBLASLt(nvjet)
# ---------------------------------------------------------------------------
print("\n[E] use_deterministic_algorithms（False 且 cuBLASLt 可用时走 nvjet）")
b, h, r, d = 128, 128, 512, 128
x = torch.randn((b, h, r), dtype=torch.bfloat16, device=DEV)
y = torch.randn((h, d, r), dtype=torch.bfloat16, device=DEV)
xf, yf = quantize(x, y, b, h, r, d)
z = torch.empty((b, h, d), dtype=torch.bfloat16, device=DEV)


def top_kernels(fn, n=3):
    fn(); torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
    evts = sorted(prof.key_averages(), key=lambda e: -e.device_time_total)
    return [e.key for e in evts[:n] if e.device_time_total > 0]


for det in (True, False):
    deep_gemm.use_deterministic_algorithms(det)
    t = bench(lambda: deep_gemm.fp8_einsum("bhr,hdr->bhd", xf, yf, z))
    ks = top_kernels(lambda: deep_gemm.fp8_einsum("bhr,hdr->bhd", xf, yf, z))
    print(f"  deterministic={det}: {t:8.1f} us  kernels={ks}")
deep_gemm.use_deterministic_algorithms(True)

# ---------------------------------------------------------------------------
# F. 性能：扫描 b（对齐 DeepGEMM test_einsum.py 的用例）
# ---------------------------------------------------------------------------
print("\n[F] 性能（FLOPs = 2*b*h*r*d，输入均已量化）")
print(f"  {'b':>6} {'h':>4} {'r':>5} {'d':>5} | {'deep_gemm us':>12} {'TFLOPS':>8} "
      f"{'GB/s':>8} | {'torch us':>10} {'x':>6}")
for (h, r, d) in ((8, 4096, 1024), (128, 512, 128)):
    for b in (4, 32, 128, 4096, 8192):
        x = torch.randn((b, h, r), dtype=torch.bfloat16, device=DEV)
        y = torch.randn((h, d, r), dtype=torch.bfloat16, device=DEV)
        xf, yf = quantize(x, y, b, h, r, d)
        z = torch.empty((b, h, d), dtype=torch.bfloat16, device=DEV)
        run_dg = lambda: deep_gemm.fp8_einsum("bhr,hdr->bhd", xf, yf, z)
        run_t = lambda: torch.einsum("bhr,hdr->bhd", x, y)
        t_dg = bench(run_dg)
        t_t = bench(run_t)
        flops = 2 * b * h * r * d
        nbytes = xf[0].numel() + xf[1].numel() * 4 + yf[0].numel() + yf[1].numel() * 4 + z.numel() * 2
        print(f"  {b:>6} {h:>4} {r:>5} {d:>5} | {t_dg:>12.1f} {flops/t_dg/1e6:>8.0f} "
              f"{nbytes/t_dg/1e3:>8.0f} | {t_t:>10.1f} {t_t/t_dg:>6.2f}")

print("\n完成。")
