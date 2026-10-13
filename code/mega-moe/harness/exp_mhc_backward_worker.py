#!/usr/bin/env python3
# P122-1：mHC 反向「per-token 融合 worker」的**真实形状**账。
#
# 融合把 K1（系数反向）、G1a（dz=dAbase 的 dgrad 部分）、K2（混合反向）融进一个 block，
# 于是 `dAbase[T,D]` 与 `dz[T,D]` 两张全局中间张量**不再往返显存**。
# 本脚本：① 逐形状算省掉的字节；② 用 H100 实测内存带宽 → 投影省时。
# 主 kernel 是 SM100 专属、仅分析；这里量的是**架构无关**的显存流量账。
import torch, time

def bench_bw(nbytes, iters=100):
    a = torch.empty(nbytes // 4, dtype=torch.float32, device='cuda')
    b = torch.empty_like(a)
    a.normal_()
    for _ in range(10): b.copy_(a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): b.copy_(a)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return 2 * nbytes / dt / 1e9, dt

print("=" * 78)
print("P122-1  mHC 反向 per-token 融合 worker：全局往返账（H100）")
print("=" * 78)
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  dev {torch.cuda.get_device_name(0)}")

HC = 4
bw, dt = bench_bw(1 << 30)
print(f"\n[H100 实测] D2D copy（1 GiB/向，读+写）: {bw:.1f} GB/s（{dt*1e6:.1f} us）")

shapes = [(2048, 512), (8192, 2048), (16384, 7168)]
print("\n[逐形状] 融合省掉的全局往返 = dAbase[T,D] 与 dz[T,D] 各 (写+读) = 4*T*D*4 B")
print(f"{'T':>7} {'H':>6} {'D=hc*H':>8} {'省掉(GB)':>10} {'投影省时(us)':>13}")
for T, H in shapes:
    D = HC * H
    nbytes = 4 * T * D * 4
    print(f"{T:>7} {H:>6} {D:>8} {nbytes/1e9:>10.4f} {nbytes/bw/1e3:>13.1f}")

print("\n[对照第 131 篇] 真实形状 T=16384 H=7168 hc=4：")
print("  * 三段 GEMM 合计 3.04 ms、整条反向访存下界 3648.9 us（第 131 篇，按其假定的可达带宽）。")
print("  * 融合只搬走元素级中间张量，不动 GEMM；它省的是 K1->K2 之间的两张 [T,D]。")

print("\n[结论]")
print("  * 元素级三段的融合是真实收益：省掉两张 [T,D] fp32 中间张量的写+读，")
print("    真实形状下约 7.52 GB —— 这是与 GEMM 无关、纯属分段引入的额外流量。")
print("  * H=64 的 CU 微基准只量到 ~6% 墙钟收益：该形状 D=256 太小，被串行 Sinkhorn 支配；")
print("    收益随 D=hc*H 线性增长（表里 T 固定时随 H 增大而放大）。")
print("  * G0/G1b 仍是真实 GEMM、du0 仍落全局——融合边界止于必须跨 token 归约/必须喂 GEMM 之外。")
