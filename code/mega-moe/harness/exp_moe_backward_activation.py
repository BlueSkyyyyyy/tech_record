#!/usr/bin/env python3
"""P144-1 实验：MoE 反向「非 GEMM 段」——SwiGLU 反向 + FP8/UE8M0 重量化。

Part A：用 torch fp64 作独立 oracle，把这段的误差来源逐层拆开
        （精确 fp64 → 前向 bf16 round 的数据路径 → FP8 UE8M0 量化）。
Part B：把这段放回**反向 single-step 预算**——把第 15/132/135 篇量过的四个 grouped GEMM
        与本篇新量的激活反向段拼成一张表，看它占多少。

主 kernel（UMMA/tcgen05/SF、融合 kernel）SM100 专属、仅分析；本文件纯 host + torch、架构无关，
H100(sm90) 真机可跑。真实内核在 src/moe_backward_activation.cu（`.sh` 段一编译运行）。
"""
import os
import time

import torch


def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def fwd_path(gate, up, dact, w):
    """前向的数据路径：gate/up 先入 bf16（镜像 kernel `:1112-1113`），返回 fp64 的 dg/du。"""
    gf = gate.to(torch.bfloat16).to(torch.float64)
    uf = up.to(torch.bfloat16).to(torch.float64)
    s = sigmoid(gf)
    d_su = dact.to(torch.float64) * w
    dg = d_su * uf * s * (1.0 + gf * (1.0 - s))
    du = d_su * s * gf
    return dg, du


def exact_path(gate, up, dact, w):
    gf = gate.to(torch.float64)
    uf = up.to(torch.float64)
    s = sigmoid(gf)
    d_su = dact.to(torch.float64) * w
    return d_su * uf * s * (1.0 + gf * (1.0 - s)), d_su * s * gf


def ue8m0_sf_inv_fp32(amax):
    """复制 common/math.cuh:114-147 的 UE8M0 幂次（E4M3 档）——用位运算。"""
    bits = amax.contiguous().view(torch.int32).to(torch.int64)
    mask = (1 << 23) - 1
    qmax_mant = 0x60 << (23 - 7)
    rounded = (bits + mask - qmax_mant) >> 23
    sf = torch.clamp(rounded, min=105 + 8) - 8
    inv = ((254 - sf) << 23).to(torch.int32).view(torch.float32)
    return sf.to(torch.int32), inv


def quant_per32(x, group=32):
    """对最后一维按 group=32 做 UE8M0 缩放 + FP8 E4M3 量化，返回反量化后的 fp32。"""
    shp = x.shape
    x = x.reshape(-1, group).to(torch.float32)
    amax = x.abs().amax(dim=1, keepdim=True)
    _, inv = ue8m0_sf_inv_fp32(amax)
    q = (x * inv).to(torch.float8_e4m3fn).to(torch.float32) / inv
    return q.reshape(shp).to(torch.float64)


def relerr(a, b):
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def part_a():
    print("== Part A：SwiGLU 反向 + 量化的误差分解（torch fp64 oracle）==")
    torch.manual_seed(0)
    shapes = [(2048, 2048), (8192, 2048), (2048, 7168)]
    for T, I in shapes:
        gate = (torch.randn(T, I, device="cuda") * 1.5)
        up = (torch.randn(T, I, device="cuda") * 1.5)
        dact = (torch.randn(T, I, device="cuda"))
        w = (0.5 + 0.5 * torch.rand(T, 1, device="cuda")).to(torch.float64)
        dg_ex, du_ex = exact_path(gate, up, dact, w)          # 真·精确（不过 bf16）
        dg_bf, du_bf = fwd_path(gate, up, dact, w)            # 前向 bf16 数据路径
        dg_q = quant_per32(dg_bf)                             # + FP8 UE8M0
        du_q = quant_per32(du_bf)
        print(f"  T={T:5d} I={I:5d} | bf16-round: dg rel={relerr(dg_bf,dg_ex):.3e} "
              f"du rel={relerr(du_bf,du_ex):.3e} (+fp8: dg {relerr(dg_q,dg_ex):.3e} "
              f"du {relerr(du_q,du_ex):.3e}) | cos dg={cos(dg_q,dg_ex):.6f} "
              f"du={cos(du_q,du_ex):.6f}")
    print("  注：bf16-round 是**前向刻意的**（gate/up 先入 bf16），反向走 straight-through；"
          "FP8 量化是喂 wgrad 的 B 操作数所需。两者都远小于算子本身的 FP8 输入噪声。")
    print()


def part_b():
    print("== Part B：反向 single-step 预算（H100 实测拼图）==")
    # 四个 grouped GEMM：第 132 篇（=第 15 篇同一套单次 launch ncu）的 single-shot 数
    d2g, d1g = 111.0, 194.3
    w2g, w1g = 500.6, 997.0        # 第 135 篇（dW2 [M=2048,N=7168]、dW1 [M=7168,N=4096]，E=8）
    gemm = d2g + d1g + w2g + w1g
    # 激活反向段：src/moe_backward_activation.cu 在 T=2048,I=2048 实测（量化路径）
    act_seg = 20.0
    print(f"  四个 grouped GEMM（T=2048,H=7168,I=2048,E=8,topk=2，单次 launch）:")
    print(f"    d2g {d2g:7.1f}us  SwiGLU 反向+量化 {act_seg:6.1f}us  "
          f"d1g {d1g:7.1f}us  w2g {w2g:7.1f}us  w1g {w1g:7.1f}us")
    print(f"    GEMM 小计 {gemm:8.1f}us + 激活段 {act_seg:.1f}us = {gemm + act_seg:8.1f}us")
    print(f"    激活段占「GEMM+激活」的 {act_seg / (gemm + act_seg) * 100:.2f}%")
    print(f"  反向通信（第 104 篇，T=4096,k=8,EP=8）：两相位 a2a ~8000us（受争用下界）")
    print(f"    → 激活段相对整步（GEMM+激活+通信）占比 < {act_seg / (gemm + act_seg + 8000) * 100:.2f}%")
    print("  结论：激活反向/重量化段**存在但很小**（~1%），它和前向 L1 epilogue（第 142 篇 24.2us）")
    print("        同量级；反向 step 的钱仍在四个 grouped GEMM（wgrad 占 83%，第 132 篇）。")
    print()


def part_c():
    print("== Part C：CUDA 内核 vs torch 元素级实现（同形状，仅参考量级）==")
    T, I = 2048, 2048
    gate = (torch.randn(T, I, device="cuda") * 1.5).to(torch.bfloat16)
    up = (torch.randn(T, I, device="cuda") * 1.5).to(torch.bfloat16)
    dact = torch.randn(T, I, device="cuda").to(torch.bfloat16)
    w = (0.5 + 0.5 * torch.rand(T, 1, device="cuda"))

    def torch_bwd():
        gf = gate.float(); uf = up.float()
        s = torch.sigmoid(gf)
        d_su = dact.float() * w
        dg = d_su * uf * s * (1 + gf * (1 - s))
        du = d_su * s * gf
        return dg, du

    for _ in range(5):
        torch_bwd()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        torch_bwd()
    torch.cuda.synchronize()
    us = (time.time() - t0) / 50 * 1e6
    mb = T * I * (6 + 4) / 1e6
    print(f"  torch 元素级（无量化，读 3×bf16 写 2×fp32）：{us:.1f}us / {mb / (us*1e-6) / 1e3:.0f} GB/s")
    print(f"  CUDA 融合核（含 UE8M0 FP8 量化，读写更少）：19.98us / ~1679 GB/s（见段一 [C]）")
    print()


if __name__ == "__main__":
    print(f"torch {torch.__version__}, cuda {torch.version.cuda}, "
          f"device {torch.cuda.get_device_name(0)}")
    part_a()
    part_b()
    part_c()
